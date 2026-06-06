# -*- coding: utf-8 -*-
"""
hybrid/eval_multigpu.py
========================
Distributed Multi-GPU Evaluation untuk Seluruh Pipeline (YOLO, Mask R-CNN, Hybrid).

Strategi Evaluasi Terdistribusi:
  - Setiap GPU memuat model (YOLO, Mask R-CNN, atau Hybrid) secara terpisah
  - Setiap GPU memproses subset gambar yang berbeda (data parallelism)
  - Hasil prediksi dikumpulkan dan dievaluasi via COCOeval di proses utama

python hybrid/eval_multigpu.py --dataset /home/my/Trainning-Models/MyFineTunning-RunPOD/datasets/me-bottle-isempty-ku3-h61lr-2-yolov11/data.yaml

tmux new-session -d -s eval_multigpu "cd Trainning-Models/MyFineTunning-RunPOD && source .venv/bin/activate && python3 -u hybrid/eval_multigpu.py --dataset /home/my/Trainning-Models/MyFineTunning-RunPOD/datasets/me-bottle-isempty-ku3-h61lr-2-yolov11/data.yaml 2>&1 | tee hybrid/eval_multigpu-h61lr-2-yolov11.log"

"""

import os, sys, gc, csv, time, argparse, pickle, tempfile
import numpy as np
import cv2

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HYBRID_DIR      = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_HYBRID_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
import torch.multiprocessing as mp

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE, NUM_CLASSES,
    get_output_dir, REPORTS_DIR, DATA_FILES_DIR
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools, load_native_coco_gt
)

SAM_MODEL_PATH = os.path.join(ROOT, "models", "sam2.1_t.pt")

MODELS_CONFIG = [
    {"label": "YOLOv8m", "key": "yolov8m", "type": "yolo_det"},
    {"label": "YOLOv9m", "key": "yolov9m", "type": "yolo_det"},
    {"label": "YOLO11l", "key": "yolo11l", "type": "yolo_det"},
    {"label": "YOLOv8m-Seg", "key": "yolov8m_seg", "type": "yolo_seg"},
    {"label": "YOLOv9c-Seg", "key": "yolov9c_seg", "type": "yolo_seg"},
    {"label": "YOLO11l-Seg", "key": "yolo11l_seg", "type": "yolo_seg"},
    {"label": "Mask R-CNN ResNet-50", "key": "maskrcnn", "type": "maskrcnn"},
    {"label": "Hybrid (YOLO11l+SAM2)", "key": "hybrid", "type": "hybrid"}
]

# ==============================================================================
# HELPERS
# ==============================================================================

def _flush_gpu(gpu_id: int, label: str):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu_id)
        free, total = torch.cuda.mem_get_info(gpu_id)
        print(f"  [GPU:{gpu_id}][MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

def _gpu_report_str(gpu_ids: list) -> str:
    from collections import Counter
    names  = [torch.cuda.get_device_name(i) for i in gpu_ids]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())

def _partition_images(img_dir: str, rank: int, world_size: int) -> list:
    all_imgs = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return all_imgs[rank::world_size]

def _resolve_img_dir(yaml_path: str) -> str:
    base = os.path.dirname(yaml_path)
    for split in ("valid", "test"):
        d = os.path.join(base, split, "images")
        if os.path.isdir(d): return d
    raise FileNotFoundError(f"Tidak ditemukan valid/ atau test/images di {base}")

def _get_model_size_mb(model_cfg: dict):
    try:
        mtype, mkey = model_cfg["type"], model_cfg["key"]
        if mtype in ["yolo_det", "yolo_seg"]:
            from ultralytics import YOLO
            pt = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            m = YOLO(pt)
            sz = sum(p.nelement() * p.element_size() for p in m.model.parameters()) + sum(b.nelement() * b.element_size() for b in m.model.buffers())
            del m; gc.collect()
            return round(sz / 1024**2, 2)
        elif mtype == "maskrcnn":
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
            in_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
            m.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
            sz = sum(p.nelement() * p.element_size() for p in m.parameters()) + sum(b.nelement() * b.element_size() for b in m.buffers())
            del m; gc.collect()
            return round(sz / 1024**2, 2)
        elif mtype == "hybrid":
            from ultralytics import YOLO, SAM
            y_m = YOLO(os.path.join(get_output_dir("yolo11l"), "weights", "best.pt"))
            s_m = SAM(SAM_MODEL_PATH)
            y_sz = sum(p.nelement() * p.element_size() for p in y_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in y_m.model.buffers())
            s_sz = sum(p.nelement() * p.element_size() for p in s_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in s_m.model.buffers())
            del y_m, s_m; gc.collect()
            return round((y_sz + s_sz) / 1024**2, 2)
    except Exception as e:
        print(f"  ⚠️ Gagal hitung ukuran: {e}")
        return "N/A"

# ==============================================================================
# WORKER: INFERENCE PER GPU
# ==============================================================================

def _infer_worker(rank: int, gpu_ids: list, model_cfg: dict, img_dir: str, image_ids: dict, tmp_dir: str):
    gpu = gpu_ids[rank]
    device_str = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    mtype, mkey, label = model_cfg["type"], model_cfg["key"], model_cfg["label"]
    
    print(f"  [GPU:{gpu}] Rank {rank} loading {label}...", flush=True)
    model_obj, sam_model = None, None
    try:
        if mtype in ["yolo_det", "yolo_seg"]:
            from ultralytics import YOLO
            model_obj = YOLO(os.path.join(get_output_dir(mkey), "weights", "best.pt"))
        elif mtype == "maskrcnn":
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            model_obj = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box = model_obj.roi_heads.box_predictor.cls_score.in_features
            model_obj.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
            in_mask = model_obj.roi_heads.mask_predictor.conv5_mask.in_channels
            model_obj.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
            pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
            model_obj.load_state_dict(torch.load(pt, map_location=device_str, weights_only=True))
            model_obj.to(device_str).eval()
        elif mtype == "hybrid":
            from ultralytics import YOLO, SAM
            model_obj = YOLO(os.path.join(get_output_dir("yolo11l"), "weights", "best.pt"))
            sam_model = SAM(SAM_MODEL_PATH)
    except Exception as e:
        print(f"  [GPU:{gpu}] ❌ Gagal load {label}: {e}", flush=True)
        with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
            pickle.dump({"dt_bbox": [], "dt_segm": [], "n_imgs": 0, "pre": [], "inf": [], "post": []}, f)
        return

    subset = _partition_images(img_dir, rank, len(gpu_ids))
    dt_bbox, dt_segm = [], []
    t_pre, t_inf, t_post = [], [], []
    
    from pycocotools import mask as maskUtils
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage
    
    for img_path in subset:
        if img_path not in image_ids: continue
        img_id = image_ids[img_path]
        pred_boxes, pred_confs, pred_clss, masks_np = [], [], [], []
        spd_pre, spd_inf, spd_post = 0.0, 0.0, 0.0
        
        if mtype in ["yolo_det", "yolo_seg"]:
            res = model_obj.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre, spd_inf, spd_post = spd.get("preprocess", 0.0), spd.get("inference", 0.0), spd.get("postprocess", 0.0)
            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)
                if mtype == "yolo_seg" and res.masks is not None:
                    m_data = res.masks.data.cpu().numpy()
                    H, W = res.orig_img.shape[:2]
                    for m in m_data:
                        if m.shape != (H, W): m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_np.append(m > 0.5)
                        
        elif mtype == "maskrcnn":
            pil = PILImage.open(img_path).convert("RGB")
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device_str)
            inf_st = time.perf_counter()
            with torch.no_grad(): preds = model_obj(t_img)[0]
            spd_inf = (time.perf_counter() - inf_st) * 1000
            scores = preds["scores"].cpu().numpy()
            keep = scores >= 0.001
            pred_boxes = preds["boxes"].cpu().numpy()[keep]
            pred_confs = scores[keep]
            pred_clss = preds["labels"].cpu().numpy()[keep] - 1
            if "masks" in preds:
                m_data = preds["masks"].cpu().numpy()[keep, 0]
                H, W = pil.size[1], pil.size[0]
                for m in m_data:
                    if m.shape != (H, W): m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_np.append(m > 0.5)
                    
        elif mtype == "hybrid":
            res = model_obj.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre, spd_inf, spd_post = spd.get("preprocess", 0.0), spd.get("inference", 0.0), spd.get("postprocess", 0.0)
            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)
                sam_st = time.perf_counter()
                try:
                    sam_res = sam_model.predict(res.orig_img, bboxes=res.boxes.xyxy, verbose=False)
                    spd_inf += (time.perf_counter() - sam_st) * 1000
                    if sam_res and sam_res[0].masks is not None:
                        m_data = sam_res[0].masks.data.cpu().numpy()
                        H, W = sam_res[0].orig_img.shape[:2]
                        for m in m_data:
                            if m.shape != (H, W): m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                            masks_np.append(m > 0.5)
                except:
                    masks_np = [None] * len(pred_boxes)
        
        t_pre.append(spd_pre); t_inf.append(spd_inf); t_post.append(spd_post)
        for i in range(len(pred_boxes)):
            score = float(pred_confs[i])
            cat_id = int(pred_clss[i]) + 1
            x1, y1, x2, y2 = pred_boxes[i].tolist()
            dt_bbox.append({"image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
            if i < len(masks_np) and masks_np[i] is not None:
                rle = maskUtils.encode(np.asfortranarray(masks_np[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                dt_segm.append({"image_id": img_id, "category_id": cat_id, "segmentation": rle, "score": score})

    with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
        pickle.dump({"dt_bbox": dt_bbox, "dt_segm": dt_segm, "pre": t_pre, "inf": t_inf, "post": t_post, "n_imgs": len(subset)}, f)

    del model_obj, sam_model
    _flush_gpu(gpu, f"{label} rank{rank}")

# ==============================================================================
# EVALUASI UTAMA PER MODEL
# ==============================================================================

def eval_model_distributed(model_cfg: dict, gpu_ids: list, coco_gt_dict: dict, image_ids: dict, img_dir: str, tmp_dir: str) -> tuple:
    print(f"\n" + "="*65)
    print(f"  Evaluating: {model_cfg['label']} (Multi-GPU)")
    print("="*65)

    world_size = len(gpu_ids)
    mb_size = _get_model_size_mb(model_cfg)
    print(f"  [Model] Size: {mb_size} MB")

    t_wall_start = time.perf_counter()
    mp.spawn(_infer_worker, args=(gpu_ids, model_cfg, img_dir, image_ids, tmp_dir), nprocs=world_size, join=True)
    total_wall = time.perf_counter() - t_wall_start

    all_dt_bbox, all_dt_segm = [], []
    all_pre, all_inf, all_post = [], [], []
    total_imgs = 0

    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f: data = pickle.load(f)
            all_dt_bbox.extend(data["dt_bbox"])
            all_dt_segm.extend(data["dt_segm"])
            all_pre.extend(data.get("pre", []))
            all_inf.extend(data.get("inf", []))
            all_post.extend(data.get("post", []))
            total_imgs += data["n_imgs"]
            os.remove(pkl)  # Clean up

    if not all_dt_bbox and not all_dt_segm:
        print("  ❌ Tidak ada prediksi.")
        return None, None

    fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"
    avg_pre = round(sum(all_pre)/len(all_pre), 2) if all_pre else "N/A"
    avg_inf = round(sum(all_inf)/len(all_inf), 2) if all_inf else "N/A"
    avg_post = round(sum(all_post)/len(all_post), 2) if all_post else "N/A"

    print(f"  [Speed] Pre={avg_pre}ms | Inf={avg_inf}ms | Post={avg_post}ms")
    print(f"  [Throughput] {fps} FPS | Latency={lat}ms")

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(); coco_gt.dataset = coco_gt_dict; coco_gt.createIndex()
        
        if all_dt_bbox:
            coco_dt = coco_gt.loadRes(all_dt_bbox)
            eval_box = COCOeval(coco_gt, coco_dt, iouType="bbox")
            eval_box.evaluate(); eval_box.accumulate(); eval_box.summarize()
            mAP50_box = round(float(eval_box.stats[1]), 4)
            mAP50_95_box = round(float(eval_box.stats[0]), 4)
        else:
            mAP50_box = mAP50_95_box = "N/A"
            
        if all_dt_segm:
            coco_dt_seg = coco_gt.loadRes(all_dt_segm)
            eval_seg = COCOeval(coco_gt, coco_dt_seg, iouType="segm")
            eval_seg.evaluate(); eval_seg.accumulate(); eval_seg.summarize()
            mAP50_mask = round(float(eval_seg.stats[1]), 4)
            mAP50_95_mask = round(float(eval_seg.stats[0]), 4)
        else:
            mAP50_mask = mAP50_95_mask = "N/A"
    except Exception as e:
        print(f"  ⚠️ COCOeval error: {e}")
        mAP50_box = mAP50_95_box = mAP50_mask = mAP50_95_mask = "ERR"

    gpu_str = _gpu_report_str(gpu_ids)
    
    det_row = {
        "Model": model_cfg["label"],
        "Model Size (MB)": mb_size,
        "mAP50-95": mAP50_95_box,
        "mAP50": mAP50_box,
        "Precision": "N/A", "Recall": "N/A",
        "Preprocess (ms)": avg_pre, "Inference (ms)": avg_inf, "Postprocess (ms)": avg_post,
        "Latency (ms)": lat, "FPS": fps, "GPUs": gpu_str, "Evaluator": "COCOeval (MultiGPU)"
    }
    
    seg_row = None
    if mAP50_mask != "N/A" or model_cfg["type"] in ["yolo_seg", "maskrcnn", "hybrid"]:
        seg_row = {
            "Model": model_cfg["label"],
            "Model Size (MB)": mb_size,
            "mAP50-95(Box)": mAP50_95_box,
            "mAP50-95(Mask)": mAP50_95_mask,
            "Preprocess (ms)": avg_pre, "Inference (ms)": avg_inf, "Postprocess (ms)": avg_post,
            "Latency (ms)": lat, "FPS": fps, "GPUs": gpu_str, "Evaluator": "COCOeval (MultiGPU)"
        }
        
    return det_row, seg_row

# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Evaluation untuk Seluruh Model")
    parser.add_argument("--gpus", type=str, default="all", help="Contoh: '0,1' atau 'all'.")
    parser.add_argument("--dataset", type=str, default="default", help="Path absolut ke data.yaml dataset evaluasi. Atau 'default' untuk menggunakan SEG_YAML dari config_shared.")
    args = parser.parse_args()

    GPU_IDS = list(range(torch.cuda.device_count())) if args.gpus.strip().lower() == "all" else [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not torch.cuda.is_available() or not GPU_IDS: print("❌ CUDA tidak tersedia."); sys.exit(1)

    eval_path = SEG_YAML if args.dataset == "default" else args.dataset
    
    ds_name = ""
    is_native_coco = False
    
    if os.path.isfile(eval_path) and eval_path.endswith('.yaml'):
        ds_name = os.path.basename(os.path.dirname(eval_path))
    elif os.path.isfile(eval_path) and eval_path.endswith('.json'):
        is_native_coco = True
        ds_name = os.path.basename(os.path.dirname(os.path.dirname(eval_path))) if "valid" in eval_path else os.path.basename(os.path.dirname(eval_path))
    elif os.path.isdir(eval_path):
        if os.path.isfile(os.path.join(eval_path, "data.yaml")):
            eval_path = os.path.join(eval_path, "data.yaml")
            ds_name = os.path.basename(os.path.dirname(eval_path))
        elif os.path.isfile(os.path.join(eval_path, "valid", "_annotations.coco.json")):
            eval_path = os.path.join(eval_path, "valid", "_annotations.coco.json")
            is_native_coco = True
            ds_name = os.path.basename(os.path.dirname(os.path.dirname(eval_path)))
        else:
            print(f"❌ Dataset tidak valid (tidak ada data.yaml atau valid/_annotations.coco.json): {eval_path}")
            sys.exit(1)
    else:
        print(f"❌ Path dataset tidak ditemukan: {eval_path}")
        sys.exit(1)

    suffix = "default" if args.dataset == "default" else ds_name

    print("=" * 65)
    print(f"  All Models — Distributed Multi-GPU Evaluation [{suffix}]")
    print(f"  GPU : {GPU_IDS} (World size: {len(GPU_IDS)})")
    print(f"  Dataset: {eval_path} (Native COCO: {is_native_coco})")
    print("=" * 65 + "\n")

    if is_native_coco:
        coco_gt_dict, image_ids = load_native_coco_gt(os.path.dirname(eval_path))
        img_dir = os.path.dirname(eval_path)
    else:
        coco_gt_dict, image_ids = build_coco_ground_truth(eval_path, split="valid")
        img_dir = _resolve_img_dir(eval_path)

    if not coco_gt_dict:
        print("❌ Gagal build COCO ground truth.")
        sys.exit(1)

    all_det_rows = []
    all_seg_rows = []

    with tempfile.TemporaryDirectory(prefix="multigpu_eval_") as tmp_dir:
        for mcfg in MODELS_CONFIG:
            d_row, s_row = eval_model_distributed(mcfg, GPU_IDS, coco_gt_dict, image_ids, img_dir, tmp_dir)
            if d_row: all_det_rows.append(d_row)
            if s_row: all_seg_rows.append(s_row)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if all_det_rows:
        det_csv = os.path.join(REPORTS_DIR, f"report_all_det_multigpu_{suffix}.csv")
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_det_rows[0].keys()))
            w.writeheader(); w.writerows(all_det_rows)
        print(f"\n✅ Det Report: {det_csv}")

    if all_seg_rows:
        seg_csv = os.path.join(REPORTS_DIR, f"report_all_seg_multigpu_{suffix}.csv")
        with open(seg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_seg_rows[0].keys()))
            w.writeheader(); w.writerows(all_seg_rows)
        print(f"✅ Seg Report: {seg_csv}")

    # ------ Generate Comparison Grid (Training Dataset) ------
    print("\n" + "="*65 + "\n  Generating Comparison Grid\n" + "="*65)
    try:
        import subprocess
        subprocess.run([sys.executable, "-u", os.path.join(ROOT, "utils", "generate_comparison_grid.py")], check=False)
    except Exception as e:
        print(f"⚠️ Gagal memanggil generate_comparison_grid: {e}")

