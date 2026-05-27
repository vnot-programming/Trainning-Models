# -*- coding: utf-8 -*-
from __future__ import annotations
"""
utils/eval_single_model.py
===========================
Skrip evaluasi sentral — pengganti semua `eval_multigpu.py` yang tersebar.

Tanggung jawab:
  1. COCOeval terdistribusi (mAP50, mAP50-95, Precision, Recall) via mp.spawn.
  2. Menulis CSV per-varian ke  : <WORKSPACE>/reports/pipeline/csv/<model>/<variant>_<type>.csv
  3. Menulis CSV kompilasi model : <WORKSPACE>/reports/pipeline/csv/kompilasi_<model>_<type>.csv
  4. Menulis CSV kompilasi ALL   : <WORKSPACE>/reports/pipeline/csv/kompilasi_ALL_<type>.csv
  5. Generate N gambar prediksi individual per sampel ke <WORKSPACE>/reports/pipeline/visuals/<model>/
  6. Generate 1 gambar panel (N varian + 1 Ground Truth) per sampel.

Cara menjalankan:
    # Evaluasi 1 keluarga model:
    python utils/eval_single_model.py --family yolo9 --gpus 0

    # Semua GPU:
    python utils/eval_single_model.py --family yolo8 --gpus all

    # Evaluasi semua keluarga (dipanggil oleh run_pipeline_parallel.py):
    python utils/eval_single_model.py --family yolo8 --gpus 0
    python utils/eval_single_model.py --family yolo9 --gpus 0
    python utils/eval_single_model.py --family yolov10 --gpus 0
    python utils/eval_single_model.py --family yolo11 --gpus 0
    python utils/eval_single_model.py --family maskrcnn --gpus 0
    python utils/eval_single_model.py --family hybrid --gpus 0

tmux:
    tmux new-session -d -s eval_yolo9 "source /data/programs/anaconda3/bin/activate && conda activate yolo_env && \\
      cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && \\
      python -u utils/eval_single_model.py --family yolo9 --gpus 0 2>&1 | tee data-files/$(cat .workspace_id)/logs/eval_single_yolo9.log"

Output CSV:
    <WORKSPACE>/reports/pipeline/csv/<model>/<variant>_detection.csv
    <WORKSPACE>/reports/pipeline/csv/<model>/<variant>_segmentation.csv
    <WORKSPACE>/reports/pipeline/csv/kompilasi_<model>_detection.csv
    <WORKSPACE>/reports/pipeline/csv/kompilasi_<model>_segmentation.csv
    <WORKSPACE>/reports/pipeline/csv/kompilasi_ALL_detection.csv
    <WORKSPACE>/reports/pipeline/csv/kompilasi_ALL_segmentation.csv

Output Visuals:
    <WORKSPACE>/reports/pipeline/visuals/<model>/<sample>_<variant>.jpg
    <WORKSPACE>/reports/pipeline/visuals/<model>/<sample>_<family>_panel.jpg
"""

import os
import sys
import gc
import csv
import time
import pickle
import tempfile
import argparse
from collections import Counter

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT       = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
import torch.multiprocessing as mp
import numpy as np
import cv2

from config_shared import (
    WORKSPACE_DIR, DET_YAML, SEG_YAML, IMAGE_SIZE, NUM_CLASSES,
    get_output_dir, MODEL_COLORS, IMAGE_SAMPLES_DIR,
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools,
)

# ==============================================================================
# DEFINISI KELUARGA MODEL
# ==============================================================================

# Setiap varian: (model_key, model_label, task_type, yaml)
# task_type: "det" | "seg" | "maskrcnn" | "hybrid"
FAMILY_VARIANTS: dict[str, list[dict]] = {
    "yolo8": [
        {"key": "yolov8m",     "label": "YOLOv8m",     "type": "det", "yaml": DET_YAML},
        {"key": "yolov8m_seg", "label": "YOLOv8m-Seg", "type": "seg", "yaml": SEG_YAML},
        {"key": "yolov8x",     "label": "YOLOv8x",     "type": "det", "yaml": DET_YAML},
        {"key": "yolov8x_seg", "label": "YOLOv8x-Seg", "type": "seg", "yaml": SEG_YAML},
    ],
    "yolo9": [
        {"key": "yolov9m",     "label": "YOLOv9m",     "type": "det", "yaml": DET_YAML},
        {"key": "yolov9c_seg", "label": "YOLOv9c-Seg", "type": "seg", "yaml": SEG_YAML},
        {"key": "yolov9e",     "label": "YOLOv9e",     "type": "det", "yaml": DET_YAML},
        {"key": "yolov9e_seg", "label": "YOLOv9e-Seg", "type": "seg", "yaml": SEG_YAML},
    ],
    "yolov10": [
        {"key": "yolov10m", "label": "YOLOv10m", "type": "det", "yaml": DET_YAML},
        {"key": "yolov10x", "label": "YOLOv10x", "type": "det", "yaml": DET_YAML},
    ],
    "yolo11": [
        {"key": "yolo11n",     "label": "YOLO11n",     "type": "det", "yaml": DET_YAML},
        {"key": "yolo11n_seg", "label": "YOLO11n-Seg", "type": "seg", "yaml": SEG_YAML},
        {"key": "yolo11l",     "label": "YOLO11l",     "type": "det", "yaml": DET_YAML},
        {"key": "yolo11l_seg", "label": "YOLO11l-Seg", "type": "seg", "yaml": SEG_YAML},
        {"key": "yolo11x",     "label": "YOLO11x",     "type": "det", "yaml": DET_YAML},
        {"key": "yolo11x_seg", "label": "YOLO11x-Seg", "type": "seg", "yaml": SEG_YAML},
    ],
    "maskrcnn": [
        {"key": "maskrcnn", "label": "Mask R-CNN ResNet-50", "type": "maskrcnn", "yaml": SEG_YAML},
    ],
    "hybrid": [
        {"key": "hybrid", "label": "Hybrid (YOLO11l+SAM2)", "type": "hybrid", "yaml": SEG_YAML},
    ],
}

# Warna per varian untuk visualisasi (BGR, cycling)
_PANEL_COLORS = [
    (0, 255, 0),    # Hijau
    (255, 0, 0),    # Biru
    (0, 0, 255),    # Merah
    (0, 255, 255),  # Kuning
    (255, 0, 255),  # Magenta
    (255, 128, 0),  # Oranye
]

SAM_MODEL_PATH = os.path.join(ROOT, "models", "sam2.1_t.pt")

# ==============================================================================
# PATH HELPERS
# ==============================================================================

def _csv_dir(family: str) -> str:
    """Folder CSV per model: <WORKSPACE>/reports/pipeline/csv/<family>/"""
    d = os.path.join(WORKSPACE_DIR, "reports", "pipeline", "csv", family)
    os.makedirs(d, exist_ok=True)
    return d


def _csv_root() -> str:
    """Root folder CSV: <WORKSPACE>/reports/pipeline/csv/"""
    d = os.path.join(WORKSPACE_DIR, "reports", "pipeline", "csv")
    os.makedirs(d, exist_ok=True)
    return d


def _visuals_dir(family: str) -> str:
    """Folder visuals per model: <WORKSPACE>/reports/pipeline/visuals/<family>/"""
    d = os.path.join(WORKSPACE_DIR, "reports", "pipeline", "visuals", family)
    os.makedirs(d, exist_ok=True)
    return d


def _csv_path_variant(family: str, variant_key: str, task_type: str) -> str:
    """Path CSV per varian: csv/<family>/<variant_key>_<detection|segmentation>.csv"""
    suffix = "detection" if task_type == "det" else "segmentation"
    fname  = f"{variant_key}_{suffix}.csv"
    return os.path.join(_csv_dir(family), fname)


def _csv_path_kompilasi(family: str, task_type: str) -> str:
    """Path CSV kompilasi per-model."""
    suffix = "detection" if task_type == "det" else "segmentation"
    return os.path.join(_csv_root(), f"kompilasi_{family}_{suffix}.csv")


def _csv_path_all(task_type: str) -> str:
    """Path CSV kompilasi ALL."""
    suffix = "detection" if task_type == "det" else "segmentation"
    return os.path.join(_csv_root(), f"kompilasi_ALL_{suffix}.csv")


# ==============================================================================
# GPU / INFERENCE HELPERS
# ==============================================================================

def _flush_gpu(rank: int, label: str):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(rank)
    print(f"  [GPU:{rank}][MemFlush] {label} ✅", flush=True)


def _gpu_report_str(gpu_ids: list) -> str:
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
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Tidak ditemukan valid/ atau test/images di {base}")


def _avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else "N/A"


# ==============================================================================
# DDP WORKER — inference per GPU
# ==============================================================================

def _infer_worker(rank: int, gpu_ids: list, variant: dict, img_dir: str,
                  image_ids: dict, tmp_dir: str):
    """
    Worker universal: menjalankan inference YOLO-det/seg, MaskRCNN, atau Hybrid.
    Menyimpan prediksi ke pickle untuk dikumpulkan di proses utama.
    """
    gpu        = gpu_ids[rank]
    device_str = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    mtype = variant["type"]
    mkey  = variant["key"]
    label = variant["label"]

    print(f"  [GPU:{gpu}] Rank {rank} loading {label}...", flush=True)

    model_obj  = None
    sam_model  = None

    try:
        if mtype in ("det", "seg"):
            from ultralytics import YOLO
            pt = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            model_obj = YOLO(pt)
        elif mtype == "maskrcnn":
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            model_obj = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box  = model_obj.roi_heads.box_predictor.cls_score.in_features
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
            pickle.dump({"preds": [], "n_imgs": 0, "pre": [], "inf": [], "post": []}, f)
        return

    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []
    t_pre, t_inf, t_post = [], [], []

    for img_path in subset:
        if img_path not in image_ids:
            continue
        img_id = image_ids[img_path]
        pred_boxes, pred_confs, pred_clss, masks_np = [], [], [], []
        spd_pre = spd_inf = spd_post = 0.0

        try:
            if mtype in ("det", "seg"):
                res = model_obj.predict(img_path, conf=0.001, iou=0.6,
                                        imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
                spd = res.speed
                spd_pre  = spd.get("preprocess",  0.0)
                spd_inf  = spd.get("inference",   0.0)
                spd_post = spd.get("postprocess", 0.0)
                if res.boxes is not None and len(res.boxes) > 0:
                    pred_boxes = res.boxes.xyxy.cpu().numpy()
                    pred_confs = res.boxes.conf.cpu().numpy()
                    pred_clss  = res.boxes.cls.cpu().numpy().astype(int)
                    if mtype == "seg" and res.masks is not None:
                        m_data = res.masks.data.cpu().numpy()
                        H, W   = res.orig_img.shape[:2]
                        for m in m_data:
                            if m.shape != (H, W):
                                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                            masks_np.append(m > 0.5)

            elif mtype == "maskrcnn":
                import torchvision.transforms.functional as TF
                from PIL import Image as PILImage
                pre_st = time.perf_counter()
                pil    = PILImage.open(img_path).convert("RGB")
                t_img  = TF.to_tensor(pil).unsqueeze(0).to(device_str)
                spd_pre = (time.perf_counter() - pre_st) * 1000
                inf_st  = time.perf_counter()
                with torch.no_grad():
                    out = model_obj(t_img)[0]
                spd_inf  = (time.perf_counter() - inf_st) * 1000
                post_st  = time.perf_counter()
                scores   = out["scores"].cpu().numpy()
                keep     = scores >= 0.001
                pred_boxes = out["boxes"].cpu().numpy()[keep]
                pred_confs = scores[keep]
                pred_clss  = out["labels"].cpu().numpy()[keep] - 1
                if "masks" in out:
                    m_data = out["masks"].cpu().numpy()[keep, 0]
                    H, W   = pil.size[1], pil.size[0]
                    for m in m_data:
                        if m.shape != (H, W):
                            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_np.append(m > 0.5)
                spd_post = (time.perf_counter() - post_st) * 1000

            elif mtype == "hybrid":
                res = model_obj.predict(img_path, conf=0.001, iou=0.6,
                                        imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
                spd = res.speed
                spd_pre  = spd.get("preprocess",  0.0)
                spd_inf  = spd.get("inference",   0.0)
                spd_post = spd.get("postprocess", 0.0)
                if res.boxes is not None and len(res.boxes) > 0:
                    pred_boxes = res.boxes.xyxy.cpu().numpy()
                    pred_confs = res.boxes.conf.cpu().numpy()
                    pred_clss  = res.boxes.cls.cpu().numpy().astype(int)
                    try:
                        sam_res = sam_model.predict(res.orig_img, bboxes=res.boxes.xyxy, verbose=False)
                        spd_inf += (time.perf_counter()) * 0  # timing sudah diambil dari YOLO
                        if sam_res and sam_res[0].masks is not None:
                            m_data = sam_res[0].masks.data.cpu().numpy()
                            H, W   = sam_res[0].orig_img.shape[:2]
                            for m in m_data:
                                if m.shape != (H, W):
                                    m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                                masks_np.append(m > 0.5)
                    except Exception as e_sam:
                        print(f"  [GPU:{gpu}] ⚠️ SAM2 error: {e_sam}", flush=True)

        except Exception as e_inf:
            print(f"  [GPU:{gpu}] ⚠️ Inferensi gagal untuk {img_path}: {e_inf}", flush=True)
            continue

        t_pre.append(spd_pre); t_inf.append(spd_inf); t_post.append(spd_post)

        from pycocotools import mask as maskUtils
        for i in range(len(pred_boxes)):
            score  = float(pred_confs[i])
            cat_id = int(pred_clss[i]) + 1
            x1, y1, x2, y2 = pred_boxes[i].tolist()
            entry = {
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       score,
            }
            if i < len(masks_np) and masks_np[i] is not None:
                rle = maskUtils.encode(np.asfortranarray(masks_np[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                entry["segmentation"] = rle
            preds.append(entry)

    with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
        pickle.dump({"preds": preds, "n_imgs": len(subset),
                     "pre": t_pre, "inf": t_inf, "post": t_post}, f)

    del model_obj, sam_model
    _flush_gpu(gpu, f"{label} rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


# ==============================================================================
# PRECISION & RECALL (IoU@0.5, manual)
# ==============================================================================

def _calc_prec_recall(all_preds: list, coco_gt_dict: dict) -> tuple:
    gt_per_class = {}
    gt_boxes_all = []
    for ann in coco_gt_dict.get("annotations", []):
        cid = ann["category_id"]
        gt_per_class[cid] = gt_per_class.get(cid, 0) + 1
        b = ann["bbox"]
        gt_boxes_all.append((ann["image_id"], cid, [b[0], b[1], b[0]+b[2], b[1]+b[3]]))

    if not gt_per_class:
        return 0.0, 0.0

    num_classes = max(gt_per_class.keys()) + 1
    total_tp = total_fp = total_gt = 0

    for cid in range(num_classes):
        cls_preds = [p for p in all_preds if p.get("category_id") == cid]
        total_gt += gt_per_class.get(cid, 0)
        if not cls_preds:
            continue
        cls_gts    = [(img_id, b) for img_id, c, b in gt_boxes_all if c == cid]
        gt_matched = set()
        for pred in sorted(cls_preds, key=lambda x: x.get("score", 0), reverse=True):
            if pred.get("score", 0) < 0.5:
                continue
            pb = pred["bbox"]
            px1, py1, pw, ph = pb[0], pb[1], pb[2], pb[3]
            px2, py2 = px1 + pw, py1 + ph
            best_iou = best_idx = -1
            for idx, (img_id, gb) in enumerate(cls_gts):
                if pred["image_id"] != img_id:
                    continue
                gx1, gy1, gx2, gy2 = gb[0], gb[1], gb[2], gb[3]
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ua = pw * ph + (gx2 - gx1) * (gy2 - gy1) - ia
                iou = ia / ua if ua > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou; best_idx = idx
            if best_iou >= 0.5 and best_idx not in gt_matched:
                total_tp += 1; gt_matched.add(best_idx)
            else:
                total_fp += 1

    prec = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    rec  = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0
    return prec, rec


# ==============================================================================
# EVALUASI SATU VARIAN (COCOeval terdistribusi)
# ==============================================================================

def eval_variant(variant: dict, gpu_ids: list, tmp_dir: str) -> tuple[dict | None, dict | None]:
    """
    Evaluasi satu varian model.
    Returns: (det_row, seg_row) — masing-masing bisa None jika task tidak relevan.
    """
    mkey  = variant["key"]
    label = variant["label"]
    mtype = variant["type"]
    yaml  = variant["yaml"]

    print("\n" + "=" * 65)
    print(f"  Evaluasi: {label}")
    print("=" * 65)

    pt_path = os.path.join(get_output_dir(mkey), "weights", "best.pt")

    if mtype not in ("maskrcnn", "hybrid") and not os.path.exists(pt_path):
        msg = f"best.pt tidak ditemukan: {pt_path}"
        print(f"  ❌ {msg}")
        send_telegram_msg(f"❌ <b>{label} Eval</b>\n{msg}")
        return None, None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return None, None

    # Build COCO GT sesuai task (det → DET_YAML, seg → SEG_YAML)
    eval_yaml = DET_YAML if mtype == "det" else yaml
    coco_gt_dict, image_ids = build_coco_ground_truth(eval_yaml, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT")
        return None, None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(eval_yaml)
    world_size = len(gpu_ids)

    # Clear temp pickles dari run sebelumnya
    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"rank{r}.pkl")
        if os.path.exists(pkl):
            os.remove(pkl)

    print(f"  [Spawn] {world_size} GPU worker secara paralel...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker,
        args=(gpu_ids, variant, img_dir, image_ids, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall

    # Kumpulkan prediksi
    all_preds  = []
    all_pre    = []; all_inf = []; all_post = []
    total_imgs = 0
    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            all_preds.extend(data["preds"])
            total_imgs += data["n_imgs"]
            all_pre.extend(data.get("pre", []))
            all_inf.extend(data.get("inf", []))
            all_post.extend(data.get("post", []))
            os.remove(pkl)

    if not all_preds:
        print("  ⚠️ Tidak ada prediksi yang berhasil dikumpulkan.")

    fps    = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_ms = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"
    avg_pre  = _avg(all_pre)
    avg_inf  = _avg(all_inf)
    avg_post = _avg(all_post)
    gpu_str  = _gpu_report_str(gpu_ids)
    size_mb  = round(os.path.getsize(pt_path) / 1e6, 2) if os.path.exists(pt_path) else "N/A"

    print(f"  [Speed] Pre={avg_pre}ms | Inf={avg_inf}ms | Post={avg_post}ms")
    print(f"  [Throughput] {fps} FPS | Latency={lat_ms}ms")

    det_row = seg_row = None

    # ── Detection metrics ─────────────────────────────────────────────────────
    dt_bbox = [p for p in all_preds if "bbox" in p]
    if dt_bbox or mtype == "det":
        try:
            mAP50, mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
        except Exception as e:
            print(f"  ⚠️ COCOeval bbox error: {e}")
            mAP50 = mAP50_95 = "ERR"
        prec, rec = _calc_prec_recall(all_preds, coco_gt_dict)
        print(f"  ✅ mAP50={mAP50}  mAP50-95={mAP50_95}  P={prec}  R={rec}")
        det_row = {
            "Model":             label,
            "Model Size (MB)":   size_mb,
            "mAP50-95":          mAP50_95,
            "mAP50":             mAP50,
            "Precision":         prec,
            "Recall":            rec,
            "Preprocess (ms)":   avg_pre,
            "Inference (ms)":    avg_inf,
            "Postprocess (ms)":  avg_post,
            "Latency (ms)":      lat_ms,
            "FPS":               fps,
            "GPUs":              gpu_str,
            "Evaluator":         "COCOeval (DDP)",
        }

    # ── Segmentation metrics ──────────────────────────────────────────────────
    dt_segm = [p for p in all_preds if "segmentation" in p]
    if dt_segm or mtype in ("seg", "maskrcnn", "hybrid"):
        try:
            mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(
                coco_gt_dict, image_ids, all_preds, iou_type="segm"
            )
            _, mAP50_95_box = evaluate_coco_predictions(
                coco_gt_dict, image_ids, all_preds, iou_type="bbox"
            )
        except Exception as e:
            print(f"  ⚠️ COCOeval segm error: {e}")
            mAP50_mask = mAP50_95_mask = mAP50_95_box = "ERR"
        print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")
        seg_row = {
            "Model":             label,
            "Model Size (MB)":   size_mb,
            "mAP50-95(Box)":     mAP50_95_box,
            "mAP50-95(Mask)":    mAP50_95_mask,
            "Preprocess (ms)":   avg_pre,
            "Inference (ms)":    avg_inf,
            "Postprocess (ms)":  avg_post,
            "Latency (ms)":      lat_ms,
            "FPS":               fps,
            "GPUs":              gpu_str,
            "Evaluator":         "COCOeval (DDP)",
        }

    return det_row, seg_row


# ==============================================================================
# CSV WRITERS
# ==============================================================================

_DET_FIELDS = [
    "Model", "Model Size (MB)", "mAP50-95", "mAP50",
    "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
    "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator",
]
_SEG_FIELDS = [
    "Model", "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)",
    "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)",
    "Latency (ms)", "FPS", "GPUs", "Evaluator",
]


def _write_csv(path: str, rows: list, fields: list, mode: str = "w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = (mode == "w") or (not os.path.exists(path))
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def save_variant_csv(family: str, variant: dict, det_row, seg_row):
    """Simpan CSV per-varian (satu baris per file)."""
    mtype = variant["type"]
    mkey  = variant["key"]
    if det_row:
        path = _csv_path_variant(family, mkey, "det")
        _write_csv(path, [det_row], _DET_FIELDS)
        print(f"  📄 Det CSV: {path}")
    if seg_row:
        path = _csv_path_variant(family, mkey, "seg")
        _write_csv(path, [seg_row], _SEG_FIELDS)
        print(f"  📄 Seg CSV: {path}")


def save_kompilasi_csv(family: str, all_det: list, all_seg: list):
    """Simpan CSV kompilasi per-model."""
    if all_det:
        path = _csv_path_kompilasi(family, "det")
        _write_csv(path, all_det, _DET_FIELDS)
        print(f"  📊 Kompilasi Det CSV [{family}]: {path}")
    if all_seg:
        path = _csv_path_kompilasi(family, "seg")
        _write_csv(path, all_seg, _SEG_FIELDS)
        print(f"  📊 Kompilasi Seg CSV [{family}]: {path}")


def append_to_all_csv(det_rows: list, seg_rows: list):
    """Append ke CSV kompilasi_ALL (append mode agar bisa dipanggil multi-kali)."""
    if det_rows:
        path = _csv_path_all("det")
        mode = "a" if os.path.exists(path) else "w"
        _write_csv(path, det_rows, _DET_FIELDS, mode=mode)
        print(f"  🌐 kompilasi_ALL_detection.csv diperbarui.")
    if seg_rows:
        path = _csv_path_all("seg")
        mode = "a" if os.path.exists(path) else "w"
        _write_csv(path, seg_rows, _SEG_FIELDS, mode=mode)
        print(f"  🌐 kompilasi_ALL_segmentation.csv diperbarui.")


# ==============================================================================
# VISUAL GENERATOR
# ==============================================================================

def _get_class_names(yaml_path: str) -> list:
    try:
        import yaml as _yaml
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return list(names)
    except Exception:
        return []


def _draw_predictions(img_bgr: np.ndarray, boxes, masks, confs, clss,
                      color: tuple, class_names: list) -> np.ndarray:
    """Gambar bounding box, mask, dan label pada gambar (in-place copy)."""
    out = img_bgr.copy()
    if masks is not None:
        overlay = out.copy()
        for m in masks:
            colored = np.zeros_like(out, dtype=np.uint8)
            colored[m > 0.5] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
        out = overlay

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf  = float(confs[i]) if confs is not None else 1.0
        cls   = int(clss[i]) if clss is not None else 0
        cname = class_names[cls % len(class_names)] if class_names else f"cls{cls}"
        label_txt = f"{cname} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        txt_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(out, label_txt, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)
    return out


def _render_single_variant(variant: dict, img_path: str, device_str: str,
                            class_names: list, color: tuple) -> np.ndarray:
    """Render prediksi satu varian model pada satu gambar."""
    mtype = variant["type"]
    mkey  = variant["key"]

    try:
        if mtype in ("det", "seg"):
            from ultralytics import YOLO
            pt    = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            model = YOLO(pt)
            res   = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                  device=device_str, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
            confs = res.boxes.conf.cpu().numpy() if res.boxes else []
            clss  = res.boxes.cls.cpu().numpy() if res.boxes else []
            masks = None
            if mtype == "seg" and res.masks is not None:
                H, W  = res.orig_img.shape[:2]
                masks = np.array([
                    cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    for m in res.masks.data.cpu().numpy()
                ])
            del model; gc.collect(); torch.cuda.empty_cache()
            img_bgr = cv2.imread(img_path)
            return _draw_predictions(img_bgr, boxes, masks, confs, clss, color, class_names)

        elif mtype == "maskrcnn":
            import torchvision
            import torchvision.transforms.functional as TF
            from PIL import Image as PILImage
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            device = torch.device(device_str)
            model  = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
            in_msk = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_msk, 256, NUM_CLASSES + 1)
            pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
            model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
            model.to(device).eval()
            pil   = PILImage.open(img_path).convert("RGB")
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t_img)[0]
            scores = out["scores"].cpu().numpy()
            keep   = scores >= 0.5
            boxes  = out["boxes"].cpu().numpy()[keep]
            confs  = scores[keep]
            clss   = out["labels"].cpu().numpy()[keep] - 1
            masks  = None
            if "masks" in out:
                H, W  = pil.size[1], pil.size[0]
                masks = np.array([
                    cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    for m in out["masks"].cpu().numpy()[keep, 0]
                ])
            del model; gc.collect(); torch.cuda.empty_cache()
            img_bgr = cv2.imread(img_path)
            return _draw_predictions(img_bgr, boxes, masks, confs, clss, color, class_names)

        elif mtype == "hybrid":
            from ultralytics import YOLO, SAM
            yolo  = YOLO(os.path.join(get_output_dir("yolo11l"), "weights", "best.pt"))
            sam   = SAM(SAM_MODEL_PATH)
            res   = yolo.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                 device=device_str, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if (res.boxes and len(res.boxes) > 0) else []
            confs = res.boxes.conf.cpu().numpy() if (res.boxes and len(res.boxes) > 0) else []
            clss  = res.boxes.cls.cpu().numpy() if (res.boxes and len(res.boxes) > 0) else []
            masks = None
            if len(boxes) > 0:
                try:
                    sr = sam.predict(res.orig_img, bboxes=res.boxes.xyxy, verbose=False)
                    if sr and sr[0].masks is not None:
                        H, W  = sr[0].orig_img.shape[:2]
                        masks = np.array([
                            cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                            for m in sr[0].masks.data.cpu().numpy()
                        ])
                except Exception as e:
                    print(f"  ⚠️ SAM2 visual error: {e}")
            del yolo, sam; gc.collect(); torch.cuda.empty_cache()
            img_bgr = cv2.imread(img_path)
            return _draw_predictions(img_bgr, boxes, masks, confs, clss, color, class_names)

    except Exception as e:
        print(f"  ⚠️ Render gagal untuk {variant['label']}: {e}")

    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((640, 640, 3), dtype=np.uint8)
    return img


def _draw_ground_truth(img_path: str, yaml_path: str) -> np.ndarray:
    """Gambar ground truth dari label YOLO (txt) atau COCO JSON."""
    import yaml as _yaml

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return np.zeros((640, 640, 3), dtype=np.uint8)
    H, W = img_bgr.shape[:2]

    # Cari label .txt (YOLO format) di dataset valid/
    try:
        base    = os.path.dirname(yaml_path)
        lbl_dir = os.path.join(base, "valid", "labels")
        if not os.path.isdir(lbl_dir):
            lbl_dir = os.path.join(base, "test", "labels")

        stem    = os.path.splitext(os.path.basename(img_path))[0]
        lbl_txt = os.path.join(lbl_dir, stem + ".txt")

        if os.path.exists(lbl_txt):
            with open(lbl_txt) as f:
                lines = f.read().strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                parts  = list(map(float, line.split()))
                cls    = int(parts[0])
                cx, cy = parts[1] * W, parts[2] * H
                bw, bh = parts[3] * W, parts[4] * H
                x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2); y2 = int(cy + bh / 2)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img_bgr, f"GT cls{cls}", (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print(f"  ⚠️ GT render warning: {e}")

    cv2.putText(img_bgr, "Ground Truth", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def generate_visuals(family: str, variants: list, gpu_ids: list):
    """
    Generate visualisasi untuk semua sampel:
    - N gambar prediksi individual per sampel
    - 1 gambar panel (N varian + 1 Ground Truth) per sampel
    """
    import matplotlib.pyplot as plt

    if not os.path.isdir(IMAGE_SAMPLES_DIR):
        print(f"  ⚠️ IMAGE_SAMPLES_DIR tidak ditemukan: {IMAGE_SAMPLES_DIR}")
        return

    samples = sorted([
        os.path.join(IMAGE_SAMPLES_DIR, f)
        for f in os.listdir(IMAGE_SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not samples:
        print(f"  ⚠️ Tidak ada gambar sampel di: {IMAGE_SAMPLES_DIR}")
        return

    device_str = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    vis_dir    = _visuals_dir(family)
    # Gunakan yaml dari varian pertama untuk GT
    gt_yaml    = variants[0]["yaml"] if variants else SEG_YAML
    class_names = _get_class_names(gt_yaml)

    n_variants = len(variants)
    # Panel columns: N varian + 1 GT
    n_cols = n_variants + 1

    print(f"\n[Visual] Generating {len(samples)} sampel × ({n_variants} varian + 1 GT) = {len(samples) * n_cols} gambar...")

    for idx, img_path in enumerate(samples, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Render setiap varian secara individual
        panels = []
        for vi, variant in enumerate(variants):
            color = _PANEL_COLORS[vi % len(_PANEL_COLORS)]
            rendered = _render_single_variant(variant, img_path, device_str, class_names, color)
            panels.append(rendered)

            # Simpan gambar individual
            out_path = os.path.join(vis_dir, f"{stem}_{variant['key']}.jpg")
            cv2.imwrite(out_path, rendered)
            print(f"  [{idx}/{len(samples)}] Individual → {os.path.basename(out_path)}")

        # 2. Ground Truth panel
        gt_panel = _draw_ground_truth(img_path, gt_yaml)

        # 3. Panel grid (matplotlib): semua varian + GT
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        for vi, (variant, panel) in enumerate(zip(variants, panels)):
            axes[vi].imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            axes[vi].set_title(variant["label"], fontsize=12, fontweight="bold")
            axes[vi].axis("off")
        # GT di kolom terakhir
        axes[-1].imshow(cv2.cvtColor(gt_panel, cv2.COLOR_BGR2RGB))
        axes[-1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[-1].axis("off")

        plt.tight_layout()
        panel_path = os.path.join(vis_dir, f"{stem}_{family}_panel.jpg")
        plt.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{idx}/{len(samples)}] Panel    → {os.path.basename(panel_path)}")

    print(f"\n✅ Visuals selesai → {vis_dir}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi sentral per keluarga model (COCOeval + Visualisasi)")
    parser.add_argument("--family", type=str, required=True,
                        choices=list(FAMILY_VARIANTS.keys()),
                        help="Keluarga model yang dievaluasi. Contoh: yolo9, yolo11, maskrcnn, hybrid")
    parser.add_argument("--gpus", type=str, default="all",
                        help="GPU yang digunakan. Contoh: '0,1' atau 'all'.")
    parser.add_argument("--skip-eval",    action="store_true", help="Lewati COCOeval (hanya visual).")
    parser.add_argument("--skip-visuals", action="store_true", help="Lewati generate visual (hanya eval).")
    args = parser.parse_args()

    # Resolve GPU IDs
    if args.gpus.strip().lower() == "all":
        GPU_IDS = list(range(torch.cuda.device_count()))
    else:
        GPU_IDS = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    if not torch.cuda.is_available() or not GPU_IDS:
        print("❌ CUDA tidak tersedia atau tidak ada GPU yang dipilih.")
        sys.exit(1)

    n_avail = torch.cuda.device_count()
    for g in GPU_IDS:
        if g >= n_avail:
            print(f"❌ GPU {g} tidak tersedia (sistem memiliki {n_avail} GPU).")
            sys.exit(1)

    family   = args.family
    variants = FAMILY_VARIANTS[family]

    print("=" * 70)
    print(f"  eval_single_model.py — Family: {family.upper()}")
    print(f"  GPU         : {GPU_IDS}")
    print(f"  World size  : {len(GPU_IDS)}")
    print(f"  Varian      : {[v['label'] for v in variants]}")
    print(f"  WORKSPACE   : {WORKSPACE_DIR}")
    print("=" * 70 + "\n")

    gc.collect()
    torch.cuda.empty_cache()

    send_telegram_msg(
        f"🚀 <b>eval_single_model [{family}] Dimulai</b>\n"
        f"GPUs: <code>{GPU_IDS}</code>\n"
        f"Varian: <code>{[v['label'] for v in variants]}</code>"
    )

    all_det_rows: list = []
    all_seg_rows: list = []

    # ── FASE 1: COCOeval per varian ───────────────────────────────────────────
    if not args.skip_eval:
        with tempfile.TemporaryDirectory(prefix=f"eval_{family}_") as tmp_dir:
            t0 = time.perf_counter()
            for variant in variants:
                send_telegram_msg(
                    f"🔍 <b>{variant['label']} Eval</b>\nMemulai DDP inference...",
                    force=False
                )
                det_row, seg_row = eval_variant(variant, GPU_IDS, tmp_dir)

                # Simpan CSV per-varian
                save_variant_csv(family, variant, det_row, seg_row)

                if det_row:
                    all_det_rows.append(det_row)
                    send_telegram_msg(
                        f"✅ <b>{variant['label']} Det Selesai</b>\n"
                        f"mAP50-95: <code>{det_row['mAP50-95']}</code>  "
                        f"mAP50: <code>{det_row['mAP50']}</code>\n"
                        f"P: <code>{det_row['Precision']}</code>  R: <code>{det_row['Recall']}</code>"
                    )
                if seg_row:
                    all_seg_rows.append(seg_row)
                    send_telegram_msg(
                        f"✅ <b>{variant['label']} Seg Selesai</b>\n"
                        f"mAP50-95(Box): <code>{seg_row['mAP50-95(Box)']}</code>\n"
                        f"mAP50-95(Mask): <code>{seg_row['mAP50-95(Mask)']}</code>"
                    )

            total_eval = round(time.perf_counter() - t0, 1)
            print(f"\n✅ COCOeval [{family}] selesai dalam {total_eval}s")

        # Simpan CSV kompilasi per-model
        save_kompilasi_csv(family, all_det_rows, all_seg_rows)

        # Append ke kompilasi ALL
        append_to_all_csv(all_det_rows, all_seg_rows)

    # ── FASE 2: Generate Visuals ──────────────────────────────────────────────
    if not args.skip_visuals:
        generate_visuals(family, variants, GPU_IDS)

    # ── Selesai ───────────────────────────────────────────────────────────────
    send_telegram_msg(
        f"🏁 <b>eval_single_model [{family}] Selesai</b>\n"
        f"Det rows: <code>{len(all_det_rows)}</code>  "
        f"Seg rows: <code>{len(all_seg_rows)}</code>\n"
        f"CSV: <code>{_csv_root()}</code>\n"
        f"Visuals: <code>{_visuals_dir(family)}</code>"
    )
    print(f"\n✅ eval_single_model [{family}] — DONE")
