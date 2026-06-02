# -*- coding: utf-8 -*-
"""
utils/golden_evaluation-new_method.py
======================================
Distributed Multi-GPU Evaluation untuk Seluruh Pipeline menggunakan metode hybrid baru (YOLO11l + SAM2).

Menggunakan model YOLO11l hasil training terbaru yang dideteksi secara dinamis dari workspace aktif.

PANDUAN EKSEKUSI AGAR TIDAK TERPUTUS (WORKFLOW TMUX + SLURM YANG BENAR):

1. Buat Sesi TMUX di Login Node (JANGAN MASUK NODE AI DULU):
   tmux new-session -s training_pipeline

2. Di dalam TMUX, hubungkan (attach) ke Node GPU (Slurm):
   cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
   ./attach_gpu.sh
   *(Anda akan seketika berada di dalam shell bash node komputasi, misal @ai2/@ai3, dengan conda env yolo_env aktif secara otomatis)*

3. Di dalam Node GPU, jalankan evaluasi secara langsung (Dinamis tanpa hardcode):
   
   # Deteksi ID workspace aktif secara dinamis:
   WS_ID=$(cat /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/.workspace_id)

   # Jalankan skrip evaluasi:
     cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && LOG_DIR=$(python3 -c "import config_shared, os; print(os.path.join(config_shared.WORKSPACE_DIR, 'logs'))") && mkdir -p "$LOG_DIR" && python3 -u utils/evaluation_hybrid_sota.py 2>&1 | tee "$LOG_DIR/4_evaluation_hybrid_sota.log"


4. Detach dari TMUX untuk meninggalkan proses di background (Aman jika terminal/laptop Anda ditutup):
   Tekan Ctrl+b, lalu tekan d (di Mac: tekan tombol control (^)+b lalu tekan d).
   (Untuk masuk kembali memantau proses nanti dari login node, jalankan: tmux attach -t training_pipeline)

"""

import os, sys, gc, csv, time, argparse, pickle, tempfile
import numpy as np
import cv2

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_UTILS_DIR      = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
import torch.multiprocessing as mp

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE, NUM_CLASSES,
    get_output_dir, REPORTS_DIR, DATA_FILES_DIR, GOLDEN_SEG_DATASET_LOCATION, GOLDEN_DET_DATASET_LOCATION,
    PAPER1_CSV_DIR, EVAL_CONF, EVAL_IOU
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools, load_native_coco_gt
)

SAM_MODEL_PATH        = os.path.join(ROOT, "models", "sam2.1_t.pt")
MOBILE_SAM_MODEL_PATH = os.path.join(ROOT, "models", "mobile_sam.pt")
YOLO11L_DET_PATH  = os.path.join(WORKSPACE_DIR, "runs", "yolo11l", "weights", "best.pt")
YOLO11L_SEG_PATH  = os.path.join(WORKSPACE_DIR, "runs", "yolo11l_seg", "weights", "best.pt")

MODELS_CONFIG = [
    # ── YOLO Detection (9 Models) ─────────────────────────────────────────────
    {"label": "YOLOv8m",      "key": "yolov8m",      "type": "yolo_det"},
    {"label": "YOLOv8x",      "key": "yolov8x",      "type": "yolo_det"},
    {"label": "YOLOv9m",      "key": "yolov9m",      "type": "yolo_det"},
    {"label": "YOLOv9e",      "key": "yolov9e",      "type": "yolo_det"},
    {"label": "YOLOv10m",     "key": "yolov10m",     "type": "yolo_det"},
    {"label": "YOLOv10x",     "key": "yolov10x",     "type": "yolo_det"},
    {"label": "YOLO11n",      "key": "yolo11n",      "type": "yolo_det"},
    {"label": "YOLO11l",      "key": "yolo11l",      "type": "yolo_det"},
    {"label": "YOLO11x",      "key": "yolo11x",      "type": "yolo_det"},

    # ── YOLO Segmentation (7 Models) ──────────────────────────────────────────
    {"label": "YOLOv8m-Seg",  "key": "yolov8m_seg",  "type": "yolo_seg"},
    {"label": "YOLOv8x-Seg",  "key": "yolov8x_seg",  "type": "yolo_seg"},
    {"label": "YOLOv9c-Seg",  "key": "yolov9c_seg",  "type": "yolo_seg"},
    {"label": "YOLOv9e-Seg",  "key": "yolov9e_seg",  "type": "yolo_seg"},
    {"label": "YOLO11n-Seg",  "key": "yolo11n_seg",  "type": "yolo_seg"},
    {"label": "YOLO11l-Seg",  "key": "yolo11l_seg",  "type": "yolo_seg"},
    {"label": "YOLO11x-Seg",  "key": "yolo11x_seg",  "type": "yolo_seg"},

    # ── Mask R-CNN (1 Model) ──────────────────────────────────────────────────
    {"label": "Mask R-CNN",   "key": "maskrcnn",     "type": "maskrcnn"},

    # ── Hybrid Detection (7 Models - YOLO Det + SAM2.1_t) ──────────────────────
    {"label": "Hybrid (YOLOv8m+SAM2.1_t)",  "key": "hybrid_yolov8m",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLOv8x+SAM2.1_t)",  "key": "hybrid_yolov8x",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLOv9m+SAM2.1_t)",  "key": "hybrid_yolov9m",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLOv9e+SAM2.1_t)",  "key": "hybrid_yolov9e",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLO11n+SAM2.1_t)",  "key": "hybrid_yolo11n",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLO11l+SAM2.1_t)",  "key": "hybrid_yolo11l",  "type": "hybrid_det"},
    {"label": "Hybrid (YOLO11x+SAM2.1_t)",  "key": "hybrid_yolo11x",  "type": "hybrid_det"},

    # ── Hybrid Segmentation (9 Models - YOLO Seg/Det + SAM2.1_t) ──────────────
    {"label": "Hybrid (YOLOv8m-Seg+SAM2.1_t)",  "key": "hybrid_yolov8m_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLOv8x-Seg+SAM2.1_t)",  "key": "hybrid_yolov8x_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLOv9c-Seg+SAM2.1_t)",  "key": "hybrid_yolov9c_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLOv9e-Seg+SAM2.1_t)",  "key": "hybrid_yolov9e_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLOv10m+SAM2.1_t)",     "key": "hybrid_yolov10m",     "type": "hybrid_seg"},
    {"label": "Hybrid (YOLOv10x+SAM2.1_t)",     "key": "hybrid_yolov10x",     "type": "hybrid_seg"},
    {"label": "Hybrid (YOLO11n-Seg+SAM2.1_t)",  "key": "hybrid_yolo11n_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLO11l-Seg+SAM2.1_t)",  "key": "hybrid_yolo11l_seg",  "type": "hybrid_seg"},
    {"label": "Hybrid (YOLO11x-Seg+SAM2.1_t)",  "key": "hybrid_yolo11x_seg",  "type": "hybrid_seg"},

    # ── Hybrid Detection (7 Models - YOLO Det + MOBILE SAM) ──────────────────────────
    {"label": "Hybrid (YOLOv8m+MobileSAM)",  "key": "hybrid_yolov8m_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLOv8x+MobileSAM)",  "key": "hybrid_yolov8x_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLOv9m+MobileSAM)",  "key": "hybrid_yolov9m_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLOv9e+MobileSAM)",  "key": "hybrid_yolov9e_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLO11n+MobileSAM)",  "key": "hybrid_yolo11n_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLO11l+MobileSAM)",  "key": "hybrid_yolo11l_mobile",  "type": "hybrid_det_mobile"},
    {"label": "Hybrid (YOLO11x+MobileSAM)",  "key": "hybrid_yolo11x_mobile",  "type": "hybrid_det_mobile"},

    # ── Hybrid Segmentation (9 Models - YOLO Seg/Det + MOBILE SAM) ──────────────────
    {"label": "Hybrid (YOLOv8m-Seg+MobileSAM)",  "key": "hybrid_yolov8m_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLOv8x-Seg+MobileSAM)",  "key": "hybrid_yolov8x_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLOv9c-Seg+MobileSAM)",  "key": "hybrid_yolov9c_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLOv9e-Seg+MobileSAM)",  "key": "hybrid_yolov9e_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLOv10m+MobileSAM)",     "key": "hybrid_yolov10m_mobile",     "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLOv10x+MobileSAM)",     "key": "hybrid_yolov10x_mobile",     "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLO11n-Seg+MobileSAM)",  "key": "hybrid_yolo11n_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLO11l-Seg+MobileSAM)",  "key": "hybrid_yolo11l_seg_mobile",  "type": "hybrid_seg_mobile"},
    {"label": "Hybrid (YOLO11x-Seg+MobileSAM)",  "key": "hybrid_yolo11x_seg_mobile",  "type": "hybrid_seg_mobile"},
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
        # print(f"  [GPU:{gpu_id}][MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

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

def _get_model_metrics(model_cfg: dict, is_seg: bool = False) -> dict:
    """Mengembalikan dict berisi weights_size_mb dan parameters_m."""
    result = {"weights_size_mb": "N/A", "parameters_m": "N/A"}
    try:
        mtype, mkey = model_cfg["type"], model_cfg["key"]
        if mtype in ["yolo_det", "yolo_seg"]:
            from ultralytics import YOLO
            if mkey == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif mkey == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            m = YOLO(pt)
            total_params = sum(p.nelement() for p in m.model.parameters())
            sz = sum(p.nelement() * p.element_size() for p in m.model.parameters()) + sum(b.nelement() * b.element_size() for b in m.model.buffers())
            del m; gc.collect()
            result["weights_size_mb"] = round(sz / 1024**2, 2)
            result["parameters_m"] = round(total_params / 1e6, 2)
        elif mtype == "maskrcnn":
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
            in_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
            m.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
            total_params = sum(p.nelement() for p in m.parameters())
            sz = sum(p.nelement() * p.element_size() for p in m.parameters()) + sum(b.nelement() * b.element_size() for b in m.buffers())
            del m; gc.collect()
            result["weights_size_mb"] = round(sz / 1024**2, 2)
            result["parameters_m"] = round(total_params / 1e6, 2)
        elif mtype in ["hybrid", "hybrid_det", "hybrid_seg"]:
            from ultralytics import YOLO, SAM
            # Pendekatan moderat untuk parameter/metrik hybrid SAM2
            yolo_key = mkey.replace("hybrid_", "")
            if yolo_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif yolo_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(yolo_key), "weights", "best.pt")
                
            y_m = YOLO(pt)
            s_m = SAM(SAM_MODEL_PATH)
            y_params = sum(p.nelement() for p in y_m.model.parameters())
            s_params = sum(p.nelement() for p in s_m.model.parameters())
            y_sz = sum(p.nelement() * p.element_size() for p in y_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in y_m.model.buffers())
            s_sz = sum(p.nelement() * p.element_size() for p in s_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in s_m.model.buffers())
            del y_m, s_m; gc.collect()
            result["weights_size_mb"] = round((y_sz + s_sz) / 1024**2, 2)
            result["parameters_m"] = round((y_params + s_params) / 1e6, 2)
        elif mtype in ["hybrid_det_mobile", "hybrid_seg_mobile"]:
            from ultralytics import YOLO, SAM
            # Parameter/metrik hybrid MobileSAM — path model berbeda dari SAM2
            yolo_key = mkey.replace("hybrid_", "").replace("_mobile", "")
            if yolo_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif yolo_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(yolo_key), "weights", "best.pt")

            y_m = YOLO(pt)
            s_m = SAM(MOBILE_SAM_MODEL_PATH)
            y_params = sum(p.nelement() for p in y_m.model.parameters())
            s_params = sum(p.nelement() for p in s_m.model.parameters())
            y_sz = sum(p.nelement() * p.element_size() for p in y_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in y_m.model.buffers())
            s_sz = sum(p.nelement() * p.element_size() for p in s_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in s_m.model.buffers())
            del y_m, s_m; gc.collect()
            result["weights_size_mb"] = round((y_sz + s_sz) / 1024**2, 2)
            result["parameters_m"] = round((y_params + s_params) / 1e6, 2)
    except Exception as e:
        print(f"  ⚠️ Gagal hitung ukuran/parameter: {e}")
    return result

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
            if mkey == "yolo11l":
                model_obj = YOLO(YOLO11L_DET_PATH)
            elif mkey == "yolo11l_seg":
                model_obj = YOLO(YOLO11L_SEG_PATH)
            else:
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
        elif mtype in ["hybrid_det", "hybrid_seg"]:
            from ultralytics import YOLO, SAM
            yolo_key = mkey.replace("hybrid_", "")
            if yolo_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif yolo_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(yolo_key), "weights", "best.pt")

            print(f"  [GPU:{gpu}] Hybrid SAM2 menggunakan {yolo_key} untuk generator prompt.")
            model_obj = YOLO(pt)
            sam_model = SAM(SAM_MODEL_PATH)
        elif mtype in ["hybrid_det_mobile", "hybrid_seg_mobile"]:
            from ultralytics import YOLO, SAM
            # Ekstrak yolo_key: hapus prefix 'hybrid_' dan suffix '_mobile'
            yolo_key = mkey.replace("hybrid_", "").replace("_mobile", "")
            if yolo_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif yolo_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(yolo_key), "weights", "best.pt")

            print(f"  [GPU:{gpu}] Hybrid MobileSAM menggunakan {yolo_key} untuk generator prompt.")
            model_obj = YOLO(pt)
            sam_model = SAM(MOBILE_SAM_MODEL_PATH)
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
        img_key = img_path if img_path in image_ids else os.path.basename(img_path)
        if img_key not in image_ids: continue
        img_id = image_ids[img_key]
        pred_boxes, pred_confs, pred_clss, masks_np = [], [], [], []
        spd_pre, spd_inf, spd_post = 0.0, 0.0, 0.0
        
        if mtype in ["yolo_det", "yolo_seg"]:
            res = model_obj.predict(img_path, conf=EVAL_CONF, iou=EVAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
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
            # 1. PREPROCESS
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pil = PILImage.open(img_path).convert("RGB")
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device_str)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            spd_pre = (t1 - t0) * 1000

            # 2. INFERENCE
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad(): preds = model_obj(t_img)[0]
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            spd_inf = (t3 - t2) * 1000

            # 3. POSTPROCESS
            torch.cuda.synchronize()
            t4 = time.perf_counter()
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
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            spd_post = (t5 - t4) * 1000
                    
        elif mtype in ["hybrid_det", "hybrid_seg", "hybrid_det_mobile", "hybrid_seg_mobile"]:
            res = model_obj.predict(img_path, conf=EVAL_CONF, iou=EVAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre = spd.get("preprocess", 0.0)
            yolo_inf = spd.get("inference", 0.0)
            yolo_post = spd.get("postprocess", 0.0)

            # Label backend SAM untuk pesan log yang informatif
            sam_backend = "MobileSAM" if "mobile" in mtype else "SAM2"

            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)

                # 1. SAM Inference (SAM2 atau MobileSAM, keduanya callable dengan API yang sama)
                torch.cuda.synchronize()
                sam_inf_st = time.perf_counter()
                try:
                    sam_res = sam_model(res.orig_img, bboxes=pred_boxes.tolist(), device=device_str, verbose=False)
                    torch.cuda.synchronize()
                    sam_inf_et = time.perf_counter()
                    sam_inf_time = (sam_inf_et - sam_inf_st) * 1000
                except Exception as e:
                    print(f"  [GPU:{gpu}] ⚠️ {sam_backend} inference error: {e}")
                    sam_res = None
                    sam_inf_time = 0.0

                # 2. SAM Postprocess
                torch.cuda.synchronize()
                sam_post_st = time.perf_counter()
                if sam_res and sam_res[0].masks is not None:
                    m_data = sam_res[0].masks.data.cpu().numpy()
                    H, W = sam_res[0].orig_img.shape[:2]
                    for m in m_data:
                        if m.shape != (H, W): m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_np.append(m > 0.5)
                else:
                    masks_np = [None] * len(pred_boxes)
                torch.cuda.synchronize()
                sam_post_et = time.perf_counter()
                sam_post_time = (sam_post_et - sam_post_st) * 1000

                spd_inf = yolo_inf + sam_inf_time
                spd_post = yolo_post + sam_post_time
            else:
                spd_inf = yolo_inf
                spd_post = yolo_post
        
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
    is_seg = "seg" in img_dir.lower()
    model_metrics = _get_model_metrics(model_cfg, is_seg)
    weights_mb = model_metrics["weights_size_mb"]
    params_m = model_metrics["parameters_m"]
    print(f"  [Model] Weights: {weights_mb} MB | Parameters: {params_m} M")

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
            os.remove(pkl)

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

    # Inisialisasi metrik
    mAP50_box = mAP50_95_box = recall_box = precision_box = "N/A"
    mAP50_mask = mAP50_95_mask = recall_mask = precision_mask = "N/A"
    
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(); coco_gt.dataset = coco_gt_dict; coco_gt.createIndex()
        
        if all_dt_bbox:
            coco_dt = coco_gt.loadRes(all_dt_bbox)
            eval_box = COCOeval(coco_gt, coco_dt, iouType="bbox")
            eval_box.evaluate(); eval_box.accumulate(); eval_box.summarize()
            mAP50_95_box = round(float(eval_box.stats[0]), 4)
            mAP50_box    = round(float(eval_box.stats[1]), 4)
            recall_box   = round(float(eval_box.stats[8]), 4)
            p_matrix = eval_box.eval['precision']
            p_iou50 = p_matrix[0, :, :, 0, 2]
            p_valid = p_iou50[p_iou50 > -1]
            precision_box = round(float(np.mean(p_valid)), 4) if len(p_valid) > 0 else "N/A"
            
        if all_dt_segm and is_seg:
            coco_dt_seg = coco_gt.loadRes(all_dt_segm)
            eval_seg = COCOeval(coco_gt, coco_dt_seg, iouType="segm")
            eval_seg.evaluate(); eval_seg.accumulate(); eval_seg.summarize()
            mAP50_95_mask = round(float(eval_seg.stats[0]), 4)
            mAP50_mask    = round(float(eval_seg.stats[1]), 4)
            recall_mask   = round(float(eval_seg.stats[8]), 4)
            p_matrix_seg = eval_seg.eval['precision']
            p_iou50_seg = p_matrix_seg[0, :, :, 0, 2]
            p_valid_seg = p_iou50_seg[p_iou50_seg > -1]
            precision_mask = round(float(np.mean(p_valid_seg)), 4) if len(p_valid_seg) > 0 else "N/A"
    except Exception as e:
        print(f"  ⚠️ COCOeval error: {e}")
        mAP50_box = mAP50_95_box = recall_box = precision_box = "ERR"
        mAP50_mask = mAP50_95_mask = recall_mask = precision_mask = "ERR"

    gpu_str = _gpu_report_str(gpu_ids)
    
    det_row = None
    if model_cfg["type"] in ["yolo_det", "hybrid_det", "hybrid_det_mobile"]:
        det_row = {
            "Model": model_cfg["label"],
            "Weights Size (MB)": weights_mb,
            "Parameters (M)": params_m,
            "mAP50-95": mAP50_95_box,
            "mAP50": mAP50_box,
            "Precision": precision_box,
            "Recall": recall_box,
            "Preprocess (ms)": avg_pre, "Inference (ms)": avg_inf, "Postprocess (ms)": avg_post,
            "Latency (ms)": lat, "FPS": fps, "GPUs": gpu_str, "Evaluator": "COCOeval (pycocotools)"
        }

    seg_row = None
    if mAP50_mask != "N/A" or model_cfg["type"] in ["yolo_seg", "maskrcnn", "hybrid_seg", "hybrid_seg_mobile"]:
        seg_row = {
            "Model": model_cfg["label"],
            "Weights Size (MB)": weights_mb,
            "Parameters (M)": params_m,
            "mAP50-95(Box)": mAP50_95_box,
            "mAP50(Box)": mAP50_box,
            "mAP50-95(Mask)": mAP50_95_mask,
            "mAP50(Mask)": mAP50_mask,
            "Precision(Mask)": precision_mask,
            "Recall(Mask)": recall_mask,
            "Preprocess (ms)": avg_pre, "Inference (ms)": avg_inf, "Postprocess (ms)": avg_post,
            "Latency (ms)": lat, "FPS": fps, "GPUs": gpu_str, "Evaluator": "COCOeval (pycocotools)"
        }
        
    return det_row, seg_row

# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Evaluation [New Method - Dual Path]")
    parser.add_argument("--gpus", type=str, default="all", help="Contoh: '0,1' atau 'all'.")
    parser.add_argument("--dataset", type=str, default="default", help="Path direktori atau file. 'default' menggunakan SEG_DATASET_LOCATION.")
    parser.add_argument("--coco", action="store_true", help="Paksa gunakan format COCO (mencari _annotations.coco.json)")
    parser.add_argument("--yolo", action="store_true", help="Paksa gunakan format YOLO (mencari data.yaml)")
    args = parser.parse_args()

    GPU_IDS = list(range(torch.cuda.device_count())) if args.gpus.strip().lower() == "all" else [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not torch.cuda.is_available() or not GPU_IDS: 
        print("❌ CUDA tidak tersedia atau tidak ada GPU."); sys.exit(1)

    n_avail = torch.cuda.device_count()
    for g in GPU_IDS:
        if g >= n_avail:
            print(f"❌ GPU {g} tidak tersedia (sistem punya {n_avail} GPU)."); sys.exit(1)

    print("=" * 65)
    print("  Distributed Multi-GPU Evaluation [New Method]")
    print("=" * 65)
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {len(GPU_IDS)}\n")

    def _prepare_dataset(raw_path, force_coco, force_yolo):
        ep = raw_path
        ds_n = ""
        is_nc = False
        
        if os.path.isdir(ep):
            ds_n = os.path.basename(os.path.normpath(ep))
            has_yaml = os.path.isfile(os.path.join(ep, "data.yaml"))
            has_coco = os.path.isfile(os.path.join(ep, "valid", "_annotations.coco.json"))
            if force_coco and has_coco:
                ep = os.path.join(ep, "valid", "_annotations.coco.json")
                is_nc = True
            elif force_coco and not has_coco:
                print(f"❌ Flag --coco aktif tapi JSON tidak ada di {ep}/valid/")
                sys.exit(1)
            elif force_yolo and has_yaml:
                ep = os.path.join(ep, "data.yaml")
            elif force_yolo and not has_yaml:
                print(f"❌ Flag --yolo aktif tapi data.yaml tidak ada di {ep}")
                sys.exit(1)
            elif has_coco:
                ep = os.path.join(ep, "valid", "_annotations.coco.json")
                is_nc = True
            elif has_yaml:
                ep = os.path.join(ep, "data.yaml")
            else:
                print(f"❌ Dataset tidak valid (tidak ada yaml/json) di: {ep}")
                sys.exit(1)
        elif os.path.isfile(ep):
            if ep.endswith('.yaml'): ds_n = os.path.basename(os.path.dirname(ep))
            elif ep.endswith('.json'):
                is_nc = True
                ds_n = os.path.basename(os.path.dirname(os.path.dirname(ep))) if "valid" in ep else os.path.basename(os.path.dirname(ep))
        else:
            print(f"❌ Path tidak ditemukan: {ep}"); sys.exit(1)
            
        if is_nc:
            gt_dict, ids = load_native_coco_gt(os.path.dirname(ep))
            i_dir = os.path.dirname(ep)
        else:
            gt_dict, ids = build_coco_ground_truth(ep, split="valid")
            i_dir = _resolve_img_dir(ep)
            
        if not gt_dict: print(f"❌ Gagal build GT untuk {ep}"); sys.exit(1)
        return {"eval_path": ep, "ds_name": ds_n, "is_native_coco": is_nc, "coco_gt_dict": gt_dict, "image_ids": ids, "img_dir": i_dir}

    if args.dataset == "default":
        print("\n[Setup] Mode Auto: Memuat Dataset Deteksi & Segmentasi secara terpisah...\n")
        det_info = _prepare_dataset(GOLDEN_DET_DATASET_LOCATION, args.coco, args.yolo)
        seg_info = _prepare_dataset(GOLDEN_SEG_DATASET_LOCATION, args.coco, args.yolo)
        suffix = "default"
    else:
        info = _prepare_dataset(args.dataset, args.coco, args.yolo)
        det_info = info
        seg_info = info
        suffix = info["ds_name"]

    print("=" * 65)
    print(f"  All Models — Distributed Multi-GPU Evaluation [New Method - Dual Path] [{suffix}]")
    print(f"  GPU : {GPU_IDS} (World size: {len(GPU_IDS)})")
    if args.dataset == "default":
        print(f"  Det Dataset: {det_info['eval_path']} (Native COCO: {det_info['is_native_coco']})")
        print(f"  Seg Dataset: {seg_info['eval_path']} (Native COCO: {seg_info['is_native_coco']})")
    else:
        print(f"  Dataset: {det_info['eval_path']} (Native COCO: {det_info['is_native_coco']})")
    print("=" * 65 + "\n")

    all_det_rows = []
    all_seg_rows = []

    with tempfile.TemporaryDirectory(prefix="multigpu_eval_") as tmp_dir:
        for mcfg in MODELS_CONFIG:
            if mcfg["type"] in ["yolo_det", "hybrid_det", "hybrid_det_mobile"]:
                target_info = det_info
            elif mcfg["type"] in ["yolo_seg", "maskrcnn", "hybrid_seg", "hybrid_seg_mobile"]:
                target_info = seg_info
            else:
                target_info = seg_info

            ep = target_info["eval_path"]

            if args.dataset != "default":
                is_det_dataset = "det" in ep.lower()
                is_seg_dataset = "seg" in ep.lower()
                if is_det_dataset and not is_seg_dataset:
                    if mcfg["type"] not in ["yolo_det", "hybrid_det", "hybrid_det_mobile"]:
                        print(f"⏭️  Skip {mcfg['label']} (bukan model deteksi)")
                        continue
                elif is_seg_dataset and not is_det_dataset:
                    if mcfg["type"] not in ["yolo_seg", "maskrcnn", "hybrid_seg", "hybrid_seg_mobile"]:
                        print(f"⏭️  Skip {mcfg['label']} (bukan model segmentasi)")
                        continue

            d_row, s_row = eval_model_distributed(mcfg, GPU_IDS, target_info["coco_gt_dict"], target_info["image_ids"], target_info["img_dir"], tmp_dir)
            if d_row: all_det_rows.append(d_row)
            if s_row: all_seg_rows.append(s_row)

    NEW_CSV_DIR = os.path.join(PAPER1_CSV_DIR, "new-method")
    os.makedirs(NEW_CSV_DIR, exist_ok=True)
    
    if all_det_rows:
        det_csv = os.path.join(NEW_CSV_DIR, "hybrid_sota_det.csv")
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_det_rows[0].keys()))
            w.writeheader(); w.writerows(all_det_rows)
        print(f"\n✅ Det Report [New Method]: {det_csv}")

    if all_seg_rows:
        seg_csv = os.path.join(NEW_CSV_DIR, "hybrid_sota_seg.csv")
        with open(seg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_seg_rows[0].keys()))
            w.writeheader(); w.writerows(all_seg_rows)
        print(f"✅ Seg Report [New Method]: {seg_csv}")

    try:
        msg = f"✅ <b>Hybrid SOTA Evaluation Finished</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>\n"
        if all_det_rows:
            msg += f"• Det Report: <code>{len(all_det_rows)} models</code>\n"
        if all_seg_rows:
            msg += f"• Seg Report: <code>{len(all_seg_rows)} models</code>\n"
        send_telegram_msg(msg)
    except Exception as e:
        print(f"⚠️ Gagal mengirim notifikasi Telegram: {e}")

    print("\n  [Visual] Note: Untuk visualisasi new-method silakan jalankan utils/evaluation_visuals_hybrid_sota.py")
