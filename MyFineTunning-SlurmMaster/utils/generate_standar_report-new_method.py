# -*- coding: utf-8 -*-
"""
utils/generate_standar_report-new_method.py
================================================================================
SKRIP GABUNGAN ORKESTRASI EVALUASI KUANTITATIF (CSV) & KUALITATIF (VISUALISASI)
================================================================================
Penulis: Google Professional Full Stack Developer Persona (Antigravity AI Agent)
Proyek: Fine-Tuning & Benchmarking 49 Model YOLO & Hybrid (SAM2) — Scopus Q1/Q2

Konsep Utama:
------------
Skrip ini menggabungkan fungsi evaluasi kuantitatif multi-GPU (mAP COCOeval) 
dan visualisasi grid kualitatif secara terpadu dan independen, hanya bergantung pada
config_shared.py dan datasets lokal.

Alur Kerja:
----------
1. Tahap 1 (Kuantitatif): Mengukur mAP50 dan mAP50-95 secara paralel multi-GPU 
   untuk YOLO, Mask R-CNN, dan Hybrid (YOLO+SAM2) pada dataset validasi standar.
   Hasil akhirnya disimpan ke laporan CSV di folder reports/paper1/csv/new-method/.
2. Tahap 2 (Kualitatif): Membaca gambar dari folder sampel tetap,
   melakukan rendering visual (masker semi-transparan, bounding box, label teks kontras),
   serta memplot grid komparasi dual-generasi (YOLOv8 dan YOLO11) ke folder visuals/.

Cara menjalankan:
----------------
Hubungkan terminal Anda ke Compute Node GPU yang telah dibooking:
$ cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils && ./attach_gpu.sh

Jalankan skrip terpadu ini:
$ python3 -u utils/generate_standar_report-new_method.py --gpus 0

atau 
cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && LOG_DIR=$(python3 -c "import config_shared, os; print(os.path.join(config_shared.WORKSPACE_DIR, 'logs'))") && mkdir -p "$LOG_DIR" && python3 -u utils/generate_standar_report-new_method.py --gpus 0 2>&1 | tee "$LOG_DIR/2_generate_standar_report-new_method.log"
"""

import os
import sys
import gc
import csv
import json
import time
import pickle
import argparse
import tempfile
import shutil
import tarfile
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ==============================================================================
# PATH SETUP — resolve ROOT dari lokasi file ini
# ==============================================================================
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, _THIS_DIR)

import torch
import torch.multiprocessing as mp

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE, NUM_CLASSES,
    get_output_dir, REPORTS_DIR, DATA_FILES_DIR, STANDAR_SEG_DATASET_LOCATION, STANDAR_DET_DATASET_LOCATION,
    PAPER1_CSV_DIR, PAPER1_VIS_DIR, EVAL_CONF, EVAL_IOU, VISUAL_CONF, VISUAL_IOU, flush_gpu,
    IMAGE_SAMPLES_DIR
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools, load_native_coco_gt
)
from ultralytics import YOLO, SAM

# ==============================================================================
# KONSTANTA LOKAL & CONFIG MODEL
# ==============================================================================
SAM_MODEL_PATH = os.path.join(ROOT, "models", "sam2.1_t.pt")
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
]

# ==============================================================================
# HELPERS (KUANTITATIF)
# ==============================================================================
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
    except Exception as e:
        print(f"  ⚠️ Gagal hitung ukuran/parameter: {e}")
    return result

# ==============================================================================
# WORKER INFERENCE PARALEL (KUANTITATIF)
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
                
            print(f"  [GPU:{gpu}] Hybrid menggunakan {yolo_key} untuk generator prompt.")
            model_obj = YOLO(pt)
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
        img_key = img_path if img_path in image_ids else os.path.basename(img_path)
        if img_key not in image_ids: continue
        img_id = image_ids[img_key]
        pred_boxes, pred_confs, pred_clss, masks_np = [], [], [], []
        spd_pre, spd_inf, spd_post = 0.0, 0.0, 0.0
        
        if mtype in ["yolo_det", "yolo_seg"]:
            res = model_obj.predict(img_path, conf=EVAL_CONF, iou=EVAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre = spd.get("preprocess", 0.0)
            spd_inf = spd.get("inference", 0.0)
            spd_post = spd.get("postprocess", 0.0)
            
            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)
            if mtype == "yolo_seg" and res.masks is not None:
                masks_np = res.masks.data.cpu().numpy()
                H, W = res.orig_img.shape[:2]
                masks_np = np.array([cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5 for m in masks_np])
                
        elif mtype == "maskrcnn":
            pil = PILImage.open(img_path).convert("RGB")
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device_str)
            
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                preds_out = model_obj(t_img)[0]
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            spd_inf = (t3 - t2) * 1000
            
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            scores = preds_out["scores"].cpu().numpy()
            keep = scores >= EVAL_CONF
            pred_boxes = preds_out["boxes"].cpu().numpy()[keep]
            pred_confs = scores[keep]
            pred_clss = preds_out["labels"].cpu().numpy()[keep] - 1
            
            if "masks" in preds_out:
                m_data = preds_out["masks"].cpu().numpy()[keep, 0]
                H, W = pil.size[1], pil.size[0]
                for m in m_data:
                    if m.shape != (H, W): m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_np.append(m > 0.5)
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            spd_post = (t5 - t4) * 1000
                    
        elif mtype in ["hybrid_det", "hybrid_seg"]:
            res = model_obj.predict(img_path, conf=EVAL_CONF, iou=EVAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre = spd.get("preprocess", 0.0)
            yolo_inf = spd.get("inference", 0.0)
            yolo_post = spd.get("postprocess", 0.0)

            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)
                
                # 1. SAM2 Inference (menggunakan model callable berdasarkan list bbox koordinat)
                torch.cuda.synchronize()
                sam_inf_st = time.perf_counter()
                try:
                    sam_res = sam_model(res.orig_img, bboxes=pred_boxes.tolist(), device=device_str, verbose=False)
                    torch.cuda.synchronize()
                    sam_inf_et = time.perf_counter()
                    sam_inf_time = (sam_inf_et - sam_inf_st) * 1000
                except Exception as e:
                    sam_res = None
                    sam_inf_time = 0.0

                # 2. SAM2 Postprocess
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

        # Simpan Bounding Box COCO format
        for idx in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[idx]
            dt_bbox.append({
                "image_id": img_id,
                "category_id": int(pred_clss[idx]) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(pred_confs[idx])
            })
            
        # Simpan Segmentation COCO format (RLE)
        if mtype in ["yolo_seg", "maskrcnn", "hybrid_seg"] and len(masks_np) > 0:
            for idx in range(len(pred_boxes)):
                if idx < len(masks_np) and masks_np[idx] is not None:
                    rle = maskUtils.encode(np.asfortranarray(masks_np[idx].astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    dt_segm.append({
                        "image_id": img_id,
                        "category_id": int(pred_clss[idx]) + 1,
                        "segmentation": rle,
                        "score": float(pred_confs[idx])
                    })
                    
        t_pre.append(spd_pre)
        t_inf.append(spd_inf)
        t_post.append(spd_post)

    with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
        pickle.dump({
            "dt_bbox": dt_bbox, "dt_segm": dt_segm, "n_imgs": len(subset),
            "pre": t_pre, "inf": t_inf, "post": t_post
        }, f)
        
    del model_obj, sam_model
    flush_gpu(gpu, f"{mkey}_rank{rank}")

# ==============================================================================
# HELPERS (KUALITATIF / VISUALISASI)
# ==============================================================================
def load_maskrcnn(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
    pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
    if os.path.exists(pt):
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
        model.to(device).eval()
        return model
    return None

CLASS_NAMES = []
COLORS = []

def init_classes_from_coco(coco_dir):
    global CLASS_NAMES, COLORS
    json_path = os.path.join(coco_dir, "valid", "_annotations.coco.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        CLASS_NAMES = ["Class"] * NUM_CLASSES
        COLORS = [(0, 255, 0)] * NUM_CLASSES
        return

    with open(json_path, 'r') as f:
        coco_gt = json.load(f)

    cats = [c for c in coco_gt['categories'] if c['id'] > 0]
    cats = sorted(cats, key=lambda x: x['id'])
    CLASS_NAMES = [c['name'] for c in cats]

    color_palette = [
        (0, 255, 0),     # Hijau
        (255, 0, 0),     # Biru
        (0, 0, 255),     # Merah
        (0, 255, 255),   # Kuning
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Oranye
        (128, 0, 128),   # Ungu
        (0, 128, 128),   # Olive
    ]
    COLORS = [color_palette[i % len(color_palette)] for i in range(len(CLASS_NAMES))]

def get_text_color(bgr_color):
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def draw_custom(image_path, boxes, masks, confs, clss, color=None):
    img = cv2.imread(image_path)
    if img is None: return None

    # Draw masks
    if masks is not None:
        overlay = img.copy()
        for i in range(len(masks)):
            m = masks[i]
            if m is None: continue
            c_color = color if color is not None else COLORS[int(clss[i]) % len(COLORS)]
            c_color_arr = np.array(c_color, dtype=np.uint8)
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[m > 0.5] = c_color_arr
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.45, 0)
        img = overlay

    # Draw boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = confs[i] if confs is not None else 1.0
        c = int(clss[i])
        c_color = color if color is not None else COLORS[c % len(COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), c_color, 2)
        label = f"{CLASS_NAMES[c % len(CLASS_NAMES)]} {conf:.2f}" if confs is not None else CLASS_NAMES[c % len(CLASS_NAMES)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 4, y1), c_color, -1)
        text_color = get_text_color(c_color)
        cv2.putText(img, label, (x1 + 2, y1 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img

def load_gt_annotations(coco_dir):
    json_path = os.path.join(coco_dir, "valid", "_annotations.coco.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        return None
    from pycocotools.coco import COCO
    return COCO(json_path)

def get_gt_data(coco_api, img_filename):
    if coco_api is None:
        return [], [], []
    img_ids = coco_api.getImgIds()
    target_id = None
    for i in img_ids:
        info = coco_api.loadImgs(i)[0]
        if info['file_name'] == img_filename:
            target_id = i
            break
    if target_id is None:
        return [], [], []
    ann_ids = coco_api.getAnnIds(imgIds=target_id)
    anns = coco_api.loadAnns(ann_ids)

    boxes, masks, clss = [], [], []
    for ann in anns:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x+w, y+h])
        clss.append(ann['category_id'] - 1)
        if 'segmentation' in ann and ann['segmentation']:
            mask = coco_api.annToMask(ann)
            masks.append(mask)
    return np.array(boxes) if boxes else [], np.array(masks) if masks else None, np.array(clss) if clss else []

def plot_grid(images, titles, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for i in range(6):
        if i < len(images) and images[i] is not None:
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(titles[i], fontsize=16)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Distributed mAP Evaluation & Visualisation pipeline")
    parser.add_argument("--gpus", type=str, default="0", help="Daftar GPU komputasi, contoh: '0' atau '0,1'")
    parser.add_argument("--skip-eval", action="store_true", help="Lewati fase evaluasi kuantitatif (CSV)")
    parser.add_argument("--skip-visual", action="store_true", help="Lewati fase visualisasi kualitatif (Grid)")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    world_size = len(gpu_ids)
    
    print("="*80)
    print("  ORKESTRASI METODE BARU: KUANTITATIF (CSV) & KUALITATIF (VISUALISASI)")
    print("="*80)
    
    # --------------------------------------------------------------------------
    # FASE 1: EVALUASI KUANTITATIF (PENGHASIL CSV)
    # --------------------------------------------------------------------------
    if not args.skip_eval:
        print("\n>>> Memulai FASE 1: Evaluasi Kuantitatif (COCOeval mAP)...")
        # 1. Bangun GT dari standard datasets menggunakan format native COCO JSON
        coco_gt_det, image_ids_det = load_native_coco_gt(os.path.join(STANDAR_DET_DATASET_LOCATION, "valid"))
        coco_gt_seg, image_ids_seg = load_native_coco_gt(os.path.join(STANDAR_SEG_DATASET_LOCATION, "valid"))
        
        if coco_gt_det is None or coco_gt_seg is None:
            print("❌ Gagal memuat Ground Truth COCO untuk standard datasets. Periksa ketersediaan berkas _annotations.coco.json!")
            sys.exit(1)
            
        img_dir_det = os.path.join(STANDAR_DET_DATASET_LOCATION, "valid")
        img_dir_seg = os.path.join(STANDAR_SEG_DATASET_LOCATION, "valid")
        
        gpu_report = _gpu_report_str(gpu_ids)
        rows = []
        
        for mcfg in MODELS_CONFIG:
            label, mkey, mtype = mcfg["label"], mcfg["key"], mcfg["type"]
            print(f"\n[Eval Kuantitatif] Running: {label}...")
            
            # Persiapkan direktori & parameter dataset
            if mtype in ["yolo_det", "hybrid_det"]:
                img_dir, image_ids, coco_gt = img_dir_det, image_ids_det, coco_gt_det
            else:
                img_dir, image_ids, coco_gt = img_dir_seg, image_ids_seg, coco_gt_seg
                
            metrics = _get_model_metrics(mcfg)
            tmp_dir = tempfile.mkdtemp(prefix=f"eval_{mkey}_")
            t0 = time.perf_counter()
            
            try:
                # Menjalankan worker inference paralel multi-GPU
                mp.spawn(
                    _infer_worker,
                    args=(gpu_ids, mcfg, img_dir, image_ids, tmp_dir),
                    nprocs=world_size,
                    join=True
                )
                
                # Mengumpulkan hasil worker pkl
                dt_bbox, dt_segm = [], []
                n_imgs = 0
                all_pre, all_inf, all_post = [], [], []
                
                for r in range(world_size):
                    pkl_path = os.path.join(tmp_dir, f"rank{r}.pkl")
                    if os.path.exists(pkl_path):
                        with open(pkl_path, "rb") as f:
                            data = pickle.load(f)
                            dt_bbox.extend(data["dt_bbox"])
                            dt_segm.extend(data["dt_segm"])
                            n_imgs += data["n_imgs"]
                            all_pre.extend(data["pre"])
                            all_inf.extend(data["inf"])
                            all_post.extend(data["post"])
                            
                elapsed = time.perf_counter() - t0
                
                # Evaluasi mAP COCOeval
                mAP50_box, mAP95_box = "N/A", "N/A"
                mAP50_mask, mAP95_mask = "N/A", "N/A"
                
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                # Bounding Box mAP
                if len(dt_bbox) > 0 and (mtype in ["yolo_det", "maskrcnn", "hybrid_det"] or (mtype in ["yolo_seg", "hybrid_seg"])):
                    try:
                        coco_gt_obj = COCO()
                        coco_gt_obj.dataset = coco_gt
                        coco_gt_obj.createIndex()
                        coco_dt = coco_gt_obj.loadRes(dt_bbox)
                        eval_box = COCOeval(coco_gt_obj, coco_dt, iouType="bbox")
                        eval_box.evaluate()
                        eval_box.accumulate()
                        eval_box.summarize()
                        mAP95_box = round(float(eval_box.stats[0]), 4)
                        mAP50_box = round(float(eval_box.stats[1]), 4)
                    except Exception as ev_err:
                        print(f"  [Eval-Error] Gagal hitung mAP Box: {ev_err}")
                        
                # Mask Segmentation mAP
                if len(dt_segm) > 0 and (mtype in ["yolo_seg", "maskrcnn", "hybrid_seg"]):
                    try:
                        coco_gt_obj = COCO()
                        coco_gt_obj.dataset = coco_gt
                        coco_gt_obj.createIndex()
                        coco_dt_seg = coco_gt_obj.loadRes(dt_segm)
                        eval_seg = COCOeval(coco_gt_obj, coco_dt_seg, iouType="segm")
                        eval_seg.evaluate()
                        eval_seg.accumulate()
                        eval_seg.summarize()
                        mAP95_mask = round(float(eval_seg.stats[0]), 4)
                        mAP50_mask = round(float(eval_seg.stats[1]), 4)
                    except Exception as ev_err:
                        print(f"  [Eval-Error] Gagal hitung mAP Mask: {ev_err}")
                        
                # Hitung Rata-rata Latensi Kecepatan
                avg_pre = round(np.mean(all_pre), 2) if all_pre else 0.0
                avg_inf = round(np.mean(all_inf), 2) if all_inf else 0.0
                avg_post = round(np.mean(all_post), 2) if all_post else 0.0
                
                rows.append({
                    "Model": label,
                    "mAP50 (Box)": mAP50_box,
                    "mAP50-95 (Box)": mAP95_box,
                    "mAP50 (Mask)": mAP50_mask,
                    "mAP50-95 (Mask)": mAP95_mask,
                    "Speed Preprocess (ms)": avg_pre,
                    "Speed Inference (ms)": avg_inf,
                    "Speed Postprocess (ms)": avg_post,
                    "Weights Size (MB)": metrics["weights_size_mb"],
                    "Parameters (M)": metrics["parameters_m"],
                    "GPUs": gpu_report,
                })
                
                print(f"  [Hasil] {label} -> mAP50(Box): {mAP50_box} | mAP50(Mask): {mAP50_mask}")
            except Exception as e:
                print(f"  [Hasil] ❌ Gagal evaluasi {label}: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                
        # Export ke CSV
        os.makedirs(PAPER1_CSV_DIR, exist_ok=True)
        csv_path = os.path.join(PAPER1_CSV_DIR, "kompilasi_new_method_standar.csv")
        fields = [
            "Model", "mAP50 (Box)", "mAP50-95 (Box)", "mAP50 (Mask)", "mAP50-95 (Mask)",
            "Speed Preprocess (ms)", "Speed Inference (ms)", "Speed Postprocess (ms)",
            "Weights Size (MB)", "Parameters (M)", "GPUs"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n>>> FASE 1 SELESAI. CSV berhasil diekspor ke: {csv_path}")
        
        # Kirim notifikasi telegram
        send_telegram_msg(
            f"📊 <b>Fase 1 Kuantitatif Selesai</b>\n"
            f"Output: <code>{os.path.basename(csv_path)}</code>\n"
            f"Total Evaluasi: {len(rows)} model."
        )

    # --------------------------------------------------------------------------
    # FASE 2: EVALUASI KUALITATIF (PENGHASIL GRID VISUALISASI)
    # --------------------------------------------------------------------------
    if not args.skip_visual:
        print("\n>>> Memulai FASE 2: Evaluasi Kualitatif (Sample Grid Visualization)...")
        print(f"    Membaca berkas sampel dari: {IMAGE_SAMPLES_DIR}")
        
        if not os.path.isdir(IMAGE_SAMPLES_DIR) or not os.listdir(IMAGE_SAMPLES_DIR):
            print(f"  ❌ Direktori image_samples kosong atau tidak ditemukan: {IMAGE_SAMPLES_DIR}")
            sys.exit(1)
            
        device_str = f"cuda:{gpu_ids[0]}"
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        
        # Inisialisasi visual output directory
        STD_VIS_DIR = os.path.join(PAPER1_VIS_DIR, "new-method", "standard")
        IMG_SAMPLE_DIR = os.path.join(STD_VIS_DIR, "images_sample")
        DET_OUT_DIR = os.path.join(STD_VIS_DIR, "detection")
        SEG_OUT_DIR = os.path.join(STD_VIS_DIR, "segmentation")
        COMP_DET_DIR = os.path.join(STD_VIS_DIR, "comparison", "detection")
        COMP_SEG_DIR = os.path.join(STD_VIS_DIR, "comparison", "segmentation")
        
        for d in [IMG_SAMPLE_DIR, DET_OUT_DIR, SEG_OUT_DIR, COMP_DET_DIR, COMP_SEG_DIR]:
            os.makedirs(d, exist_ok=True)
            
        init_classes_from_coco(STANDAR_SEG_DATASET_LOCATION)
        
        # Load Models (Detection)
        print("  [Visual] Loading Detection Models...")
        yolo8_det  = YOLO(os.path.join(get_output_dir("yolov8m"), "weights", "best.pt"))
        yolo9_det  = YOLO(os.path.join(get_output_dir("yolov9m"), "weights", "best.pt"))
        yolo11_det = YOLO(YOLO11L_DET_PATH)
        mrcnn_det  = load_maskrcnn(device)
        
        # Load Models (Segmentation)
        print("  [Visual] Loading Segmentation Models...")
        yolo8_seg  = YOLO(os.path.join(get_output_dir("yolov8m_seg"), "weights", "best.pt"))
        yolo9_seg  = YOLO(os.path.join(get_output_dir("yolov9c_seg"), "weights", "best.pt"))
        yolo11_seg = YOLO(YOLO11L_SEG_PATH)
        sam2_model = SAM(SAM_MODEL_PATH)
        
        # Load GT COCO Anotasi
        coco_det = load_gt_annotations(STANDAR_DET_DATASET_LOCATION)
        coco_seg = load_gt_annotations(STANDAR_SEG_DATASET_LOCATION)
        
        img_names = sorted([f for f in os.listdir(IMAGE_SAMPLES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  [Visual] Memproses visualisasi untuk {len(img_names)} gambar...")
        
        for idx, img_name in enumerate(img_names, 1):
            img_path = os.path.join(IMAGE_SAMPLES_DIR, img_name)
            print(f"    [{idx}/{len(img_names)}] Visual rendering: {img_name}...")
            shutil.copy(img_path, os.path.join(IMG_SAMPLE_DIR, img_name))
            
            # ------------------------------------------------------------------
            # DETEKSI VISUALISASI
            # ------------------------------------------------------------------
            # 1. YOLOv8m
            res8_det = yolo8_det.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b8 = res8_det.boxes.xyxy.cpu().numpy() if res8_det.boxes else []
            c8 = res8_det.boxes.conf.cpu().numpy() if res8_det.boxes else []
            cls8 = res8_det.boxes.cls.cpu().numpy().astype(int) if res8_det.boxes else []
            img8_det = draw_custom(img_path, b8, None, c8, cls8, color=(0, 255, 0))
            if img8_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolov8m_{img_name}"), img8_det)
            
            # 2. YOLOv9m
            res9_det = yolo9_det.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b9 = res9_det.boxes.xyxy.cpu().numpy() if res9_det.boxes else []
            c9 = res9_det.boxes.conf.cpu().numpy() if res9_det.boxes else []
            cls9 = res9_det.boxes.cls.cpu().numpy().astype(int) if res9_det.boxes else []
            img9_det = draw_custom(img_path, b9, None, c9, cls9, color=(255, 0, 0))
            if img9_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolov9m_{img_name}"), img9_det)
            
            # 3. YOLO11l (New)
            res11_det = yolo11_det.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b11 = res11_det.boxes.xyxy.cpu().numpy() if res11_det.boxes else []
            c11 = res11_det.boxes.conf.cpu().numpy() if res11_det.boxes else []
            cls11 = res11_det.boxes.cls.cpu().numpy().astype(int) if res11_det.boxes else []
            img11_det = draw_custom(img_path, b11, None, c11, cls11, color=(0, 0, 255))
            if img11_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolo11l_{img_name}"), img11_det)
            
            # 4. Mask R-CNN
            from PIL import Image as PILImage
            import torchvision.transforms.functional as TF
            pil = PILImage.open(img_path).convert("RGB")
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = mrcnn_det(t_img)[0]
            scores_mrcnn = preds["scores"].cpu().numpy()
            keep = scores_mrcnn >= VISUAL_CONF
            bmrcnn = preds["boxes"].cpu().numpy()[keep]
            cmrcnn = scores_mrcnn[keep]
            clsmrcnn = preds["labels"].cpu().numpy()[keep] - 1
            mmrcnn = preds["masks"].cpu().numpy()[keep, 0] if "masks" in preds else None
            
            img_mrcnn_det = draw_custom(img_path, bmrcnn, None, cmrcnn, clsmrcnn, color=(0, 255, 255))
            if img_mrcnn_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"maskrcnn_{img_name}"), img_mrcnn_det)
            
            # 5. Hybrid Detection V8 (YOLOv8m + SAM2)
            sam_m_det_v8 = None
            if len(b8) > 0:
                try:
                    sam_res_det_v8 = sam2_model(res8_det.orig_img, bboxes=b8.tolist(), device=device_str, verbose=False)[0]
                    if sam_res_det_v8.masks is not None:
                        sam_m_det_v8 = sam_res_det_v8.masks.data.cpu().numpy()
                        h, w = sam_res_det_v8.orig_img.shape[:2]
                        sam_m_det_v8 = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m_det_v8])
                except Exception as e:
                    print(f"  [SAM2-Det-V8-Error] Gagal segmentasi hybrid det v8 untuk {img_name}: {e}")
            img_hybrid_det_v8 = draw_custom(img_path, b8, sam_m_det_v8, c8, cls8, color=(255, 0, 255))
            if img_hybrid_det_v8 is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"hybrid_v8_{img_name}"), img_hybrid_det_v8)
            
            # 5.1 Hybrid Detection V11 (YOLO11l + SAM2)
            sam_m_det_v11 = None
            if len(b11) > 0:
                try:
                    sam_res_det_v11 = sam2_model(res11_det.orig_img, bboxes=b11.tolist(), device=device_str, verbose=False)[0]
                    if sam_res_det_v11.masks is not None:
                        sam_m_det_v11 = sam_res_det_v11.masks.data.cpu().numpy()
                        h, w = sam_res_det_v11.orig_img.shape[:2]
                        sam_m_det_v11 = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m_det_v11])
                except Exception as e:
                    print(f"  [SAM2-Det-V11-Error] Gagal segmentasi hybrid det v11 untuk {img_name}: {e}")
            img_hybrid_det_v11 = draw_custom(img_path, b11, sam_m_det_v11, c11, cls11, color=(255, 0, 255))
            if img_hybrid_det_v11 is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"hybrid_v11_{img_name}"), img_hybrid_det_v11)
            
            # 6. GT Detection
            gt_b_det, _, gt_c_det = get_gt_data(coco_det, img_name)
            img_gt_det = draw_custom(img_path, gt_b_det, None, None, gt_c_det, color=(255, 255, 0))
            
            # Plot Grid Det V8 & V11
            plot_grid(
                [img8_det, img9_det, img11_det, img_mrcnn_det, img_hybrid_det_v8, img_gt_det],
                ["YOLOv8m", "YOLOv9m", "YOLO11l (New)", "Mask R-CNN", "Hybrid-v8 (YOLOv8m+SAM2)", "Ground Truth"],
                os.path.join(COMP_DET_DIR, f"grid_det_v8_{img_name}")
            )
            plot_grid(
                [img8_det, img9_det, img11_det, img_mrcnn_det, img_hybrid_det_v11, img_gt_det],
                ["YOLOv8m", "YOLOv9m", "YOLO11l (New)", "Mask R-CNN", "Hybrid-v11 (YOLO11l+SAM2)", "Ground Truth"],
                os.path.join(COMP_DET_DIR, f"grid_det_v11_{img_name}")
            )

            # ------------------------------------------------------------------
            # SEGMENTASI VISUALISASI
            # ------------------------------------------------------------------
            # 1. YOLOv8m-Seg
            res8_seg = yolo8_seg.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b8s = res8_seg.boxes.xyxy.cpu().numpy() if res8_seg.boxes else []
            c8s = res8_seg.boxes.conf.cpu().numpy() if res8_seg.boxes else []
            cls8s = res8_seg.boxes.cls.cpu().numpy().astype(int) if res8_seg.boxes else []
            m8s = res8_seg.masks.data.cpu().numpy() if res8_seg.masks else None
            if m8s is not None:
                h, w = res8_seg.orig_img.shape[:2]
                m8s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m8s])
            else: m8s_resized = None
            img8_seg = draw_custom(img_path, b8s, m8s_resized, c8s, cls8s, color=(0, 255, 0))
            if img8_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolov8m_seg_{img_name}"), img8_seg)
            
            # 2. YOLOv9c-Seg
            res9_seg = yolo9_seg.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b9s = res9_seg.boxes.xyxy.cpu().numpy() if res9_seg.boxes else []
            c9s = res9_seg.boxes.conf.cpu().numpy() if res9_seg.boxes else []
            cls9s = res9_seg.boxes.cls.cpu().numpy().astype(int) if res9_seg.boxes else []
            m9s = res9_seg.masks.data.cpu().numpy() if res9_seg.masks else None
            if m9s is not None:
                h, w = res9_seg.orig_img.shape[:2]
                m9s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m9s])
            else: m9s_resized = None
            img9_seg = draw_custom(img_path, b9s, m9s_resized, c9s, cls9s, color=(255, 0, 0))
            if img9_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolov9c_seg_{img_name}"), img9_seg)

            # 3. YOLO11l-Seg
            res11_seg = yolo11_seg.predict(img_path, conf=VISUAL_CONF, iou=VISUAL_IOU, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            b11s = res11_seg.boxes.xyxy.cpu().numpy() if res11_seg.boxes else []
            c11s = res11_seg.boxes.conf.cpu().numpy() if res11_seg.boxes else []
            cls11s = res11_seg.boxes.cls.cpu().numpy().astype(int) if res11_seg.boxes else []
            m11s = res11_seg.masks.data.cpu().numpy() if res11_seg.masks else None
            if m11s is not None:
                h, w = res11_seg.orig_img.shape[:2]
                m11s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m11s])
            else: m11s_resized = None
            img11_seg = draw_custom(img_path, b11s, m11s_resized, c11s, cls11s, color=(0, 0, 255))
            if img11_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolo11l_seg_{img_name}"), img11_seg)
            
            # 4. Mask R-CNN Seg
            if mmrcnn is not None:
                h, w = pil.size[1], pil.size[0]
                mmrcnn_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in mmrcnn])
            else: mmrcnn_resized = None
            img_mrcnn_seg = draw_custom(img_path, bmrcnn, mmrcnn_resized, cmrcnn, clsmrcnn, color=(0, 255, 255))
            if img_mrcnn_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"maskrcnn_{img_name}"), img_mrcnn_seg)

            # 5. Hybrid Seg V8 (YOLOv8m-Seg + SAM2)
            sam_m_seg_v8 = None
            if len(b8s) > 0:
                try:
                    sam_res_seg_v8 = sam2_model(res8_seg.orig_img, bboxes=b8s.tolist(), device=device_str, verbose=False)[0]
                    if sam_res_seg_v8.masks is not None:
                        sam_m_seg_v8 = sam_res_seg_v8.masks.data.cpu().numpy()
                        h, w = sam_res_seg_v8.orig_img.shape[:2]
                        sam_m_seg_v8 = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m_seg_v8])
                except Exception as e:
                    print(f"  [SAM2-Seg-V8-Error] Gagal segmentasi hybrid seg v8 untuk {img_name}: {e}")
            img_hybrid_seg_v8 = draw_custom(img_path, b8s, sam_m_seg_v8, c8s, cls8s, color=(255, 0, 255))
            if img_hybrid_seg_v8 is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"hybrid_v8_{img_name}"), img_hybrid_seg_v8)

            # 5.1 Hybrid Seg V11 (YOLO11l-Seg + SAM2)
            sam_m_seg_v11 = None
            if len(b11s) > 0:
                try:
                    sam_res_seg_v11 = sam2_model(res11_seg.orig_img, bboxes=b11s.tolist(), device=device_str, verbose=False)[0]
                    if sam_res_seg_v11.masks is not None:
                        sam_m_seg_v11 = sam_res_seg_v11.masks.data.cpu().numpy()
                        h, w = sam_res_seg_v11.orig_img.shape[:2]
                        sam_m_seg_v11 = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m_seg_v11])
                except Exception as e:
                    print(f"  [SAM2-Seg-V11-Error] Gagal segmentasi hybrid seg v11 untuk {img_name}: {e}")
            img_hybrid_seg_v11 = draw_custom(img_path, b11s, sam_m_seg_v11, c11s, cls11s, color=(255, 0, 255))
            if img_hybrid_seg_v11 is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"hybrid_v11_{img_name}"), img_hybrid_seg_v11)

            # 6. GT Seg
            gt_b_seg, gt_m_seg, gt_c_seg = get_gt_data(coco_seg, img_name)
            img_gt_seg = draw_custom(img_path, gt_b_seg, gt_m_seg, None, gt_c_seg, color=(255, 255, 0))

            # Plot Grid Seg V8 & V11
            plot_grid(
                [img8_seg, img9_seg, img11_seg, img_mrcnn_seg, img_hybrid_seg_v8, img_gt_seg],
                ["YOLOv8m-Seg", "YOLOv9c-Seg", "YOLO11l-Seg", "Mask R-CNN", "Hybrid-v8 (YOLOv8m-Seg+SAM2)", "Ground Truth"],
                os.path.join(COMP_SEG_DIR, f"grid_seg_v8_{img_name}")
            )
            plot_grid(
                [img8_seg, img9_seg, img11_seg, img_mrcnn_seg, img_hybrid_seg_v11, img_gt_seg],
                ["YOLOv8m-Seg", "YOLOv9c-Seg", "YOLO11l-Seg", "Mask R-CNN", "Hybrid-v11 (YOLO11l-Seg+SAM2)", "Ground Truth"],
                os.path.join(COMP_SEG_DIR, f"grid_seg_v11_{img_name}")
            )
            
            flush_gpu(gpu_ids[0], f"visuals_{img_name}")
            
        print(f"\n>>> FASE 2 SELESAI. Visualisasi tersimpan di: {STD_VIS_DIR}")

if __name__ == "__main__":
    main()
