# -*- coding: utf-8 -*-
"""
utils/generate_report_single_model.py
================================================================================
SKRIP SENTRAL EVALUASI KUANTITATIF & VISUALISASI KUALITATIF MULTI-MODEL
================================================================================
Penulis: Google Professional Full Stack Developer Persona (Antigravity AI Agent)
Proyek: Fine-Tuning & Benchmarking 49 Model YOLO & Hybrid (SAM2) — Scopus Q1/Q2

Deskripsi Fungsional:
--------------------
Skrip ini dirancang secara terintegrasi untuk mengeksekusi dua pilar evaluasi:
  1. Pilar Visualisasi Kualitatif (Sample-Based):
     - Membangun panel grid komparatif gabungan (_panel.jpg) per model family.
     - Menyusun visualisasi dengan urutan Ground Truth (GT) di posisi awal (indeks pertama).
     - Menyematkan Title Bar minimalis putih bersih dengan perataan tengah (Horizontal Center).
     - Menampilkan label teks bounding box kontras otomatis (hitam/putih) terhadap warna box.
     - Mengabaikan penyimpanan gambar prediksi individual demi efisiensi VRAM dan kapasitas disk.
     - Menyimpan seluruh hasil visualisasi secara hierarkis ke dalam subfolder family masing-masing.

  2. Pilar Evaluasi Kuantitatif (Metric-Based):
     - Mengalkulasi metrik mAP50 dan mAP50-95 (COCOeval) secara paralel terdistribusi.
     - Mengekspor laporan evaluasi CSV berstruktur hierarkis ke subfolder tujuan.
     - Menghasilkan laporan kompilasi per-family serta laporan gabungan global (ALL).

Petunjuk Pengoperasian (CLI Usage):
----------------------------------
Langkah 1: Hubungkan terminal Anda ke Compute Node GPU yang telah dibooking:
           $ cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
           $ ./attach_gpu.sh

Langkah 2: Jalankan skrip evaluasi dengan parameter yang sesuai:
           # A. Evaluasi seluruh varian dalam satu family (Contoh: YOLOv8)
           $ python3 -u utils/generate_report_single_model.py --family yolo8 --gpus 0

           # B. Evaluasi varian tertentu secara spesifik (Contoh: YOLOv9m & YOLOv9e)
           $ python3 -u utils/generate_report_single_model.py --family yolo9 --variants yolov9m,yolov9e --gpus 0

           # C. Eksekusi orkestrasi otomatis untuk seluruh family sekaligus (Pipeline Mode)
                WS_ID=$(cat .workspace_id)
                LOG_DIR="data-files/MyFineTunning-${WS_ID}/logs"
                mkdir -p "$LOG_DIR"
                python3 -u utils/generate_report_single_model.py --family all 2>&1 | tee "$LOG_DIR/1_generate_report_single_model.log"
           # D. 
                cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && LOG_DIR=$(python3 -c "import config_shared, os; print(os.path.join(config_shared.WORKSPACE_DIR, 'logs'))") && mkdir -p "$LOG_DIR" && python3 -u utils/generate_report_single_model.py 2>&1 | tee "$LOG_DIR/1_generate_report_single_model.log"
                
Struktur Direktori Output (di dalam WORKSPACE_DIR):
-------------------------------------------------
reports/pipeline/
├── csv/
│   ├── yolo8/
│   │   ├── yolov8m_detection.csv
│   │   ├── yolov8m-seg_segmentation.csv
│   │   ├── yolov8x_detection.csv
│   │   └── yolov8x-seg_segmentation.csv
│   ├── yolo9/
│   │   ├── yolov9m_detection.csv
│   │   ├── yolov9c-seg_segmentation.csv
│   │   ├── yolov9e_detection.csv
│   │   └── yolov9e-seg_segmentation.csv
│   ├── yolov10/
│   │   ├── yolov10m_detection.csv
│   │   └── yolov10x_detection.csv
│   ├── yolo11/
│   │   ├── yolo11n_detection.csv
│   │   ├── yolo11n-seg_segmentation.csv
│   │   ├── yolo11l_detection.csv
│   │   ├── yolo11l-seg_segmentation.csv
│   │   ├── yolo11x_detection.csv
│   │   └── yolo11x-seg_segmentation.csv
│   ├── maskrcnn/
│   │   └── maskrcnn_segmentation.csv
│   ├── hybrid/
│   │   └── hybrid_segmentation.csv
│   ├── rtdetr/
│   │   └── rtdetr_detection.csv
│   ├── kompilasi_yolo8_detection.csv
│   ├── kompilasi_yolo8_segmentation.csv
│   ├── kompilasi_yolo9_detection.csv
│   ├── kompilasi_yolo9_segmentation.csv
│   ├── kompilasi_yolov10_detection.csv
│   ├── kompilasi_yolo11_detection.csv
│   ├── kompilasi_yolo11_segmentation.csv
│   ├── kompilasi_maskrcnn_segmentation.csv
│   ├── kompilasi_hybrid_segmentation.csv
│   ├── kompilasi_rtdetr_detection.csv
│   ├── kompilasi_ALL_detection.csv
│   └── kompilasi_ALL_segmentation.csv
└── visuals/
    ├── yolo8/
    │   └── <sample_id>_yolo8_panel.jpg     ← Panel Grid: GT (indeks awal) + Varian YOLOv8
    ├── yolo9/
    │   └── <sample_id>_yolo9_panel.jpg     ← Panel Grid: GT (indeks awal) + Varian YOLOv9
    ├── yolov10/
    │   └── <sample_id>_yolov10_panel.jpg   ← Panel Grid: GT (indeks awal) + Varian YOLOv10
    ├── yolo11/
    │   └── <sample_id>_yolo11_panel.jpg    ← Panel Grid: GT (indeks awal) + Varian YOLO11
    ├── maskrcnn/
    │   └── <sample_id>_maskrcnn_panel.jpg  ← Panel Grid: GT (indeks awal) + Mask R-CNN
    ├── hybrid/
    │   └── <sample_id>_hybrid_panel.jpg    ← Panel Grid: GT (indeks awal) + Varian Hybrid
    └── rtdetr/
        └── <sample_id>_rtdetr_panel.jpg    ← Panel Grid: GT (indeks awal) + RT-DETR-L
"""

from __future__ import annotations

import os
import sys
import gc
import csv
import time
import pickle
import argparse
import tempfile
from pathlib import Path
from typing import Optional

# ==============================================================================
# PATH SETUP — resolve ROOT dari lokasi file ini (utils/generate_report_single_model.py)
# ==============================================================================
_THIS_FILE = os.path.abspath(__file__)
_UTILS_DIR = os.path.dirname(_THIS_FILE)
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config_shared import (
    WORKSPACE_DIR,
    REPORTS_DIR,
    IMAGE_SAMPLES_DIR,
    DET_YAML,
    SEG_YAML,
    IMAGE_SIZE,
    get_output_dir,
    MODEL_COLORS,
    NUM_CLASSES,
    VISUAL_NUM_SAMPLES,
    EVAL_DATASET_LOCATION,
    CSV_REPORT_FIELDS,
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth,
    evaluate_coco_predictions,
    check_pycocotools,
    load_native_coco_gt,
)

# ==============================================================================
# KONSTANTA PATH MODEL
# ==============================================================================
SAM_MODEL_PATH = os.path.join(ROOT, "models", "sam2.1_t.pt")
MOBILE_SAM_MODEL_PATH = os.path.join(ROOT, "models", "mobile_sam.pt")
YOLO11L_DET_PATH  = os.path.join(WORKSPACE_DIR, "runs", "yolo11l", "weights", "best.pt")
YOLO11L_SEG_PATH  = os.path.join(WORKSPACE_DIR, "runs", "yolo11l_seg", "weights", "best.pt")

# ==============================================================================
# KONSTANTA — Definisi Semua Family & Varian
# ==============================================================================

# Struktur: family_key → list of variant specs
# Setiap spec: {"key": str, "label": str, "task": "det" | "seg"}
FAMILY_VARIANTS: dict[str, list[dict]] = {
    "yolo8": [
        {"key": "yolov8m",     "label": "YOLOv8m",     "task": "det"},
        {"key": "yolov8m_seg", "label": "YOLOv8m-Seg", "task": "seg"},
        {"key": "yolov8x",     "label": "YOLOv8x",     "task": "det"},
        {"key": "yolov8x_seg", "label": "YOLOv8x-Seg", "task": "seg"},
    ],
    "yolo9": [
        {"key": "yolov9m",     "label": "YOLOv9m",     "task": "det"},
        {"key": "yolov9c_seg", "label": "YOLOv9c-Seg", "task": "seg"},
        {"key": "yolov9e",     "label": "YOLOv9e",     "task": "det"},
        {"key": "yolov9e_seg", "label": "YOLOv9e-Seg", "task": "seg"},
    ],
    "yolov10": [
        {"key": "yolov10m", "label": "YOLOv10m", "task": "det"},
        {"key": "yolov10x", "label": "YOLOv10x", "task": "det"},
    ],
    "yolo11": [
        {"key": "yolo11n",     "label": "YOLO11n",     "task": "det"},
        {"key": "yolo11n_seg", "label": "YOLO11n-Seg", "task": "seg"},
        {"key": "yolo11l",     "label": "YOLO11l",     "task": "det"},
        {"key": "yolo11l_seg", "label": "YOLO11l-Seg", "task": "seg"},
        {"key": "yolo11x",     "label": "YOLO11x",     "task": "det"},
        {"key": "yolo11x_seg", "label": "YOLO11x-Seg", "task": "seg"},
    ],
    "maskrcnn": [
        {"key": "maskrcnn", "label": "Mask R-CNN", "task": "seg"},
    ],
    "hybrid": [
        {"key": "hybrid_yolov8m",     "label": "Hybrid (YOLOv8m+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolov8m_seg", "label": "Hybrid (YOLOv8m-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolov8x",     "label": "Hybrid (YOLOv8x+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolov8x_seg", "label": "Hybrid (YOLOv8x-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolov9m",     "label": "Hybrid (YOLOv9m+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolov9c_seg", "label": "Hybrid (YOLOv9c-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolov9e",     "label": "Hybrid (YOLOv9e+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolov9e_seg", "label": "Hybrid (YOLOv9e-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolov10m",    "label": "Hybrid (YOLOv10m+SAM2.1_t)",    "task": "seg"},
        {"key": "hybrid_yolov10x",    "label": "Hybrid (YOLOv10x+SAM2.1_t)",    "task": "seg"},
        {"key": "hybrid_yolo11n",     "label": "Hybrid (YOLO11n+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolo11n_seg", "label": "Hybrid (YOLO11n-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolo11l",     "label": "Hybrid (YOLO11l+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolo11l_seg", "label": "Hybrid (YOLO11l-Seg+SAM2.1_t)", "task": "seg"},
        {"key": "hybrid_yolo11x",     "label": "Hybrid (YOLO11x+SAM2.1_t)",     "task": "seg"},
        {"key": "hybrid_yolo11x_seg", "label": "Hybrid (YOLO11x-Seg+SAM2.1_t)", "task": "seg"},
        
        # ── Hybrid Detection + MobileSAM ──────────────────────────────────────────
        {"key": "hybrid_yolov8m_mobile",      "label": "Hybrid (YOLOv8m+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolov8x_mobile",      "label": "Hybrid (YOLOv8x+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolov9m_mobile",      "label": "Hybrid (YOLOv9m+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolov9e_mobile",      "label": "Hybrid (YOLOv9e+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolo11n_mobile",      "label": "Hybrid (YOLO11n+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolo11l_mobile",      "label": "Hybrid (YOLO11l+MobileSAM)",      "task": "seg"},
        {"key": "hybrid_yolo11x_mobile",      "label": "Hybrid (YOLO11x+MobileSAM)",      "task": "seg"},
        
        # ── Hybrid Segmentation + MobileSAM ───────────────────────────────────────
        {"key": "hybrid_yolov8m_seg_mobile",  "label": "Hybrid (YOLOv8m-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolov8x_seg_mobile",  "label": "Hybrid (YOLOv8x-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolov9c_seg_mobile",  "label": "Hybrid (YOLOv9c-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolov9e_seg_mobile",  "label": "Hybrid (YOLOv9e-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolov10m_mobile",     "label": "Hybrid (YOLOv10m+MobileSAM)",     "task": "seg"},
        {"key": "hybrid_yolov10x_mobile",     "label": "Hybrid (YOLOv10x+MobileSAM)",     "task": "seg"},
        {"key": "hybrid_yolo11n_seg_mobile",  "label": "Hybrid (YOLO11n-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolo11l_seg_mobile",  "label": "Hybrid (YOLO11l-Seg+MobileSAM)",  "task": "seg"},
        {"key": "hybrid_yolo11x_seg_mobile",  "label": "Hybrid (YOLO11x-Seg+MobileSAM)",  "task": "seg"},
    ],
    # RT-DETR — Paper 3: Edge Transformer vs CNN Benchmark
    # task="det" karena RT-DETR hanya melakukan deteksi (bukan segmentasi native).
    # Evaluasi menggunakan branch yang sama dengan YOLO det (Ultralytics RTDETR API-compatible).
    "rtdetr": [
        {"key": "rtdetr_l", "label": "RT-DETR-L", "task": "det"},
    ],
}

# ==============================================================================
# HELPERS — Shared
# ==============================================================================

def _flush_gpu(rank: int, label: str = "") -> None:
    """Bersihkan memori GPU dan tampilkan status."""
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(rank)
        free, total = torch.cuda.mem_get_info(rank)
        print(f"  [GPU:{rank}][MemFlush] {label} ✅ free={free/1e9:.2f}GB", flush=True)
    except Exception as e:
        print(f"  [GPU:{rank}][MemFlush] {label} ⚠️ {e}", flush=True)


def _avg_ms(lst: list) -> float | str:
    """Hitung rata-rata dari list waktu (ms). Kembalikan 'N/A' jika kosong."""
    return round(sum(lst) / len(lst), 2) if lst else "N/A"


def _gpu_report_str(gpu_ids: list) -> str:
    """Format GPU IDs + nama GPU untuk laporan CSV."""
    try:
        import torch
        from collections import Counter
        names  = [torch.cuda.get_device_name(i) for i in gpu_ids]
        counts = Counter(names)
        return ", ".join(f"{c}x {n}" for n, c in counts.items())
    except Exception:
        return str(gpu_ids)


def _resolve_img_dir(yaml_path: str) -> str:
    """Cari folder valid/ atau test/images dari path YAML dataset."""
    base = os.path.dirname(yaml_path)
    for split in ("valid", "test"):
        d = os.path.join(base, split, "images")
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"Tidak ditemukan folder valid/ atau test/images di: {base}"
    )


def _partition_images(img_dir: str, rank: int, world_size: int) -> list:
    """Bagi daftar gambar secara merata ke setiap GPU-rank."""
    all_imgs = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return all_imgs[rank::world_size]


def _get_csv_dir(family: str) -> str:
    """Kembalikan path folder CSV untuk satu family. Buat jika belum ada."""
    d = os.path.join(REPORTS_DIR, "csv", family)
    os.makedirs(d, exist_ok=True)
    return d


def _get_visuals_dir(family: str = None) -> str:
    """Kembalikan path folder output visual, opsional dengan subdirektori family. Buat jika belum ada."""
    if family:
        d = os.path.join(REPORTS_DIR, "visuals", family)
    else:
        d = os.path.join(REPORTS_DIR, "visuals")
    os.makedirs(d, exist_ok=True)
    return d


# ==============================================================================
# MODUL 1 — VISUALISASI: Per gambar sampel → N prediksi individual + 1 panel grid
# ==============================================================================

def _get_contrast_color(bgr_color: tuple) -> tuple:
    """Tentukan warna teks (hitam/putih) paling kontras dengan warna BGR background."""
    B, G, R = bgr_color
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return (0, 0, 0) if brightness > 127.5 else (255, 255, 255)


def _load_class_names(yaml_path: str) -> dict:
    """Load class names dictionary dari YAML dataset."""
    try:
        import yaml
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                names_data = data["names"]
                if isinstance(names_data, dict):
                    return {int(k): v for k, v in names_data.items()}
                elif isinstance(names_data, list):
                    return {i: v for i, v in enumerate(names_data)}
    except Exception as e:
        print(f"  [Warning] Gagal memuat nama kelas dari {yaml_path}: {e}")
    return {}


def _draw_predictions_on_image(img_bgr, result, theme_color: tuple) -> None:
    """Gambar bounding box, label, dan mask pada image (in-place)."""
    import cv2
    import numpy as np

    H, W = img_bgr.shape[:2]
    names = result.names

    # Segmentation Masks
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.float32), (W, H),
                    interpolation=cv2.INTER_NEAREST
                )
            bool_mask = (mask > 0.5).astype(np.uint8)
            colored = np.zeros_like(img_bgr)
            colored[bool_mask == 1] = theme_color
            cv2.addWeighted(colored, 0.45, img_bgr, 1.0, 0, img_bgr)

    # Bounding Boxes + Labels
    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        boxes_cls  = result.boxes.cls.cpu().numpy().astype(int)

        # Contrast-safe text color
        text_color = _get_contrast_color(theme_color)

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            conf    = float(boxes_conf[i])
            cls_id  = int(boxes_cls[i])
            cls_name = names.get(cls_id, f"cls{cls_id}")

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), theme_color, 2)

            label_txt = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
            cv2.putText(img_bgr, label_txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)


def generate_visuals_for_family(
    family: str,
    variants: list[dict],
    gpu_ids: list[int],
    n_samples: int = VISUAL_NUM_SAMPLES,
    conf: float = 0.5,
) -> None:
    """
    Untuk setiap gambar sampel, hasilkan:
    - N gambar prediksi individual (satu per varian model)
    - 1 gambar panel grid (N varian + 1 Ground Truth)

    Parameter
    ---------
    family   : Nama family model (misal "yolo8")
    variants : List spec varian dari FAMILY_VARIANTS[family]
    gpu_ids  : List ID GPU yang tersedia
    n_samples: Jumlah gambar sampel yang diproses
    conf     : Confidence threshold prediksi
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    visuals_dir = _get_visuals_dir(family)
    names_dict = _load_class_names(SEG_YAML)

    # Validasi direktori samples
    if not os.path.isdir(IMAGE_SAMPLES_DIR):
        print(f"[Visual] ⚠️  IMAGE_SAMPLES_DIR tidak ditemukan: {IMAGE_SAMPLES_DIR}")
        return

    # Ambil gambar sampel
    all_imgs = sorted([
        os.path.join(IMAGE_SAMPLES_DIR, f)
        for f in os.listdir(IMAGE_SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    samples = all_imgs[:n_samples]

    if not samples:
        print(f"[Visual] ⚠️  Tidak ada gambar sampel di: {IMAGE_SAMPLES_DIR}")
        return

    print(f"\n[Visual] Family={family} | {len(variants)} varian | {len(samples)} sampel")

    for sample_path in samples:
        base = os.path.splitext(os.path.basename(sample_path))[0]
        print(f"\n  [Visual] Memproses sampel: {base}")

        gt_img = cv2.imread(sample_path)
        if gt_img is None:
            print(f"  [Visual] ⚠️  Gagal membaca gambar: {sample_path}")
            continue

        panel_images  = []  # Untuk panel grid akhir
        panel_labels  = []  # Label tiap panel cell

        # --- Ground Truth panel ---
        panel_images.append(gt_img.copy())
        panel_labels.append("Ground Truth")

        # --- Prediksi per varian ---
        gpu_cycle  = gpu_ids[0] if gpu_ids else 0
        device_str = f"cuda:{gpu_cycle}"

        for spec in variants:
            model_key   = spec["key"]
            model_label = spec["label"]
            theme_color = MODEL_COLORS.get(model_key, (255, 255, 0))

            # ──────────────────────────────────────────────────────────────
            # Tentukan pt_path berdasarkan tipe family:
            #
            # • YOLO normal   → get_output_dir(model_key)/weights/best.pt
            # • Mask R-CNN    → WORKSPACE_DIR/runs/maskrcnn/weights/best.pt
            # • Hybrid        → Hybrid BUKAN model terlatih tersendiri.
            #                   Ia memakai weights YOLO dasar yang sudah ada
            #                   (misal hybrid_yolov8m → runs/yolov8m/weights/best.pt)
            #                   lalu di-refine oleh SAM2.
            # ──────────────────────────────────────────────────────────────
            is_maskrcnn = (family == "maskrcnn")
            is_hybrid   = model_key.startswith("hybrid_")

            if is_hybrid:
                # Ekstrak base YOLO key: "hybrid_yolov8m_mobile" → "yolov8m"
                yolo_base_key = model_key.replace("hybrid_", "").replace("_mobile", "")
                pt_path = os.path.join(get_output_dir(yolo_base_key), "weights", "best.pt")
            else:
                pt_path = os.path.join(get_output_dir(model_key), "weights", "best.pt")

            if not os.path.exists(pt_path):
                print(f"  [Visual] ⚠️  best.pt tidak ditemukan untuk {model_label}: {pt_path}")
                blank = gt_img.copy()
                H_b, W_b = blank.shape[:2]
                na_text = "Weights Not Found (N/A)"
                (tw, th), _ = cv2.getTextSize(na_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cx, cy = (W_b - tw) // 2, (H_b + th) // 2
                cv2.putText(blank, na_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2, cv2.LINE_AA)
                panel_images.append(blank)
                panel_labels.append(f"{model_label} ⚠ N/A")
                continue

            try:
                # ── MASK R-CNN: Gunakan TorchVision loader (bukan YOLO) ──
                if is_maskrcnn:
                    import torch
                    import torchvision.transforms.functional as TF
                    from PIL import Image as PILImage

                    sys.path.insert(0, os.path.join(ROOT, "mask-r-cnn"))
                    from maskrcnn_builder import build_model

                    device = torch.device(device_str)
                    mrcnn_model = build_model(device)
                    mrcnn_model.load_state_dict(
                        torch.load(pt_path, map_location=device, weights_only=True)
                    )
                    mrcnn_model.eval()

                    pil_img = PILImage.open(sample_path).convert("RGB")
                    W_i, H_i = pil_img.size
                    t_img = TF.to_tensor(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        preds_out = mrcnn_model(t_img)[0]

                    scores = preds_out["scores"].cpu().numpy()
                    labels_arr = preds_out["labels"].cpu().numpy()
                    boxes_arr  = preds_out["boxes"].cpu().numpy()
                    masks_arr  = preds_out["masks"].cpu().numpy()

                    pred_img = cv2.imread(sample_path)
                    keep = scores >= conf
                    for i, (sc, lb, bx, mk) in enumerate(
                        zip(scores[keep], labels_arr[keep], boxes_arr[keep], masks_arr[keep])
                    ):
                        bin_mask = (mk[0] > 0.5).astype(np.uint8)
                        if bin_mask.shape != (H_i, W_i):
                            bin_mask = cv2.resize(bin_mask, (W_i, H_i),
                                                  interpolation=cv2.INTER_NEAREST)
                        colored = np.zeros_like(pred_img)
                        colored[bin_mask == 1] = theme_color
                        cv2.addWeighted(colored, 0.45, pred_img, 1.0, 0, pred_img)

                        x1, y1, x2, y2 = map(int, bx)
                        cls_id = int(lb) - 1  # Konversi 1-indexed TorchVision ke 0-indexed YOLO
                        cls_name = names_dict.get(cls_id, f"cls{cls_id}")
                        text_color = _get_contrast_color(theme_color)

                        cv2.rectangle(pred_img, (x1, y1), (x2, y2), theme_color, 2)
                        lbl_txt = f"{cls_name} {sc:.2f}"
                        (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(pred_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
                        cv2.putText(pred_img, lbl_txt, (x1 + 2, y1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

                    del mrcnn_model
                    gc.collect()

                # ── HYBRID: YOLO base + SAM2/MobileSAM ───────────────────
                elif is_hybrid:
                    from ultralytics import YOLO as _YOLO, SAM

                    yolo_model = _YOLO(pt_path)
                    is_mobile = "_mobile" in model_key
                    sam_pt = MOBILE_SAM_MODEL_PATH if is_mobile else SAM_MODEL_PATH
                    sam_model  = SAM(sam_pt)

                    det_res = yolo_model.predict(
                        sample_path, conf=conf, imgsz=IMAGE_SIZE,
                        device=device_str, verbose=False
                    )[0]

                    pred_img = det_res.orig_img.copy()
                    H_i, W_i = pred_img.shape[:2]

                    if det_res.boxes is not None and len(det_res.boxes) > 0:
                        pred_boxes = det_res.boxes.xyxy
                        pred_confs = det_res.boxes.conf.cpu().numpy()
                        pred_clss  = det_res.boxes.cls.cpu().numpy().astype(int)

                        # Kirim list Python koordinat absolut agar aman dari distorsi scale internal SAM
                        sam_res = sam_model.predict(
                            det_res.orig_img, bboxes=pred_boxes.tolist(), verbose=False
                        )

                        if sam_res and sam_res[0].masks is not None:
                            sam_masks = sam_res[0].masks.data.cpu().numpy()
                        else:
                            sam_masks = []

                        text_color = _get_contrast_color(theme_color)

                        for idx_b in range(len(pred_boxes)):
                            sc       = float(pred_confs[idx_b])
                            cls_id   = int(pred_clss[idx_b])
                            cls_name = det_res.names.get(cls_id, f"cls{cls_id}")

                            if idx_b < len(sam_masks):
                                mk = sam_masks[idx_b]
                                if mk.shape != (H_i, W_i):
                                    mk = cv2.resize(mk.astype(np.float32), (W_i, H_i),
                                                    interpolation=cv2.INTER_NEAREST)
                                bin_mask = (mk > 0.5).astype(np.uint8)
                                colored = np.zeros_like(pred_img)
                                colored[bin_mask == 1] = theme_color
                                cv2.addWeighted(colored, 0.45, pred_img, 1.0, 0, pred_img)

                            bx_arr = det_res.boxes.xyxy.cpu().numpy()[idx_b]
                            x1, y1, x2, y2 = map(int, bx_arr)
                            cv2.rectangle(pred_img, (x1, y1), (x2, y2), theme_color, 2)
                            lbl_txt = f"{cls_name} {sc:.2f}"
                            (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(pred_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
                            cv2.putText(pred_img, lbl_txt, (x1 + 2, y1 - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

                    del yolo_model, sam_model
                    gc.collect()

                # ── YOLO (semua family lainnya) ───────────────────────────
                else:
                    model = YOLO(pt_path)
                    result = model.predict(
                        sample_path, conf=conf, imgsz=IMAGE_SIZE,
                        device=gpu_cycle, verbose=False
                    )[0]
                    pred_img = result.orig_img.copy()
                    _draw_predictions_on_image(pred_img, result, theme_color)
                    del model, result
                    gc.collect()

                # Simpan gambar individual: <base>_<model_key>.jpg
                # individual_path = os.path.join(visuals_dir, f"{base}_{model_key}.jpg")
                # cv2.imwrite(individual_path, pred_img)
                # print(f"    ➡️  Individual: {os.path.basename(individual_path)}")
                
                # Lewati penyimpanan gambar individual demi efisiensi visualisasi

                panel_images.append(pred_img.copy())
                panel_labels.append(model_label)

            except Exception as e:
                print(f"  [Visual] ❌ Gagal {model_label}: {e}")
                blank = gt_img.copy()
                H_b, W_b = blank.shape[:2]
                err_text = f"Inference Error: {type(e).__name__}"
                font_sc = 0.55
                (tw, th), _ = cv2.getTextSize(err_text, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 2)
                cx, cy = (W_b - tw) // 2, (H_b + th) // 2
                cv2.putText(blank, err_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_sc, (0, 0, 200), 2, cv2.LINE_AA)
                panel_images.append(blank)
                panel_labels.append(f"{model_label} ✗ Error")

        # --- Bangun panel grid ---
        panel_path = os.path.join(visuals_dir, f"{base}_{family}_panel.jpg")
        _build_panel_grid(panel_images, panel_labels, panel_path)
        print(f"    🖼️  Panel grid: {os.path.basename(panel_path)}")


# Palet warna title bar per index panel (BGR) — dipakai berurutan
_PANEL_BAR_COLORS = [
    (80,  80,  80),   # 0: Abu-abu tua  → Ground Truth
    (200, 100,   0),  # 1: Biru tua     → Varian 1
    (  0, 180,   0),  # 2: Hijau        → Varian 2
    (  0,   0, 200),  # 3: Merah        → Varian 3
    (200,   0, 200),  # 4: Magenta      → Varian 4
    (  0, 165, 255),  # 5: Oranye       → Varian 5
    (255, 140,   0),  # 6: Biru langit  → Varian 6
    ( 30, 200, 200),  # 7: Kuning emas  → Varian 7
    (180,   0, 180),  # 8: Ungu         → Varian 8
    (  0, 210, 210),  # 9: Teal         → Varian 9
]


def _build_panel_grid(images: list, labels: list, out_path: str, cols: int = 3) -> None:
    """
    Bangun dan simpan panel grid dari daftar gambar.
    Caption nama model diletakkan sebagai COLOR TITLE BAR di ATAS setiap panel
    (konsisten dengan desain comparison grid di eval_unu_helpers.py).

    Parameter
    ---------
    images   : List gambar BGR (numpy array)
    labels   : Label teks untuk tiap sel
    out_path : Path file output .jpg
    cols     : Jumlah kolom grid
    """
    import cv2
    import numpy as np
    import math

    if not images:
        print("  [Panel] ⚠️  Tidak ada gambar untuk dijadikan panel.")
        return

    # Resize semua gambar ke ukuran yang sama
    target_h, target_w = 480, 640
    # Tinggi title bar di atas setiap panel
    bar_h = 32

    resized = []
    for img in images:
        r = cv2.resize(img, (target_w, target_h))
        resized.append(r)

    n    = len(resized)
    rows = math.ceil(n / cols)

    # Padding jika jumlah gambar tidak genap kelipatan cols
    while len(resized) < rows * cols:
        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized.append(blank)
        labels.append("")

    # Tambahkan TITLE BAR di ATAS setiap panel
    # Warna bar putih bersih, teks hitam tebal (merah gelap jika error/N/A)
    paneled = []
    for i, (img, lbl) in enumerate(zip(resized, labels)):
        bar_color = (255, 255, 255) # Putih bersih
        text_color = (0, 0, 0)      # Teks hitam

        if lbl and ("N/A" in lbl or "Error" in lbl or "✗" in lbl or "⚠" in lbl):
            text_color = (0, 0, 180) # Merah gelap untuk status error/N/A

        # Buat bar kosong
        bar = np.full((bar_h, target_w, 3), bar_color, dtype=np.uint8)

        # Tulis teks label di bar
        if lbl:
            # Hitung ukuran teks untuk penempatan vertikal & horizontal center yang tepat
            font_scale = 0.60
            thickness  = 1 # Tidak perlu di-bold
            (tw, th), baseline = cv2.getTextSize(
                lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Posisi X & Y di tengah bar (Horizontal & Vertical Center)
            text_x = (target_w - tw) // 2
            text_y = (bar_h + th) // 2 - baseline // 2
            # Teks utama (tanpa shadow hitam, terpusat sempurna)
            cv2.putText(bar, lbl, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

        # Gabungkan: bar atas + gambar
        cell = np.vstack([bar, img])
        paneled.append(cell)

    # Susun baris dan gabungkan
    row_imgs = []
    for r in range(rows):
        row_slice = paneled[r * cols: (r + 1) * cols]
        row_imgs.append(np.hstack(row_slice))

    grid = np.vstack(row_imgs)
    cv2.imwrite(out_path, grid)


# ==============================================================================
# MODUL 2 — EVALUASI COCOeval: Menghitung metrik per varian
# ==============================================================================

def _infer_worker_det(
    rank: int, gpu_ids: list, pt_path: str,
    img_dir: str, image_ids: dict, tmp_dir: str, _barrier: str
) -> None:
    """Worker inferens deteksi — dijalankan via mp.spawn, simpan hasil ke pickle."""
    import torch
    import torch.multiprocessing as mp
    from ultralytics import YOLO

    gpu = gpu_ids[rank]
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens deteksi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []
    speed_pre, speed_inf, speed_post = [], [], []

    t_start = time.perf_counter()
    for img_path in subset:
        if img_path not in image_ids:
            continue
        try:
            result = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, device=gpu, verbose=False)
            if result:
                spd = result[0].speed
                speed_pre.append(spd.get("preprocess", 0.0))
                speed_inf.append(spd.get("inference",  0.0))
                speed_post.append(spd.get("postprocess", 0.0))
                if result[0].boxes is not None:
                    boxes = result[0].boxes.xyxy.cpu().numpy()
                    confs = result[0].boxes.conf.cpu().numpy()
                    clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                    for box, conf, cls in zip(boxes, confs, clss):
                        preds.append({"image": img_path, "pred_box": box.tolist(),
                                      "pred_cls": int(cls), "pred_conf": float(conf)})
        except Exception as e:
            print(f"  [GPU:{gpu}] ⚠️  Gagal pada {img_path}: {e}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    pkl_path = os.path.join(tmp_dir, f"det_rank{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf, "speed_post": speed_post}, f)
    del model
    _flush_gpu(gpu, f"det rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _infer_worker_seg(
    rank: int, gpu_ids: list, pt_path: str,
    img_dir: str, image_ids: dict, tmp_dir: str
) -> None:
    """Worker inferens segmentasi — dijalankan via mp.spawn, simpan hasil ke pickle."""
    import torch
    from ultralytics import YOLO

    gpu = gpu_ids[rank]
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens segmentasi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []
    speed_pre, speed_inf, speed_post = [], [], []

    t_start = time.perf_counter()
    for img_path in subset:
        if img_path not in image_ids:
            continue
        try:
            result = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, device=gpu, verbose=False)
            if result:
                spd = result[0].speed
                speed_pre.append(spd.get("preprocess", 0.0))
                speed_inf.append(spd.get("inference",  0.0))
                speed_post.append(spd.get("postprocess", 0.0))
                if result[0].masks is not None:
                    masks = result[0].masks.data.cpu().numpy()
                    boxes = result[0].boxes.xyxy.cpu().numpy()
                    confs = result[0].boxes.conf.cpu().numpy()
                    clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                    for mask, box, conf, cls in zip(masks, boxes, confs, clss):
                        preds.append({"image": img_path, "pred_mask": mask,
                                      "pred_box": box.tolist(),
                                      "pred_cls": int(cls), "pred_conf": float(conf)})
        except Exception as e:
            print(f"  [GPU:{gpu}] ⚠️  Gagal pada {img_path}: {e}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi-mask dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    pkl_path = os.path.join(tmp_dir, f"seg_rank{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf, "speed_post": speed_post}, f)
    del model
    _flush_gpu(gpu, f"seg rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _infer_worker_maskrcnn_spawn(
    rank: int, gpu_ids: list, pt_path: str,
    img_dir: str, image_ids: dict, tmp_dir: str
) -> None:
    """Worker inferens Mask R-CNN — memproses subset gambar secara paralel."""
    import torch
    import torchvision
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage
    import cv2
    import numpy as np
    
    gpu = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens Mask R-CNN...", flush=True)

    # Tambahkan path pencarian agar dapat mengimpor maskrcnn_builder
    sys.path.insert(0, os.path.join(ROOT, "mask-r-cnn"))
    from maskrcnn_builder import build_model
    
    model = build_model(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds = []
    speed_pre, speed_inf, speed_post = [], [], []
    
    t_start = time.perf_counter()
    with torch.no_grad():
        for img_path in subset:
            if img_path not in image_ids:
                continue
            try:
                # Preprocess
                t0 = time.perf_counter()
                pil = PILImage.open(img_path).convert("RGB")
                W, H = pil.size
                t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
                t1 = time.perf_counter()
                
                # Inference
                preds_output = model(t_img)[0]
                t2 = time.perf_counter()
                
                # Postprocess
                scores = preds_output["scores"].cpu().numpy()
                keep = scores >= 0.5
                
                boxes  = preds_output["boxes"].cpu().numpy()[keep]
                confs  = scores[keep]
                labels = preds_output["labels"].cpu().numpy()[keep] - 1  # 0-indexed
                masks  = preds_output["masks"].cpu().numpy()[keep]
                
                for i in range(len(confs)):
                    score = float(confs[i])
                    cat_id = int(labels[i])
                    x1, y1, x2, y2 = boxes[i].tolist()
                    
                    bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                    if bin_mask.shape != (H, W):
                        bin_mask = cv2.resize(bin_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    preds.append({
                        "image": img_path,
                        "pred_mask": bin_mask > 0.5,
                        "pred_box": [x1, y1, x2, y2],
                        "pred_cls": cat_id,
                        "pred_conf": score
                    })
                t3 = time.perf_counter()
                
                speed_pre.append((t1 - t0) * 1000)
                speed_inf.append((t2 - t1) * 1000)
                speed_post.append((t3 - t2) * 1000)
            except Exception as e:
                print(f"  [GPU:{gpu}] ⚠️ Gagal pada {img_path}: {e}", flush=True)
                
    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi-mask dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    pkl_path = os.path.join(tmp_dir, f"seg_rank{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf, "speed_post": speed_post}, f)
    
    del model
    _flush_gpu(gpu, f"maskrcnn rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _infer_worker_hybrid_spawn(
    rank: int, gpu_ids: list, pt_path: str,
    img_dir: str, image_ids: dict, tmp_dir: str, model_key: str
) -> None:
    """Worker inferens Hybrid (YOLO + SAM2/MobileSAM) — memproses subset gambar secara paralel."""
    import torch
    from ultralytics import YOLO, SAM
    import cv2
    import numpy as np
    
    gpu = gpu_ids[rank]
    device_str = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens Hybrid dengan model_key={model_key}...", flush=True)

    # 1. Load YOLO model
    model = YOLO(pt_path)
    
    # 2. Load SAM model dynamically
    is_mobile = "_mobile" in model_key
    sam_pt = MOBILE_SAM_MODEL_PATH if is_mobile else SAM_MODEL_PATH
    sam_model = SAM(sam_pt)
    
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds = []
    speed_pre, speed_inf, speed_post = [], [], []
    
    t_start = time.perf_counter()
    for img_path in subset:
        if img_path not in image_ids:
            continue
        try:
            # YOLO predict dynamically computes speed metrics
            res = model.predict(img_path, conf=0.5, iou=0.6, imgsz=IMAGE_SIZE, device=device_str, verbose=False)[0]
            spd = res.speed
            spd_pre = spd.get("preprocess", 0.0)
            spd_inf = spd.get("inference", 0.0)
            spd_post = spd.get("postprocess", 0.0)
            
            if res.boxes is not None and len(res.boxes) > 0:
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_confs = res.boxes.conf.cpu().numpy()
                pred_clss = res.boxes.cls.cpu().numpy().astype(int)
                
                # Refine bounding boxes using SAM2
                sam_st = time.perf_counter()
                sam_res = sam_model.predict(res.orig_img, bboxes=res.boxes.xyxy, verbose=False)
                sam_elapsed_ms = (time.perf_counter() - sam_st) * 1000
                
                # Add SAM2 inference speed to overall inference latency
                spd_inf += sam_elapsed_ms
                
                if sam_res and sam_res[0].masks is not None:
                    m_data = sam_res[0].masks.data.cpu().numpy()
                    H, W = sam_res[0].orig_img.shape[:2]
                    for idx_box in range(len(pred_boxes)):
                        score = float(pred_confs[idx_box])
                        cat_id = int(pred_clss[idx_box])
                        box_xyxy = pred_boxes[idx_box].tolist()
                        
                        if idx_box < len(m_data):
                            m = m_data[idx_box]
                            if m.shape != (H, W):
                                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                            mask_bin = m > 0.5
                        else:
                            mask_bin = np.zeros((H, W), dtype=bool)
                            
                        preds.append({
                            "image": img_path,
                            "pred_mask": mask_bin,
                            "pred_box": box_xyxy,
                            "pred_cls": cat_id,
                            "pred_conf": score
                        })
            
            speed_pre.append(spd_pre)
            speed_inf.append(spd_inf)
            speed_post.append(spd_post)
        except Exception as e:
            print(f"  [GPU:{gpu}] ⚠️ Gagal pada {img_path}: {e}", flush=True)
            
    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi-mask dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    pkl_path = os.path.join(tmp_dir, f"seg_rank{rank}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf, "speed_post": speed_post}, f)
    
    del model, sam_model
    _flush_gpu(gpu, f"hybrid rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _collect_pkl_results(tmp_dir: str, prefix: str, world_size: int) -> dict:

    """Kumpulkan dan merge hasil pickle dari semua rank."""
    all_preds, all_pre_t, all_inf_t, all_post_t = [], [], [], []
    total_imgs = 0
    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"{prefix}_rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            all_preds.extend(data["preds"])
            total_imgs += data["n_imgs"]
            all_pre_t.extend(data.get("speed_pre",  []))
            all_inf_t.extend(data.get("speed_inf",  []))
            all_post_t.extend(data.get("speed_post", []))
    return {
        "preds": all_preds,
        "n_imgs": total_imgs,
        "avg_pre_ms":  _avg_ms(all_pre_t),
        "avg_inf_ms":  _avg_ms(all_inf_t),
        "avg_post_ms": _avg_ms(all_post_t),
    }


def _get_model_metrics(model_key: str, family: str) -> dict:
    """Mengembalikan dict berisi weights_size_mb dan parameters_m secara dinamis."""
    result = {"weights_size_mb": "N/A", "parameters_m": "N/A"}
    try:
        import torch
        # 1. Tentukan path model
        is_hybrid = model_key.startswith("hybrid_")
        is_maskrcnn = (family == "maskrcnn")
        
        if is_maskrcnn:
            pt = os.path.join(WORKSPACE_DIR, "runs", "maskrcnn", "weights", "best.pt")
        elif is_hybrid:
            yolo_base_key = model_key.replace("hybrid_", "").replace("_mobile", "")
            if yolo_base_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif yolo_base_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(yolo_base_key), "weights", "best.pt")
        else:
            if model_key == "yolo11l":
                pt = YOLO11L_DET_PATH
            elif model_key == "yolo11l_seg":
                pt = YOLO11L_SEG_PATH
            else:
                pt = os.path.join(get_output_dir(model_key), "weights", "best.pt")
                
        # 2. Hitung parameter & weight size
        if is_maskrcnn:
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
            in_box = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
            in_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
            m.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
            if os.path.exists(pt):
                m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
            total_params = sum(p.nelement() for p in m.parameters())
            sz = sum(p.nelement() * p.element_size() for p in m.parameters()) + sum(b.nelement() * b.element_size() for b in m.buffers())
            del m; gc.collect()
            result["weights_size_mb"] = round(sz / 1024**2, 2)
            result["parameters_m"] = round(total_params / 1e6, 2)
        elif is_hybrid:
            from ultralytics import YOLO, SAM
            y_m = YOLO(pt)
            
            is_mobile = "_mobile" in model_key
            sam_pt = MOBILE_SAM_MODEL_PATH if is_mobile else SAM_MODEL_PATH
            s_m = SAM(sam_pt)
            
            y_params = sum(p.nelement() for p in y_m.model.parameters())
            s_params = sum(p.nelement() for p in s_m.model.parameters())
            y_sz = sum(p.nelement() * p.element_size() for p in y_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in y_m.model.buffers())
            s_sz = sum(p.nelement() * p.element_size() for p in s_m.model.parameters()) + sum(b.nelement() * b.element_size() for b in s_m.model.buffers())
            del y_m, s_m; gc.collect()
            result["weights_size_mb"] = round((y_sz + s_sz) / 1024**2, 2)
            result["parameters_m"] = round((y_params + s_params) / 1e6, 2)
        else:
            from ultralytics import YOLO
            m = YOLO(pt)
            total_params = sum(p.nelement() for p in m.model.parameters())
            sz = sum(p.nelement() * p.element_size() for p in m.model.parameters()) + sum(b.nelement() * b.element_size() for b in m.model.buffers())
            del m; gc.collect()
            result["weights_size_mb"] = round(sz / 1024**2, 2)
            result["parameters_m"] = round(total_params / 1e6, 2)
    except Exception as e:
        print(f"  ⚠️ Gagal hitung ukuran/parameter untuk {model_key}: {e}")
    return result


def _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, predictions, iou_type="bbox") -> dict:
    """
    Evaluate predictions using COCOeval and return a dictionary of metrics.
    Metrics returned: mAP50, mAP50_95, precision, recall
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import copy
    import numpy as np

    result = {
        "mAP50": "N/A",
        "mAP50_95": "N/A",
        "precision": "N/A",
        "recall": "N/A"
    }

    if not predictions:
        return result

    try:
        # Filter empty segmentations for segm evaluation to prevent pycocotools crash
        if iou_type == "segm":
            coco_gt_dict = copy.deepcopy(coco_gt_dict)
            valid_anns = []
            for ann in coco_gt_dict.get("annotations", []):
                seg = ann.get("segmentation")
                if seg and (isinstance(seg, dict) or (isinstance(seg, list) and len(seg) > 0)):
                    valid_anns.append(ann)
            coco_gt_dict["annotations"] = valid_anns

        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        
        coco_preds = []
        for pred in predictions:
            img_path = pred["image"]
            if img_path not in image_ids:
                continue
            
            img_id = image_ids[img_path]
            cls = pred["pred_cls"]
            conf = pred["pred_conf"]
            
            if iou_type == "bbox":
                bbox = pred["pred_box"]
                x1, y1, x2, y2 = bbox
                coco_preds.append({
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": conf,
                })
            elif iou_type == "segm":
                if "pred_mask" not in pred or pred["pred_mask"] is None:
                    continue
                mask = pred["pred_mask"]
                if isinstance(mask, dict):
                    rle = mask
                else:
                    from pycocotools import mask as maskUtils
                    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                
                bbox = pred["pred_box"]
                x1, y1, x2, y2 = bbox
                coco_preds.append({
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": rle,
                    "score": conf,
                })
                
        if not coco_preds:
            return result

        coco_dt = coco_gt.loadRes(coco_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        mAP50 = round(float(coco_eval.stats[1]), 4)
        mAP50_95 = round(float(coco_eval.stats[0]), 4)
        recall = round(float(coco_eval.stats[8]), 4)
        
        # Precision IoU=0.50
        p_matrix = coco_eval.eval['precision']
        p_iou50 = p_matrix[0, :, :, 0, 2] # IoU=0.50, all areas, maxDets=100
        p_valid = p_iou50[p_iou50 > -1]
        precision = round(float(np.mean(p_valid)), 4) if len(p_valid) > 0 else "N/A"
        
        result["mAP50"] = mAP50
        result["mAP50_95"] = mAP50_95
        result["precision"] = precision
        result["recall"] = recall
    except Exception as e:
        print(f"  [COCOeval-Extended] ⚠️ Gagal evaluasi {iou_type}: {e}")
        
    return result


def evaluate_variant_detection(
    gpu_ids: list, tmp_dir: str, model_key: str, model_label: str
) -> Optional[dict]:
    """
    Evaluasi COCOeval untuk satu varian model deteksi.

    Returns: dict row laporan, atau None jika gagal.
    """
    import torch
    import torch.multiprocessing as mp

    print(f"\n{'='*65}")
    print(f"  Eval Detection: {model_label}")
    print(f"{'='*65}")

    pt_path = os.path.join(get_output_dir(model_key), "weights", "best.pt")
    if not os.path.exists(pt_path):
        msg = f"best.pt tidak ditemukan untuk {model_label}: {pt_path}"
        print(f"  ❌ {msg}")
        send_telegram_msg(f"❌ <b>{model_label} Eval</b>\n{msg}")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools belum terinstal. Jalankan: pip install pycocotools")
        return None

    print("  [GT] Membangun COCO ground truth deteksi...")
    eval_valid_dir = os.path.join(EVAL_DATASET_LOCATION, "valid")
    coco_json_path = os.path.join(eval_valid_dir, "_annotations.coco.json")
    if os.path.exists(coco_json_path):
        print(f"  [GT] Menggunakan dataset terpadu COCO dari {eval_valid_dir}...")
        img_dir = eval_valid_dir
        coco_gt_dict, image_ids = load_native_coco_gt(img_dir)
        if image_ids:
            image_ids = {os.path.join(img_dir, k): v for k, v in image_ids.items()}
    else:
        coco_gt_dict, image_ids = build_coco_ground_truth(DET_YAML, split="valid")
        img_dir = _resolve_img_dir(DET_YAML)

    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker deteksi...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker_det,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir, ""),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall

    collected   = _collect_pkl_results(tmp_dir, "det", world_size)
    all_preds   = collected["preds"]
    total_imgs  = collected["n_imgs"]
    avg_pre_ms  = collected["avg_pre_ms"]
    avg_inf_ms  = collected["avg_inf_ms"]
    avg_post_ms = collected["avg_post_ms"]

    print(f"  [Collect] {len(all_preds)} prediksi dari {total_imgs} gambar")

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_per_img_ms = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"

    print("  [COCOeval] Menghitung mAP50 & mAP50-95...")
    coco_metrics = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    mAP50 = coco_metrics["mAP50"]
    mAP50_95 = coco_metrics["mAP50_95"]
    prec = coco_metrics["precision"]
    rec = coco_metrics["recall"]
    print(f"  ✅ mAP50={mAP50}  mAP50-95={mAP50_95}  Precision={prec}  Recall={rec}")

    metrics = _get_model_metrics(model_key, family="yolo_det")
    weights_size_mb = metrics["weights_size_mb"]
    parameters_m = metrics["parameters_m"]

    return {
        "Model":            model_label,
        "Model Size (MB)":  weights_size_mb,
        "Parameters (M)":   parameters_m,
        "mAP50-95":         mAP50_95,
        "mAP50":            mAP50,
        "Precision":        prec,
        "Recall":           rec,
        "Preprocess (ms)":  avg_pre_ms,
        "Inference (ms)":   avg_inf_ms,
        "Postprocess (ms)": avg_post_ms,
        "Latency (ms)":     lat_per_img_ms,
        "FPS":              throughput_fps,
        "GPUs":             _gpu_report_str(gpu_ids),
        "Evaluator":        "COCOeval (MultiGPU)",
    }


def evaluate_variant_segmentation(
    gpu_ids: list, tmp_dir: str, model_key: str, model_label: str
) -> Optional[dict]:
    """
    Evaluasi COCOeval untuk satu varian model segmentasi.

    Returns: dict row laporan, atau None jika gagal.
    """
    import torch
    import torch.multiprocessing as mp

    print(f"\n{'='*65}")
    print(f"  Eval Segmentation: {model_label}")
    print(f"{'='*65}")

    pt_path = os.path.join(get_output_dir(model_key), "weights", "best.pt")
    if not os.path.exists(pt_path):
        msg = f"best.pt tidak ditemukan untuk {model_label}: {pt_path}"
        print(f"  ❌ {msg}")
        send_telegram_msg(f"❌ <b>{model_label} Eval</b>\n{msg}")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools belum terinstal. Jalankan: pip install pycocotools")
        return None

    print("  [GT] Membangun COCO ground truth segmentasi...")
    eval_valid_dir = os.path.join(EVAL_DATASET_LOCATION, "valid")
    coco_json_path = os.path.join(eval_valid_dir, "_annotations.coco.json")
    if os.path.exists(coco_json_path):
        print(f"  [GT] Menggunakan dataset terpadu COCO dari {eval_valid_dir}...")
        img_dir = eval_valid_dir
        coco_gt_dict, image_ids = load_native_coco_gt(img_dir)
        if image_ids:
            image_ids = {os.path.join(img_dir, k): v for k, v in image_ids.items()}
    else:
        coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
        img_dir = _resolve_img_dir(SEG_YAML)

    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker_seg,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall

    collected   = _collect_pkl_results(tmp_dir, "seg", world_size)
    all_preds   = collected["preds"]
    total_imgs  = collected["n_imgs"]
    avg_pre_ms  = collected["avg_pre_ms"]
    avg_inf_ms  = collected["avg_inf_ms"]
    avg_post_ms = collected["avg_post_ms"]

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_per_img_ms = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"

    print("  [COCOeval] Menghitung mAP Mask & Box...")
    coco_metrics_seg = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="segm")
    mAP50_mask = coco_metrics_seg["mAP50"]
    mAP50_95_mask = coco_metrics_seg["mAP50_95"]
    prec_mask = coco_metrics_seg["precision"]
    rec_mask = coco_metrics_seg["recall"]

    coco_metrics_box = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    mAP50_box = coco_metrics_box["mAP50"]
    mAP50_95_box = coco_metrics_box["mAP50_95"]
    prec_box = coco_metrics_box["precision"]
    rec_box = coco_metrics_box["recall"]

    print(f"  ✅ Box: mAP50={mAP50_box} mAP50-95={mAP50_95_box} | Mask: mAP50={mAP50_mask} mAP50-95={mAP50_95_mask}")

    metrics = _get_model_metrics(model_key, family="yolo_seg")
    weights_size_mb = metrics["weights_size_mb"]
    parameters_m = metrics["parameters_m"]

    return {
        "Model":            model_label,
        "Model Size (MB)":  weights_size_mb,
        "Parameters (M)":   parameters_m,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50(Box)":        mAP50_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "mAP50(Mask)":       mAP50_mask,
        "Precision(Box)":    prec_box,
        "Recall(Box)":       rec_box,
        "Precision(Mask)":   prec_mask,
        "Recall(Mask)":      rec_mask,
        "Preprocess (ms)":  avg_pre_ms,
        "Inference (ms)":   avg_inf_ms,
        "Postprocess (ms)": avg_post_ms,
        "Latency (ms)":     lat_per_img_ms,
        "FPS":              throughput_fps,
        "GPUs":             _gpu_report_str(gpu_ids),
        "Evaluator":        "COCOeval (MultiGPU)",
    }


def evaluate_maskrcnn_segmentation(
    gpu_ids: list, tmp_dir: str, model_key: str, model_label: str
) -> Optional[dict]:
    """
    Evaluasi COCOeval untuk Mask R-CNN.
    """
    import torch
    import torch.multiprocessing as mp

    print(f"\n{'='*65}")
    print(f"  Eval Mask R-CNN: {model_label}")
    print(f"{'='*65}")

    pt_path = os.path.join(WORKSPACE_DIR, "runs", "maskrcnn", "weights", "best.pt")
    if not os.path.exists(pt_path):
        msg = f"best.pt tidak ditemukan untuk {model_label}: {pt_path}"
        print(f"  ❌ {msg}")
        send_telegram_msg(f"❌ <b>{model_label} Eval</b>\n{msg}")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools belum terinstal.")
        return None

    print("  [GT] Membangun COCO ground truth segmentasi...")
    eval_valid_dir = os.path.join(EVAL_DATASET_LOCATION, "valid")
    coco_json_path = os.path.join(eval_valid_dir, "_annotations.coco.json")
    if os.path.exists(coco_json_path):
        print(f"  [GT] Menggunakan dataset terpadu COCO dari {eval_valid_dir}...")
        img_dir = eval_valid_dir
        coco_gt_dict, image_ids = load_native_coco_gt(img_dir)
        if image_ids:
            image_ids = {os.path.join(img_dir, k): v for k, v in image_ids.items()}
    else:
        coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
        img_dir = _resolve_img_dir(SEG_YAML)

    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi Mask R-CNN...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker_maskrcnn_spawn,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall

    collected   = _collect_pkl_results(tmp_dir, "seg", world_size)
    all_preds   = collected["preds"]
    total_imgs  = collected["n_imgs"]
    avg_pre_ms  = collected["avg_pre_ms"]
    avg_inf_ms  = collected["avg_inf_ms"]
    avg_post_ms = collected["avg_post_ms"]

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else 0.0
    lat_per_img_ms = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else 0.0

    print("  [COCOeval] Menghitung mAP Mask & Box...")
    coco_metrics_seg = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="segm")
    mAP50_mask = coco_metrics_seg["mAP50"]
    mAP50_95_mask = coco_metrics_seg["mAP50_95"]
    prec_mask = coco_metrics_seg["precision"]
    rec_mask = coco_metrics_seg["recall"]

    coco_metrics_box = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    mAP50_box = coco_metrics_box["mAP50"]
    mAP50_95_box = coco_metrics_box["mAP50_95"]
    prec_box = coco_metrics_box["precision"]
    rec_box = coco_metrics_box["recall"]

    print(f"  ✅ Box: mAP50={mAP50_box} mAP50-95={mAP50_95_box} | Mask: mAP50={mAP50_mask} mAP50-95={mAP50_95_mask}")

    metrics = _get_model_metrics(model_key, family="maskrcnn")
    weights_size_mb = metrics["weights_size_mb"]
    parameters_m = metrics["parameters_m"]

    return {
        "Model":            model_label,
        "Model Size (MB)":  weights_size_mb,
        "Parameters (M)":   parameters_m,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50(Box)":        mAP50_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "mAP50(Mask)":       mAP50_mask,
        "Precision(Box)":    prec_box,
        "Recall(Box)":       rec_box,
        "Precision(Mask)":   prec_mask,
        "Recall(Mask)":      rec_mask,
        "Preprocess (ms)":  avg_pre_ms,
        "Inference (ms)":   avg_inf_ms,
        "Postprocess (ms)": avg_post_ms,
        "Latency (ms)":     lat_per_img_ms,
        "FPS":              throughput_fps,
        "GPUs":             _gpu_report_str(gpu_ids),
        "Evaluator":        "COCOeval (MultiGPU)",
    }


def evaluate_hybrid_segmentation(
    gpu_ids: list, tmp_dir: str, model_key: str, model_label: str
) -> Optional[dict]:
    """
    Evaluasi COCOeval untuk Hybrid (YOLO + SAM2/MobileSAM).
    """
    import torch
    import torch.multiprocessing as mp

    print(f"\n{'='*65}")
    print(f"  Eval Hybrid: {model_label}")
    print(f"{'='*65}")

    # Resolve yolo_key from hybrid key (e.g. hybrid_yolov8m_mobile -> yolov8m)
    yolo_key = model_key.replace("hybrid_", "").replace("_mobile", "")
    pt_path = os.path.join(get_output_dir(yolo_key), "weights", "best.pt")
    if not os.path.exists(pt_path):
        msg = f"best.pt tidak ditemukan untuk {model_label}: {pt_path}"
        print(f"  ❌ {msg}")
        send_telegram_msg(f"❌ <b>{model_label} Eval</b>\n{msg}")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools belum terinstal.")
        return None

    print("  [GT] Membangun COCO ground truth segmentasi...")
    eval_valid_dir = os.path.join(EVAL_DATASET_LOCATION, "valid")
    coco_json_path = os.path.join(eval_valid_dir, "_annotations.coco.json")
    if os.path.exists(coco_json_path):
        print(f"  [GT] Menggunakan dataset terpadu COCO dari {eval_valid_dir}...")
        img_dir = eval_valid_dir
        coco_gt_dict, image_ids = load_native_coco_gt(img_dir)
        if image_ids:
            image_ids = {os.path.join(img_dir, k): v for k, v in image_ids.items()}
    else:
        coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
        img_dir = _resolve_img_dir(SEG_YAML)

    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi Hybrid...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker_hybrid_spawn,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir, model_key),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall

    collected   = _collect_pkl_results(tmp_dir, "seg", world_size)
    all_preds   = collected["preds"]
    total_imgs  = collected["n_imgs"]
    avg_pre_ms  = collected["avg_pre_ms"]
    avg_inf_ms  = collected["avg_inf_ms"]
    avg_post_ms = collected["avg_post_ms"]

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else 0.0
    lat_per_img_ms = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else 0.0

    print("  [COCOeval] Menghitung mAP Mask & Box...")
    coco_metrics_seg = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="segm")
    mAP50_mask = coco_metrics_seg["mAP50"]
    mAP50_95_mask = coco_metrics_seg["mAP50_95"]
    prec_mask = coco_metrics_seg["precision"]
    rec_mask = coco_metrics_seg["recall"]

    coco_metrics_box = _evaluate_coco_predictions_extended(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    mAP50_box = coco_metrics_box["mAP50"]
    mAP50_95_box = coco_metrics_box["mAP50_95"]
    prec_box = coco_metrics_box["precision"]
    rec_box = coco_metrics_box["recall"]

    print(f"  ✅ Box: mAP50={mAP50_box} mAP50-95={mAP50_95_box} | Mask: mAP50={mAP50_mask} mAP50-95={mAP50_95_mask}")

    metrics = _get_model_metrics(model_key, family="hybrid")
    weights_size_mb = metrics["weights_size_mb"]
    parameters_m = metrics["parameters_m"]

    return {
        "Model":            model_label,
        "Model Size (MB)":  weights_size_mb,
        "Parameters (M)":   parameters_m,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50(Box)":        mAP50_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "mAP50(Mask)":       mAP50_mask,
        "Precision(Box)":    prec_box,
        "Recall(Box)":       rec_box,
        "Precision(Mask)":   prec_mask,
        "Recall(Mask)":      rec_mask,
        "Preprocess (ms)":  avg_pre_ms,
        "Inference (ms)":   avg_inf_ms,
        "Postprocess (ms)": avg_post_ms,
        "Latency (ms)":     lat_per_img_ms,
        "FPS":              throughput_fps,
        "GPUs":             _gpu_report_str(gpu_ids),
        "Evaluator":        "COCOeval (MultiGPU)",
    }


# ==============================================================================
# MODUL 3 — KOMPILASI CSV: Per-varian, per-model, ALL
# ==============================================================================

# Definisi kolom CSV untuk masing-masing tipe tugas
_DET_FIELDS = [
    "Model", "Model Size (MB)", "Parameters (M)", "mAP50-95", "mAP50",
    "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
    "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
]
_SEG_FIELDS = [
    "Model", "Model Size (MB)", "Parameters (M)",
    "mAP50-95(Box)", "mAP50(Box)", "mAP50-95(Mask)", "mAP50(Mask)",
    "Precision(Box)", "Recall(Box)", "Precision(Mask)", "Recall(Mask)",
    "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)",
    "Latency (ms)", "FPS", "GPUs", "Evaluator"
]


def _write_csv(path: str, fields: list, rows: list[dict], mode: str = "w") -> None:
    """Tulis satu atau lebih row ke file CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = mode == "w" or not os.path.exists(path)
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_variant_csv(family: str, model_key: str, task: str, row: dict) -> str:
    """
    Simpan CSV per-varian (tanpa prefix).
    Contoh: reports/pipeline/csv/yolo8/yolov8m_detection.csv
    """
    csv_dir  = _get_csv_dir(family)
    filename = f"{model_key}_{task}.csv"
    path     = os.path.join(csv_dir, filename)
    fields   = _DET_FIELDS if task == "detection" else _SEG_FIELDS
    _write_csv(path, fields, [row], mode="w")
    print(f"  📄 Variant CSV: {path}")
    return path


def compile_family_csv(
    family: str,
    all_det_rows: list[dict],
    all_seg_rows: list[dict],
) -> None:
    """
    Simpan CSV kompilasi per-family (tanpa prefix).
    Contoh: reports/pipeline/csv/kompilasi_yolo8_detection.csv
    """
    csv_base = os.path.join(REPORTS_DIR, "csv")
    os.makedirs(csv_base, exist_ok=True)

    if all_det_rows:
        path = os.path.join(csv_base, f"kompilasi_{family}_detection.csv")
        _write_csv(path, _DET_FIELDS, all_det_rows, mode="w")
        print(f"  📑 Kompilasi Family DET: {path}")

    if all_seg_rows:
        path = os.path.join(csv_base, f"kompilasi_{family}_segmentation.csv")
        _write_csv(path, _SEG_FIELDS, all_seg_rows, mode="w")
        print(f"  📑 Kompilasi Family SEG: {path}")


def compile_all_csv(
    global_det_rows: list[dict],
    global_seg_rows: list[dict],
) -> None:
    """
    Simpan CSV kompilasi ALL dari seluruh family.
    Output: reports/pipeline/csv/kompilasi_ALL_detection.csv
             reports/pipeline/csv/kompilasi_ALL_segmentation.csv
    """
    csv_base = os.path.join(REPORTS_DIR, "csv")
    os.makedirs(csv_base, exist_ok=True)

    if global_det_rows:
        path = os.path.join(csv_base, "kompilasi_ALL_detection.csv")
        _write_csv(path, _DET_FIELDS, global_det_rows, mode="w")
        print(f"\n  🗂️  Kompilasi ALL DET: {path}")

    if global_seg_rows:
        path = os.path.join(csv_base, "kompilasi_ALL_segmentation.csv")
        _write_csv(path, _SEG_FIELDS, global_seg_rows, mode="w")
        print(f"  🗂️  Kompilasi ALL SEG: {path}")


# Definisi kolom CSV gabungan (single model) — field unified detection + segmentation
_SINGLE_MODEL_FIELDS = CSV_REPORT_FIELDS


def compile_all_single_model_csv(
    global_det_rows: list[dict],
    global_seg_rows: list[dict],
) -> None:
    """
    Gabungkan kompilasi_ALL_detection.csv dan kompilasi_ALL_segmentation.csv
    ke dalam satu file kompilasi_ALL_single_model.csv dengan field unified.

    Field mapping:
      - Detection: mAP50-95 -> mAP50-95 (Box), mAP50 -> mAP50 (Box), Mask fields N/A
      - Segmentation: mAP50-95(Box) -> mAP50-95 (Box), mAP50-95(Mask) -> mAP50-95 (Mask)
        mAP50 (Box) dan mAP50 (Mask) diisi N/A (karena pipeline seg tidak menghitung mAP50)
      - Weights Size (MB) dan Parameters (M) diambil dari "Model Size (MB)" bila tersedia
      Output: reports/pipeline/csv/kompilasi_ALL_single_model.csv
    """
    csv_base = os.path.join(REPORTS_DIR, "csv")
    os.makedirs(csv_base, exist_ok=True)

    unified_rows: list[dict] = []

    # --- Mapping baris Detection ---
    for row in global_det_rows:
        unified_rows.append({
            "Model":             row.get("Model", "N/A"),
            "Weights Size (MB)": row.get("Model Size (MB)", "N/A"),
            "Parameters (M)":    row.get("Parameters (M)", "N/A"),
            "mAP50-95(Box)":     row.get("mAP50-95", "N/A"),
            "mAP50(Box)":        row.get("mAP50", "N/A"),
            "mAP50-95(Mask)":    "N/A",
            "mAP50(Mask)":       "N/A",
            "Precision(Box)":    row.get("Precision", "N/A"),
            "Recall(Box)":       row.get("Recall", "N/A"),
            "Precision(Mask)":   "N/A",
            "Recall(Mask)":      "N/A",
            "Preprocess (ms)":   row.get("Preprocess (ms)", "N/A"),
            "Inference (ms)":    row.get("Inference (ms)", "N/A"),
            "Postprocess (ms)":  row.get("Postprocess (ms)", "N/A"),
            "Latency (ms)":      row.get("Latency (ms)", "N/A"),
            "FPS":               row.get("FPS", "N/A"),
            "GPUs":              row.get("GPUs", "N/A"),
            "Evaluator":         row.get("Evaluator", "N/A"),
        })

    # --- Mapping baris Segmentation ---
    for row in global_seg_rows:
        unified_rows.append({
            "Model":             row.get("Model", "N/A"),
            "Weights Size (MB)": row.get("Model Size (MB)", "N/A"),
            "Parameters (M)":    row.get("Parameters (M)", "N/A"),
            "mAP50-95(Box)":     row.get("mAP50-95(Box)", "N/A"),
            "mAP50(Box)":        row.get("mAP50(Box)", "N/A"),
            "mAP50-95(Mask)":    row.get("mAP50-95(Mask)", "N/A"),
            "mAP50(Mask)":       row.get("mAP50(Mask)", "N/A"),
            "Precision(Box)":    row.get("Precision(Box)", "N/A"),
            "Recall(Box)":       row.get("Recall(Box)", "N/A"),
            "Precision(Mask)":   row.get("Precision(Mask)", "N/A"),
            "Recall(Mask)":      row.get("Recall(Mask)", "N/A"),
            "Preprocess (ms)":   row.get("Preprocess (ms)", "N/A"),
            "Inference (ms)":    row.get("Inference (ms)", "N/A"),
            "Postprocess (ms)":  row.get("Postprocess (ms)", "N/A"),
            "Latency (ms)":      row.get("Latency (ms)", "N/A"),
            "FPS":               row.get("FPS", "N/A"),
            "GPUs":              row.get("GPUs", "N/A"),
            "Evaluator":         row.get("Evaluator", "N/A"),
        })

    if unified_rows:
        path = os.path.join(csv_base, "kompilasi_ALL_single_model.csv")
        _write_csv(path, _SINGLE_MODEL_FIELDS, unified_rows, mode="w")
        print(f"  🗂️  Kompilasi ALL Single Model: {path}")
    else:
        print("  ⚠️  Tidak ada data untuk kompilasi_ALL_single_model.csv (det & seg kosong).")


# ==============================================================================
# PIPELINE UTAMA — Jalankan per family
# ==============================================================================

def run_family(
    family: str,
    gpu_ids: list[int],
    skip_eval: bool = False,
    skip_visual: bool = False,
    variant_filter: Optional[list[str]] = None,
    n_samples: int = VISUAL_NUM_SAMPLES,
) -> tuple[list[dict], list[dict]]:
    """
    Jalankan evaluasi + visualisasi untuk satu family model.

    Returns
    -------
    (det_rows, seg_rows) — list row laporan yang berhasil dihitung.
    """
    if family not in FAMILY_VARIANTS:
        print(f"[Pipeline] ❌ Family '{family}' tidak dikenali. "
              f"Pilihan: {list(FAMILY_VARIANTS.keys())}")
        return [], []

    variants = FAMILY_VARIANTS[family]

    # Terapkan filter varian jika ada
    if variant_filter:
        variants = [v for v in variants if v["key"] in variant_filter]
        if not variants:
            print(f"[Pipeline] ⚠️  Tidak ada varian yang cocok dengan filter: {variant_filter}")
            return [], []

    print(f"\n{'═'*70}")
    print(f"  🚀 FAMILY: {family.upper()} | {len(variants)} varian | GPUs: {gpu_ids}")
    print(f"{'═'*70}")

    send_telegram_msg(
        f"🚀 <b>Report Pipeline Dimulai</b>\n"
        f"Family: <code>{family}</code>\n"
        f"Varian: <code>{[v['label'] for v in variants]}</code>\n"
        f"GPUs: <code>{gpu_ids}</code>"
    )

    family_det_rows: list[dict] = []
    family_seg_rows: list[dict] = []

    # Buat tmp dir bersama untuk pickle antar worker
    with tempfile.TemporaryDirectory(prefix=f"{family}_eval_") as tmp_dir:
        for spec in variants:
            model_key   = spec["key"]
            model_label = spec["label"]
            task        = spec["task"]
            task_name   = "detection" if task == "det" else "segmentation"

            print(f"\n[Pipeline] Varian: {model_label} (task={task_name})")

            row = None
            if not skip_eval:
                try:
                    if family == "maskrcnn":
                        row = evaluate_maskrcnn_segmentation(gpu_ids, tmp_dir, model_key, model_label)
                    elif family == "hybrid":
                        row = evaluate_hybrid_segmentation(gpu_ids, tmp_dir, model_key, model_label)
                    elif task == "det":
                        row = evaluate_variant_detection(gpu_ids, tmp_dir, model_key, model_label)
                    else:
                        row = evaluate_variant_segmentation(gpu_ids, tmp_dir, model_key, model_label)
                except Exception as e:
                    print(f"[Pipeline] ❌ Evaluasi gagal untuk {model_label}: {e}")

                if row:
                    # Simpan CSV per-varian
                    save_variant_csv(family, model_key, task_name, row)
                    if task == "det":
                        family_det_rows.append(row)
                    else:
                        family_seg_rows.append(row)

                    send_telegram_msg(
                        f"✅ <b>{model_label} Selesai</b>\n"
                        f"Task: <code>{task_name}</code>\n"
                        + (f"mAP50-95: <code>{row.get('mAP50-95', 'N/A')}</code>\n"
                           f"Precision: <code>{row.get('Precision', 'N/A')}</code>\n"
                           f"Recall: <code>{row.get('Recall', 'N/A')}</code>\n"
                           if task == "det" else
                           f"mAP50-95(Box): <code>{row.get('mAP50-95(Box)', 'N/A')}</code>\n"
                           f"mAP50-95(Mask): <code>{row.get('mAP50-95(Mask)', 'N/A')}</code>\n")
                        + f"FPS: <code>{row.get('FPS', 'N/A')}</code>"
                    )
                else:
                    print(f"[Pipeline] ⚠️  Tidak ada row laporan untuk {model_label}.")

    # Kompilasi CSV per-family
    if family_det_rows or family_seg_rows:
        compile_family_csv(family, family_det_rows, family_seg_rows)

    # Visualisasi (terpisah dari evaluasi agar VRAM sudah bebas)
    # generate_visuals_for_family() kini mendukung SEMUA family:
    #   - YOLO (yolo8/9/10/11/rtdetr) → YOLO() Ultralytics
    #   - maskrcnn → TorchVision loader (maskrcnn_builder.build_model)
    #   - hybrid   → YOLO base (strip prefix "hybrid_") + SAM2
    if not skip_visual:
        try:
            generate_visuals_for_family(family, variants, gpu_ids, n_samples=n_samples)
        except Exception as e:
            print(f"[Pipeline] ⚠️  Visualisasi gagal untuk family {family}: {e}")

    return family_det_rows, family_seg_rows


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Laporan Evaluasi & Visualisasi per Model (generate_report_single_model.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--family", type=str, default="all",
        help=(
            f"Family model yang akan diproses. "
            f"Pilihan: {list(FAMILY_VARIANTS.keys())} atau 'all'. "
            f"Default: all."
        )
    )
    parser.add_argument(
        "--variants", type=str, default="",
        help=(
            "Filter varian spesifik dalam family (pisahkan dengan koma). "
            "Contoh: --variants yolov8m,yolov8x. "
            "Default: semua varian dalam family."
        )
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="GPU yang digunakan (pisahkan dengan koma). Contoh: --gpus 0,1. Default: 0."
    )
    parser.add_argument(
        "--samples", type=int, default=VISUAL_NUM_SAMPLES,
        help=f"Jumlah gambar sampel untuk visualisasi. Default: {VISUAL_NUM_SAMPLES}."
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Lewati evaluasi COCOeval, hanya jalankan visualisasi."
    )
    parser.add_argument(
        "--skip-visual", action="store_true",
        help="Lewati visualisasi, hanya jalankan evaluasi COCOeval."
    )
    args = parser.parse_args()

    # ── Resolusi GPU list ─────────────────────────────────────────────────────
    try:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    except ValueError as e:
        print(f"❌ Format --gpus tidak valid: {e}. Gunakan format '0' atau '0,1'.")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available() or not gpu_ids:
        print("❌ CUDA tidak tersedia atau GPU list kosong.")
        sys.exit(1)

    n_avail = torch.cuda.device_count()
    for g in gpu_ids:
        if g >= n_avail:
            print(f"❌ GPU {g} tidak tersedia (sistem punya {n_avail} GPU).")
            sys.exit(1)

    # ── Resolusi family list ──────────────────────────────────────────────────
    if args.family.strip().lower() == "all":
        families_to_run = list(FAMILY_VARIANTS.keys())
    else:
        families_to_run = [f.strip() for f in args.family.split(",") if f.strip()]

    # ── Resolusi filter varian ────────────────────────────────────────────────
    variant_filter = None
    if args.variants.strip():
        variant_filter = [v.strip() for v in args.variants.split(",") if v.strip()]

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  GENERATE REPORT SINGLE MODEL — Pipeline Evaluasi & Visualisasi")
    print("═" * 70)
    print(f"  Families     : {families_to_run}")
    print(f"  Variant Filter: {variant_filter if variant_filter else 'Semua'}")
    print(f"  GPUs          : {gpu_ids}")
    print(f"  Samples       : {args.samples}")
    print(f"  Skip Eval     : {args.skip_eval}")
    print(f"  Skip Visual   : {args.skip_visual}")
    print(f"  WORKSPACE_DIR : {WORKSPACE_DIR}")
    print(f"  REPORTS_DIR   : {REPORTS_DIR}")
    print("═" * 70 + "\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Jalankan semua family ─────────────────────────────────────────────────
    global_det_rows: list[dict] = []
    global_seg_rows: list[dict] = []

    t_start = time.perf_counter()
    for family in families_to_run:
        det_rows, seg_rows = run_family(
            family       = family,
            gpu_ids      = gpu_ids,
            skip_eval    = args.skip_eval,
            skip_visual  = args.skip_visual,
            variant_filter = variant_filter,
            n_samples    = args.samples,
        )
        global_det_rows.extend(det_rows)
        global_seg_rows.extend(seg_rows)

    # ── Kompilasi ALL ─────────────────────────────────────────────────────────
    if not args.skip_eval:
        compile_all_csv(global_det_rows, global_seg_rows)
        compile_all_single_model_csv(global_det_rows, global_seg_rows)

    total_elapsed = round(time.perf_counter() - t_start, 1)

    print(f"\n{'═'*70}")
    print(f"  ✅ SELESAI — Total waktu: {total_elapsed}s")
    print(f"  📂 Output CSV  : {os.path.join(REPORTS_DIR, 'csv')}")
    print(f"  📂 Output Visual: {os.path.join(REPORTS_DIR, 'visuals')}")
    print("═" * 70)

    send_telegram_msg(
        f"🏁 <b>Report Pipeline Selesai!</b>\n"
        f"Total waktu: <code>{total_elapsed}s</code>\n"
        f"Families: <code>{families_to_run}</code>\n"
        f"Det rows: <code>{len(global_det_rows)}</code> | "
        f"Seg rows: <code>{len(global_seg_rows)}</code>\n"
        f"Output: <code>{REPORTS_DIR}</code>"
    )


if __name__ == "__main__":
    main()
