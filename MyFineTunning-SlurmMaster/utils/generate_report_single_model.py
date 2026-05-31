# -*- coding: utf-8 -*-
"""
utils/generate_report_single_model.py
=======================================
Skrip sentral untuk:
  1. Menghasilkan visualisasi prediksi per gambar sampel:
     - N gambar prediksi individual (satu per varian model)
     - 1 panel grid (semua varian + Ground Truth)
  2. Mengompilasi laporan CSV dengan struktur hierarkis:
     - Per-varian: reports/pipeline/csv/<family>/<variant>_<task>.csv
     - Per-model:  reports/pipeline/csv/kompilasi_<family>_<task>.csv
     - ALL:        reports/pipeline/csv/kompilasi_ALL_<task>.csv

Cara menjalankan:
    # Satu family (semua varian):
    python -u utils/generate_report_single_model.py --family yolo8 --gpus 0

    # Family spesifik dengan varian terpilih:
    python -u utils/generate_report_single_model.py --family yolo9 --variants yolov9m,yolov9e --gpus 0

    # Semua family sekaligus:
    python -u utils/generate_report_single_model.py --family all --gpus 0

Struktur output (di dalam WORKSPACE_DIR):
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
    │   ├── kompilasi_yolo8_detection.csv
    │   ├── kompilasi_yolo8_segmentation.csv
    │   ├── kompilasi_yolo9_detection.csv
    │   ├── kompilasi_yolo9_segmentation.csv
    │   ├── kompilasi_yolov10_detection.csv
    │   ├── kompilasi_yolo11_detection.csv
    │   ├── kompilasi_yolo11_segmentation.csv
    │   ├── kompilasi_maskrcnn_segmentation.csv
    │   ├── kompilasi_hybrid_segmentation.csv
    │   ├── kompilasi_ALL_detection.csv
    │   └── kompilasi_ALL_segmentation.csv
    └── visuals/
        ├── A1_yolov8m.jpg
        ├── A1_yolov8m-seg.jpg
        ├── A1_yolov8_panel.jpg          ← Grid: semua varian + GT
        └── ...
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
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth,
    evaluate_coco_predictions,
    check_pycocotools,
)

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


def _get_visuals_dir() -> str:
    """Kembalikan path folder output visual. Buat jika belum ada."""
    d = os.path.join(REPORTS_DIR, "visuals")
    os.makedirs(d, exist_ok=True)
    return d


# ==============================================================================
# MODUL 1 — VISUALISASI: Per gambar sampel → N prediksi individual + 1 panel grid
# ==============================================================================

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def generate_visuals_for_family(
    family: str,
    variants: list[dict],
    gpu_ids: list[int],
    n_samples: int = 5,
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

    visuals_dir = _get_visuals_dir()

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
        gpu_cycle = gpu_ids[0] if gpu_ids else 0  # Gunakan GPU pertama untuk visualisasi
        for spec in variants:
            model_key   = spec["key"]
            model_label = spec["label"]
            pt_path     = os.path.join(get_output_dir(model_key), "weights", "best.pt")

            if not os.path.exists(pt_path):
                print(f"  [Visual] ⚠️  best.pt tidak ditemukan untuk {model_label}: {pt_path}")
                # Tambahkan blank image ke panel agar grid tetap konsisten
                blank = gt_img.copy()
                cv2.putText(blank, f"{model_label}: N/A", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                panel_images.append(blank)
                panel_labels.append(f"{model_label} (N/A)")
                continue

            try:
                theme_color = MODEL_COLORS.get(model_key, (255, 255, 0))
                model = YOLO(pt_path)
                result = model.predict(
                    sample_path, conf=conf, imgsz=IMAGE_SIZE,
                    device=gpu_cycle, verbose=False
                )[0]

                # Gambar prediksi individual
                pred_img = result.orig_img.copy()
                _draw_predictions_on_image(pred_img, result, theme_color)

                # Simpan gambar individual: <base>_<model_key>.jpg
                individual_path = os.path.join(visuals_dir, f"{base}_{model_key}.jpg")
                cv2.imwrite(individual_path, pred_img)
                print(f"    ➡️  Individual: {os.path.basename(individual_path)}")

                panel_images.append(pred_img.copy())
                panel_labels.append(model_label)

                del model, result
                gc.collect()

            except Exception as e:
                print(f"  [Visual] ❌ Gagal {model_label}: {e}")
                blank = gt_img.copy()
                cv2.putText(blank, f"{model_label}: ERROR", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                panel_images.append(blank)
                panel_labels.append(f"{model_label} (Error)")

        # --- Bangun panel grid ---
        panel_path = os.path.join(visuals_dir, f"{base}_{family}_panel.jpg")
        _build_panel_grid(panel_images, panel_labels, panel_path)
        print(f"    🖼️  Panel grid: {os.path.basename(panel_path)}")


def _build_panel_grid(images: list, labels: list, out_path: str, cols: int = 3) -> None:
    """
    Bangun dan simpan panel grid dari daftar gambar.

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

    # Tambahkan label ke setiap sel
    for i, (img, lbl) in enumerate(zip(resized, labels)):
        if lbl:
            cv2.putText(img, lbl, (8, target_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, lbl, (8, target_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

    # Susun baris dan gabungkan
    row_imgs = []
    for r in range(rows):
        row_slice = resized[r * cols: (r + 1) * cols]
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
    img_dir: str, image_ids: dict, tmp_dir: str, yolo_key: str
) -> None:
    """Worker inferens Hybrid (YOLO + SAM2) — memproses subset gambar secara paralel."""
    import torch
    from ultralytics import YOLO, SAM
    import cv2
    import numpy as np
    
    gpu = gpu_ids[rank]
    device_str = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens Hybrid dengan yolo_key={yolo_key}...", flush=True)

    # 1. Load YOLO model
    model = YOLO(pt_path)
    
    # 2. Load SAM2 model
    sam_model_path = os.path.join(ROOT, "models", "sam2.1_t.pt")
    sam_model = SAM(sam_model_path)
    
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
    coco_gt_dict, image_ids = build_coco_ground_truth(DET_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(DET_YAML)
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
    mAP50, mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    print(f"  ✅ mAP50={mAP50}  mAP50-95={mAP50_95}")

    # Hitung Precision & Recall manual (IoU@0.5)
    gt_det: dict[str, dict] = {}
    for img_path, img_id in image_ids.items():
        gt_det[img_path] = {"boxes": [], "clss": []}
        for ann in coco_gt_dict["annotations"]:
            if ann["image_id"] == img_id:
                b = ann["bbox"]
                gt_det[img_path]["boxes"].append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
                gt_det[img_path]["clss"].append(ann["category_id"] - 1)

    gt_per_class: dict[int, int] = {}
    for g in gt_det.values():
        for c in g["clss"]:
            gt_per_class[c] = gt_per_class.get(c, 0) + 1

    total_tp, total_fp, total_gt = 0, 0, 0
    num_classes = max(gt_per_class.keys()) + 1 if gt_per_class else 1
    for cls_id in range(num_classes):
        cls_preds    = [p for p in all_preds if p["pred_cls"] == cls_id]
        cls_gt_count = gt_per_class.get(cls_id, 0)
        total_gt    += cls_gt_count
        if not cls_preds:
            continue
        gt_matched    = set()
        gt_boxes_cls  = [
            (ip, idx, box)
            for ip, g in gt_det.items()
            for idx, (box, cls) in enumerate(zip(g["boxes"], g["clss"]))
            if cls == cls_id
        ]
        for pred in sorted(cls_preds, key=lambda x: x["pred_conf"], reverse=True):
            best_iou, best_key = 0.0, None
            pb = pred["pred_box"]
            for gi_path, gi_idx, gb in gt_boxes_cls:
                if pred["image"] != gi_path:
                    continue
                x1 = max(pb[0], gb[0]); y1 = max(pb[1], gb[1])
                x2 = min(pb[2], gb[2]); y2 = min(pb[3], gb[3])
                iw = max(0, x2 - x1);   ih = max(0, y2 - y1)
                ia = iw * ih
                pa = (pb[2]-pb[0]) * (pb[3]-pb[1])
                ga = (gb[2]-gb[0]) * (gb[3]-gb[1])
                ua = pa + ga - ia
                iou = ia / ua if ua > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_key = iou, (gi_path, gi_idx)
            if best_iou >= 0.5 and best_key not in gt_matched:
                total_tp += 1; gt_matched.add(best_key)
            else:
                total_fp += 1

    prec = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    rec  = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0
    print(f"  ✅ Precision={prec}  Recall={rec}")

    size_mb = round(os.path.getsize(pt_path) / 1e6, 2)
    return {
        "Model":            model_label,
        "Model Size (MB)":  size_mb,
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
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
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
    mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="segm"
    )
    _, mAP50_95_box = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="bbox"
    )
    print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")

    size_mb = round(os.path.getsize(pt_path) / 1e6, 2)
    return {
        "Model":            model_label,
        "Model Size (MB)":  size_mb,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
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
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
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
    mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="segm"
    )
    _, mAP50_95_box = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="bbox"
    )
    print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")

    size_mb = round(os.path.getsize(pt_path) / 1e6, 2)
    return {
        "Model":            model_label,
        "Model Size (MB)":  size_mb,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
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
    Evaluasi COCOeval untuk Hybrid (YOLO + SAM2).
    """
    import torch
    import torch.multiprocessing as mp

    print(f"\n{'='*65}")
    print(f"  Eval Hybrid: {model_label}")
    print(f"{'='*65}")

    # Resolve yolo_key from hybrid key (e.g. hybrid_yolov8m -> yolov8m)
    yolo_key = model_key.replace("hybrid_", "")
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
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO ground truth")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi Hybrid...")
    t_wall = time.perf_counter()
    mp.spawn(
        _infer_worker_hybrid_spawn,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir, yolo_key),
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
    mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="segm"
    )
    _, mAP50_95_box = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="bbox"
    )
    print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")

    # Model size is YOLO size + SAM2 size
    y_size = os.path.getsize(pt_path) / 1e6
    sam_model_path = os.path.join(ROOT, "models", "sam2.1_t.pt")
    s_size = os.path.getsize(sam_model_path) / 1e6 if os.path.exists(sam_model_path) else 0.0
    size_mb = round(y_size + s_size, 2)

    return {
        "Model":            model_label,
        "Model Size (MB)":  size_mb,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
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
    "Model", "Model Size (MB)", "mAP50-95", "mAP50",
    "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
    "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
]
_SEG_FIELDS = [
    "Model", "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)",
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


# ==============================================================================
# PIPELINE UTAMA — Jalankan per family
# ==============================================================================

def run_family(
    family: str,
    gpu_ids: list[int],
    skip_eval: bool = False,
    skip_visual: bool = False,
    variant_filter: Optional[list[str]] = None,
    n_samples: int = 5,
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
    if not skip_visual:
        if family in ["maskrcnn", "hybrid"]:
            print(f"[Pipeline] ℹ️  Visualisasi untuk {family} dilakukan oleh skrip terpisah. Melewati visualisasi Ultralytics.")
        else:
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
        "--samples", type=int, default=5,
        help="Jumlah gambar sampel untuk visualisasi. Default: 5."
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
