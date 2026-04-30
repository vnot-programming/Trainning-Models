# -*- coding: utf-8 -*-
"""
main.py  (ROOT)
===============
Setup awal satu kali: verifikasi dataset, buat folder output, cetak ringkasan.
Jalankan ini SEKALI sebelum menjalankan sub-modul fine-tuning per model.

Cara pakai:
    python main.py
"""

import os
import sys
from datetime import datetime

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# BUAT TIMESTAMP & TULIS .workspace_id
# Harus dilakukan SEBELUM import config_shared agar semua sub-script
# membaca timestamp yang sama dari file ini.
# ==============================================================================
_ROOT = os.path.dirname(os.path.abspath(__file__))
_WS_ID_FILE = os.path.join(_ROOT, ".workspace_id")

if os.path.exists(_WS_ID_FILE):
    with open(_WS_ID_FILE) as _f:
        _ts = _f.read().strip()
    print(f"[Setup] Melanjutkan workspace yang sudah ada: MyFineTunning-{_ts}")
else:
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(_WS_ID_FILE, "w") as _f:
        _f.write(_ts)
    print(f"[Setup] Workspace baru dibuat: MyFineTunning-{_ts}")

from config_shared import (
    WORKSPACE_DIR, DET_DATASET_LOCATION, SEG_DATASET_LOCATION,
    DET_YAML, SEG_YAML, EPOCHS, IMAGE_SIZE, NUM_CLASSES,
    YOLO_BATCH_SIZE, MASKRCNN_BATCH_SIZE,
    DATASETS_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR
)

print("=" * 65)
print("  MyFineTunning — Setup & Verifikasi")
print("=" * 65)

# ==============================================================================
# 1. VERIFIKASI DATASET
# ==============================================================================
print("\n[Setup] Verifikasi dataset...")
errors = []

for label, path in [
    ("DET dataset",  DET_DATASET_LOCATION),
    ("SEG dataset",  SEG_DATASET_LOCATION),
    ("DET YAML",     DET_YAML),
    ("SEG YAML",     SEG_YAML),
]:
    if os.path.exists(path):
        print(f"  ✅ {label}: {path}")
    else:
        print(f"  ❌ {label} TIDAK DITEMUKAN: {path}")
        errors.append(path)

if errors:
    print(f"\n❌ {len(errors)} path tidak ditemukan. Edit config_shared.py")
    sys.exit(1)

# ==============================================================================
# 2. BUAT FOLDER OUTPUT
# ==============================================================================
print("\n[Setup] Membuat folder output workspace dan direktori global...")
workspace_folders = [
    "runs/yolov8m",
    "runs/yolov8m_seg",
    "runs/yolov9m",
    "runs/yolov9m_seg",
    "runs/yolo11m",
    "runs/yolo11m_seg",
    "runs/maskrcnn/weights",
]
for folder in workspace_folders:
    full = os.path.join(WORKSPACE_DIR, folder)
    os.makedirs(full, exist_ok=True)
    print(f"  📁 {full}")

global_folders = [DATASETS_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR]
for folder in global_folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  📁 {folder}")

# ==============================================================================
# 3. RINGKASAN KONFIGURASI
# ==============================================================================
print(f"""
[Setup] Konfigurasi aktif:
  EPOCHS           : {EPOCHS}
  IMAGE_SIZE        : {IMAGE_SIZE}
  NUM_CLASSES       : {NUM_CLASSES}
  YOLO_BATCH_SIZE   : {YOLO_BATCH_SIZE}  (DDP total, dibagi ke semua GPU)
  MASKRCNN_BATCH    : {MASKRCNN_BATCH_SIZE}  (Single GPU cuda:0)
  WORKSPACE_DIR     : {WORKSPACE_DIR}

[Setup] Urutan eksekusi yang BENAR (jalankan satu per satu):
  1. cd yolo/yolo8   && python -u main.py 2>&1 | tee yolov8_train.log
  2. cd yolo/yolo9   && python -u main.py 2>&1 | tee yolov9_train.log
  3. cd yolo/yolo11  && python -u main.py 2>&1 | tee yolo11_train.log
  4. cd mask-r-cnn   && python -u main.py 2>&1 | tee maskrcnn_train.log
  5. cd hybrid       && python -u main.py 2>&1 | tee hybrid_eval.log

  Setiap script menghasilkan CSV report di:
  {REPORTS_DIR}
""")

print("✅ Setup selesai. Siap menjalankan fine-tuning per model.")
