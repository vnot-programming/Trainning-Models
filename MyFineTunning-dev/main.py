# -*- coding: utf-8 -*-
"""
main.py  (ROOT)
===============
Setup awal satu kali: verifikasi dataset, buat folder output, cetak ringkasan.
Jalankan ini SEKALI sebelum menjalankan sub-modul fine-tuning per model.

Cara pakai:
    python main.py
atau
    source /home/my/Computer-Vision/MyFineTunning-dev/.venv/bin/activate
    python /home/my/Computer-Vision/MyFineTunning-dev/main.py
tmux new-session -d -s main_session "cd /home/my/Trainning-Models/MyFineTunning-dev && python3 main.py 2>&1 | tee MainReport.log"

YOLO8 = tmux new-session -d -s yolo8training "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Trainning-Models/MyFineTunning-dev/yolo/yolo8 && python -u main.py 2>&1 | tee yolo8training.log"
YOLO9 = tmux new-session -d -s yolo9training "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Trainning-Models/MyFineTunning-dev/yolo/yolo9 && python -u main.py 2>&1 | tee yolo9training.log"
YOLO11 = tmux new-session -d -s yolo11training "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Trainning-Models/MyFineTunning-dev/yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log"
MaskRCNN = tmux new-session -d -s masktraining "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Trainning-Models/MyFineTunning-dev/mask-r-cnn && python -u train_multigpu.py 2>&1 | tee masktraining.log"

"""

import os
import sys
from datetime import datetime
from telegram_utils import send_telegram_msg
from dataset_setup import setup_all_datasets

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# BUAT TIMESTAMP & TULIS .workspace_id
# Harus dilakukan SEBELUM import config_shared agar semua sub-script
# membaca timestamp yang sama dari file ini.
# ==============================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
_WS_ID_FILE = os.path.join(ROOT, ".workspace_id")

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
    DATASETS_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR, IMAGE_SAMPLES_DIR
)

print("=" * 65)
print("  MyFineTunning — Setup & Verifikasi")
print("=" * 65)

# ==============================================================================
# 1. DATASET — Auto-download jika belum ada
# ==============================================================================
print("\n[Setup] Memeriksa ketersediaan dataset...")
try:
    _datasets = setup_all_datasets()
    # Update config_shared paths jika lokasi hasil download berbeda
    DET_DATASET_LOCATION = _datasets["det_location"]
    SEG_DATASET_LOCATION = _datasets["seg_location"]
    DET_YAML             = _datasets["det_yaml"]
    SEG_YAML             = _datasets["seg_yaml"]
except RuntimeError as _e:
    print(f"\n❌ Dataset gagal disiapkan: {_e}")
    send_telegram_msg(f"❌ <b>Setup Gagal</b>\nDataset tidak tersedia:\n<code>{_e}</code>")
    sys.exit(1)

# ==============================================================================
# 2. BUAT FOLDER OUTPUT
# ==============================================================================
print("\n[Setup] Membuat folder output workspace dan direktori global...")
workspace_folders = [
    "runs/yolov8m",
    "runs/yolov8m_seg",
    "runs/yolov9m",
    "runs/yolov9c_seg",
    "runs/yolo11m",
    "runs/yolo11m_seg",
    "runs/maskrcnn/weights",
]
for folder in workspace_folders:
    full = os.path.join(WORKSPACE_DIR, folder)
    os.makedirs(full, exist_ok=True)
    print(f"  📁 {full}")

global_folders = [DATASETS_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR, IMAGE_SAMPLES_DIR]
for folder in global_folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  📁 {folder}")

print("\n[Setup] Menyiapkan 10 gambar sampel untuk visualisasi (IMAGE_SAMPLES_DIR)...")
import random
import shutil
img_dir_test = os.path.join(SEG_DATASET_LOCATION, "test", "images")
img_dir_valid = os.path.join(SEG_DATASET_LOCATION, "valid", "images")

# Utamakan test, jika tidak ada gunakan valid
source_img_dir = img_dir_test if os.path.isdir(img_dir_test) else img_dir_valid
suffix = "test" if source_img_dir == img_dir_test else "valid"

if os.path.isdir(source_img_dir):
    all_imgs = [os.path.join(source_img_dir, f) for f in os.listdir(source_img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(all_imgs) >= 10:
        samples = random.sample(all_imgs, 10)
        for i, img_path in enumerate(samples, 1):
            ext = os.path.splitext(img_path)[1]
            new_name = f"image{i}-{suffix}{ext}"
            dest_path = os.path.join(IMAGE_SAMPLES_DIR, new_name)
            shutil.copy2(img_path, dest_path)
        print(f"  ✅ 10 gambar berhasil disalin dan di-rename ke {IMAGE_SAMPLES_DIR}")
    else:
        print(f"  ⚠️ Gambar di {source_img_dir} kurang dari 10.")
else:
    print(f"  ⚠️ Folder sumber tidak ditemukan: {source_img_dir}")

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

[Setup] Urutan training (Default menggunakan GPU 0 | Jika Multi maka Paralel Aktif):
  1. YOLO8   : cd yolo/yolo8  && python -u main.py 2>&1 | tee yolo8training.log
  2. YOLO9   : cd yolo/yolo9  && python -u main.py 2>&1 | tee yolo9training.log
  3. YOLO11  : cd yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log
  4. MaskRCNN: cd mask-r-cnn  && python -u train_multigpu.py 2>&1 | tee maskrcnntraining.log
  5. Hybrid  : cd hybrid      && python -u main.py 2>&1 | tee hybridtraining.log

  💡 Tips GPU: Tambahkan '--device 1,2' jika ingin menggunakan GPU nomor 1 dan 2 saja.

  🛠️ Background Execution (tmux):
  - YOLO8 : tmux new-session -d -s yolo8training "source .venv/bin/activate && cd yolo/yolo8 && python -u main.py 2>&1 | tee yolo8training.log"
  - YOLO9 : tmux new-session -d -s yolo9training "source .venv/bin/activate && cd yolo/yolo9 && python -u main.py 2>&1 | tee yolo9training.log"
  - YOLO11: tmux new-session -d -s yolo11training "source .venv/bin/activate && cd yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log"
  - MaskRCNN: tmux new-session -d -s maskrcnntraining "source .venv/bin/activate && cd . && python -u mask-r-cnn/train_multigpu.py 2>&1 | tee maskrcnntraining.log"
  - Hybrid: tmux new-session -d -s hybridtraining "source .venv/bin/activate && cd hybrid && python -u main.py 2>&1 | tee hybridtraining.log"

  Setiap script menghasilkan CSV report di:
  {REPORTS_DIR}
""")

print("✅ Setup selesai. Siap menjalankan fine-tuning per model.")
send_telegram_msg(f"✅ <b>Setup Selesai!</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>\nSiap menjalankan training pipeline.")
