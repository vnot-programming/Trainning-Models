# -*- coding: utf-8 -*-
"""
rtdetr/train_rtdetr.py
======================
Fine-tuning RT-DETR-L pada dataset botol plastik (RVM) — Paper 3.

Arsitektur: Real-Time DEtection TRansformer (RT-DETR) oleh Ultralytics.
- Menggantikan NMS dengan attention mechanism murni (ViT-based encoder).
- Model dikomparasi langsung melawan YOLO11n+MobileSAM pada Jetson Orin Nano.

Cara menjalankan (WAJIB melalui Slurm — baca SLURM_Guide.md):
    python -u train_rtdetr.py 2>&1 | tee train_rtdetr.log
    python -u train_rtdetr.py --device 0 2>&1 | tee train_rtdetr.log
    python -u train_rtdetr.py --skip-eval
"""

import os
import sys
import gc

# Matikan torch dynamo sebelum import torch (konsisten dengan seluruh pipeline)
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Tambahkan ROOT ke sys.path agar config_shared dapat diimpor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── GPU Fan Manager (opsional, jika tersedia) ────────────────────────────────
try:
    from gpu_fan_manager import start_fan_manager
    start_fan_manager()
except ImportError:
    print("[Warning] gpu_fan_manager.py tidak ditemukan di ROOT — dilewati.")

# ── Import konfigurasi sentral (Single Source of Truth) ─────────────────────
# SEMUA nilai di bawah ini WAJIB dibaca dari config_shared.py, TIDAK boleh hardcode.
from config_shared import (
    WORKSPACE_DIR,
    DET_YAML,
    MODELS_DIR,
    EPOCHS,
    IMAGE_SIZE,
    RTDETR_BATCH_SIZE,   # Batch khusus RT-DETR (Transformer) — lebih kecil dari YOLO_BATCH_SIZE
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    get_output_dir,
    compress_run,
    parse_device,
    download_and_move_model,
    REPORTS_DIR,
)
from telegram_utils import get_yolo_callbacks, send_telegram_msg

import argparse
import torch
from ultralytics import RTDETR, settings

# Arahkan semua download bobot Ultralytics ke MODELS_DIR yang terdefinisi di config_shared
settings.update({"weights_dir": MODELS_DIR})

# ── Nama kunci model (konsisten dengan FAMILY_VARIANTS di eval_single_model.py) ─
# Kunci ini menentukan folder output: WORKSPACE_DIR/runs/rtdetr_l/
MODEL_KEY  = "rtdetr_l"
MODEL_FILE = "rtdetr-l.pt"      # Nama file bobot dasar (ada di ALL_BASE_MODELS)
MODEL_LABEL = "RT-DETR-L (Paper 3 — Edge Transformer)"

# ── Argument Parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RT-DETR-L Fine-tuning untuk RVM Dataset")
parser.add_argument(
    "--device", type=str, default=None,
    help="GPU index: '0', '1,2', 'cpu'. Default: semua GPU yang tersedia."
)
parser.add_argument(
    "--skip-eval", action="store_true",
    help="Lewati evaluasi otomatis setelah training selesai."
)
args = parser.parse_args()

# ── Resolusi device ──────────────────────────────────────────────────────────
if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    n = torch.cuda.device_count()
    DEVICE = list(range(n)) if n > 1 else (0 if n == 1 else "cpu")

print(f"[Device] RT-DETR-L → {DEVICE}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_gpu_report_str(device) -> str:
    """Format nama GPU untuk laporan CSV."""
    if device == "cpu":
        return "1x CPU"
    from collections import Counter
    ids = [device] if isinstance(device, int) else device
    gpu_names = [torch.cuda.get_device_name(i) for i in ids]
    counts = Counter(gpu_names)
    return ", ".join(f"{count}x {name}" for name, count in counts.items())


def _flush(label: str):
    """Bersihkan VRAM dan tampilkan konfirmasi."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[MemFlush] {label} ✅", flush=True)


def _train() -> str:
    """
    Jalankan fine-tuning RT-DETR-L.
    - Mendukung resume otomatis dari last.pt jika training sebelumnya terputus.
    - Membaca semua hyperparameter dari config_shared.py (zero hardcode).
    - Menyimpan flag training_done.flag setelah selesai.

    Returns:
        str — Path ke file best.pt hasil training.
    """
    out_dir   = get_output_dir(MODEL_KEY)
    best_pt   = os.path.join(out_dir, "weights", "best.pt")
    last_pt   = os.path.join(out_dir, "weights", "last.pt")
    done_flag = os.path.join(out_dir, "training_done.flag")

    # ── Skip jika sudah selesai sebelumnya ────────────────────────────────────
    if os.path.exists(done_flag) and os.path.exists(best_pt):
        print(f"\n[SKIP] {MODEL_LABEL}: training sudah selesai.\n  best.pt: {best_pt}")
        return best_pt

    # Auto-scale workers agar tidak melebihi batas QoS Slurm
    n_gpus = len(DEVICE) if isinstance(DEVICE, list) else 1
    train_workers = max(1, NUM_WORKERS // n_gpus)

    # ── Resume dari last.pt jika ada ─────────────────────────────────────────
    if os.path.exists(last_pt):
        print(f"\n[RESUME] {MODEL_LABEL}: melanjutkan dari last.pt\n  {last_pt}")
        model = RTDETR(last_pt)
        for k, v in get_yolo_callbacks(MODEL_LABEL).items():
            model.add_callback(k, v)
        # RT-DETR mendukung `resume=True` melalui API Ultralytics
        model.train(resume=True, patience=EARLY_STOPPING_PATIENCE, workers=train_workers)

    else:
        # ── Training dari bobot dasar ─────────────────────────────────────────
        print(f"\n{'='*65}\n  {MODEL_LABEL}\n{'='*65}")
        model_path = download_and_move_model(MODEL_FILE)
        model = RTDETR(model_path)

        for k, v in get_yolo_callbacks(MODEL_LABEL).items():
            model.add_callback(k, v)

        # ── Catatan konfigurasi RT-DETR vs YOLO ──────────────────────────────
        # RT-DETR menggunakan arsitektur Transformer, bukan CNN.
        # Augmentasi seperti mosaic/mixup berlebihan justru dapat merusak konvergensi
        # karena attention mechanism sudah menangkap global context secara alami.
        # Oleh karena itu, kita tidak mematikan mosaic, tapi membiarkan Ultralytics
        # mengelola augmentasi secara default untuk RTDETR (sudah dikalibrasi).
        # CATATAN: RT-DETR-L (Transformer 108 GFLOPs) menggunakan RTDETR_BATCH_SIZE
        # bukan YOLO_BATCH_SIZE karena attention mechanism O(N²) jauh lebih boros VRAM.
        # Data empiris: batch=30 & batch=15 → OOM; batch=7 → stabil di 16.9GB VRAM.
        # Nilai RTDETR_BATCH_SIZE=8 memberikan margin keamanan ~13GB VRAM pada V100 32GB.
        model.train(
            data=DET_YAML,                                  # Dibaca dari config_shared
            epochs=EPOCHS,                                  # Dibaca dari config_shared
            imgsz=IMAGE_SIZE,                               # Dibaca dari config_shared
            batch=RTDETR_BATCH_SIZE,                        # Khusus RT-DETR — dari config_shared
            project=os.path.dirname(out_dir),              # Dibaca dari get_output_dir
            name=os.path.basename(out_dir),                # Dibaca dari get_output_dir
            exist_ok=True,
            device=DEVICE,
            patience=EARLY_STOPPING_PATIENCE,              # Dibaca dari config_shared
            workers=train_workers,
        )

    # ── Tandai training selesai ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(done_flag), exist_ok=True)
    with open(done_flag, "w") as f:
        f.write("done")

    result = str(model.trainer.best) if hasattr(model, "trainer") and model.trainer else best_pt
    del model
    _flush(MODEL_LABEL)
    return result


# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  RT-DETR-L Fine-tuning Pipeline (Paper 3 — Edge Transformer)")
print("="*65)
print(f"  Dataset (DET) : {DET_YAML}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Batch Size    : {RTDETR_BATCH_SIZE}  (RTDETR_BATCH_SIZE — khusus Transformer)")
print(f"  Image Size    : {IMAGE_SIZE}")
print(f"  Workspace     : {WORKSPACE_DIR}")
print("="*65 + "\n")

# ── Jalankan Training ─────────────────────────────────────────────────────────
best_weights = _train()

# ── Simpan path best_det ke file referensi (untuk pipeline downstream) ────────
det_path_ref = os.path.join(get_output_dir(MODEL_KEY), "weights", "best_path.txt")
os.makedirs(os.path.dirname(det_path_ref), exist_ok=True)
with open(det_path_ref, "w") as f:
    f.write(best_weights)
print(f"[Info] Path RT-DETR best.pt disimpan ke: {det_path_ref}")

# ── Evaluasi menggunakan generate_report_single_model.py ─────────────────────
os.makedirs(REPORTS_DIR, exist_ok=True)

if not args.skip_eval:
    print("\n" + "="*65)
    print(f"  Evaluasi RT-DETR-L via generate_report_single_model.py")
    print("="*65)
    import subprocess
    eval_gpu  = args.device if args.device is not None else "all"
    eval_script = os.path.join(ROOT, "utils", "generate_report_single_model.py")

    if not os.path.exists(eval_script):
        print(f"⚠️ Skrip evaluasi tidak ditemukan: {eval_script}")
    else:
        try:
            subprocess.run(
                [sys.executable, "-u", eval_script,
                 "--family", "rtdetr",
                 "--gpus",   str(eval_gpu)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️ Evaluasi RT-DETR gagal: {e}")
else:
    print("\n[Skip] Evaluasi dilewati karena argumen --skip-eval aktif.")

# ── Kompres folder hasil training ─────────────────────────────────────────────
try:
    compress_run(MODEL_KEY)
except Exception as e:
    print(f"⚠️ Gagal kompres {MODEL_KEY}: {e}")

print(f"\n✅ RT-DETR-L Pipeline (Paper 3) selesai.")
send_telegram_msg(
    f"✅ <b>RT-DETR-L Training Finished (Paper 3)</b>\n"
    f"Workspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>\n"
    f"best.pt: <code>{best_weights}</code>"
)
