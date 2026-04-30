# -*- coding: utf-8 -*-
"""
config_shared.py
================
Konfigurasi bersama untuk semua sub-modul fine-tuning.

WORKSPACE_DIR dibuat otomatis dengan suffix timestamp pada saat root main.py
dijalankan pertama kali. Semua sub-script membaca timestamp dari file
.workspace_id yang disimpan di root MyFineTunning, sehingga semua modul
menggunakan workspace yang SAMA dalam satu sesi run.
"""

import os
from datetime import datetime

# ==============================================================================
# ENVIRONMENT — Matikan torch._dynamo SEBELUM torch di-import
# ==============================================================================
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# WORKSPACE — Berbasis timestamp
# ==============================================================================
# Direktori root MyFineTunning (lokal / RunPOD)
_FINETUNING_ROOT = os.environ.get(
    "FINETUNING_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__)))
)
_WORKSPACE_ID_FILE = os.path.join(_FINETUNING_ROOT, ".workspace_id")

# Baca timestamp dari file jika sudah ada (sub-script ikut workspace root main)
# Jika belum ada (pertama kali root main.py dijalankan), buat baru.
if os.path.exists(_WORKSPACE_ID_FILE):
    with open(_WORKSPACE_ID_FILE) as _f:
        _TIMESTAMP = _f.read().strip()
else:
    _TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Path workspace persisten di RunPOD Network Volume
_BASE_DIR     = os.environ.get("WORKSPACE_BASE", "/workspace")
WORKSPACE_DIR = os.path.join(_BASE_DIR, f"MyFineTunning-{_TIMESTAMP}")

# ==============================================================================
# PATHS — Dataset
# ==============================================================================
DET_DATASET_LOCATION = os.environ.get(
    "DET_DATASET",
    "/root/MyTrainEngine/me-bottle-isempty-ku3-7"
)
SEG_DATASET_LOCATION = os.environ.get(
    "SEG_DATASET",
    "/root/MyTrainEngine/segpoligon-me-bottle-isempty3-5"
)

DET_YAML = os.path.join(DET_DATASET_LOCATION, "data.yaml")
SEG_YAML = os.path.join(SEG_DATASET_LOCATION, "data.yaml")

# ==============================================================================
# HYPERPARAMETER
# ==============================================================================
EPOCHS              = 100
IMAGE_SIZE          = 640
NUM_CLASSES         = 7
YOLO_BATCH_SIZE     = 64   # DDP total (dibagi ke semua GPU oleh Ultralytics)
MASKRCNN_BATCH_SIZE = 4    # Single GPU cuda:0
NUM_WORKERS         = 16


# ==============================================================================
# HELPERS
# ==============================================================================
def parse_device(device_str: str):
    """
    Konversi string device dari argumen CLI ke format yang diterima:
      - Ultralytics YOLO: list int  → [0], [1,2], [0,1,2]
      - torch.device     : "cuda:0", "cuda:1", "cpu"

    Contoh input:
      "0"     → [0]       (single GPU)
      "1,2"   → [1, 2]    (multi-GPU DDP)
      "0,1,2" → [0, 1, 2]
      "cpu"   → "cpu"

    Returns:
      list[int] | str
    """
    if device_str.strip().lower() == "cpu":
        return "cpu"
    try:
        ids = [int(d.strip()) for d in device_str.split(",") if d.strip()]
        return ids if len(ids) > 1 else ids[0]
    except ValueError:
        raise ValueError(f"Format device tidak valid: '{device_str}'. "
                         f"Gunakan format: '0', '1,2', atau 'cpu'")


def get_output_dir(model_key: str) -> str:
    """
    Kembalikan path direktori output untuk satu model dalam workspace saat ini.
    model_key contoh: "yolov8m", "yolov8m_seg", "yolov9m", "yolo11m", "maskrcnn"
    """
    return os.path.join(WORKSPACE_DIR, "runs", model_key)


def compress_run(model_key: str) -> str:
    """
    Kompres folder runs/{model_key} menjadi file .tar.gz di dalam workspace.
    Dipanggil setelah training + evaluasi satu model selesai.

    Returns:
        str — Path ke file .tar.gz yang dihasilkan, atau "" jika gagal.
    """
    import tarfile
    run_dir  = get_output_dir(model_key)
    tar_path = os.path.join(WORKSPACE_DIR, "runs", f"{model_key}.tar.gz")

    if not os.path.isdir(run_dir):
        print(f"[Compress] ⚠️  Folder tidak ditemukan: {run_dir}")
        return ""

    print(f"[Compress] Mengompres {run_dir} → {tar_path} ...")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(run_dir, arcname=model_key)
        size_mb = round(os.path.getsize(tar_path) / 1e6, 2)
        print(f"[Compress] ✅ Selesai: {tar_path} ({size_mb} MB)")
        return tar_path
    except Exception as e:
        print(f"[Compress] ❌ Gagal: {e}")
        return ""


def save_yolo_visual_samples(
    model_pt: str,
    model_key: str,
    img_dir: str,
    n: int = 5,
    conf: float = 0.5,
) -> None:
    """
    Ambil n gambar random dari img_dir, jalankan prediksi YOLO,
    simpan hasilnya ke runs/visuals/{model_key}_XX_{basename}.png.

    Parameters
    ----------
    model_pt  : str  — Path ke file best.pt
    model_key : str  — Prefix nama file output, misal "yolov8m"
    img_dir   : str  — Direktori gambar (test/images atau valid/images)
    n         : int  — Jumlah sampel random (default 5)
    conf      : float — Confidence threshold prediksi
    """
    import random, gc
    from ultralytics import YOLO

    visual_dir = os.path.join(WORKSPACE_DIR, "runs", "visuals")
    os.makedirs(visual_dir, exist_ok=True)

    # Cari gambar yang ada
    if not os.path.isdir(img_dir):
        print(f"[Visual] ⚠️  img_dir tidak ditemukan: {img_dir}")
        return

    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not all_imgs:
        print(f"[Visual] ⚠️  Tidak ada gambar di: {img_dir}")
        return

    samples = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"\n[Visual] {model_key} — {len(samples)} sampel dari {img_dir}")

    try:
        import cv2, numpy as np, matplotlib
        matplotlib.use("Agg")   # Non-interactive backend
        import matplotlib.pyplot as plt

        model = YOLO(model_pt)
        for idx, img_path in enumerate(samples, 1):
            base = os.path.splitext(os.path.basename(img_path))[0]
            result = model.predict(img_path, conf=conf, verbose=False)
            img_plot = result[0].plot()   # BGR numpy array

            # Simpan langsung via cv2 (tanpa matplotlib overhead)
            out_path = os.path.join(visual_dir, f"{model_key}_{idx:02d}_{base}.png")
            import cv2 as _cv2
            _cv2.imwrite(out_path, img_plot)
            print(f"  [{idx}/{len(samples)}] → {out_path}")

        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        print(f"[Visual] ❌ Gagal untuk {model_key}: {e}")


def compress_visuals() -> str:
    """
    Kompres seluruh folder runs/visuals/ menjadi visuals.tar.gz.
    Dipanggil di akhir hybrid/main.py (langkah terakhir pipeline).

    Returns:
        str — Path ke file .tar.gz, atau "" jika gagal.
    """
    import tarfile
    visual_dir = os.path.join(WORKSPACE_DIR, "runs", "visuals")
    tar_path   = os.path.join(WORKSPACE_DIR, "runs", "visuals.tar.gz")

    if not os.path.isdir(visual_dir):
        print(f"[Compress] ⚠️  visuals/ tidak ditemukan: {visual_dir}")
        return ""

    n_files = sum(1 for f in os.listdir(visual_dir) if f.endswith(".png"))
    print(f"\n[Compress] Mengompres visuals/ ({n_files} PNG) → {tar_path} ...")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(visual_dir, arcname="visuals")
        size_mb = round(os.path.getsize(tar_path) / 1e6, 2)
        print(f"[Compress] ✅ visuals.tar.gz: {size_mb} MB")
        return tar_path
    except Exception as e:
        print(f"[Compress] ❌ Gagal compress visuals: {e}")
        return ""


def measure_yolo_complexity(pt_path: str, img_size: int = 640) -> dict:
    """
    Ukur Parameters (M), GFLOPs, dan Peak VRAM (GB) dari YOLO model best.pt.

    Metode:
      - model.info(verbose=False) → (layers, params, gradients, flops)
        Ini membaca langsung dari model summary Ultralytics — akurat untuk
        fused model (yang selalu menampilkan 0 gradients setelah load).
      - Peak VRAM: torch.cuda.mem_get_info() sebelum & sesudah predict,
        kemudian ambil selisih minimum (total - free_before).

    Returns: dict {"params_m", "gflops", "vram_gb"}
    """
    import torch, gc
    import numpy as np
    empty = {"params_m": "N/A", "gflops": "N/A", "vram_gb": "N/A"}
    if not os.path.exists(pt_path):
        return empty
    try:
        from ultralytics import YOLO

        model = YOLO(pt_path)

        # ── Parameters & GFLOPs ────────────────────────────────────────────────
        # PENTING: get_flops() harus dipanggil dengan m.model (nn.Module),
        # bukan m (YOLO wrapper) — wrapper selalu mengembalikan 0.
        # get_num_params() juga dari m.model.
        from ultralytics.utils.torch_utils import get_num_params, get_flops
        params_m = round(get_num_params(model.model) / 1e6, 3)
        gflops   = round(get_flops(model.model, imgsz=img_size), 2)

        # ── Peak VRAM via pynvml ───────────────────────────────────────────────
        # max_memory_allocated() hanya menangkap alokasi proses ini sendiri.
        # pynvml membaca langsung dari driver (sama dengan nvidia-smi) →
        # mencerminkan VRAM yang benar-benar digunakan oleh model ini.
        vram_gb = "N/A"
        if torch.cuda.is_available():
            dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            # Warmup
            model.predict(dummy, verbose=False, imgsz=img_size)
            torch.cuda.synchronize()
            try:
                from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                                    nvmlDeviceGetMemoryInfo)
                nvmlInit()
                handle    = nvmlDeviceGetHandleByIndex(0)
                mem_info  = nvmlDeviceGetMemoryInfo(handle)
                vram_gb   = round(mem_info.used / 1e9, 2)
            except Exception:
                # Fallback: torch peak alloc
                torch.cuda.reset_peak_memory_stats(0)
                model.predict(dummy, verbose=False, imgsz=img_size)
                torch.cuda.synchronize()
                vram_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"params_m": params_m, "gflops": gflops, "vram_gb": vram_gb}
    except Exception as e:
        print(f"[Complexity] ⚠️  Gagal untuk {pt_path}: {e}")
        return empty



def measure_vram_peak(fn_inference, device_id: int = 0) -> float:
    """
    Ukur peak VRAM (GB) selama fn_inference() dipanggil.

    Parameters
    ----------
    fn_inference : callable — Fungsi yang menjalankan satu forward pass
    device_id    : int      — GPU index (default 0)

    Returns: float (GB), atau -1.0 jika CUDA tidak tersedia
    """
    import torch
    if not torch.cuda.is_available():
        return -1.0
    torch.cuda.reset_peak_memory_stats(device_id)
    fn_inference()
    torch.cuda.synchronize(device_id)
    return round(torch.cuda.max_memory_allocated(device_id) / 1e9, 2)
