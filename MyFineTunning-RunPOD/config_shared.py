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
ROOT = os.environ.get(
    "ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__)))
)
_WORKSPACE_ID_FILE = os.path.join(ROOT, ".workspace_id")

# Baca timestamp dari file jika sudah ada (sub-script ikut workspace root main)
# Jika belum ada (pertama kali root main.py dijalankan), buat baru.
if os.path.exists(_WORKSPACE_ID_FILE):
    with open(_WORKSPACE_ID_FILE) as _f:
        _TIMESTAMP = _f.read().strip()
else:
    _TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Path workspace lokal — di dalam data-files/ agar output terpusat
_BASE_DIR     = os.environ.get(
    "WORKSPACE_BASE",
    os.path.join(ROOT, "data-files")
)
WORKSPACE_DIR = os.path.join(_BASE_DIR, f"MyFineTunning-{_TIMESTAMP}")

# ==============================================================================
# PATHS — Project Structure
# ==============================================================================
DATASETS_DIR      = os.path.join(ROOT, "datasets")
MODELS_DIR        = os.path.join(ROOT, "models")
DATA_FILES_DIR    = os.path.join(ROOT, "data-files")
REPORTS_DIR       = os.path.join(WORKSPACE_DIR, "reports", "pipeline")
PAPER1_CSV_DIR    = os.path.join(WORKSPACE_DIR, "reports", "paper1", "csv")
PAPER1_VIS_DIR    = os.path.join(WORKSPACE_DIR, "reports", "paper1", "visuals")
VISUALS_DIR       = os.path.join(WORKSPACE_DIR, "visuals")
IMAGE_SAMPLES_DIR = os.path.join(WORKSPACE_DIR, "image_samples")

# ==============================================================================
# PATHS — Dataset
# ==============================================================================
DET_DATASET_LOCATION = os.environ.get(
    "DET_DATASET",
    os.path.join(DATASETS_DIR, "me-bottle-isempty-ku3-8")
)
SEG_DATASET_LOCATION = os.environ.get(
    "SEG_DATASET",
    os.path.join(DATASETS_DIR, "segpoligon-me-bottle-isempty3-7")
)

DET_YAML = os.path.join(DET_DATASET_LOCATION, "data.yaml")
SEG_YAML = os.path.join(SEG_DATASET_LOCATION, "data.yaml")

STANDAR_DET_DATASET_LOCATION = os.environ.get(
    "STANDAR_DET_DATASET",
    os.path.join(DATASETS_DIR, "standard_datasets_det")
)
STANDAR_SEG_DATASET_LOCATION = os.environ.get(
    "STANDAR_SEG_DATASET",
    os.path.join(DATASETS_DIR, "standard_datasets_seg")
)

GOLDEN_DET_DATASET_LOCATION = os.environ.get(
    "GOLDEN_DET_DATASET",
    os.path.join(DATASETS_DIR, "golden_dataset_det")
)
GOLDEN_SEG_DATASET_LOCATION = os.environ.get(
    "GOLDEN_SEG_DATASET",
    os.path.join(DATASETS_DIR, "golden_dataset_seg")
)

# ==============================================================================
# PATHS — Pipeline (untuk run_pipeline.py)
# ==============================================================================
VENV_ACTIVATE_PATH = os.path.join(ROOT, ".venv", "bin", "activate")

PIPELINE_JOBS = [
    {
        "name": "yolo8",
        "session": "yolo8training",
        "workdir": os.path.join(ROOT, "yolo", "yolo8"),
        "script": "main.py",
        "logfile": "yolo8training.log",
    },
    {
        "name": "yolo9",
        "session": "yolo9training",
        "workdir": os.path.join(ROOT, "yolo", "yolo9"),
        "script": "main.py",
        "logfile": "yolo9training.log",
    },
    {
        "name": "yolo11",
        "session": "yolo11training",
        "workdir": os.path.join(ROOT, "yolo", "yolo11"),
        "script": "main.py",
        "logfile": "yolo11training.log",
    },
    {
        "name": "maskrcnn",
        "session": "masktraining",
        "workdir": os.path.join(ROOT, "mask-r-cnn"),
        "script": "train_multigpu.py",
        "logfile": "train_multigpu.log",
    },
]

# ==============================================================================
# GPU Memory Cleanup Config (untuk run_pipeline.py)
# ==============================================================================
GPU_IDLE_THRESHOLD_MIB = 500    # Max VRAM terpakai agar dianggap idle
GPU_CLEANUP_TIMEOUT    = 120    # Timeout (detik) menunggu GPU idle
GPU_CLEANUP_POLL_SEC   = 5      # Interval polling nvidia-smi saat cleanup
GPU_COOLDOWN_SEC       = 15     # Jeda (detik) setelah training selesai

# ==============================================================================
# HYPERPARAMETER Test
# ==============================================================================
# EPOCHS              = 2 # 100
# IMAGE_SIZE          = 640
# NUM_CLASSES         = 7
# YOLO_BATCH_SIZE     = 80    # 160 menyebabkan OOM pada YOLO11l (Large) di VRAM 20GB. 80 / 5 = 16 per GPU.
# MASKRCNN_BATCH_SIZE = 8     # Diturunkan ke 8 agar lebih aman (menghindari OOM).
# NUM_WORKERS         = 10    # 16

# ==============================================================================
# HYPERPARAMETER RunPOD
# ==============================================================================
# EPOCHS              = 100
# IMAGE_SIZE          = 640
# NUM_CLASSES         = 7
# YOLO_BATCH_SIZE     = 96   # DDP total (dibagi ke semua GPU oleh Ultralytics)
# MASKRCNN_BATCH_SIZE = 10    # 4
# NUM_WORKERS         = 14    # 16

# ==============================================================================
# HYPERPARAMETER RunPOD - Opsi Lain
# ==============================================================================
EPOCHS              = 100
IMAGE_SIZE          = 640
NUM_CLASSES         = 7
YOLO_BATCH_SIZE     = 64   # DDP total (dibagi ke semua GPU oleh Ultralytics)
MASKRCNN_BATCH_SIZE = 8    # 4
NUM_WORKERS         = 14    # 16

# Cara menghitung manualnya didasarkan pada **kapasitas VRAM GPU** dan **jumlah CPU Core** yang tersedia. Berikut adalah panduan hitungan manual untuk meningkatkan performa di RunPod Anda:

# ### 1. Menghitung `YOLO_BATCH_SIZE` (Total Batch)
# YOLO menggunakan DDP (Distributed Data Parallel), jadi angkanya adalah total untuk 4 GPU.

# *   **Rumus:** `(Target VRAM per GPU / VRAM saat ini) * Batch per GPU saat ini * Jumlah GPU`
# *   **Logika Manual:**
#     *   Saat ini: Batch total **80** (artinya **20 per GPU**) menggunakan **10-15GB**.
#     *   Kapasitas A5000: **24GB**. Kita ingin target aman di **20GB** (agar tidak OOM saat validasi).
#     *   Jika 20 per GPU = 12GB, maka untuk mencapai 20GB: `(20GB / 12GB) * 20 = ~33 per GPU`.
#     *   **Rekomendasi:** Gunakan **batch 40 per GPU**.
#     *   **Hitungan:** `40 * 4 GPU = 160`.
#     *   *Kenapa VRAM sekarang cuma 10-15GB?* Karena batch 20 terlalu kecil untuk GPU 24GB. GPU Anda sedang "santai".

# ### 2. Menghitung `MASKRCNN_BATCH_SIZE`
# MaskRCNN jauh lebih berat daripada YOLO karena memproses region proposal (RPN).

# *   **Logika Manual:**
#     *   MaskRCNN biasanya memakan VRAM besar. Di GPU 24GB, biasanya bisa menampung **8 s/d 12 image per GPU**.
#     *   Jika Anda menggunakan 4 GPU: `8 image * 4 GPU = 32`.
#     *   **Rekomendasi:** Naikkan secara bertahap ke **32** atau **48**.
#     *   Jika Anda set ke **16** (seperti sekarang), itu artinya cuma **4 image per GPU**. Ini sangat rendah untuk spek A5000.

# ### 3. Menghitung `NUM_WORKERS` (CPU Power)
# Ini adalah "asisten" yang menyiapkan gambar sebelum dikirim ke GPU.

# *   **Rumus Aman:** `2 * Jumlah GPU` s/d `4 * Jumlah GPU`.
# *   **Rumus Maksimal:** `Total CPU Cores / Jumlah GPU`.
# *   **Logika Manual:**
#     *   Anda punya **96 Core** dan **4 GPU**. Artinya jatah maksimal per GPU adalah `96 / 4 = 24 core`.
#     *   Namun, jangan gunakan semua core agar sistem tidak *hang*.
#     *   **Rekomendasi:** Gunakan **8 s/d 12 worker per GPU**.
#     *   **Hitungan:** `8 * 4 GPU = 32` atau `12 * 4 GPU = 48`.
#     *   Gunakan **32** atau **48** jika Anda merasa GPU Utilization (di `nvidia-smi`) sering turun di bawah 90%.

# ---

# ### Kesimpulan Rekomendasi "Gas Pol" (Safe Mode):
# Jika Anda ingin memaksimalkan 4x RTX A5000 tanpa sering OOM:

# 1.  **`YOLO_BATCH_SIZE = 160`** (40 per GPU) -> VRAM akan terisi sekitar **18-20GB**.
# 2.  **`MASKRCNN_BATCH_SIZE = 32`** (8 per GPU) -> MaskRCNN sangat haus memori, 8 per GPU sudah cukup berat.
# 3.  **`NUM_WORKERS = 32`** -> Sudah sangat cukup untuk menyuplai data ke 4 GPU secara simultan.

# **Tips:** Selalu perhatikan kolom `GPU-Util` di `nvidia-smi`. Jika nilainya **95-100%**, berarti settingan Anda sudah optimal. Jika nilainya rendah (misal 60%), naikkan `NUM_WORKERS`.

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


# Konstanta Warna Spesifik Tiap Model (Format BGR untuk OpenCV)
MODEL_COLORS = {
    "yolov8m":     (255,   0,   0),   # Biru
    "yolov8m_seg": (255,   0,   0),
    "yolov9m":     (  0,   0, 255),   # Merah
    "yolov9c_seg": (  0,   0, 255),
    "yolo11l":     (  0, 255,   0),   # Hijau
    "yolo11l_seg": (  0, 255,   0),
    "maskrcnn":    (255,   0, 255),   # Magenta/Ungu
    "hybrid":      (  0, 165, 255),   # Orange
}

def save_yolo_visual_samples(
    model_pt: str,
    model_key: str,
    img_dir: str,          # Parameter ini tetap ada demi backward compatibility, tapi kita akan pakai IMAGE_SAMPLES_DIR
    n: int = 10,           # Diubah jadi 10 default
    conf: float = 0.5,
) -> None:
    """
    Render gambar dari IMAGE_SAMPLES_DIR secara berurutan, jalankan prediksi YOLO,
    gambar custom bounding box/mask dengan warna tema model, lalu simpan hasilnya.
    """
    import gc
    from ultralytics import YOLO

    visual_dir = VISUALS_DIR
    os.makedirs(visual_dir, exist_ok=True)

    # Gunakan IMAGE_SAMPLES_DIR yang telah dipersiapkan oleh main.py
    target_img_dir = IMAGE_SAMPLES_DIR
    if not os.path.isdir(target_img_dir):
        print(f"[Visual] ⚠️  IMAGE_SAMPLES_DIR tidak ditemukan: {target_img_dir}")
        return

    # Ambil semua gambar sampel yang ada, urutkan (image1, image2, ...)
    all_imgs = sorted([os.path.join(target_img_dir, f) for f in os.listdir(target_img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    if not all_imgs:
        print(f"[Visual] ⚠️  Tidak ada gambar sampel di: {target_img_dir}")
        return

    samples = all_imgs[:n]
    print(f"\n[Visual] {model_key} — {len(samples)} sampel dari {target_img_dir}")

    # Ambil warna tema untuk model ini (fallback ke cyan jika tidak dikenali)
    theme_color = MODEL_COLORS.get(model_key, (255, 255, 0))

    try:
        import cv2, numpy as np

        model = YOLO(model_pt)
        for idx, img_path in enumerate(samples, 1):
            base = os.path.splitext(os.path.basename(img_path))[0]
            result = model.predict(img_path, conf=conf, verbose=False)[0]
            
            # Original Image
            img_bgr = result.orig_img.copy()
            overlay = img_bgr.copy()
            H, W = img_bgr.shape[:2]
            
            names = result.names  # Dictionary class index ke nama class
            
            # Jika ada segmentation masks
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
                for i, mask in enumerate(masks):
                    if mask.shape != (H, W):
                        mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    bool_mask = (mask > 0.5).astype(np.uint8)
                    colored_mask = np.zeros_like(overlay)
                    colored_mask[bool_mask == 1] = theme_color
                    cv2.addWeighted(colored_mask, 0.45, overlay, 1.0, 0, overlay)

            # Gambar Bounding Boxes dan Label
            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                boxes_conf = result.boxes.conf.cpu().numpy()
                boxes_cls  = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                    confidence = float(boxes_conf[i])
                    cls_id = int(boxes_cls[i])
                    
                    # Dapatkan nama class
                    cls_name = names.get(cls_id, f"cls{cls_id}")
                    
                    # Gambar kotak
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_color, 2)
                    
                    # Teks Label
                    label_txt = f"{cls_name} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    # Background text
                    cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
                    # Text putih
                    cv2.putText(overlay, label_txt, (x1 + 2, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = os.path.join(visual_dir, f"{model_key}_{idx:02d}_{base}.png")
            cv2.imwrite(out_path, overlay)
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
    visual_dir = VISUALS_DIR
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


def download_and_move_model(model_name: str) -> str:
    """
    Pastikan model dasar (misal 'yolov8m.pt') tersedia di MODELS_DIR.
    Jika tidak ada, download menggunakan YOLO dari ultralytics (ke dir aktif),
    lalu pindahkan file tersebut ke MODELS_DIR.
    
    Returns:
        Absolute path ke model di dalam MODELS_DIR.
    """
    import shutil
    from ultralytics.utils.downloads import attempt_download_asset
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = os.path.join(MODELS_DIR, model_name)
    
    if os.path.exists(target_path):
        print(f"[Model] ✅ Model sudah ada di: {target_path}")
        return target_path
        
    print(f"[Model] ⏳ Mendownload {model_name}...")
    try:
        # Download ke current working directory
        downloaded_path = attempt_download_asset(model_name)
        if os.path.exists(downloaded_path):
            print(f"[Model] ✅ Berhasil didownload ke {downloaded_path}. Memindahkan ke {MODELS_DIR}...")
            shutil.move(downloaded_path, target_path)
            return target_path
    except Exception as e:
        print(f"[Model] ❌ Gagal mendownload model: {e}")
    
    return target_path


# Daftar semua model dasar yang dibutuhkan oleh pipeline
ALL_BASE_MODELS = [
    "yolov8m.pt",
    "yolov8m-seg.pt",
    "yolov9m.pt",
    "yolov9c-seg.pt",
    "yolo11l.pt",
    "yolo11l-seg.pt",
    "sam2.1_t.pt",
]

def ensure_all_base_models():
    """Download semua model dasar ke MODELS_DIR jika belum ada."""
    print("\n[Models] Memeriksa ketersediaan semua model dasar...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_name in ALL_BASE_MODELS:
        download_and_move_model(model_name)
    print("[Models] ✅ Semua model dasar tersedia.\n")
