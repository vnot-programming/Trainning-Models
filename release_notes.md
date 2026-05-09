# Release Notes

---

## 🏷️ v0.2.0 — `MyFineTunning_20260509`
- **Date:** May 9, 2026
- **Status:** ✅ Release (Stable)
- **Branch:** `main`

### 🚀 What's Changed

#### ✅ Multi-GPU Evaluation Pipeline — Standardization
- **`run_pipeline_multi.py`**: Script orkestrator baru untuk menjalankan semua 5 evaluasi model secara **sekuensial** dengan venv otomatis, `--skip`, `--gpus` args, dan log ringkasan akhir.
- **Latency Dinamis**: Mengganti nilai statis `N/A (Distributed)` dengan data timing **nyata** pada seluruh pipeline:
  - `yolo8/eval_multigpu.py`, `yolo9/eval_multigpu.py`, `yolo11/eval_multigpu.py`: Mengekstrak `result.speed` (preprocess/inference/postprocess) per gambar dari setiap rank, dikumpulkan via pickle, dirata-rata di main process.

#### ✅ Bug Fix: CSV Writer `ValueError`
- **Root cause**: `seg_fields` (fieldnames `DictWriter`) tidak mencakup kolom `Preprocess (ms)`, `Inference (ms)`, `Postprocess (ms)` yang baru ditambahkan ke `seg_row`.
- **Fixed di**: `yolo8/eval_multigpu.py`, `yolo9/eval_multigpu.py`, `yolo11/eval_multigpu.py`.

#### ✅ Fix: Multi-GPU Reporting di `main.py`
- `yolo8/main.py`, `yolo9/main.py`, `yolo11/main.py`: Mengganti hardcoded `device=0` dan `get_gpu_report_str(0)` menjadi `device=DEVICE` dan `get_gpu_report_str(DEVICE)` agar laporan deteksi mencerminkan semua GPU yang terpakai.

#### ✅ Hybrid Evaluation — Major Fix
- **Masalah lama**: Field timing tertukar — `Preprocess (ms)` berisi YOLO wall-clock total, `Inference (ms)` berisi SAM2 wall-clock total (≥200ms, tidak representatif).
- **Perbaikan**:
  - Worker mengekstrak `result[0].speed["preprocess"]`, `["inference"]`, `["postprocess"]` dari YOLO secara per-gambar.
  - `Inference (ms)` = YOLO_inference + SAM2_wall_clock (komponen serial pipeline).
  - `Postprocess (ms)` = YOLO postprocess (NMS).
  - Precision & Recall dihitung secara manual dari matching prediksi bbox vs GT (IoU ≥ 0.5) — menggantikan nilai `N/A (MultiGPU)`.
- `seg_row` ditambah kolom `Preprocess`, `Inference`, `Postprocess`.

#### ✅ Simplifikasi Label Model
Semua label model di CSV report dipersingkat untuk konsistensi publikasi:
| Sebelum | Sesudah |
|---|---|
| `YOLOv8m (Fine-tuned COCOeval)` | `YOLOv8m` |
| `YOLOv9m (Fine-tuned COCOeval)` | `YOLOv9m` |
| `YOLO11m (Fine-tuned COCOeval)` | `YOLO11m` |
| `Mask R-CNN ResNet-50 FPN-v2 (DDP Fine-tuned, MultiGPU Eval)` | `Mask R-CNN ResNet-50 FPN-v2` |
| `Hybrid (YOLO11m+SAM2, MultiGPU)` | `Hybrid (YOLO11m+SAM2)` |

#### ✅ Dokumentasi
- `run_pipeline.py`, `run_pipeline_multi.py`: Log filename distandarisasi (`run_pipeline.log`, `run_pipeline_multi.log`).
- `run_pipeline_multi.py`: tmux session name diperbarui ke `run_pipeline_multi`.

---

## 🏷️ v0.1.1 — `MyFineTunning_20260505_000000`
- **Date:** May 5, 2026
- **Status:** ✅ Release (Stable)

### 📝 Summary
Major updates to training infrastructure with hybrid approach integration, custom upload functionality, and codebase cleanup.

### 🚀 What's Changed
- **Hybrid Approach**: Added hybrid model implementation (YOLO11 + SAM2) with documentation and performance metrics.
- **Custom Upload**: New `custom_upload.py` module for flexible model upload workflows.
- **Code Cleanup**: Removed unnecessary cache files, logs, and deprecated scripts from mask-r-cnn module.
- **RClone Integration**: Added `rclone.conf.example` for cloud storage synchronization.
- **Updated Core Modules**: `config_shared.py`, `main.py`, `run_pipeline.py`, YOLO variants (yolo8, yolo9, yolo11).
- **Documentation**: Updated README with current project structure and workflow guidelines.

---
**GitHub:** `https://github.com/vnot-programming/Trainning-Models`
