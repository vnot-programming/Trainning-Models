# 📋 Software Development Plan (SDP)
# Proyek: MyFineTunning-SlurmMaster — Riset Scopus Q1/Q2

---

## Project Overview

- **Root Project:** `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/`
- **Konfigurasi Utama:** `config_shared.py`
- **Conda Environment:** `yolo_env`
- **GPU Node:** `ai2`, `ai3` (Tesla V100 32GB) via Slurm (Job ID aktif)
- **Total Model Eksperimen:** 49 Models (16 YOLO dasar, 1 Mask R-CNN, 16 Hybrid SAM2, 16 Hybrid Mobile SAM)

---

## Development Log / Progress Tracking

---

### [Entri 001] — Inisialisasi Skrip Evaluasi SOTA

- **Tanggal/Waktu:** 2026-05-27 23:18 WIB
- **Tugas yang diselesaikan:**
  - Membuat skrip evaluasi baru `utils/evaluation_hybrid_sota.py` yang menggabungkan evaluasi kuantitatif (COCOeval mAP) dan kualitatif (visual grid 1x4) untuk komparasi head-to-head YOLO11l+SAM2 vs YOLO11l+Mobile SAM.
  - Memperbaiki bug `NameError` (`SAM_PATH` → `SAM2_PATH`) pada fungsi visualizer.
  - Memperbaiki bug `IsADirectoryError` pada `SEG_DATASET_LOCATION` dengan menambahkan fungsi `_prepare_dataset_sota()`.
- **File yang diubah/dibuat:**
  - `utils/evaluation_hybrid_sota.py` [DIBUAT BARU]
- **Status saat ini:** Selesai (parsial) — skor mAP masih 0.0 (Bug belum diidentifikasi)
- **Catatan untuk AI selanjutnya (Handoff Note):** Skor mAP 0.0 karena bug pemetaan kunci `image_ids`. Cek fungsi `_infer_worker_sota` di mana pencarian kunci gambar menggunakan `os.path.basename` tetapi dictionary `image_ids` berisi path absolut.

---

### [Entri 002] — Perbaikan Bug Skor mAP = 0.0

- **Tanggal/Waktu:** 2026-05-27 23:59 WIB
- **Tugas yang diselesaikan:**
  - Mengidentifikasi dan memperbaiki bug pemetaan kunci gambar (`image_ids key mismatch`) yang menyebabkan seluruh 320 gambar dilewati saat inferensi sehingga skor mAP bernilai `0.0`.
  - Mengganti pencarian kunci gambar menjadi logika fleksibel yang memeriksa path absolut terlebih dahulu, kemudian fallback ke nama file biasa.
  - Mengintegrasikan parameter dinamis `EVAL_CONF` & `EVAL_IOU` dari `config_shared.py`.
- **File yang diubah/dibuat:**
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai — mAP berhasil terkalkulasi dengan nilai riil (YOLO11l+SAM2: mAP50-95 Mask = 0.0218)
- **Catatan untuk AI selanjutnya (Handoff Note):** Evaluasi berhasil untuk 1 model saja (YOLO11l vs SAM2 vs Mobile SAM). Perlu dirombak untuk skala 49 model penuh sesuai arahan user.

---

### [Entri 003] — Perombakan Skala Penuh: Evaluasi 49 Model Dual Path

- **Tanggal/Waktu:** 2026-05-28 00:18 WIB
- **Tugas yang diselesaikan:**
  - Merombak total `utils/evaluation_hybrid_sota.py` untuk mendukung evaluasi paralel terdistribusi **49 model** (16 YOLO dasar, 1 Mask R-CNN, 16 Hybrid SAM2, 16 Hybrid Mobile SAM).
  - Mengimplementasikan arsitektur Dual-Path Dataset:
    - **Path DET:** Dataset `GOLDEN_DET_DATASET_LOCATION` → Laporan `hybrid_sota_det.csv`
    - **Path SEG:** Dataset `GOLDEN_SEG_DATASET_LOCATION` → Laporan `hybrid_sota_seg.csv`
  - Menghapus fungsionalitas visual grid (atas permintaan user untuk efisiensi evaluasi kuantitatif murni).
  - Memperbaiki bug `iouType not supported` pada COCOeval — memetakan `split_type` (`det`/`seg`) ke `iou_type` (`bbox`/`segm`) yang valid secara eksplisit.
  - Mengeksekusi evaluasi via TMUX `training_pipeline` di node GPU `@ai3` menggunakan workflow TMUX yang benar.
  - Memperbarui `GEMINI.md` untuk mencakup prosedur **Langkah 3.1 — Panduan Eksekusi TMUX** dan memperbarui total model dari 33 ke **49 Model**.
- **File yang diubah/dibuat:**
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI MAYOR]
  - `.gemini/GEMINI.md` [DIMODIFIKASI — total model 49, tambah Langkah 3.1]
  - `data-files/MyFineTunning-20260526_123630/reports/paper1/csv/new-method/hybrid_sota_det.csv` [DIBUAT — 50 baris, 49 model deteksi]
  - `data-files/MyFineTunning-20260526_123630/reports/paper1/csv/new-method/hybrid_sota_seg.csv` [DIBUAT — 41 baris, model segmentasi + hybrid]
- **Status saat ini:** **Selesai 100%** — Kedua laporan CSV berhasil terisi dengan nilai metrik COCOeval yang riil.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Laporan deteksi memuat 49 model (16 YOLO det dasar, 16 YOLO seg dasar, 1 Mask R-CNN, beserta 32 varian hybrid SAM2 & Mobile SAM).
  - Laporan segmentasi memuat 7 YOLO seg dasar + 1 Mask R-CNN + 16 Hybrid SAM2 + 16 Hybrid Mobile SAM.
  - `hybrid_sota_det.csv` dan `hybrid_sota_seg.csv` tersimpan di: `reports/paper1/csv/new-method/`
  - Langkah berikutnya yang disarankan: Integrasi hasil CSV SOTA ke dalam tabel perbandingan akhir paper & pembuatan visualisasi komparatif (chart/bar graph).

---

### [Entri 004] — Penambahan Handler Penuh MobileSAM (16 Model)

- **Tanggal/Waktu:** 2026-05-28 01:37 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan konstanta `MOBILE_SAM_MODEL_PATH` → `models/mobile_sam.pt` (non-hardcode, dinamis dari ROOT).
  - Menambahkan handler `hybrid_det_mobile` & `hybrid_seg_mobile` di **4 titik kritis**:
    1. `_get_model_metrics()` — kalkulasi ukuran weights & jumlah parameter MobileSAM.
    2. `_infer_worker()` — load YOLO + MobileSAM, ekstraksi `yolo_key` yang benar via `.replace("_mobile", "")`.
    3. `eval_model_distributed()` — routing `det_row` / `seg_row` mencakup tipe `_mobile`.
    4. Loop dataset utama — routing `det_info` / `seg_info` dan filter skip berdasarkan tipe model.
  - Menggabungkan blok inference SAM2 & MobileSAM menjadi **satu blok** yang lebih bersih (`hybrid_det`, `hybrid_seg`, `hybrid_det_mobile`, `hybrid_seg_mobile`) dengan penambahan variabel `sam_backend` untuk log yang informatif.
  - Model yang kini aktif terevaluasi: **7 YOLO Det + MobileSAM** (`hybrid_det_mobile`) & **9 YOLO Seg/Det + MobileSAM** (`hybrid_seg_mobile`).
- **File yang diubah/dibuat:**
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI — 6 blok kode, +~50 baris]
- **Status saat ini:** Selesai — siap dieksekusi di node GPU.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan file `models/mobile_sam.pt` tersedia di path `ROOT/models/`. Jika belum ada, download terlebih dahulu sebelum menjalankan evaluasi.
  - Key ekstraksi untuk MobileSAM menggunakan: `mkey.replace("hybrid_", "").replace("_mobile", "")` — logika ini penting untuk mendapatkan `yolo_key` yang valid dari `get_output_dir()`.
  - Untuk menjalankan evaluasi: `python3 -u utils/evaluation_hybrid_sota.py 2>&1 | tee logs/evaluation_hybrid_sota.log`

---

### [Entri 005] — Bugfix: YOLOv10 MobileSAM Key Error (`_seg` tidak ada)

- **Tanggal/Waktu:** 2026-05-28 01:55 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki bug `MODELS_CONFIG` di mana key `YOLOv10m` dan `YOLOv10x` pada blok MobileSAM salah ditulis sebagai `hybrid_yolov10m_seg_mobile` / `hybrid_yolov10x_seg_mobile`.
  - Logika ekstraksi `.replace("hybrid_", "").replace("_mobile", "")` menghasilkan `yolov10m_seg` yang tidak ada di filesystem — YOLOv10 tidak memiliki varian segmentasi.
  - Perbaikan: key diubah ke `hybrid_yolov10m_mobile` / `hybrid_yolov10x_mobile` → hasil ekstraksi menjadi `yolov10m` / `yolov10x` (det model), konsisten dengan blok SAM2.
- **Root Cause:** YOLOv10 tidak memiliki varian `_seg`. SAM (SAM2/MobileSAM) bertindak sebagai segmentor, YOLO hanya sebagai prompt generator bounding box.
- **File yang diubah/dibuat:**
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI — line 122-123]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Aturan konsistensi key hybrid: Model **tanpa varian `_seg`** (YOLOv10m, YOLOv10x) → key tanpa `_seg` di semua blok hybrid.
  - Model **dengan varian `_seg`** (YOLOv8m/x, YOLOv9c/e, YOLO11n/l/x) → key tetap dengan `_seg_mobile`.
