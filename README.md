# 🤖 Computer Vision Research & Training Models (Multi-Environment)

[![Version](https://img.shields.io/badge/version-v0.3.0-blue?style=for-the-badge)](https://github.com/vnot-programming/Trainning-Models/releases)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange?style=for-the-badge)](https://github.com/vnot-programming/Trainning-Models)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

Repositori pusat untuk proses *Fine-Tuning* model, evaluasi komparatif SOTA (State of the Art), dan eksperimen *Computer Vision* berskala besar — secara khusus ditargetkan untuk publikasi ilmiah internasional standar **Scopus Q1/Q2**.

Repositori ini telah diarsiteki ulang untuk mendukung berbagai lingkungan komputasi (*Multi-Environment*) dan diintegrasikan dengan aplikasi evaluasi visual interaktif **RVM (Reverse Vending Machine)**.

---

## 🚀 Rilis & Status Saat Ini
- **Versi Rilis Aktif:** `v0.3.0-dev`
- **Fokus Utama:** Evaluasi komparatif berskala besar pada infrastruktur HPC (High-Performance Computing) Slurm, optimalisasi arsitektur *hybrid* segmentasi, serta visualisasi web berbasis Cloudflare Tunnel.
- **Rilis Log:** Detail riwayat perubahan tercatat pada [`release_notes.md`](./release_notes.md).

---

## 🖥️ Pembagian Lingkungan Kerja (Multi-Environment)
Repositori dibagi menjadi 3 sub-workspace utama untuk menyesuaikan target infrastruktur:

1. **`MyFineTunning-SlurmMaster/` (Workspace Aktif Utama)**
   - Dioptimalkan sepenuhnya untuk sistem kluster HPC Slurm (contoh: Tesla V100 32GB GPU nodes `ai2` dan `ai3` di lingkungan kampus KU).
   - Dilengkapi biner Cloudflare Tunnel untuk akses eksternal, sistem *Auto-Booking Daemon* GPU, pemantauan *queue* berbasis terminal, serta aplikasi web RVM.
2. **`MyFineTunning-RunPOD/`**
   - Konfigurasi khusus yang dioptimalkan untuk GPU *cloud instance* RunPOD (misal multi-GPU RTX A5000 / A6000) dengan kapasitas throughput tinggi.
3. **`MyFineTunning-dev/`**
   - Lingkungan *sandbox* pengembangan lokal untuk modifikasi kode cepat, pembuatan unit test, dan verifikasi alur data dasar.

---

## 🧠 Evaluasi Komparatif 49 Model (Scopus Q1/Q2 Target)
Pipeline riset ini mengevaluasi secara paralel dan head-to-head total **49 model** yang dibagi ke dalam empat kategori arsitektural:

| Kategori Model | Jumlah Model | Deskripsi Teknis |
| :--- | :---: | :--- |
| **YOLO Dasar (Deteksi & Segmentasi)** | 16 Model | Varian YOLOv8 (m, x, m-seg, x-seg), YOLOv9 (m, e, c-seg, e-seg), YOLOv10 (m, x), dan YOLO11 (n, l, x, n-seg, l-seg, x-seg) sebagai pembanding dasar (*baseline*). |
| **Mask R-CNN** | 1 Model | Representasi arsitektur Two-Stage klasik berbasis **ResNet-50 FPN V2** (TorchVision) dengan DDP (Distributed Data Parallel) evaluator. |
| **Hybrid SAM2** | 16 Model | Kombinasi 16 model YOLO di atas (sebagai generator kotak pembatas/*prompt generator*) dengan **SAM 2.1 Tiny (`sam2.1_t.pt`)** untuk segmentasi presisi tingkat piksel. |
| **Hybrid Mobile SAM** | 16 Model | Kombinasi 16 model YOLO dengan **Mobile SAM (`mobile_sam.pt`)** untuk segmentasi presisi tingkat piksel dengan beban VRAM yang sangat ringan. |

*Metrik Tambahan:* Skrip latihan mandiri terintegrasi penuh untuk **RT-DETR-L** dengan batas memori adaptif (`RTDETR_BATCH_SIZE = 12/14`) guna mencegah CUDA OOM pada arsitektur Transformer.

---

## 📊 Metrik Geometri Lanjutan (Standard Baru)
Selain metrik COCO standar (mAP50, mAP50-95 untuk Box dan Mask), sistem ini menyuntikkan kalkulasi metrik kontur tepi (*boundary*) secara dinamis untuk model segmentasi instansi:
- **Boundary IoU:** Mengukur kualitas presisi garis luar (*contour*) antara prediksi mask dan ground-truth dengan kernel adaptif `0.02 * sqrt(H² + W²)`.
- **Boundary AP:** Rata-rata presisi pada ambang batas Boundary IoU ≥ 0.5 (Standard publikasi SOTA).

---

## 📱 Aplikasi Web RVM (Reverse Vending Machine)
Berada di dalam `MyFineTunning-SlurmMaster/RVM`, aplikasi ini menyajikan evaluasi kualitatif interaktif yang ramah pengguna:
- **Backend (Flask API - Port 8502):** Dilengkapi **GPU Inference Queue (FIFO)**. Menjamin hanya 1 proses inferensi berjalan di GPU pada satu waktu untuk menghindari CUDA OOM saat diakses banyak pengguna eksternal.
- **Frontend (Port 8501):** Antarmuka premium berbasis *Bio-Digital Minimalism 2026* dengan dukungan Dual-Theme (Dark/Light), kompatibilitas buta warna (*Color Blind Safety*), mode perbandingan visual (hingga 5 model vs Ground Truth secara berdampingan), serta fitur ekspor gambar grid berkualitas tinggi.
- **Akses Publik (Cloudflare Tunnel):**
  - **Frontend:** [https://front-rvm.penelitian.my.id](https://front-rvm.penelitian.my.id)
  - **Backend API:** [https://backend-rvm.penelitian.my.id](https://backend-rvm.penelitian.my.id)

---

## 🛠️ Struktur Direktori Proyek
```
Trainning-Models/
├── MyFineTunning-SlurmMaster/     # Workspace aktif kluster Slurm (KU-Slurm)
│   ├── RVM/                       # Aplikasi Web Reverse Vending Machine
│   │   ├── backend/               # Flask API Server & Inference Queue
│   │   └── frontend/              # Web Interface (HTML, CSS, JS)
│   ├── docs/                      # Rencana pengembangan (SDP.md) & rencana publikasi
│   ├── utils/                     # Skrip pembantu, evaluasi baru, & auto-booking
│   │   └── slurm/                 # Pemantauan status antrean (pretty_squeue.py)
│   ├── yolo/                      # Varian kode training YOLO (v8, v9, v10, v11)
│   ├── mask-r-cnn/                # Kode training & evaluasi Mask R-CNN
│   ├── rtdetr/                    # Kode training & evaluasi RT-DETR-L
│   ├── datasets/                  # Lokasi dataset lokal (training, standard, golden, coco)
│   ├── models/                    # Penyimpanan bobot dasar (.pt)
│   ├── config_shared.py           # Konfigurasi Bersama (Single Source of Truth)
│   └── start_rvm.sh               # Launcher otomatis layanan RVM & Cloudflared
├── MyFineTunning-RunPOD/          # Workspace teroptimasi untuk cloud RunPOD
├── MyFineTunning-dev/             # Workspace pengembangan sandbox lokal
├── README.md                      # Dokumentasi utama repositori (File ini)
└── release_notes.md               # Catatan rilis versi repositori
```

---

## 🏃 Panduan Penggunaan di Kluster Slurm (KU-Slurm)

### 1. Perintah CLI Global `slurm`
Untuk mempermudah manajemen, utilitas `myslurm.sh` dapat dipanggil secara global melalui terminal di login node dengan mengetikkan:
```bash
slurm
```
Perintah ini akan menampilkan menu interaktif:
- Melakukan booking GPU baru ke antrean Slurm.
- Memantau antrean global cluster & antrean pribadi secara realtime.
- Melakukan attach/kill pada sesi tmux komputasi.
- Mengaktifkan, mematikan, atau memantau status RVM Web Services (Frontend, Backend, dan Cloudflare Tunnel).

### 2. Cara Booking GPU & Masuk Compute Node
Untuk melakukan pemesanan GPU secara aman dan tahan terhadap terputusnya sesi SSH:
```bash
# Inisialisasi Booking melalui daemon latar belakang (tmux session: gpu_booking)
slurm -> Pilih Menu 1 (Booking GPU Baru)

# Setelah booking aktif (Notifikasi Telegram dikirim), masuk ke compute node:
slurm -> Pilih Menu 2 (Masuk / Attach ke Node GPU)
# Otomatis berpindah ke shell compute node (@ai2/@ai3) dengan conda environment 'yolo_env' aktif.
```

### 3. Eksekusi Pipeline Training & Evaluasi
Di dalam compute node GPU:
```bash
# Jalankan seluruh pipeline training & evaluasi secara paralel/sekuensial terkelola:
cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster
python3 run_pipeline_parallel.py 2>&1 | tee run_pipeline_parallel.log

# Menjalankan evaluasi saja tanpa melatih ulang model (sangat berguna untuk analisis cepat):
python3 run_pipeline_parallel.py --eval-only --gpus 0
```

---

## 🔄 Alur Pengembangan Git (STRICT)
- **Branch Kerja:** `dev` (Semua agen AI dan pengembang wajib bekerja di branch ini).
- **Prosedur Push:** Perubahan diunggah ke branch `dev`. Sinkronisasi ke branch `main` atau `master` hanya boleh dilakukan atas perintah/konfirmasi tertulis langsung dari pengguna.

---

*Dikelola oleh Antigravity Agent | Diperbarui secara dinamis: Juni 2026*
