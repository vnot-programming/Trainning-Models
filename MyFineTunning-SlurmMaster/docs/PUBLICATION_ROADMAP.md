# Peta Jalan Publikasi & Integrasi SOTA Edge AI (2026-2027)
**Proyek:** Reverse Vending Machine (RVM) Tanpa Konveyor (Konteks Teknik Lingkungan)

---

## BAGIAN I: PETA JALAN TIGA PUBLIKASI SCOPUS Q1/Q2

### 1. Paper 1: The Cloud-Edge Hybrid Ecosystem
- **Judul Tentatif:** *Achieving Zero-Sorting-Error in Reverse Vending Machines using Cloud-Tethered Zero-Shot Segmentation.*
- **Konfigurasi Model:** YOLO11l (Detection Prompt) + SAM2.1_t (Segment Anything 2).
- **Infrastruktur Target:** Edge Node (Raspberry Pi untuk *capture*) terhubung ke Edge Server/Cloud via 5G/Wi-Fi.
- **Fokus Riset:** Mengedepankan akurasi absolut (*Mask* dan *Box mAP*) di mana RVM bertindak sebagai agen IoT pasif, sementara beban inferensi berat diselesaikan di sisi *server*.

### 2. Paper 2: The Autonomous Green-Edge (Standalone)
- **Judul Tentatif:** *Democratizing Low-Power Autonomous Waste Sorting: An Evaluation of YOLO11 Nano Segmentation for Edge Computing.*
- **Konfigurasi Model:** YOLO11n-Seg (Native Segmentation).
- **Infrastruktur Target:** Murni berjalan *offline* pada Raspberry Pi 5 atau Jetson Nano.
- **Fokus Riset:** Mengorbankan sebagian kecil akurasi demi latensi ultra-rendah (<30ms) dan jejak memori super kecil (~6 MB). Menekankan aspek keberlanjutan energi (*energy sustainability*) untuk penyebaran RVM mandiri bertenaga surya.

### 3. Paper 3: SOTA Hybrid-Edge (Next-Gen AI)
- **Judul Tentatif:** *Pushing Zero-Shot to the Edge: A Comparative Study of CNNs (MobileSAM) vs Vision Transformers (RT-DETR) for Micro-RVM Architectures.*
- **Konfigurasi Model:** (YOLO11n + MobileSAM) VS (RT-DETR).
- **Infrastruktur Target:** NVIDIA Jetson Orin Nano (Memanfaatkan Tensor Cores secara lokal).
- **Fokus Riset:** Mambawa kecanggihan arsitektur hibrida dari *Cloud* langsung ke *Edge device* tanpa koneksi internet. Mengadu kompresi arsitektur *ViT-Tiny* milik MobileSAM melawan kemampuan atensi global dari arsitektur *Real-Time DEtection TRansformer* (RT-DETR).

---

## BAGIAN II: RENCANA TEKNIS INSTALASI & FINE-TUNING (PAPER 3)

Untuk merealisasikan eksekusi **Paper 3**, kita akan mengekspansi *pipeline* Slurm Master yang saat ini berjalan. Seluruh API difasilitasi oleh ekosistem `ultralytics` secara mulus.

### A. Rencana Eksekusi Mobile SAM (Mobile Segment Anything)
*MobileSAM* diciptakan dengan mengganti komponen *Image Encoder* Transformer (ViT-H/ViT-L) yang berat pada SAM asli dengan *Vision Transformer* berukuran sangat mini (*ViT-Tiny*), mengecilkan ukurannya di bawah 40MB.

1. **Persiapan Model Weights:**
   - Karena ekosistem Ultralytics sudah mensertifikasi MobileSAM, kita akan mengunduh bobot resminya: `mobile_sam.pt`.
   - Perintah unduh: `wget https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt -P /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/models/`
2. **Arsitektur Skrip Hibrida:**
   - Modifikasi tidak membutuhkan pembuatan skrip atau folder baru (`hybrid_mobilesam` tidak diperlukan).
   - Skrip utama `utils/evaluation_hybrid_sota.py` telah mendukung secara *native* infrastruktur pemanggilan MobileSAM (melalui kondisional `sam_backend = "MobileSAM" if "mobile" in mtype else "SAM2"`). Kita hanya perlu mengaktifkan pengujian metriknya secara langsung pada pipeline yang ada.
3. **Fokus Evaluasi:**
   - Menghitung penurunan waktu inferensi (*inference time drop*) dari SAM2 (~80ms) menjadi MobileSAM (estimasi <15ms) pada lingkungan GPU V100, untuk merefleksikan kecepatannya pada Edge GPU (Jetson).

### B. Rencana Eksekusi RT-DETR (Real-Time DETR)
RT-DETR adalah varian transformer *end-to-end* pertama yang sanggup berjalan *real-time*. Model ini membuang kelemahan arsitektur YOLO (NMS - *Non-Maximum Suppression* lambat) dan murni menggunakan *attention mechanism*.

1. **Instalasi dan Setup:**
   - Ultralytics mendukung RT-DETR secara *native*. Kita akan memakai versi `rtdetr-l.pt` (Large) atau `rtdetr-resnet50.pt`.
2. **Pembuatan Pipeline Skrip (Fine-tuning):**
   - Membuat direktori `rtdetr`.
   - Membuat skrip `train_rtdetr.py`. Karena RT-DETR memiliki perilaku hiperparameter yang sedikit berbeda (tidak menggunakan *mosaic* berlebihan karena arsitektur transformer global), kita akan memodifikasi konfigurasi *data augmentation*.
   - Eksekusi *training* menggunakan `mp.spawn` pada Slurm V100 GPU:
     ```python
     from ultralytics import RTDETR
     model = RTDETR("rtdetr-l.pt")
     model.train(data=DET_YAML, epochs=100, imgsz=640)
     ```
3. **Integrasi ke Evaluator Global:**
   - Menambahkan struktur kamus (dictionary) RT-DETR ke dalam variabel `FAMILY_VARIANTS` di berbagai skrip evaluasi agar metriknya dapat disejajarkan langsung (*head-to-head*) melawan YOLO11. Target berkas yang wajib dimodifikasi:
     - `docs/PUBLICATION_ROADMAP.md`
     - `utils/generate_report_single_model.py`
     - `utils/eval_single_model.py`
     - `utils/standar_evaluation_visuals-new_method.py`
     - `utils/standar_evaluation-new_method.py`
     - `utils/golden_evaluation-new_method.py`
     - `utils/golden_evaluation_visuals-new_method.py`

---
**Dokumen dikelola oleh:** AI Research Assistant (Antigravity)
**Tujuan:** Panduan eksekusi infrastruktur *Machine Learning* untuk publikasi Scopus.
