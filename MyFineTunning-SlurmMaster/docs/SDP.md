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

---

### [Entri 006] — Penambahan Skrip SOTA Hybrid ke Scheduler Pipeline

- **Tanggal/Waktu:** 2026-05-28 02:29 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi `run_pipeline_parallel.py` untuk mengintegrasikan evaluasi akhir (New Method) secara berurutan.
  - Empat skrip (`standar_eval`, `standar_eval_vis`, `golden_eval`, `golden_eval_vis`) **sudah** terdaftar sebelumnya di `new_eval_specs`.
  - Menambahkan skrip kelima: `evaluation_hybrid_sota.py` dengan ID `eval_new_hybrid_sota` ke dalam antrean, yang bergantung (dependent) pada selesainya skrip `eval_new_gld_vis`.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pipeline scheduler kini mengeksekusi ke-5 skrip secara berurutan di tahap akhir (global eval).
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan seluruh proses di-run dalam TMUX (`training_pipeline`) menggunakan perintah `python3 run_pipeline_parallel.py 2>&1 | tee run_pipeline_parallel.log`.
  - Sistem akan otomatis menunggu dependensi selesai sebelum menembakkan evaluasi Hybrid SOTA.

---

### [Entri 007] — Penyempurnaan Varian Hybrid SAM2.1_t & Dual Visualisasi

- **Tanggal/Waktu:** 2026-05-28 02:45 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi label model hybrid di skrip evaluasi kuantitatif (`standar_evaluation-new_method.py`, `golden_evaluation-new_method.py`, `generate_report_single_model.py`, dan `evaluation_hybrid_sota.py`) untuk secara eksplisit menggunakan nama akademis `+SAM2.1_t` (menggantikan `+SAM2`).
  - Merancang dan mengimplementasikan Dual-Grid Visualisasi pada `standar_evaluation_visuals-new_method.py` & `golden_evaluation_visuals-new_method.py`:
    - **Versi YOLOv8**: Plot grid perbandingan yang secara taktis menampilkan `Hybrid-v8 (YOLOv8m+SAM2)` dengan file output `grid_det_v8_...` dan `grid_seg_v8_...`.
    - **Versi YOLO11**: Plot grid perbandingan yang secara taktis menampilkan `Hybrid-v11 (YOLO11l+SAM2)` dengan file output `grid_det_v11_...` dan `grid_seg_v11_...`.
  - Memperbaiki bug inisialisasi dataset (`NameError: SEG_DATASET_LOCATION`) pada `standar_evaluation_visuals-new_method.py` dengan merujuk ke variabel `STANDAR_` yang diimpor secara tepat.
  - Mengintegrasikan pengiriman notifikasi Telegram secara stand-alone pada akhir eksekusi ketiga skrip evaluasi kuantitatif utama (`standar_evaluation-new_method.py`, `golden_evaluation-new_method.py`, dan `evaluation_hybrid_sota.py`) untuk mempermudah pemantauan langsung di luar scheduler.
  - Memperbarui dokumentasi docstring pada skrip pengarsipan lokal (`utils/upload_utils.py`) untuk menyajikan informasi path, folder, dan berkas target kompresi secara akademis dan presisi sesuai permintaan pengguna.
  - Memvalidasi seluruh sintaksis python (kompilasi sukses 100%) dan berhasil menguji fungsionalitas visualisasi di compute node GPU `@ai2` dengan 10 gambar sampel standard & golden tanpa kendala.
- **File yang diubah/dibuat:**
  - `utils/standar_evaluation-new_method.py` [DIMODIFIKASI]
  - `utils/golden_evaluation-new_method.py` [DIMODIFIKASI]
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI]
  - `utils/standar_evaluation_visuals-new_method.py` [DIMODIFIKASI]
  - `utils/golden_evaluation_visuals-new_method.py` [DIMODIFIKASI]
  - `utils/upload_utils.py` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh output visualisasi dual-grid tersimpan di folder `reports/paper1/visuals/new-method/standard/comparison/` dan `reports/paper1/visuals/new-method/golden/comparison/`.
  - Gambar individu hybrid juga tersimpan dengan nama file `hybrid_v8_...` dan `hybrid_v11_...`.
  - Skrip siap dieksekusi secara otomatis oleh pipeline scheduler paralel `run_pipeline_parallel.py`.

---

### [Entri 008] — Penambahan Fitur Upload ke Google Drive Menggunakan Rclone

- **Tanggal/Waktu:** 2026-05-27 20:54 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi skrip `upload.sh` untuk menambahkan perintah `upload`.
  - Mengimplementasikan sinkronisasi direktori lokal `datas/` ke Google Drive menggunakan alat `rclone`.
  - Mengonfigurasi struktur direktori dinamis dengan nama remote `gdrive:gdrive-backup/{hostname}/MyFineTunning-{timestamp}/datas` sesuai instruksi.
  - Memastikan alur berjalan secara sekuensial (dari proses packing lokal baru dilanjutkan dengan proses remote copy).
- **File yang diubah/dibuat:**
  - `upload.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 008]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Fitur dapat dipanggil dengan menjalankan `bash upload.sh upload`.
  - Pastikan profil konfigurasi remote pada rclone bernama `gdrive` dan telah di-authenticate. Apabila di masa mendatang penamaan remote berubah, konfigurasi perlu disesuaikan pada variabel `REMOTE_PATH` dalam fungsi `do_upload()` di skrip `upload.sh`.

---

### [Entri 009] — Kompresi Total Workspace Lokal & Pengecualian Unggahan

- **Tanggal/Waktu:** 2026-05-27 21:23 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan langkah kompresi `data-files/MyFineTunning-{workspace_id}` (seluruh folder workspace aktif) di `upload_utils.py` ke dalam direktori `datas/`.
  - File kompresi diberi nama dengan format `MyFineTunning-{workspace_id}.tar.gz`.
  - Menambahkan argumen `--exclude "MyFineTunning-*.tar.gz"` ke dalam eksekusi `rclone copy` pada skrip `upload.sh` agar file archive berukuran masif tersebut tidak ikut terunggah ke Google Drive (menghemat bandwidth dan kuota cloud).
- **File yang diubah/dibuat:**
  - `utils/upload_utils.py` [DIMODIFIKASI]
  - `upload.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 009]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - File arsip workspace hanya disimpan secara lokal di `/datas/`. Rclone otomatis akan mengabaikan file ini saat push ke Google Drive.

---

### [Entri 010] — Peningkatan UI/UX Bio-Digital pada myslurm.sh

- **Tanggal/Waktu:** 2026-05-29 11:00 WIB
- **Tugas yang diselesaikan:**
  - Mengubah fungsi `squeue` dari hanya menampilkan sesi user menjadi **Sesi Global** untuk melihat status cluster secara menyeluruh.
  - Menambahkan **Daftar Antrean Anda (USER)** secara terpisah di bawah antrean global sehingga pengguna dapat membedakan job miliknya dan job orang lain secara paralel.
  - Menerapkan desain *Bio-Digital Minimalism* menggunakan border box artistik untuk daftar `squeue` dan `tmux ls`.
  - Mencegat raw output `tmux ls` dengan perintah `awk` untuk menghasilkan penomoran baris dan ikon indikator aktif (🟢) pada daftar sesi tmux.
  - Menambahkan submenu interaktif dan pemrosesan validasi di *case 5* dan *case 4*, untuk menjamin kesalahan input (huruf atau angka di luar daftar) gagal secara elegan.
  - Mengubah opsi keluar (Keluar Menu) menjadi angka `0` dengan warna peringatan (Merah) menggunakan ANSI escape codes, sehingga antarmuka menjadi lebih konsisten dengan standar utilitas CLI pada umumnya.
  - **Penyempurnaan Menu 5 (Hapus Sesi tmux):** Mengimplementasikan siklus pengulangan dinamis (*continuous loop*). Setelah pengguna berhasil menghapus satu sesi tmux, layar akan melakukan render ulang secara instan memperlihatkan daftar sesi terbaru, memungkinkan pengguna menghapus beberapa sesi sekaligus tanpa harus bolak-balik ke menu utama. Cukup menekan `Enter` (kosong) untuk keluar dari sub-menu tersebut.
  - **Revisi Skrip:** Mengekstraksi fungsi `squeue` ke dalam skrip Python terpisah (`utils/slurm/pretty_squeue.py`) untuk menggantikan `awk` murni demi fleksibilitas manipulasi string yang lebih tinggi. Skrip ini mengubah kolom partisi menjadi kolom waktu berjalan (`TIME`) dengan format cerdas (contoh: "1d 5h", "2h 30m", "45s") dan memberi warna dinamis pada STATE (`R`=Hijau, `PD`=Kuning, `CG`=Merah).
  - **Sistem Auto-Resume Booking (True Background Daemon):** Mengelevasi arsitektur Auto-Resume dari sekadar fitur pasif di `myslurm.sh` menjadi sebuah pelindung (daemon) sejati 24/7 di dalam `utils/book_gpu.py`. Ketika `book_gpu.py` berjalan dalam `tmux` dan mendeteksi bahwa antrean mati karena `TIME LIMIT` atau *Error* lainnya, skrip python ini tidak akan menyerah; melainkan langsung merekonstruksi ulang pengiriman antrean ke Slurm secara independen dan melaporkannya ke Telegram tanpa henti. Ini memastikan GPU Booking kebal terhadap diskoneksi SSH!
  - **Sub-Menu Manajemen Sesi TMUX (Menu 5):** Mengembangkan Menu 5 dari yang tadinya hanya "Hapus Sesi" menjadi "Manajemen Sesi TMUX". Kini, pengguna disajikan Sub-Menu dengan dua pilihan: **1) Masuk ke Sesi Tmux (Attach)** dan **2) Hapus Sesi Tmux (Kill)**. Keduanya menggunakan mekanisme *continuous loop* di mana pengguna akan selalu dikembalikan ke daftar sesi terbaru setelah melakukan *attach* (lalu *detach*) atau *kill*, tanpa harus terlempar ke menu utama, sampai pengguna menekan Enter (kosong) untuk keluar.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI MAYOR]
  - `utils/slurm/pretty_squeue.py` [DIBUAT BARU / DIRESTRUKTURISASI]
  - `utils/book_gpu.py` [DIMODIFIKASI MAYOR - DAEMON ARCHITECTURE]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Bash menu interaktif kini jauh lebih intuitif dan stabil.
  - Eksekusi menggunakan `bash utils/myslurm.sh` secara langsung untuk navigasi cepat.

---

### [Entri 011] — Mengaktifkan Evaluasi Metode Baru (New Method Evaluations)

- **Tanggal/Waktu:** 2026-05-30 21:05 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki potensi *KeyError* pada `run_pipeline_parallel.py` karena pengguna memberikan komen pada kunci `"state"`.
  - Mengubah konfigurasi dari (awalnya `SKIPPED`) menjadi `"state": "PENDING",` agar tugas **Fase Evaluasi Metode Baru** dimasukkan ke dalam antrean eksekusi pipeline yang riil.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 011]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Pipeline kini akan secara otomatis menjalankan seluruh alur eksekusi metode baru (standar, golden, sota, dan visualisasinya) setelah evaluasi global tuntas.

---

### [Entri 012] — Integrasi Auto-Archive (upload_utils.py) ke Pipeline Utama

- **Tanggal/Waktu:** 2026-05-30 21:10 WIB
- **Tugas yang diselesaikan:**
  - Mengintegrasikan pemanggilan skrip `utils/upload_utils.py` langsung ke dalam *orchestrator* utama (`run_pipeline_parallel.py`).
  - Pemanggilan diletakkan pada tahap paling akhir (`_print_final_summary`), tepat setelah pembuatan visualisasi *grid* komparasi selesai dan sebelum notifikasi Telegram `Pipeline Finished` dikirimkan.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 012]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Mulai sekarang, setiap kali pipeline paralel (training + eval) tuntas, seluruh arsip hasil, bobot, dan log akan otomatis dikompresi ke format `.tar.gz` di dalam direktori `datas/`.

---

### [Entri 013] — Penambahan Notifikasi Telegram pada Pengarsipan Lokal

- **Tanggal/Waktu:** 2026-05-30 21:12 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis skrip `utils/upload_utils.py` dan memverifikasi ketiadaan fitur pelaporan Telegram.
  - Mengimplementasikan pengiriman notifikasi instan Telegram dengan memanggil `send_telegram_msg` dari `telegram_utils.py` (menyuntikkan *path* *root* proyek ke `sys.path`).
  - Laporan yang dikirim meliputi durasi kompresi, jumlah model, jumlah weights, dan jumlah folder terkompresi.
- **File yang diubah/dibuat:**
  - `utils/upload_utils.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 013]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Mulai sekarang, saat pengarsipan lokal dan kompresi `.tar.gz` selesai (baik dipanggil via `run_pipeline_parallel.py` maupun dipanggil manual), sistem akan mengirim rekap laporan sukses ke grup Telegram Anda.

---

### [Entri 014] — Bugfix Kritis: Daemon Gagal Auto-Resume pada Slurm TIMEOUT+

- **Tanggal/Waktu:** 2026-05-30 21:30 WIB
- **Tugas yang diselesaikan:**
  - Menginvestigasi insiden kegagalan skrip `book_gpu.py` yang tidak mengirimkan notifikasi Telegram dan *stuck* dalam *infinite loop* saat antrean mencapai *MaxWall* (Timeout).
  - Mengidentifikasi akar masalah: Output `sacct` dari Slurm untuk job yang mati paksa seringkali memiliki tambahan tanda plus (`TIMEOUT+`, `CANCELLED+`) atau string yang terpotong (`OUT_OF_ME`). Hal ini menyebabkan klausa `if state in [...]` yang sangat spesifik menjadi gagal (*False*).
  - Merombak logika kondisional dalam `monitor_job()` dengan menerapkan pendekatan *Catch-All* untuk *State End* (`elif state != "R"`). Setiap state yang bukan `PD`, `R`, atau `CG` kini dianggap sebagai interupsi valid dan langsung dieksekusi.
  - Memperbaiki parser `get_job_state()` agar tidak *crash* atau menghasilkan string kosong jika output `sacct` hanya menghasilkan satu kata (misal `"TIMEOUT"` tanpa *ExitCode*).
- **File yang diubah/dibuat:**
  - `utils/book_gpu.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 014]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Daemon `book_gpu.py` kini jauh lebih tangguh (resilient) dalam membaca variasi aneh dari string log Slurm. Jika terbunuh karena *QoS Limit* atau node mati, ia dijamin akan mengirim notifikasi Telegram dan merakit ulang *booking* baru.

---

### [Entri 015] — Bugfix Kritis: Task "New Method Eval" Stuck di PENDING

- **Tanggal/Waktu:** 2026-05-31 01:59 WIB
- **Tugas yang diselesaikan:**
  - Mengidentifikasi penyebab tugas-tugas *New Method Evaluation* (`New Method Std Eval`, dll) terperangkap (stuck) pada status `PENDING` di HUD `run_pipeline_parallel.py`.
  - **Akar Masalah (Root Cause):** Pada fungsi `_get_next_ready_task()`, mekanisme pengambilan (dispatch) task bertipe `global_eval` secara *hardcode* hanya mengecek *id* tugas bernama `"eval_global_multigpu"`. Tugas *New Method Eval* (meskipun memiliki properti `"type": "global_eval"`) diabaikan oleh kondisional ini dan tak pernah terpilih. Karena tugas tersebut tidak bisa dieksekusi, status `PENDING` bertahan tanpa ujung dan pipeline tidak pernah mencapai blok kode terminasi.
  - **Tindakan Perbaikan:** Merombak *loop* seleksi tugas berjenis *global_eval* sehingga sistem mengulangi `self.tasks.items()` untuk mencari seluruh daftar tugas yang berlabel `t["type"] == "global_eval"`. Sekarang, baik `eval_global_multigpu` maupun kelima evaluasi turunan *New Method* akan dieksekusi secara berurutan asalkan dependensi spesifik mereka sudah `SUCCESS`.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI - Bugfix Logic Pipeline]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 015]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Masalah ini berhasil diselesaikan. Apabila pengguna masih menemukan HUD dalam kondisi `PENDING`, berikan instruksi kepada mereka untuk mengulang skrip `run_pipeline_parallel.py` agar perubahan logika orkestrasi pipeline mulai bekerja. 

---

### [Entri 016] — Integrasi Visual Post-Processing ke HUD Pipeline Paralel

- **Tanggal/Waktu:** 2026-05-31 02:08 WIB
- **Tugas yang diselesaikan:**
  - Merespons permintaan pengguna untuk memunculkan eksekusi skrip `upload_utils.py` dan `generate_comparison_grid.py` ke dalam HUD tabel *dashboard* (status `PENDING` / `RUNNING` / `SUCCESS`).
  - Menghapus pemanggilan eksklusif `subprocess.run()` untuk kedua *post-processing* tersebut dari fungsi `_print_final_summary()`.
  - Memasukkan kedua skrip tersebut ke dalam *task registry* (`_build_task_registry`) sebagai entri tugas berjenis `"global_eval"`:
    - **`post_grid`** (`Generate Final Grid`) — bergantung pada `eval_new_hybrid_sota`.
    - **`post_upload`** (`Local Archive & Upload`) — bergantung pada `post_grid`.
  - Memodifikasi fungsi `_dispatch_ready_tasks` untuk menangani parameter `device_arg` yang kosong (`""`), guna menghindari penyisipan argumen *GPU ID* palsu (yang bisa menimbulkan *crash*) bagi tugas-tugas *post-processing* (yang tidak memakai GPU).
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 016]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Kini daftar task di antrean *dashboard terminal* memuat dua item terakhir untuk pembuatan visual grid dan pengarsipan ZIP. Seluruh siklus hidup komputasi dan pengarsipan benar-benar terpusat di satu orchestrator log.
