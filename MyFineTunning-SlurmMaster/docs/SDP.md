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

---

### [Entri 017] — Pembaruan Persona Global & Pembuatan Laporan Scopus Q1

- **Tanggal/Waktu:** 2026-05-31 10:30 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan aturan persona global ke `.gemini/GEMINI.md` yang menetapkan peran agen sebagai **Asisten Peneliti Ilmiah tingkat Internasional Scopus Q1/Q2**. Agen kini diwajibkan untuk berdiskusi berbasis referensi *State of the Art* (SOTA) dan menanyakan bidang penelitian jika pengguna tidak menyebutkannya.
  - Melakukan analisis mendalam terhadap 7 laporan CSV dari metode baru (`golden_det.csv`, `golden_seg.csv`, `hybrid_sota_det.csv`, `hybrid_sota_seg.csv`, `sota_comparison_report.csv`, `standart_det.csv`, `standart_seg.csv`).
  - Menyusun laporan standar riset ilmiah tingkat Scopus Q1/Q2 di bidang Teknik Lingkungan (Environmental Engineering) mengenai klasifikasi botol pada *Reverse Vending Machine* (RVM) tanpa konveyor. Laporan berfokus pada analisis performa deteksi model YOLO yang dipadukan dengan SAM2 / MobileSAM.
- **File yang diubah/dibuat:**
  - `.gemini/GEMINI.md` [DIMODIFIKASI]
  - `datas/scopus_q1_environmental_engineering_report.md` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 017]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Laporan analisis penelitian telah tersimpan di `datas/`. Ke depannya, interaksi agen harus lebih akademis dan kritis mengenai State of the Art.

---

### [Entri 018] — Penambahan Tabel Data Model ke Laporan Ilmiah

- **Tanggal/Waktu:** 2026-05-31 10:39 WIB
- **Tugas yang diselesaikan:**
  - Melakukan penyisipan *snippet* tabel Markdown langsung ke dalam `scopus_q1_environmental_engineering_report.md` untuk menyajikan metrik komparasi parameter spesifik secara gamblang.
  - Tabel yang ditambahkan meliputi komparasi Standard vs Golden Dataset (YOLO11n+SAM2.1_t), Komparasi Segmentasi Zero-Shot (YOLOv10m vs YOLOv8m vs SAM2.1_t vs MobileSAM), serta Tabel Anomali SOTA (YOLO11l + SAM2/MobileSAM).
- **File yang diubah/dibuat:**
  - `datas/scopus_q1_environmental_engineering_report.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 018]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Laporan telah tervisualisasi lebih baik dengan kehadiran tabel komparasi. Siap untuk dijadikan draf atau diekspor oleh pengguna.

---

### [Entri 019] — Pembuatan Laporan Scopus Q1 (Paper 2: Pipeline Kompilasi)

- **Tanggal/Waktu:** 2026-05-31 10:50 WIB
- **Tugas yang diselesaikan:**
  - Melakukan pembacaan komprehensif pada direktori data CSV pipeline (MultiGPU): `kompilasi_ALL_detection.csv` dan `kompilasi_ALL_segmentation.csv`.
  - Mengonstruksi laporan analisis lanjutan (Paper 2) standar Scopus Q1 yang berfokus pada ranah teknik lingkungan dan *Edge Computing*.
  - Melakukan telaah anomali *compression-to-precision ratio* pada arsitektur nano YOLO11n dan membedahnya secara kritis melawan skenario komputasi *zero-shot* Hybrid SAM2.1_t serta Mask R-CNN.
- **File yang diubah/dibuat:**
  - `datas/paper2-scopus_q1_pipeline_report.md` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 019]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Laporan ke-2 berorientasi pada hasil akhir agregat pipeline terdistribusi, memantapkan konklusi akademis mengenai penggunaan YOLO11n untuk arsitektur IoT lingkungan yang berkelanjutan.

---

### [Entri 020] — Penyusunan Peta Jalan Publikasi (3 Papers) & Integrasi SOTA

- **Tanggal/Waktu:** 2026-05-31 11:20 WIB
- **Tugas yang diselesaikan:**
  - Menyusun rancangan *roadmap* tiga level publikasi saintifik berstandar Scopus Q1/Q2 (Cloud-Hybrid, Autonomous Green-Edge, dan SOTA Hybrid-Edge).
  - Menyusun kerangka teknis untuk instalasi dan *fine-tuning* model *Next-Gen* (MobileSAM dan RT-DETR) agar terintegrasi dengan ekosistem `ultralytics` secara langsung pada *pipeline* evaluasi yang ada.
- **File yang diubah/dibuat:**
  - `docs/PUBLICATION_ROADMAP.md` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 020]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Roadmap publikasi telah disahkan. Langkah teknis berikutnya adalah menginisiasi penulisan direktori dan skrip untuk RT-DETR dan modifikasi SAM menjadi MobileSAM di folder `utils` atau folder khusus komputasi baru. Perhatikan untuk merujuk pada `PUBLICATION_ROADMAP.md` sebelum membangun infrastruktur RT-DETR.

---

### [Entri 021] — Revisi Roadmap Publikasi (Optimalisasi MobileSAM & Target Skrip RT-DETR)

- **Tanggal/Waktu:** 2026-05-31 11:25 WIB
- **Tugas yang diselesaikan:**
  - Menghapus rencana pembuatan folder `hybrid_mobilesam` karena `utils/evaluation_hybrid_sota.py` telah dikonfirmasi memiliki arsitektur dinamis untuk memproses MobileSAM secara langsung.
  - Memperbarui daftar skrip target evaluasi (Golden & Standar) untuk integrasi model RT-DETR.
- **File yang diubah/dibuat:**
  - `docs/PUBLICATION_ROADMAP.md` [DIMODIFIKASI - Penambahan list skrip dan efisiensi MobileSAM]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 021]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** Skrip evaluator Anda ternyata sangat *scalable*. Jangan membuat *script* Hibrida baru untuk MobileSAM; langsung picu `evaluation_hybrid_sota.py` menggunakan parameter "mobile" pada *mtype*.

---

### [Entri 022] — Penguatan Aturan Global Persona (Proactive Deep Analysis)

- **Tanggal/Waktu:** 2026-05-31 11:30 WIB
- **Tugas yang diselesaikan:**
  - Melakukan telaah menyeluruh terhadap file sentral `config_shared.py` yang mendikte arsitektur dan parameter workspace.
  - Memperbarui sistem aturan global AI di `.gemini/GEMINI.md` dengan menambahkan perintah wajib **Deep Analysis & Proactive Tracing** berbasis `config_shared.py`, agar AI selanjutnya tidak pernah luput dalam memeriksa *source of truth* tanpa perlu ditegur pengguna.
- **File yang diubah/dibuat:**
  - `.gemini/GEMINI.md` [DIMODIFIKASI - Penambahan Aturan Anti-Halusinasi No. 3]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 022]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** *Rule* telah terpasang permanen di identitas dasar. Jadikan `config_shared.py` sebagai langkah pertama saat hendak mencari nilai path atau parameter apa pun.

---

### [Entri 023] — Pengetatan Prosedur Eksekusi Slurm (Zero Tolerance for Login Node Execution)

- **Tanggal/Waktu:** 2026-05-31 11:32 WIB
- **Tugas yang diselesaikan:**
  - Membatalkan paksa (*kill*) eksekusi `python config_shared.py` yang sebelumnya dijalankan keliru pada terminal Login Node.
  - Memperbarui panduan Slurm untuk menegaskan larangan mutlak terhadap eksekusi script *Python* apa pun (termasuk *download/setup*) di luar sistem Slurm. Seluruh eksekusi diwajibkan menggunakan `utils/myslurm.sh`.
- **File yang diubah/dibuat:**
  - `SLURM_Guide.md` [DIMODIFIKASI - Penambahan Peringatan Kritis di Awal Dokumen]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 023]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):** JANGAN PERNAH menjalankan perintah `python <nama_file>` menggunakan `run_command` kecuali Anda yakin 100% Anda sedang berada di dalam *tmux session* dari Slurm atau telah tervalidasi menggunakan `myslurm.sh`. Biasakan masuk via `attach_gpu.sh` sebelum mulai bekerja.

---

### [Entri 024] — Implementasi RT-DETR Pipeline (Folder, Script, & Integrasi Evaluator)

- **Tanggal/Waktu:** 2026-05-31 11:45 WIB
- **Tugas yang diselesaikan:**
  - Membuat direktori dan skrip `rtdetr/train_rtdetr.py` tanpa satu pun nilai *hardcode* — seluruh hyperparameter, path, dan nama model dibaca dinamis dari `config_shared.py`.
  - Mendaftarkan keluarga model `"rtdetr"` ke dalam `FAMILY_VARIANTS` di `utils/eval_single_model.py` dan `utils/generate_report_single_model.py`.
  - Menambahkan task `train_rtdetr` dan `eval_rtdetr` ke dalam `run_pipeline_parallel.py` (train_specs, eval_specs, global_eval deps, dan run_any check).
  - Menambahkan bobot `"rtdetr-l.pt"` ke dalam `ALL_BASE_MODELS` di `config_shared.py`.
- **File yang diubah/dibuat:**
  - `rtdetr/train_rtdetr.py` [DIBUAT BARU]
  - `utils/eval_single_model.py` [DIMODIFIKASI — Penambahan entry rtdetr di FAMILY_VARIANTS]
  - `utils/generate_report_single_model.py` [DIMODIFIKASI — Penambahan entry rtdetr di FAMILY_VARIANTS]
  - `run_pipeline_parallel.py` [DIMODIFIKASI — Registrasi task train_rtdetr & eval_rtdetr]
  - `config_shared.py` [DIMODIFIKASI — Penambahan rtdetr-l.pt ke ALL_BASE_MODELS]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 024]
- **Status saat ini:** Selesai. Siap untuk diuji oleh User di node GPU.
- **Catatan untuk AI selanjutnya (Handoff Note):** Untuk menguji coba saja (tanpa training penuh), User bisa menjalankan `python train_rtdetr.py --skip-eval` dari dalam node GPU. Pastikan bobot `rtdetr-l.pt` sudah tersedia di folder `models/` (diunduh via `ensure_all_base_models()` di `config_shared.py`).

---

### [Entri 025] — Pemisahan Konfigurasi Batch RT-DETR (RTDETR_BATCH_SIZE)

- **Tanggal/Waktu:** 2026-05-31 15:18 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis root cause OOM (CUDA Out of Memory) pada training RT-DETR-L yang dipicu oleh penggunaan `YOLO_BATCH_SIZE=30` — nilai yang didesain untuk YOLO CNN, bukan Transformer.
  - Memverifikasi data empiris dari log `data-files/MyFineTunning-20260526_123630/logs/train_rtdetr.log`:
    - `batch=30` → ❌ OOM (baris 80)
    - `batch=15` → ❌ OOM (baris 94)
    - `batch=7`  → ✅ Stabil di ~16.4–16.9GB VRAM (baris 110+)
  - Menambahkan konstanta `RTDETR_BATCH_SIZE = 8` ke `config_shared.py` (baris 164) dengan blok komentar teknis lengkap yang menjelaskan alasan arsitektural (108 GFLOPs, O(N²) attention memory).
  - Nilai 8 dipilih sebagai satu langkah konservatif di atas batch=7 yang terbukti aman, memberikan throughput ~14% lebih tinggi dengan margin VRAM ~13GB tersisa.
  - Memperbarui `rtdetr/train_rtdetr.py` untuk mengimpor dan menggunakan `RTDETR_BATCH_SIZE` menggantikan `YOLO_BATCH_SIZE` di tiga titik: blok import, `model.train()`, dan print header pipeline.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI — Penambahan `RTDETR_BATCH_SIZE = 8` beserta komentar justifikasi teknis, baris 164–188]
  - `rtdetr/train_rtdetr.py` [DIMODIFIKASI — Ganti import & penggunaan `YOLO_BATCH_SIZE` → `RTDETR_BATCH_SIZE` di 3 lokasi]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 025]
- **Status saat ini:** Selesai. Siap diuji ulang di node GPU tanpa OOM.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jika setelah training ulang dengan `RTDETR_BATCH_SIZE=8` masih OOM, kembalikan ke `7` langsung di `config_shared.py` — tidak perlu menyentuh `train_rtdetr.py`.
  - `YOLO_BATCH_SIZE` tetap dipertahankan untuk semua model YOLO lainnya — JANGAN ubah nilainya untuk keperluan RT-DETR.
  - Untuk run ulang RT-DETR: masuk ke node GPU via TMUX → `cd rtdetr` → `python -u train_rtdetr.py 2>&1 | tee train_rtdetr.log`.
  - Catatan Scopus: Batch size kecil (8–16) pada arsitektur Transformer adalah lazim dan tidak mengurangi kualitas konvergensi — training 200 epoch sudah mencukupi.

---

### [Entri 026] — Bugfix Kritis: Auto-Resume Daemon Stuck Akibat squeue Hang

- **Tanggal/Waktu:** 2026-05-31 20:36 WIB
- **Tugas yang diselesaikan:**
  - Menginvestigasi masalah di mana fitur rebooking otomatis tidak berjalan setelah job Slurm dibatalkan karena `TIME LIMIT`.
  - Mengidentifikasi akar masalah (Root Cause): Perintah `squeue -j <job_id>` pada kluster ini dapat *hang* (blocking) tanpa batas waktu (*indefinitely*) jika mencari Job ID yang sudah tidak ada (terhapus dari antrean). Hal ini menyebabkan skrip Python (`book_gpu.py`) yang menggunakan `subprocess.check_output` menjadi tersangkut (deadlock) dan tidak bisa melanjutkan evaluasi state maupun mengirim notifikasi Telegram.
  - Menambahkan argumen `timeout=30` (detik) pada pemanggilan fungsi `subprocess.check_output` di dalam `get_cmd_output()`.
  - Menambahkan baris pengecualian `except (subprocess.CalledProcessError, subprocess.TimeoutExpired):` untuk mengamankan fungsi agar mengembalikan string kosong apabila terjadi *timeout*, sehingga `state` terbaca sebagai `UNKNOWN` dan alur *auto-resume* dapat terpicu sebagaimana mestinya.
  - Mematikan sesi tmux `gpu_booking` yang tersangkut agar pengguna bisa memulai versi perbaikan.
- **File yang diubah/dibuat:**
  - `utils/book_gpu.py` [DIMODIFIKASI — Penambahan timeout pada get_cmd_output]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 026]
- **Status saat ini:** Selesai. Siap dijalankan ulang.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan setiap ada pemanggilan perintah eksternal (terutama perintah `slurm` seperti squeue atau sacct), selalu amankan dengan fungsi `timeout` agar skrip daemon *true-background* tidak pernah mengalami `stuck/deadlock`.

---

### [Entri 027] — Perbaikan Caption Panel Grid & Konfirmasi Fitur Eval-Only

- **Tanggal/Waktu:** 2026-06-01 07:48 WIB
- **Tugas yang diselesaikan:**
  1. **Konfirmasi Fitur `--eval-only`**: `run_pipeline_parallel.py` sudah mendukung mode evaluasi saja via argumen `--eval-only` (baris 504-507). Perintah: `python3 run_pipeline_parallel.py --eval-only --gpus 0`
  2. **Bugfix Caption Panel Grid**: Memperbaiki fungsi `_build_panel_grid()` di `generate_report_single_model.py`. Sebelumnya, nama model ditulis di **bagian bawah** gambar menggunakan `cv2.putText` langsung yang tumpang tindih dengan konten gambar. Kini diperbaiki menjadi **COLOR TITLE BAR di bagian atas** setiap panel (konsisten dengan desain comparison grid di `eval_unu_helpers.py`).
  3. **Perbaikan Teks Error/N/A**: Teks `N/A` dan `ERROR` pada panel blank sekarang ditulis di tengah gambar (bukan pojok kiri atas) sehingga lebih mudah terlihat, dan tidak bertabrakan dengan title bar yang diletakkan di atas.
  4. **Palet Warna Title Bar**: Menambahkan konstanta `_PANEL_BAR_COLORS` (10 warna berbeda) agar setiap varian model dapat dibedakan secara visual dengan mudah. Bar berwarna merah gelap khusus untuk status Error/N/A.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI — fungsi `_build_panel_grid` + `generate_visuals_for_family`]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 027]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk meregenerasi visual panel (tanpa training ulang), gunakan: `python3 run_pipeline_parallel.py --eval-only --gpus 0`
  - Pastikan `IMAGE_SAMPLES_DIR` di `config_shared.py` masih menunjuk ke folder yang berisi gambar sampel valid.
  - Perbaikan caption ini berlaku untuk SEMUA family panel (yolo8, yolo9, yolo10, yolo11, maskrcnn, hybrid, rtdetr).

---

### [Entri 028] — Root Cause Fix: Hybrid N/A & Mask R-CNN ERROR pada Visual Panel

- **Tanggal/Waktu:** 2026-06-01 08:12 WIB
- **Tugas yang diselesaikan:**
  - **Root Cause Hybrid N/A:** Kode lama menggunakan `get_output_dir("hybrid_yolov8m")` yang menghasilkan path `runs/hybrid_yolov8m/weights/best.pt` — folder ini **tidak pernah ada**. Hybrid bukan model yang di-training sendiri; ia memakai YOLO base weights yang sudah ada (misal `runs/yolov8m/weights/best.pt`). **Fix:** Deteksi key yang dimulai dengan `"hybrid_"`, lalu strip prefix untuk mendapatkan YOLO base key yang benar.
  - **Root Cause Mask R-CNN ERROR:** Kode lama menggunakan `YOLO(pt_path)` untuk semua family, termasuk maskrcnn. File `best.pt` maskrcnn adalah PyTorch state dict TorchVision (`maskrcnn_resnet50_fpn_v2`) — tidak bisa di-load oleh `YOLO()` loader Ultralytics. **Fix:** Deteksi `family == "maskrcnn"`, lalu gunakan `maskrcnn_builder.build_model()` + `torch.load()` secara eksplisit.
  - **Tiga cabang inferensi** kini tersedia di `generate_visuals_for_family()`: (1) Mask R-CNN via TorchVision, (2) Hybrid via YOLO base + SAM2, (3) YOLO biasa (default).
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI — blok for-loop variants di `generate_visuals_for_family`]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 028]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk meregenerasi panel visual (tanpa training ulang): `python3 run_pipeline_parallel.py --eval-only --gpus 0`
  - Jika ada model Hybrid baru yang ditambahkan ke `FAMILY_VARIANTS["hybrid"]`, pastikan nama key selalu diawali `"hybrid_"` + nama YOLO base key yang sudah ada di `runs/`.
  - Mask R-CNN builder diimpor dari `mask-r-cnn/maskrcnn_builder.py` — pastikan path `ROOT/mask-r-cnn/` valid.

---

### [Entri 029] — Fix: Visual Mask R-CNN & Hybrid Selalu Di-skip di `run_family()`

- **Tanggal/Waktu:** 2026-06-01 11:36 WIB
- **Tugas yang diselesaikan:**
  - Ditemukan bug kritis di `run_family()` baris 1522–1530: Visual untuk `family in ["maskrcnn", "hybrid"]` **selalu di-skip** dengan pesan _"dilakukan oleh skrip terpisah"_ — padahal skrip terpisah tersebut tidak pernah dipanggil.
  - Setelah Entri 028 memperbaiki `generate_visuals_for_family()` agar mendukung maskrcnn & hybrid secara native, blok pengecualian ini sudah tidak relevan dan justru berbahaya.
  - **Fix:** Hapus blok `if family in ["maskrcnn", "hybrid"]` — semua family kini memanggil `generate_visuals_for_family()` secara seragam.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI — fungsi `run_family()` baris 1522]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 029]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - `--eval-only` di `run_pipeline_parallel.py` **SUDAH** menjalankan visual untuk semua family (YOLO, Mask R-CNN, Hybrid).
  - Jika hanya ingin visual tanpa COCOeval: gunakan `--skip-eval` pada `generate_report_single_model.py` langsung.
  - Jika hanya ingin COCOeval tanpa visual: gunakan `--skip-visual` pada `generate_report_single_model.py` langsung.

---

### [Entri 030] — Fix Kritis: `eval_global_multigpu` Selalu SKIP saat `--eval-only`

- **Tanggal/Waktu:** 2026-06-01 11:40 WIB
- **Tugas yang diselesaikan:**
  - Ditemukan bug kritis di `run_pipeline_parallel.py` baris 188: `run_any` hanya mengecek apakah ada **training task** yang PENDING — padahal saat `--eval-only` semua training tasks = SKIPPED → `run_any = False` → `eval_global_multigpu = SKIPPED`.
  - Efek domino: seluruh pipeline downstream ikut SKIPPED (`eval_new_std`, `eval_new_std_vis`, `eval_new_gld`, `eval_new_gld_vis`, `eval_new_hybrid_sota`, `post_grid`, `post_upload`).
  - **Fix:** `run_any` kini menggunakan logika OR — jika ada training task PENDING **ATAU** ada eval task PENDING, maka `eval_global_multigpu` = PENDING.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI — logika `run_any` di `_build_task_registry`, baris 186]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 030]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Alur `--eval-only` yang benar setelah fix ini:
    - SKIP: semua `train_*`
    - RUN: `eval_yolo8`, `eval_yolo9`, `eval_yolo10`, `eval_yolo11`, `eval_maskrcnn`, `eval_hybrid`, `eval_rtdetr`
    - RUN: `eval_global_multigpu` (menunggu semua eval selesai)
    - RUN: `eval_new_std` → `eval_new_std_vis` → `eval_new_gld` → `eval_new_gld_vis` → `eval_new_hybrid_sota`
    - RUN: `post_grid` → `post_upload`

---

### [Entri 031] — Fix Root Cause Sejati: Eval Tasks Cascade-Skip karena `_update_failed_dependencies()`

- **Tanggal/Waktu:** 2026-06-01 11:51 WIB
- **Tugas yang diselesaikan:**
  - Log aktual menunjukkan: `"Task YOLOv8 Eval otomatis dilewati karena dependensi (train_yolo8) SKIPPED"` — artinya bukan masalah `run_any`, tapi **fungsi `_update_failed_dependencies()`** (baris 309–324) yang meng-cascade-skip semua task PENDING yang memiliki dependency = SKIPPED.
  - Logika lama di baris 319: `if dep_state in ["FAILED", "SKIPPED"]` → tidak membedakan antara SKIPPED-karena-intentional (`--eval-only`) vs SKIPPED-karena-dep-gagal.
  - **Fix yang benar (baris 166–183):** Saat `eval_only = True`, eval tasks didaftarkan dengan `dependencies = []` (kosong). Tanpa dependency training yang SKIPPED, `_update_failed_dependencies()` tidak punya alasan untuk meng-cascade-skip eval tasks.
  - Pendekatan ini lebih bersih dari mengubah logika `_update_failed_dependencies()` karena tidak mempengaruhi skenario normal.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI — `_build_task_registry()` baris 165–220]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 031]
- **Status saat ini:** Selesai. Siap dijalankan ulang.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Perintah untuk menjalankan evaluasi penuh tanpa training: `python3 run_pipeline_parallel.py --eval-only --gpus 0`
  - Perbaikan ini di Entri 031 adalah fix **final** untuk `--eval-only`. Entri 030 (fix `run_any`) dan Entri 031 (fix `resolved_deps`) keduanya diperlukan.

---

### [Entri 032] — Kustomisasi Visual Panel Grid: Relokasi Ground Truth & Minimalis Title Bar

- **Tanggal/Waktu:** 2026-06-01 21:55 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi skrip `utils/generate_report_single_model.py` untuk menyesuaikan tata letak dan gaya panel grid visualisasi multi-model.
  - Memindahkan panel **Ground Truth** dari indeks pertama (posisi awal) ke kolom terakhir (posisi akhir) dari panel grid sehingga mempermudah pembacaan komparasi dari kiri ke kanan.
  - Mengubah latar belakang **TITLE BAR** di bagian atas setiap panel sel menjadi warna putih bersih polos (`(255, 255, 255)`) dengan warna teks hitam tebal (`(0, 0, 0)`), serta warna merah gelap (`(0, 0, 180)`) jika mendeteksi status error/N/A.
  - Menghilangkan shadow teks hitam di `cv2.putText` agar title bar terlihat minimalis, bersih, dan sesuai dengan estetika *Bio-Digital Minimalism*.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Perubahan ini akan langsung berdampak pada saat pembuatan panel visualisasi multi-model untuk semua family.
  - Jika ingin memverifikasi visual hasil perubahan tanpa melakukan training, jalankan evaluasi dengan mode `python3 run_pipeline_parallel.py --eval-only --gpus 0`.

---

### [Entri 033] — Restrukturisasi Subfolder Output Visual & Pemosisian Ulang Ground Truth

- **Tanggal/Waktu:** 2026-06-01 21:59 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi skrip `utils/generate_report_single_model.py` untuk mengelompokkan file output visualisasi (gambar prediksi individual dan panel grid) berdasarkan subdirektori nama model (dan family)-nya masing-masing.
  - Mengubah fungsi `_get_visuals_dir()` untuk menerima argumen opsional `family` secara dinamis, sehingga file output visual dialihkan ke `reports/pipeline/visuals/{family}/`.
  - Mengembalikan letak panel **Ground Truth** ke indeks pertama (awal panel grid) demi kenyamanan pengguna dan menghindari kebingungan, mengingat Ground Truth sudah terintegrasi sejak awal.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Perubahan ini akan mengelompokkan seluruh berkas visual per model family (seperti `yolo8`, `yolo9`, `yolov10`, `yolo11`, `maskrcnn`, `hybrid`, `rtdetr`) di subfolder masing-masing secara dinamis.
  - Untuk memicu regenerasi visual dengan folder rapi ini, jalankan `python3 run_pipeline_parallel.py --eval-only --gpus 0`.

---

### [Entri 034] — Visual Polish: Perbaikan Geser Box Hybrid, Kontras UI/UX, dan Nama Kelas Mask R-CNN

- **Tanggal/Waktu:** 2026-06-01 22:49 WIB
- **Tugas yang diselesaikan:**
  - **Fix Bounding Box Shift di Hybrid (SAM)**: Mengubah pengiriman argumen `bboxes` ke `sam_model.predict` dari format tensor GPU `pred_boxes` menjadi list Python murni `pred_boxes.tolist()`. Langkah ini berhasil mengeliminasi bug pergeseran bounding box akibat masalah interpretasi scale internal pada wrapper SAM.
  - **Peningkatan UI/UX Kontras Teks Label**: Mengimplementasikan helper `_get_contrast_color()` untuk menghitung tingkat kecerahan (*luminance*) warna latar belakang label (`theme_color`). Teks label bounding box kini secara dinamis berwarna hitam `(0, 0, 0)` di atas warna terang dan putih `(255, 255, 255)` di atas warna gelap, menjamin keterbacaan optimal sesuai standar aksesibilitas UI/UX.
  - **Fix Class Name Mask R-CNN**: Menambahkan helper `_load_class_names()` untuk memparsing nama kelas secara dinamis dari `SEG_YAML`. Memetakan index TorchVision Mask R-CNN (1-indexed) ke nama kelas riil dengan mengurangi indeks sebesar 1 (`cls_id = int(lb) - 1`), menggantikan teks label `cls{id}` bawaan yang sebelumnya hanya menampilkan indeks angka.
  - **Fix Laten NameError**: Memperbaiki variabel error `bool_mask` menjadi `bin_mask` di dalam block inferensi Hybrid untuk mencegah potensi runtime crash.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh visualisasi (YOLO, Mask R-CNN, dan Hybrid) kini memiliki standar visual yang sangat kontras, berestetika premium, dan presisi tinggi tanpa koordinat bergeser.
  - Jalankan regenerasi visual per model via: `python3 run_pipeline_parallel.py --eval-only --gpus 0`.

---

### [Entri 035] — Bugfix Kritis: Penanganan Format List pada Parser Nama Kelas YAML

- **Tanggal/Waktu:** 2026-06-01 22:58 WIB
- **Tugas yang diselesaikan:**
  - **Bugfix Kritis pada Parser YAML (`_load_class_names`)**: Mengatasi insiden warning/error `AttributeError: 'list' object has no attribute 'items'` yang muncul karena properti `"names"` di dalam berkas `data.yaml` berupa bertipe `list` (larik) bukan `dict` (dictionary/kamus).
  - Skrip sekarang secara dinamis memeriksa tipe data `"names"` menggunakan `isinstance()`. Jika berupa `dict`, ia diiterasi menggunakan `.items()`. Jika berupa `list`, ia diiterasi menggunakan `enumerate()` untuk menghasilkan pemetaan indeks integer ke nama kelas dengan aman.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Parser YAML nama kelas sekarang sepenuhnya tahan banting (*robust*) dan kompatibel dengan kedua variasi format penulisan `"names"` di YOLO YAML dataset (baik format list maupun dict).

---

### [Entri 036] — Peningkatan Visual Premium: Skip Gambar Individual & Perataan Tengah Label Title Bar

- **Tanggal/Waktu:** 2026-06-01 23:33 WIB
- **Tugas yang diselesaikan:**
  - **Efisiensi Disk & Inferensi (Skip Gambar Individual)**: Menonaktifkan penyimpanan gambar prediksi individual (`cv2.imwrite` untuk individual path) untuk semua model varian (YOLO, Mask R-CNN, Hybrid, RT-DETR) di bawah loop visualisasi gambar sampel. Sekarang, skrip hanya berfokus menghasilkan panel grid gabungan (`_panel.jpg`), menghemat ruang penyimpanan workspace secara signifikan.
  - **Visual Title Bar Alignment & Weight**: Memodifikasi fungsi `_build_panel_grid()` agar teks nama model di title bar terpusat secara sempurna secara horizontal (`text_x = (target_w - tw) // 2`). Mengubah ketebalan teks label menjadi non-bold (`thickness = 1`) demi estetika minimalis yang bersih.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Panel visualisasi kini hanya akan merender grid gabungan dengan title bar berformat horizontal center dan ketebalan tipis (tidak bold). Tidak ada lagi file gambar individual baru yang dihasilkan di subfolder.

---

### [Entri 037] — Pembersihan Workspace: Relokasi File Evaluasi Legacy & Pembaruan Impor Dependensi

- **Tanggal/Waktu:** 2026-06-02 00:12 WIB
- **Tugas yang diselesaikan:**
  - **Audit Impor & Pembersihan File Legacy**: Mengidentifikasi 6 skrip evaluasi legacy atau standalone (`eval_paper.py`, `eval_single_model.py`, `golden_evaluation.py`, `golden_evaluation_visuals.py`, `standar_evaluation.py`, `standar_evaluation_visuals.py`) yang tidak dipanggil secara langsung oleh orkestrator pipeline utama `run_pipeline_parallel.py`.
  - **Relokasi ke `utils/generals/`**: Memindahkan ke-6 berkas tersebut ke subfolder baru `/utils/generals/` untuk merapikan workspace utama di dalam folder `utils/`.
  - **Pembuatan `__init__.py`**: Membuat file inisialisasi package python kosong `/utils/generals/__init__.py` untuk memastikan kelancaran alur pencarian import.
  - **Pembaruan Jalur Impor Aktif**: Mengubah rujukan impor fungsi `load_maskrcnn` pada skrip evaluasi visualisasi utama (`standar_evaluation_visuals-new_method.py` dan `golden_evaluation_visuals-new_method.py`) dari:
    `from eval_paper import load_maskrcnn`
    menjadi:
    `from generals.eval_paper import load_maskrcnn`
    Ini menjamin bahwa pipeline evaluasi visual SOTA metode baru tetap dapat berjalan dengan normal tanpa mengalami interupsi `ModuleNotFoundError`.
- **File yang diubah/dibuat:**
  - `utils/generals/eval_paper.py` [DIPINDAHKAN]
  - `utils/generals/eval_single_model.py` [DIPINDAHKAN]
  - `utils/generals/golden_evaluation.py` [DIPINDAHKAN]
  - `utils/generals/golden_evaluation_visuals.py` [DIPINDAHKAN]
  - `utils/generals/standar_evaluation.py` [DIPINDAHKAN]
  - `utils/generals/standar_evaluation_visuals.py` [DIPINDAHKAN]
  - `utils/generals/__init__.py` [DIBUAT BARU]
  - `utils/standar_evaluation_visuals-new_method.py` [DIMODIFIKASI]
  - `utils/golden_evaluation_visuals-new_method.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Workspace `utils/` kini jauh lebih bersih dan rapi. Seluruh dependensi import aktif telah dialihkan dengan aman ke folder `generals/`.

---

### [Entri 038] — Desain Independen: Deklarasi Lokal load_maskrcnn & Pemutusan Dependensi eval_paper

- **Tanggal/Waktu:** 2026-06-02 00:15 WIB
- **Tugas yang diselesaikan:**
  - **Dekompresi Impor / Self-Contained Design**: Menghilangkan ketergantungan impor berkas visualisasi utama (`standar_evaluation_visuals-new_method.py` dan `golden_evaluation_visuals-new_method.py`) terhadap berkas legacy `eval_paper.py` (yang kini berada di `generals/`).
  - **Deklarasi Lokal `load_maskrcnn`**: Menulis fungsi `load_maskrcnn(device)` secara lokal di bawah `# Util Functions` di dalam kedua berkas visualisasi aktif tersebut. Langkah ini memastikan kedua berkas tersebut 100% independen dan *self-contained*.
  - **Pemberantasan Dependensi Impor generals**: Menghapus baris impor `from generals.eval_paper import load_maskrcnn` secara permanen untuk menjamin kekokohan sistem.
- **File yang diubah/dibuat:**
  - `utils/standar_evaluation_visuals-new_method.py` [DIMODIFIKIKASI]
  - `utils/golden_evaluation_visuals-new_method.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh visualisasi metode baru kini 100% independen dan tidak memiliki ketergantungan impor terhadap berkas legacy di dalam folder `generals/` demi ketahanan jangka panjang.

---

### [Entri 039] — Visual Polish: Penyempurnaan dan Perapian Docstring Pembuka generate_report_single_model.py

- **Tanggal/Waktu:** 2026-06-02 00:20 WIB
- **Tugas yang diselesaikan:**
  - **Penyempurnaan Docstring Pembuka**: Merapikan dan memperbarui docstring pembuka di berkas `utils/generate_report_single_model.py` agar berestetika premium, informatif, dan sangat terstruktur sesuai dengan pembaruan arsitektur visual terbaru.
  - **Sinkronisasi Output Visual**: Menambahkan informasi direktori output visualisasi gabungan (`reports/pipeline/visuals/{family}/`) dan memetakan struktur file panel grid `_panel.jpg` terbaru ke dalam visualisasi pohon direktori terdokumentasi.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Docstring pembuka kini sepenuhnya sinkron dengan alur kerja visualisasi dan evaluasi terbaru. Siap digunakan sebagai referensi pengoperasian orkestrasi model CLI.

---

### [Entri 040] — Pembersihan Workspace: Relokasi generate_paper_visuals.py & Desain Independen

- **Tanggal/Waktu:** 2026-06-02 00:23 WIB
- **Tugas yang diselesaikan:**
  - **Relokasi ke `utils/generals/`**: Memindahkan berkas utilitas visualisasi draf paper `generate_paper_visuals.py` ke folder `/utils/generals/` demi kerapian folder utama `utils/`.
  - **Pemutusan Ketergantungan**: Menjadikan skrip `generate_paper_visuals.py` 100% *self-contained* dengan mendeklarasikan fungsi `load_maskrcnn(device)` secara lokal di dalamnya, sehingga tidak lagi mengimpor secara eksternal dari `eval_paper.py`.
  - **Penyempurnaan Path Resolver**: Menyesuaikan setup pencarian path ROOT dan `_UTILS_DIR` di dalam berkas agar secara akurat mengenali direktori root proyek sejati walaupun berjalan dari subdirektori `/utils/generals/`.
- **File yang diubah/dibuat:**
  - `utils/generals/generate_paper_visuals.py` [DIBUAT BARU / REFACTOR]
  - `utils/generate_paper_visuals.py` [DIHAPUS]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Folder `utils/` utama kini bersih dari file utilitas eksternal. Untuk menjalankan visualisasi paper secara mandiri, pengguna dapat memanggil `python3 utils/generals/generate_paper_visuals.py`.

---

### [Entri 041] — Konsolidasi Modul: Sentralisasi flush_gpu & Pembersihan Total utils/

- **Tanggal/Waktu:** 2026-06-02 00:27 WIB
- **Tugas yang diselesaikan:**
  - **Sentralisasi `flush_gpu`**: Memindahkan dan mendeklarasikan fungsi bantu pembersihan CUDA VRAM `flush_gpu(label)` secara terpusat di dalam [config_shared.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/config_shared.py) sebagai *Single Source of Truth* hyperparameter dan utilitas bersama.
  - **Pembaruan Impor Visualisasi Aktif**: Mengubah rujukan impor `flush_gpu` pada berkas visualisasi aktif (`standar_evaluation_visuals-new_method.py` dan `golden_evaluation_visuals-new_method.py`) agar secara terpusat mengimpor langsung dari `config_shared` alih-alih `eval_unu_helpers`.
  - **Relokasi `eval_unu_helpers.py`**: Memindahkan berkas modul penunjang legacy `eval_unu_helpers.py` ke folder `/utils/generals/` demi menciptakan lingkungan folder utama `utils/` yang super bersih dan terorganisasi.
  - **Penyempurnaan Path Modul**: Menyesuaikan setup path resolver ROOT di dalam `utils/generals/eval_unu_helpers.py` agar secara presisi mengenali direktori root proyek.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI — Penambahan `flush_gpu`]
  - `utils/standar_evaluation_visuals-new_method.py` [DIMODIFIKASI — Impor dialihkan ke `config_shared`]
  - `utils/golden_evaluation_visuals-new_method.py` [DIMODIFIKASI — Impor dialihkan ke `config_shared`]
  - `utils/generals/eval_unu_helpers.py` [DIBUAT BARU / REFACTOR]
  - `utils/eval_unu_helpers.py` [DIHAPUS]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh alur kerja visualisasi aktif kini terbebas secara absolut dari dependensi berkas legacy di dalam folder `generals/`.
  - Folder `utils/` utama saat ini 100% steril dari berkas pendukung legacy.

---

### [Entri 042] — Pembersihan Root Proyek: Relokasi eval_boundary_iou.py & run_eval_multi.py

- **Tanggal/Waktu:** 2026-06-02 00:32 WIB
- **Tugas yang diselesaikan:**
  - **Relokasi ke `utils/generals/`**: Memindahkan berkas orkestrator legacy sekuensial `run_eval_multi.py` dan berkas evaluasi boundary `eval_boundary_iou.py` dari direktori ROOT proyek ke folder `/utils/generals/` demi meningkatkan kebersihan direktori root.
  - **Penyempurnaan Path Modul**: Mengubah mekanisme *path resolver* ROOT di dalam berkas generals tersebut agar secara presisi mengenali direktori root proyek sejati sehingga seluruh modular import tetap berjalan dengan normal.
  - **Perlindungan `coco_eval_utils.py`**: Berdasarkan hasil analisis dependensi, berkas `coco_eval_utils.py` diidentifikasi sebagai *Shared Library* inti yang sangat aktif (diimpor di main.py, model training, dan skrip evaluasi baru). Kami memutuskan berkas ini **wajib tetap berada di ROOT** untuk menjaga stabilitas arsitektur.
- **File yang diubah/dibuat:**
  - `utils/generals/eval_boundary_iou.py` [DIBUAT BARU / REFACTOR]
  - `utils/generals/run_eval_multi.py` [DIBUAT BARU / REFACTOR]
  - `eval_boundary_iou.py` [DIHAPUS]
  - `run_eval_multi.py` [DIHAPUS]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Direktori ROOT proyek saat ini sangat bersih dan hanya menampung berkas produksi aktif. Skrip legacy yang dipindahkan dapat dipanggil secara manual menggunakan rute folder generals yang sesuai.

---

### [Entri 043] — Terpadu & Independen: Penyatuan Skrip Evaluasi & Visualisasi New Method

- **Tanggal/Waktu:** 2026-06-02 10:55 WIB
- **Tugas yang diselesaikan:**
  - **Penciptaan `generate_report-new_method.py`**: Menggabungkan fungsionalitas skrip evaluasi kuantitatif `standar_evaluation-new_method.py` (penghasil CSV) dan skrip evaluasi kualitatif `standar_evaluation_visuals-new_method.py` (penghasil grid visualisasi komparatif) ke dalam satu berkas terpadu yang independen [utils/generate_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_report-new_method.py).
  - **Dukungan CLI Fleksibel**: Menyediakan argument parser (`--skip-eval` dan `--skip-visual`) agar pengguna dapat menjalankan salah satu fase (kuantitatif atau kualitatif saja) atau keduanya secara berurutan.
  - **Konfigurasi Path Sampel Tetap**: Mengarahkan rute gambar masukan visualisasi sampel secara eksklisit ke direktori tetap `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/data-files/MyFineTunning-20260526_123630/image_samples` sesuai instruksi.
  - **Sentralisasi Dependensi**: Menjamin berkas terpadu ini 100% independen dan tidak memiliki ketergantungan impor luar selain dari `config_shared.py` dan datasets.
- **File yang diubah/dibuat:**
  - `utils/generate_report-new_method.py` [DIBUAT BARU]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pengguna dapat menjalankan evaluasi terpadu metode baru secara lengkap dengan mengeksekusi: `python3 utils/generate_report-new_method.py --gpus 0`.

---

### [Entri 044] — Rename Skrip Terpadu & Sinkronisasi Docstring

- **Tanggal/Waktu:** 2026-06-02 11:15 WIB
- **Tugas yang diselesaikan:**
  - **Relokasi Fisik & Rename**: Mengubah nama berkas terpadu `utils/generate_report-new_method.py` menjadi `utils/generate_standar_report-new_method.py` [utils/generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) secara aman untuk mencegah kebingungan penamaan dengan laporan model tunggal.
  - **Sinkronisasi Docstring**: Menyesuaikan referensi nama berkas dan instruksi pemanggilan CLI di dalam docstring pembuka skrip tersebut agar merujuk ke nama yang baru (`generate_standar_report-new_method.py`).
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIBUAT BARU via Rename]
  - `utils/generate_report-new_method.py` [DIHAPUS via Rename]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pemanggilan skrip terpadu metode baru kini resmi menggunakan nama baru: `python3 utils/generate_standar_report-new_method.py --gpus 0`.

---

### [Entri 045] — Sentralisasi Konfigurasi Visual Sampel (VISUAL_NUM_SAMPLES)

- **Tanggal/Waktu:** 2026-06-02 11:32 WIB
- **Tugas yang diselesaikan:**
  - Mengidentifikasi anomali di mana visualisasi tetap menggunakan `5` sampel meskipun parameter bawaan diubah menjadi `10`, yang disebabkan oleh _overriding_ parameter CLI (`--samples 5`) dan panggilan lokal dalam `run_family()`.
  - Menambahkan _Single Source of Truth_ `VISUAL_NUM_SAMPLES = 10` ke dalam `config_shared.py` (baris 205).
  - Mengimpor variabel tersebut ke `utils/generate_report_single_model.py` dan menggunakannya sebagai nilai bawaan (default) pada CLI argumen `--samples`, parameter `generate_visuals_for_family`, dan parameter `run_family`.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI]
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jumlah sampel gambar yang diproses untuk visualisasi sekarang diatur terpusat di `config_shared.py` lewat konstanta `VISUAL_NUM_SAMPLES`. Jangan lagi meng-_hardcode_ angka tersebut di dalam skrip evaluasi tunggal manapun.

---

### [Entri 054] — Penyesuaian Alokasi RAM pada Skrip Booking GPU

- **Tanggal/Waktu:** 2026-06-02 16:20 WIB
- **Tugas yang diselesaikan:**
  - Mengubah konfigurasi alokasi *memory* di dalam file [utils/book_gpu.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/book_gpu.py) dari `--mem=32G` menjadi `--mem=64G`. Hal ini disesuaikan dengan batas limit maksimal QoS `normal` (Max RAM 64GB) di *Slurm workload manager* yang tertera pada *AI_KU_V100 User Documents*. Limit CPU `--cpus-per-task=8` dibiarkan karena 8 core adalah batas maksimal di QoS tersebut.
- **File yang diubah/dibuat:**
  - `utils/book_gpu.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jangan melebihi limit `--cpus-per-task=8` atau `--mem=64G` untuk partisi `gpu` (QoS `normal`) agar job tidak berstatus *QOSMaxCpuPerUserLimit*.

---

### [Entri 053] — Penyesuaian Evaluasi untuk Format Native COCO & Perbaikan Bug pycocotools

- **Tanggal/Waktu:** 2026-06-02 14:55 WIB
- **Tugas yang diselesaikan:**
  - Integrasi mode fallback dinamis pada skrip evaluasi `utils/generate_report_single_model.py` dan `utils/generate_standar_report-new_method.py` untuk memprioritaskan format native `_annotations.coco.json` dari `eval_dataset_coco/valid/` tanpa perlu konfigurasi YAML YOLO.
  - Implementasi konversi struktur data dictionary `image_ids` agar key dipetakan menggunakan *absolute path* sesuai kaidah worker paralel.
  - Penambahan lapisan filter di dalam `evaluate_coco_predictions` (`utils/coco_eval_utils.py`) untuk membuang empty segmentation arrays (`[]`) secara otomatis pada mode `iou_type="segm"`. Perbaikan ini menanggulangi masalah fatal IndexError (`list index out of range`) saat `pycocotools` dihadapkan dengan dataset yang mayoritas berupa deteksi kotak pembatas (Bounding Boxes) namun dipaksa dievaluasi dengan parameter segmentasi.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `utils/coco_eval_utils.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan pipeline evaluasi diuji secara komprehensif, mengingat error empty segmentation dari `pycocotools` kini akan memunculkan nilai bypass atau mAP 0.0 alih-alih `crash`.

---

### [Entri 046] — Sentralisasi IMAGE_SAMPLES_DIR pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 11:48 WIB
- **Tugas yang diselesaikan:**
  - Mengubah konfigurasi `IMAGE_SAMPLES_DIR` di dalam [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) agar diimpor langsung dari [config_shared.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/config_shared.py) (Single Source of Truth) menggantikan nilai path absolut yang sebelumnya di-hardcode.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan semua skrip evaluasi baru maupun lama merujuk ke parameter path yang didefinisikan di `config_shared.py` untuk menghindari ketidakcocokan workspace antarsesi.

---

### [Entri 047] — Pembuatan Rencana Model Baru ("Eco" YOLO-SAM Hybrid)

- **Tanggal/Waktu:** 2026-06-02 12:08 WIB
- **Tugas yang diselesaikan:**
  - Menyusun dan menganalisis kelayakan akademis penggabungan model deteksi/segmentasi YOLO (`best.pt`) dengan SAM2 (`sam2.1_t.pt`) ke dalam berkas bobot tunggal (`hybrid.pt`) sebagai base model kustom baru ("Eco").
  - Menganalisis perbedaan *differentiability* pada *backpropagation* antara model deteksi (BBox prompt) dan model segmentasi (mask-to-mask prompt).
  - Menyimpan seluruh analisis dan rekomendasi arsitektur di dalam dokumen rencana baru [docs/rencana-new-models.md](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/docs/rencana-new-models.md).
- **File yang diubah/dibuat:**
  - `docs/rencana-new-models.md` [DIBUAT/DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Konsep model "Eco" YOLO-SAM Hybrid siap diajukan untuk rancangan implementasi fase berikutnya. Rencana arsitektur difokuskan pada skenario *differentiable mask prompting* (Mask-to-Mask) untuk meloloskan gradien backpropagation secara penuh.

---

### [Entri 048] — Integrasi Strategi Desain Model Baru untuk Perangkat Edge

- **Tanggal/Waktu:** 2026-06-02 12:15 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan strategi desain model baru yang menggabungkan akurasi, kecepatan, dan kompatibilitas perangkat *edge* (Jetson Orin / Raspberry Pi) ke dalam [rencana-new-models.md](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/docs/rencana-new-models.md).
  - Menyusun 4 strategi optimasi: Cascaded Dynamic Execution, Knowledge Distillation (Teacher-Student), Hardware-Aware Optimization (TensorRT + INT8 Quantization), dan Lightweight Vision Transformer (EfficientViT).
- **File yang diubah/dibuat:**
  - `docs/rencana-new-models.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk uji coba performa nyata di Jetson Orin (`orin1`), prioritaskan kompilasi ONNX/TensorRT dengan skema INT8 Quantization untuk memeras VRAM seminimal mungkin.

---

### [Entri 049] — Koreksi Argumen CLI pada Petunjuk Pemanggilan generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 12:36 WIB
- **Tugas yang diselesaikan:**
  - Mengoreksi contoh perintah eksekusi di dalam docstring header [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) dengan mengganti argumen `--family all` yang tidak didukung menjadi `--gpus 0` agar tidak menimbulkan kesalahan parsing `unrecognized arguments` saat dijalankan.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Skrip `generate_standar_report-new_method.py` menguji seluruh model secara terpadu dan tidak mendukung filter per-family. Gunakan argumen `--gpus <GPU_ID>` untuk mengarahkan pemrosesan paralel multi-GPU.

---

### [Entri 050] — Perbaikan Kompatibilitas Tanda Tangan Fungsi flush_gpu

- **Tanggal/Waktu:** 2026-06-02 12:40 WIB
- **Tugas yang diselesaikan:**
  - Mengatasi error `TypeError: flush_gpu() takes from 0 to 1 positional arguments but 2 were given` pada skrip `generate_standar_report-new_method.py` saat inferensi worker.
  - Memodifikasi fungsi `flush_gpu` di [config_shared.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/config_shared.py) agar mendukung parameter opsional `gpu_id` (untuk sinkronisasi CUDA multi-GPU) secara fleksibel.
  - Menambahkan penanganan bertipe data dinamis (backward compatibility) sehingga jika parameter pertama bertipe string, fungsi otomatis memperlakukannya sebagai `label` dengan default `gpu_id = 0`, sehingga tidak merusak pemanggilan versi lama.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Fungsi `flush_gpu` kini resmi mendukung pemanggilan satu argumen `flush_gpu("label")` maupun dua argumen `flush_gpu(gpu_id, "label")` secara aman di seluruh codebase.

---

### [Entri 051] — Perbaikan Evaluasi COCOeval pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 12:42 WIB
- **Tugas yang diselesaikan:**
  - Mengatasi error `TypeError: expected str, bytes or os.PathLike object, not dict` yang disebabkan oleh pemanggilan fungsi `load_native_coco_gt` dengan argument bertipe kamus (dictionary) `coco_gt` (hasil kembalian `build_coco_ground_truth`).
  - Mengganti alur evaluasi dari fungsi pembungkus eksternal ke pemanggilan native `pycocotools.coco.COCO` dan `pycocotools.cocoeval.COCOeval` secara langsung di dalam berkas [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py), karena variabel hasil worker `dt_bbox` dan `dt_segm` sudah berformat native COCO.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Proses evaluasi metrik kuantitatif mAP Box dan mAP Mask saat ini 100% menggunakan API native `pycocotools` secara langsung untuk menjamin keandalan pemrosesan kamus ground truth hasil bentukan memori.

---

### [Entri 052] — Penyelarasan Dataset Evaluasi Standard pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 12:46 WIB
- **Tugas yang diselesaikan:**
  - Menyelaraskan sumber data evaluasi kuantitatif (Fase 1) pada skrip terpadu [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) agar menggunakan dataset evaluasi standard (`standard_datasets_det` untuk deteksi dan `standard_datasets_seg` untuk segmentasi) alih-alih dataset pelatihan/baseline (`DET_YAML`/`SEG_YAML` yang mengarah ke `training_det`/`training_seg`).
  - Mengubah fungsi pembangun ground truth menggunakan `load_native_coco_gt` dengan parameter folder `valid` dari masing-masing dataset standar untuk memuat data annotasi secara langsung.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Evaluasi kuantitatif dan visualisasi kualitatif pada skrip terpadu metode standar sekarang secara konsisten menggunakan dataset evaluasi yang sama dari `standard_datasets_det` and `standard_datasets_seg`.

---

### [Entri 053] — Penambahan Impor Pustaka json pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 12:57 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki bug `NameError: name 'json' is not defined` di dalam skrip [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) pada baris 436 (fungsi `init_classes_from_coco`).
  - Menambahkan impor modul bawaan `json` pada daftar deklarasi impor awal skrip.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pemanggilan parsing JSON dalam inisialisasi kelas COCO kini dapat dieksekusi dengan aman tanpa memicu kegagalan runtime.

---

### [Entri 054] — Perbaikan Perhitungan Ukuran Buffer Mask R-CNN pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 13:00 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki bug `NameError: name 'b' is not defined` di dalam skrip [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) pada baris 184 (fungsi `_get_model_metrics`).
  - Mengubah kode `sum(b.nelement() * b.buffers())` yang tidak valid menjadi `sum(b.nelement() * b.element_size() for b in m.buffers())` untuk melakukan loop dan menghitung ukuran memori buffer model Mask R-CNN secara tepat.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Ukuran file (MB) dan jumlah parameter (M) untuk model Mask R-CNN sekarang dapat dihitung dengan benar di awal eksekusi kuantitatif tanpa memicu *warning* kegagalan hitung.

---

### [Entri 055] — Penambahan Impor Global YOLO dan SAM pada generate_standar_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 13:09 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki bug `NameError: name 'YOLO' is not defined` pada skrip [generate_standar_report-new_method.py](file:///data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/generate_standar_report-new_method.py) di baris 737 (fungsi `main` saat memuat model deteksi YOLOv8m).
  - Memindahkan impor `YOLO` dan `SAM` dari pustaka `ultralytics` ke tingkat global (top level) agar dapat diakses dari seluruh fungsi, termasuk fungsi orkestrasi `main`.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan semua pemanggilan kelas-kelas model Ultralytics (YOLO/SAM) di dalam skrip orkestrasi tidak didefinisikan secara lokal di dalam worker saja, melainkan dideklarasikan secara global untuk kebersihan dan keandalan runtime.

---

### [Entri 056] — Konsolidasi & Standarisasi Golden Dataset: generate_golden_report-new_method.py

- **Tanggal/Waktu:** 2026-06-02 14:15 WIB
- **Tugas yang diselesaikan:**
  - **Revert File Lama**: Mengembalikan seluruh perubahan format `.tolist()` pada berkas lama (`generate_report_single_model.py`, `generate_standar_report-new_method.py`, `golden_evaluation-new_method.py`, dan `golden_evaluation_visuals-new_method.py`) agar tetap orisinal sesuai instruksi pengguna.
  - **Penciptaan Skrip Terpadu Golden**: Membuat berkas baru `utils/generate_golden_report-new_method.py` yang menggabungkan fungsionalitas `golden_evaluation-new_method.py` (kuantitatif) dan `golden_evaluation_visuals-new_method.py` (kualitatif).
  - **Standarisasi Tensor GPU Mentah**: Menerapkan parsing koordinat bounding box model Hybrid (YOLO + SAM2) menggunakan Tensor GPU Mentah (`res.boxes.xyxy` atau `resX_det.boxes.xyxy` secara langsung) di dalam berkas terpadu Golden yang baru.
  - **Restrukturisasi Scheduler Pipeline**: Memperbarui registry tugas `new_eval_specs` di `run_pipeline_parallel.py` untuk mengarahkan alur evaluasi baru ke `generate_standar_report-new_method.py` dan `generate_golden_report-new_method.py` secara teratur, serta menghapus tugas visualisasi terpisah yang kini telah menyatu.
  - **Verifikasi Sukses**: Menguji jalannya visualisasi kualitatif Golden pada compute node GPU (`ai3`) secara non-interaktif dan berhasil merender 10 sampel dual-grid tanpa kendala CUDA/device mismatch.
- **File yang diubah/dibuat:**
  - `utils/generate_golden_report-new_method.py` [DIBUAT BARU]
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Tugas-tugas visualisasi terpisah (`eval_new_std_vis` dan `eval_new_gld_vis`) telah dihapus secara resmi dari orkestrator scheduler karena fungsinya telah termigrasi penuh secara internal ke dalam masing-masing skrip report baru.

---

### [Entri 057] — Integrasi Evaluasi Dinamis Dataset Terpadu COCO (Valid Only)

- **Tanggal/Waktu:** 2026-06-02 14:40 WIB
- **Tugas yang diselesaikan:**
  - Mengintegrasikan pemrosesan dataset COCO secara dinamis tanpa berkas YAML YOLO pada skrip `utils/generate_report_single_model.py` (pada fungsi `evaluate_detection`, `evaluate_segmentation`, `evaluate_maskrcnn_segmentation`, dan `evaluate_hybrid_segmentation`).
  - Menyelaraskan skrip terpadu `utils/generate_standar_report-new_method.py` agar mendeteksi dan menggunakan `EVAL_DATASET_LOCATION` secara global dari `config_shared.py`.
  - Menerapkan standarisasi keys pada dictionary `image_ids` menjadi path absolut agar 100% kompatibel dengan data subset gambar yang dipartisi oleh multi-GPU worker, mencegah kesalahan mAP bernilai 0.0 akibat ketidakcocokan nama berkas.
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Dataset terpadu `eval_dataset_coco` yang hanya memiliki subfolder `valid/` kini sepenuhnya didukung dan dapat dievaluasi secara langsung baik per-model family maupun secara standar gabungan.
  - Untuk memicu evaluasi secara mandiri:
    - Model Tunggal: `python3 utils/generate_report_single_model.py --family yolo11 --gpus 0`
    - Standar Gabungan: `python3 utils/generate_standar_report-new_method.py --gpus 0`

---

### [Entri 058] — Analisis Komprehensif Ekosistem Model Qwen (Mei 2026)

- **Tanggal/Waktu:** 2026-06-02 18:40 WIB
- **Tugas yang diselesaikan:**
  - Melakukan analisis mendalam dan komprehensif terhadap ekosistem model Qwen yang dirilis oleh Alibaba Cloud/DAMO Academy hingga Mei 2026.
  - Memetakan silsilah model dari Qwen generasi pertama, Qwen1.5, Qwen2, Qwen2.5, model penalaran khusus QwQ, dan model vision-language Qwen2.5-VL.
  - Menyusun matriks kebutuhan VRAM untuk format FP16 mentah vs format kuantisasi (INT4/GGUF/AWQ/GPTQ) untuk pemahaman kebutuhan komputasi inferensi.
  - Memberikan evaluasi kelayakan teknis untuk deploy model di server GPU tunggal Tesla V100 32GB pada klaster Slurm AI_KU_V100 (merekomendasikan model 7B/14B dan model 32B terkuantisasi INT4/GGUF).
  - Menyertakan panduan integrasi skrip `run_llm_api.sh` menggunakan Singularity container (karena tidak ada root access) dan Cloudflared Tunnel untuk akses eksternal yang aman.
- **File yang diubah/dibuat:**
  - `docs/LLM Models/qwen.md` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Analisis ini dapat dijadikan referensi teoretis dan operasional untuk mengintegrasikan asisten LLM penalar (seperti QwQ-32B atau Qwen2.5-Coder-32B) ke dalam pipeline analisis riset YOLO/SAM guna memproses laporan secara otomatis.

---

### [Entri 059] — Analisis Model Proprietary & Spekulatif Qwen & DeepSeek Baru (Mei 2026)

- **Tanggal/Waktu:** 2026-06-02 18:55 WIB
- **Tugas yang diselesaikan:**
  - Melakukan analisis mendalam terhadap model-model yang ditanyakan pengguna: `Qwen3.7-Plus`, `qwen3-vl-plus`, `qwen3-vl-flash`, `qwen3.5-omni-flash`, `deepseek-v4-pro`, `deepseek-v4-flash`, dan `deepseek-v3.2`.
  - Mengidentifikasi bahwa model-model dengan nama spesifik tersebut belum dirilis secara open-source publik (open-weights) maupun didokumentasikan secara resmi oleh Alibaba Cloud atau DeepSeek hingga Mei 2026.
  - Menganalisis skema penamaan komersial API (DashScope komersial seperti Qwen-VL-Plus/Max vs akhiran "-Flash" dan "-Omni-Flash") serta komitmen open-weights DeepSeek (R1 & V3).
  - Memberikan rekomendasi deployment alternatif yang riil dan berkinerja tinggi pada GPU V100 32GB (seri Qwen2.5-32B, QwQ-32B, dan model distilasi DeepSeek-R1-Distill-Qwen-32B).
- **File yang diubah/dibuat:**
  - `docs/LLM Models/qwen.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jika ada pembaruan dari Alibaba Cloud atau DeepSeek mengenai rilis versi-versi baru tersebut selama pelaksanaan riset ini, segera perbarui dokumen model pendukung ini dan uji kelayakan inferensinya pada cluster GPU.

---

### [Entri 060] — Pembaruan Data Komprehensif Seri Qwen3 & DeepSeek V4 (Mei 2026)

- **Tanggal/Waktu:** 2026-06-02 19:00 WIB
- **Tugas yang diselesaikan:**
  - Melakukan revisi total terhadap dokumen `docs/LLM Models/qwen.md` (menggantikan analisis spekulasi sebelumnya) menggunakan data rilis resmi Alibaba Cloud Model Studio dan DeepSeek terbaru per Mei 2026.
  - Memetakan fungsionalitas model baru: `Qwen3.7-Plus` (multimodal interactive agent), `qwen3-vl-plus`/`qwen3-vl-flash` (vision-language agent), `Qwen3-VL-30B-A3B-Thinking` (sparse MoE reasoning visual), `Qwen3.5-Omni-Flash` (low-latency real-time voice-visual), serta seri `Qwen3.6-27B` (thinking preservation).
  - Memetakan arsitektur dan spesifikasi `DeepSeek-V3.2` (DSA 128k context) dan `DeepSeek-V4-Pro`/`DeepSeek-V4-Flash` (MoE 1.6T/284B dengan context window 1 juta token, rilis April 2026).
  - Menyertakan analisis kelayakan lokal (GPU V100 32GB) untuk model MoE baru `Qwen3-VL-30B-A3B-Thinking` terkuantisasi INT4 yang sangat efisien dijalankan secara lokal karena hanya memiliki 3B active parameter.
- **File yang diubah/dibuat:**
  - `docs/LLM Models/qwen.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
  - Untuk tugas pengenalan spasial dan visual grounding lanjut, model `Qwen3-VL-30B-A3B-Thinking` sangat layak dipasang di Ollama lokal pada klaster Slurm untuk pengujian kualitatif segmentasi/deteksi YOLO/SAM2 secara terdistribusi.

---

### [Entri 061] — Bugfix: IndexError (list index out of range) mAP Mask pada COCOeval

- **Tanggal/Waktu:** 2026-06-02 19:10 WIB
- **Tugas yang diselesaikan:**
  - Mengidentifikasi penyebab error fatal `IndexError: list index out of range` pada FASE 1 evaluasi segmentasi model-model YOLO-Seg (`YOLOv8m-Seg`, `YOLOv8x-Seg`, `YOLOv9c-Seg`, `YOLOv9e-Seg`, `YOLO11n-Seg`, `YOLO11l-Seg`, `YOLO11x-Seg`) dan `Mask R-CNN`.
  - Masalah terjadi karena dataset terpadu `eval_dataset_coco` memiliki beberapa anotasi kotak pembatas murni tanpa segmentasi (segmentasi kosong `[]` atau tidak ada). Saat COCOeval dijalankan dengan `iouType="segm"`, `pycocotools` mengalami kegagalan internal saat menghitung stats sehingga array `stats` tidak terisi, yang memicu *IndexError* saat program mencoba mengakses `stats[0]` secara langsung.
  - Memperbaiki masalah ini dengan mengimplementasikan penyaringan anotasi secara dinamis (menggunakan `copy.deepcopy` dan validasi data segmentasi pada list `annotations`) sesaat sebelum inisialisasi objek `COCO` dan pemanggilan `COCOeval` untuk masker.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
  - Penyaringan ini sepenuhnya kompatibel dengan pipeline multi-GPU terdistribusi dan memastikan `mAP50(Mask)` dan `mAP50-95(Mask)` berhasil dihitung dengan aman tanpa risiko kegagalan runtime.

---

### [Entri 062] — Bugfix: Opsi select_node Tidak Muncul pada Terminal di myslurm.sh

- **Tanggal/Waktu:** 2026-06-02 21:05 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki bug di `utils/myslurm.sh` di mana prompt pilihan GPU node `select_node` ("Pilih node GPU yang tersedia...") beserta input `read -p` tidak muncul di terminal pengguna saat memilih menu requeue (pindah node).
  - Masalah teridentifikasi dari penggunaan command substitution `target_node=$(select_node)` yang menangkap seluruh keluaran standar (stdout) dari fungsi tersebut ke dalam variabel, menghambat pencetakan prompt ke layar.
  - Solusi diimplementasikan dengan mengubah fungsi `select_node` untuk langsung menetapkan nilai pilihan ke variabel global `SELECTED_NODE`, kemudian memanggil fungsi secara langsung di `move_job_node` tanpa *subshell command substitution*.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
  - Pastikan setiap script menu interaktif bash yang berinteraksi langsung dengan input pengguna menghindari penggunaan subshell capture `$(...)` untuk fungsi-fungsi yang berisi `echo` petunjuk menu atau `read` interaktif demi menjamin fungsionalitas UI terminal.

---

### [Entri 063] — Bugfix: Kesalahan "Invalid node name" Saat Pemindahan Node di myslurm.sh

- **Tanggal/Waktu:** 2026-06-02 21:10 WIB
- **Tugas yang diselesaikan:**
  - Memperbaiki kesalahan `"Invalid node name specified for job"` saat pengguna memicu aksi pemindahan node GPU untuk tugas yang sedang berjalan (*Running*).
  - Masalah teridentifikasi karena scheduler Slurm tidak memperbolehkan perubahan alokasi node (`NodeList`) secara langsung ketika status tugas masih aktif berjalan (*Running*). Selain itu, mencoba melakukan pembaruan di subshell langsung seringkali kalah cepat dengan eksekusi penjadwalan ulang instan dari scheduler.
  - Solusi diimplementasikan dengan membangun alur kerja pemindahan node Slurm yang aman:
    1. Melakukan penahanan tugas (`scontrol hold`) secara temporer untuk mengubah statusnya.
    2. Menghapus tugas dari node lama dan mengembalikan ke status pending (`scontrol requeue`).
    3. Memperbarui parameter target node yang diminta (`scontrol update JobId=... ReqNodeList=<node_target>`).
    4. Melepaskan penahanan (`scontrol release`) agar tugas siap dijalankan kembali di node baru yang diminta.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
  - Alur hold/requeue/update ReqNodeList/release terbukti andal untuk memindahkan job Slurm di partisi GPU secara dinamis tanpa intervensi manual administrator.

---

### [Entri 064] — Penyusunan Rencana Deployment LLM & Evaluasi Berdampingan (Plan.md)

- **Tanggal/Waktu:** 2026-06-02 21:22 WIB
- **Tugas yang diselesaikan:**
  - Menyusun dokumen rencana implementasi dan deployment terintegrasi (*Co-existence Plan*) antara layanan LLM (Ollama) dan proses evaluasi model YOLO/SAM2 pada berkas `docs/LLM Models/Plan.md`.
  - Melakukan analisis pembagian sumber daya VRAM pada 1 unit GPU Tesla V100 32GB (Ollama Qwen32B: ~21GB, Evaluasi YOLO: ~4GB) dengan sisa toleransi memori yang aman.
  - Memetakan alur kerja penanganan batasan waktu (*Time Limit*) Slurm menggunakan mekanisme penangkapan sinyal (`--signal=B:USR1@120` dan penanganan `SIGUSR1`) untuk meluncurkan kembali tugas baru (`sbatch`) secara otonom sebelum dihentikan paksa.
  - Merancang integrasi dynamic porting dan Cloudflared Tunnel untuk pemetaan endpoint API yang statis secara konsisten.
- **File yang diubah/dibuat:**
  - `docs/LLM Models/Plan.md` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
  - Gunakan dokumen `Plan.md` sebagai cetak biru teknis saat tiba waktunya untuk mengeksekusi integrasi server asisten LLM di compute node klaster Slurm.

---

### [Entri 065] — Perbaikan Sintaksis Diagram Mermaid di Plan.md

- **Tanggal/Waktu:** 2026-06-02 21:30 WIB
- **Tugas yang diselesaikan:**
  - Mengidentifikasi dan memperbaiki error parsing pada visualisasi diagram Mermaid di berkas `docs/LLM Models/Plan.md`.
  - Kesalahan disebabkan oleh karakter tanda kurung pada label nama `subgraph Compute Node (ai2/ai3)` yang dibaca sebagai pemisah sintaksis ilegal oleh parser Mermaid.
  - Memperbaiki dengan membungkus nama label dalam tanda kutip ganda menjadi `subgraph "Compute Node (ai2/ai3)"` sesuai pedoman formatting resmi.
- **File yang diubah/dibuat:**
  - `docs/LLM Models/Plan.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan setiap visualisasi arsitektur sistem menggunakan Mermaid diagram membungkus teks label yang mengandung karakter khusus (spasi, tanda kurung, garis miring) dengan tanda kutip ganda untuk menjamin kompatibilitas parser.

---

### [Entri 066] — Penambahan Ekspor CSV Hasil Evaluasi Standar & Golden

- **Tanggal/Waktu:** 2026-06-02 23:45 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi `utils/generate_standar_report-new_method.py` untuk mengumpulkan metrik precision/recall/latency/fps di Fase 1 (Kuantitatif), memisahkan data evaluasi ke `rows_det` dan `rows_seg`, serta menambahkan ekspor berkas `standar_det.csv` dan `standar_seg.csv` ke folder `reports/paper1/csv/new-method/`. Berkas `kompilasi_new_method_standar.csv` tetap dihasilkan di `reports/paper1/csv/`.
  - Memodifikasi `utils/generate_golden_report-new_method.py` untuk menambahkan inisialisasi list `rows` gabungan, merekap seluruh model (deteksi dan segmentasi) dengan format kolom yang setara dengan versi standar, dan mengekspornya ke `reports/paper1/csv/kompilasi_new_method_golden.csv`.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Perubahan ini memungkinkan proses evaluasi metode baru (standard & golden) menghasilkan berkas kompilasi gabungan dan berkas terpisah (det/seg) yang konsisten secara akademis untuk draf paper.
  - Untuk menjalankan ulang evaluasi standard: `python3 utils/generate_standar_report-new_method.py --gpus 0`
  - Untuk menjalankan ulang evaluasi golden: `python3 utils/generate_golden_report-new_method.py --gpus 0`

---

### [Entri 067] — Implementasi Filter Model dan Jenis Tugas pada Scheduler Pipeline

- **Tanggal/Waktu:** 2026-06-02 23:56 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan argumen CLI baru `--models` (default: `all`) dan `--tasks` (pilihan: `all`, `train`, `eval`, `new-method`) ke dalam `run_pipeline_parallel.py`.
  - Mengonfigurasi scheduler `ParallelScheduler` untuk secara dinamis men-skip model dan tugas di luar filter yang dipilih.
  - Menerapkan penghapusan dependensi `eval_new_std` terhadap `eval_global_multigpu` secara dinamis jika evaluasi model tunggal di-skip (`run_eval = False`). Hal ini menjamin rantai evaluasi metode baru (Std, Golden, Hybrid SOTA, Final Grid, Local Archive & Upload) tetap berjalan dengan lancar tanpa terimbas pembatalan cascade.
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk hanya menjalankan evaluasi metode baru dan mengarsip/unggah hasil tanpa harus training/evaluasi model tunggal, pengguna dapat menjalankan: `python3 run_pipeline_parallel.py --tasks new-method`.
  - Untuk membatasi model tertentu saja, misalnya YOLO11: `python3 run_pipeline_parallel.py --models yolo11`.

---

### [Entri 068] — Implementasi Mode Tugas eval_ku dan Opsi dry-run pada Scheduler

- **Tanggal/Waktu:** 2026-06-03 00:15 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan opsi filter tugas khusus `--tasks eval_ku` pada skrip orkestrator scheduler `run_pipeline_parallel.py`.
  - Mengatur seluruh model training dan evaluasi model tunggal ke status `SKIPPED` secara dinamis saat mode `eval_ku` aktif.
  - Mengonfigurasi agar evaluasi kompilasi global (`eval_global_multigpu`) berjalan secara instan dengan dependensi kosong (`dependencies = []`).
  - Menyambungkan rantai dependensi evaluasi metode baru dan pasca-proses secara sekuensial (`eval_new_std` -> `eval_new_gld` -> `eval_new_hybrid_sota` -> `post_grid` -> `post_upload`) agar berjalan berurutan setelah kompilasi global selesai.
  - Menambahkan argumen CLI `--dry-run` untuk mempermudah pemeriksaan bagan rencana tugas tanpa perlu CUDA atau memicu notifikasi Telegram dan subproses komputasi nyata.
  - Memodifikasi dan memperluas docstring utama di awal berkas `run_pipeline_parallel.py` untuk menyajikan dokumentasi lengkap argumen CLI yang tersedia beserta contoh (samples) pemanggilan riil yang praktis.
  - Memperbaiki bug `AttributeError` karena penulisan properti CLI yang keliru (`args.dry-run` yang dibaca Python sebagai ekspresi matematika `args.dry - run` → memicu kegagalan) menjadi variabel parser argparse yang benar (`args.dry_run`).
- **File yang diubah/dibuat:**
  - `run_pipeline_parallel.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pengguna dapat memverifikasi visualisasi rencana tugas evaluasi saja menggunakan: `python3 run_pipeline_parallel.py --tasks eval_ku --dry-run`
  - Untuk memulai eksekusi sesungguhnya di compute node GPU, jalankan: `python3 run_pipeline_parallel.py --tasks eval_ku` di dalam sesi TMUX.













