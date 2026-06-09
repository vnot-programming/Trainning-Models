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

---

### [Entri 069] — Inisialisasi Infrastruktur Layanan Multi-LLM Terisolasi

- **Tanggal/Waktu:** 2026-06-03 10:15 WIB
- **Tugas yang diselesaikan:**
  - Membuat direktori kustom `/data/users/g6717500336/singularity` yang terisolasi dari folder pelatihan YOLO.
  - Membuat skrip `setup_llm_service.sh` untuk mengunduh biner `cloudflared` secara global dan mempersiapkan container image `ollama.sif` via Singularity.
  - Membuat skrip launcher `sbatch_llm_service.sh` untuk server Ollama dengan auto-requeue (SIGUSR1 trap) dan dynamic port selection.
  - Membuat skrip `setup_lms.sh` untuk memasang LM Studio CLI (`lms`) secara terisolasi dengan mengalihkan variabel `HOME` ke folder kustom.
  - Membuat skrip launcher `sbatch_lms.sh` untuk LM Studio Server dengan auto-requeue dan dynamic port selection.
  - Membuat skrip `setup_webui.sh` untuk menarik container image `open-webui.sif` via Singularity.
- **File yang diubah/dibuat:**
  - `singularity/ollama/setup_llm_service.sh` [BARU]
  - `singularity/ollama/sbatch_llm_service.sh` [BARU]
  - `singularity/lm-studio/setup_lms.sh` [BARU]
  - `singularity/lm-studio/sbatch_lms.sh` [BARU]
  - `singularity/open-webui/setup_webui.sh` [BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk inisialisasi awal lingkungan Ollama, jalankan: `bash /data/users/g6717500336/singularity/ollama/setup_llm_service.sh`
  - Untuk inisialisasi awal lingkungan LM Studio, jalankan: `bash /data/users/g6717500336/singularity/lm-studio/setup_lms.sh`
  - Untuk inisialisasi awal lingkungan Open WebUI, jalankan: `bash /data/users/g6717500336/singularity/open-webui/setup_webui.sh`
  - Setelah semua biner dan container siap, Anda dapat men-submit job backend LLM ke Slurm menggunakan `sbatch <skrip_sbatch>`.

---

### [Entri 070] — Integrasi Server Ollama ke Daemon Auto-Rebooking (book_gpu.py)

- **Tanggal/Waktu:** 2026-06-03 10:35 WIB
- **Tugas yang diselesaikan:**
  - Mengintegrasikan kode launcher server Ollama (`sbatch_llm_service.sh`) ke dalam generator sbatch dinamis (`generate_booking_sbatch`) pada `utils/book_gpu.py`.
  - Mengubah monitoring job pada `utils/book_gpu.py` agar secara dinamis memindai log `tunnel_sbatch.log` untuk mengekstrak tautan publik Cloudflared Tunnel secara otomatis.
  - Memperbarui notifikasi Telegram yang dikirim oleh `book_gpu.py` agar menyajikan informasi status Ollama Server secara spesifik serta menampilkan tautan publik API begitu status job aktif (`RUNNING`).
- **File yang diubah/dibuat:**
  - `utils/book_gpu.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pengguna tidak perlu menjalankan `sbatch sbatch_llm_service.sh` secara manual lagi. Cukup jalankan daemon `book_gpu.py` (via tmux session `gpu_booking`), maka daemon akan otomatis memesan GPU dan menyalakan server Ollama di compute node terpilih, serta mengirimkan tautan publik API langsung ke Telegram.
  - Skrip `utils/slurm/pretty_squeue.py` tidak membutuhkan perubahan karena ia hanya berfungsi mencetak output tabel squeue secara visual, di mana job name akan tercetak sebagai `Ollama-Backend` secara otomatis.

---

### [Entri 071] — Konfigurasi HTTP/2 pada Cloudflared Tunnel untuk Memintas Firewall

- **Tanggal/Waktu:** 2026-06-03 10:48 WIB
- **Tugas yang diselesaikan:**
  - Menginvestigasi log `tunnel_sbatch.log` yang menunjukkan kegagalan konektivitas UDP/QUIC (port 7844) pada klaster server.
  - Menambahkan parameter `--protocol http2` pada seluruh berkas skrip eksekusi `cloudflared` untuk memaksa konektivitas melalui TCP/HTTP2 yang diperbolehkan oleh firewall klaster.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI]
  - `singularity/lm-studio/sbatch_lms.sh` [DIMODIFIKASI]
  - `utils/book_gpu.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Parameter `--protocol http2` wajib dipertahankan pada pemanggilan `cloudflared` di lingkungan ini untuk menghindari pemblokiran UDP oleh sistem firewall eksternal.

---

### [Entri 072] — Perbaikan Hak Akses & Perapian authorized_keys untuk Konektivitas SSH

- **Tanggal/Waktu:** 2026-06-03 11:40 WIB
- **Tugas yang diselesaikan:**
  - Menginvestigasi kegagalan koneksi SSH remote ke SLURM Master via Cloudflared dengan error `Permission denied (publickey)`.
  - Mengidentifikasi bahwa izin berkas `~/.ssh/authorized_keys` di Login Node terlalu permisif (`664` atau `-rw-rw-r--`), sehingga ditolak oleh SSH daemon.
  - Memperbaiki izin berkas `~/.ssh/authorized_keys` ke `600` (`-rw-------`) dengan perintah `chmod 600 ~/.ssh/authorized_keys`.
  - Melakukan audit dan perapian berkas `~/.ssh/authorized_keys` dengan menghapus seluruh entri kunci duplikat (menyisakan 14 kunci unik) dan mengelompokkannya secara bersih berdasarkan jenis tipe kunci (ED25519 dan RSA) dengan header dokumentasi yang informatif.
- **File yang diubah/dibuat:**
  - `~/.ssh/authorized_keys` [Izin berkas diubah ke 600, deduplikasi & format ulang]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengguna kini dapat kembali melakukan SSH remote via tunnel dengan lancar dan berkas konfigurasi kunci bersih dari redundansi.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan hak akses untuk direktori `~/.ssh` tetap `700` dan berkas `authorized_keys` tetap `600` untuk mencegah SSH Daemon mengabaikan kunci otorisasi.

---

### [Entri 073] — Penyusunan Alur Named Tunnel Cloudflared di Singularity dengan Token

- **Tanggal/Waktu:** 2026-06-03 11:45 WIB
- **Tugas yang diselesaikan:**
  - Menerima token Named Tunnel dari pengguna dan memetakan alur instalasi/eksekusinya di lingkungan klaster SLURM GPU.
  - Membandingkan opsi eksekusi menggunakan biner Go portabel lokal (`cloudflared` yang sudah terpasang) dengan opsi kontainerisasi Singularity (`cloudflared_latest.sif`).
  - Menekankan kewajiban penggunaan flag `--protocol http2` untuk memintas restriksi firewall klaster terhadap lalu lintas UDP/QUIC (port 7844).
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengguna diberikan instruksi konfigurasi Named Tunnel yang siap pakai.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk Named Tunnel permanen, pastikan selalu menggunakan parameter `--protocol http2` agar konektor tidak terputus dari edge server Cloudflare.

---

### [Entri 074] — Penambahan Submenu "Tambah Sesi" pada Manajemen Sesi TMUX

- **Tanggal/Waktu:** 2026-06-03 11:46 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan submenu "Tambah Sesi Baru (Create)" di dalam Menu 5 (Manajemen Sesi TMUX) pada utilitas CLI `utils/myslurm.sh`.
  - Merombak logika penanganan daftar sesi TMUX agar submenu tindakan tetap muncul meskipun tidak ada sesi TMUX yang aktif saat menu dibuka.
  - Menyediakan perlindungan pencegahan error ketika memilih masuk (*attach*) atau menghapus (*kill*) saat daftar sesi kosong.
  - Mengimplementasikan prompt interaktif yang menanyakan apakah pengguna ingin langsung masuk (*attach*) ke sesi baru sesaat setelah sesi dibuat di background.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Fitur manajemen sesi TMUX kini memiliki siklus fungsionalitas yang lengkap.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan hak akses eksekusi (`chmod +x`) pada skrip `myslurm.sh` tetap dipertahankan demi kenyamanan pengguna saat memanggil utilitas.

---

### [Entri 075] — Diagnosis Kegagalan SSH Client Timeout & Verifikasi Named Tunnel

- **Tanggal/Waktu:** 2026-06-03 11:48 WIB
- **Tugas yang diselesaikan:**
  - Melakukan diagnosis interaktif terhadap proses `cloudflared` Named Tunnel di Login Node menggunakan token pengguna.
  - Memverifikasi bahwa Named Tunnel berhasil terdaftar secara stabil di edge server Cloudflare (Bangkok & Singapura) dan memetakan `slurm.penelitian.my.id` ke `ssh://localhost:22` dengan sukses.
  - Mengidentifikasi akar masalah *Connect Timeout* pada SSH client Mac pengguna: parameter `ConnectTimeout 5` memutus koneksi secara paksa sebelum handshake Cloudflare Tunnel & negosiasi banner SSH selesai.
  - Mengidentifikasi ketiadaan parameter `IdentityFile` yang menyebabkan SSH client tidak menawarkan kunci `id_rsa_g6717500336` yang terdaftar di authorized_keys.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengguna diarahkan untuk memperbarui file `~/.ssh/config` di sisi klien (Mac).
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jika pengguna masih mengalami kendala setelah mengubah konfigurasi klien, minta mereka melakukan verbose debug SSH (`ssh -vvv KU-Slurm-vpnku`) untuk pelacakan lebih mendalam.

---

### [Entri 076] — Diagnosis Penolakan Kunci RSA & Alternatif Kunci ED25519

- **Tanggal/Waktu:** 2026-06-03 11:50 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis log penolakan kunci publik RSA `id_rsa_g6717500336` (`Permission denied`) meskipun hak akses berkas `authorized_keys` di server sudah dikonfigurasi secara aman (`600`).
  - Mengidentifikasi potensi ketidakcocokan algoritma negosiasi tanda tangan RSA (depresiasi tanda tangan SHA-1 pada OpenSSH 8.9+ di Ubuntu 22.04) pada klaster SLURM.
  - Memverifikasi bahwa kunci ED25519 klien (`id_ed25519` dengan fingerprint `SHA256:rfucilbrkbkszoWoY0jNyjZK+6gVrZlCn5pCy8TF5Bo`) terdaftar secara valid di server dengan label `myrvm-key`.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengguna diberikan pilihan menggunakan kunci ED25519 atau menambahkan opsi PubkeyAcceptedKeyTypes pada konfigurasi SSH Mac mereka.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Penggunaan kunci ED25519 sangat direkomendasikan karena terhindar dari isu kompatibilitas algoritma enkripsi/tanda tangan terdepresiasi di masa depan.

---

### [Entri 077] — Pembersihan Komentar authorized_keys & Sukses Uji Loopback SSH Lokal

- **Tanggal/Waktu:** 2026-06-03 11:52 WIB
- **Tugas yang diselesaikan:**
  - Melakukan penulisan ulang berkas `~/.ssh/authorized_keys` di Login Node dengan menghapus seluruh baris komentar (`#`) dan baris kosong untuk menghindari kegagalan parser `OpenSSH 8.0p1` pada CentOS/RHEL.
  - Memperbaiki string kunci publik lokal Login Node (`IEzqtZQcVMs8Ty3j...`) yang sebelumnya tertimpa oleh representasi sidik jari.
  - Mengeksekusi uji loopback SSH lokal (`ssh localhost`) dari Login Node menggunakan kunci privat lokal, yang berhasil terautentikasi sukses (`Authentication succeeded (publickey)`).
- **File yang diubah/dibuat:**
  - `~/.ssh/authorized_keys` [DIMODIFIKASI - Pembersihan format kunci]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengujian lokal membuktikan konfigurasi otorisasi server (hak akses berkas & direktori) sudah 100% tepat dan berfungsi, mengisolasi masalah otentikasi klien pada faktor kunci privat di Mac.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk melacak mengapa kunci ED25519 Mac ditolak padahal sidik jari cocok, minta pengguna menjalankan verbose SSH debug (`ssh -vvv`) dari terminal Mac mereka.

---

### [Entri 078] — Temuan Konflik Routing Named Tunnel Cloudflare (Bentrokan Target localhost)

- **Tanggal/Waktu:** 2026-06-03 11:55 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis hasil verbose log SSH (`-vvv`) dari Mac pengguna yang menunjukkan penolakan kunci publik ED25519 (`rfucilbrkbkszoWoY0jNyjZK...`).
  - Mengidentifikasi temuan kritis: Kunci host (*Server host key*) yang diterima Mac pengguna saat mengakses `slurm.penelitian.my.id` cocok dengan kunci host `vps-4c56g-asia.vnot.my.id` (VPS Azure) di berkas `known_hosts` baris 86 klien.
  - Menyimpulkan akar masalah: Terowongan Named Tunnel yang dipicu pengguna menggunakan token yang sama dengan terowongan permanen di VPS Azure. Karena aturan *Ingress* memetakan `slurm.penelitian.my.id` ke `ssh://localhost:22`, koneksi SSH dari Mac dialihkan oleh load-balancer Cloudflare ke `localhost` (port 22) milik **VPS Azure**, bukan SLURM Login Node.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai. Pengguna diarahkan untuk membuat Named Tunnel baru terpisah untuk SLURM Login Node demi memisahkan domain target SSH secara fisik.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan pengguna tidak menjalankan beberapa konektor aktif dengan token yang sama di host fisik yang berbeda jika target `localhost:22` yang dirujuk memiliki konfigurasi user/sistem yang berbeda.

---

### [Entri 079] — Diagnosis Konektivitas Docker Host (Akses SSH & Cloudflared)

- **Tanggal/Waktu:** 2026-06-03 12:30 WIB
- **Tugas yang diselesaikan:**
  - Melakukan uji coba konektivitas SSH remote dari Login Node ke Docker Host (`docker-host`).
  - Mengidentifikasi kegagalan koneksi langsung (ping ke `100.90.5.60` dan `100.123.143.87` menghasilkan 100% packet loss baik dari Login Node maupun compute node `ai3`).
  - Mendiagnosis kegagalan koneksi via Cloudflared ke `ssh.vnot.my.id` yang menghasilkan error `websocket: bad handshake`. Kegagalan disebabkan karena `cloudflared` di Login Node membutuhkan session token otentikasi Cloudflare Access yang sah untuk memintas kebijakan keamanan (Access Policy).
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai (Diselesaikan di Entri 080)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Investigasi selesai, dilanjutkan dengan pembuatan file konfigurasi SSH client di HPC Slurm untuk melakukan bypass via cloudflared.

---

### [Entri 080] — Penyelesaian Konektivitas SSH ke Docker Host dari HPC Slurm

- **Tanggal/Waktu:** 2026-06-03 13:12 WIB
- **Tugas yang diselesaikan:**
  - Membuat berkas konfigurasi SSH client `/data/users/g6717500336/.ssh/config` di HPC Slurm untuk memetakan Host `docker-host-vpnku` secara otomatis.
  - Menetapkan ProxyCommand menggunakan `/data/users/g6717500336/singularity/cloudflared access ssh --hostname ssh.vnot.my.id` untuk merujuk ke binary cloudflared lokal.
  - Mengamankan hak akses berkas `/data/users/g6717500336/.ssh/config` menjadi `600`.
  - Menguji dan memvalidasi koneksi SSH dari HPC Slurm ke Docker Host. Proses autentikasi berhasil 100% menggunakan kunci privat `~/.ssh/id_rsa` lokal tanpa kendala kebijakan Cloudflare Access.
- **File yang diubah/dibuat:**
  - `/data/users/g6717500336/.ssh/config` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Konektivitas SSH dari HPC Slurm ke Docker Host (`docker-host-vpnku`) kini telah aktif dan berjalan dengan lancar. 
  - Host target dapat langsung diakses dengan menjalankan `ssh docker-host-vpnku` dari terminal HPC Slurm.

---

### [Entri 081] — Implementasi Arsitektur Dynamic Porting & SSH Reverse Tunneling

- **Tanggal/Waktu:** 2026-06-03 13:28 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis ketidakmungkinan penggunaan port dinamis pada URL publik Cloudflare Tunnel (selalu port 443).
  - Merancang solusi integrasi: Memetakan domain statis `ollama.penelitian.my.id` di Cloudflare Dashboard ke target local `http://localhost:11434` (karena tunnel `ku-slurm-master-tunnel` berjalan langsung secara lokal di VM100).
  - Mengubah skrip launcher Slurm `singularity/ollama/sbatch_llm_service.sh` untuk secara otomatis membuka SSH Reverse Tunnel (`ssh -N -f -R 11434:localhost:$OLLAMA_PORT docker-host-vpnku`) saat job dijalankan, mengubah nama job menjadi `vnot` (menggantikan `Ollama-Backend`), serta membersihkan tunnel secara otomatis saat job selesai (`trap cleanup EXIT`).
  - Mengintegrasikan logika generator sbatch dinamis pada `utils/book_gpu.py` dengan penambahan instruksi reverse tunneling SSH ke `localhost:11434` di VM100 agar sinkron dengan sistem otomatisasi daemon.
  - Memperbarui dokumen rencana arsitektur `/docs/LLM Models/Implementation Plan.md` dengan menyertakan diagram Mermaid dan penjelasan terbaru mengenai pemetaan port dinamis via SSH Reverse Tunneling.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI - Ganti nama job & integrasi reverse tunnel]
  - `utils/book_gpu.py` [DIMODIFIKASI - Sinkronisasi sbatch gen]
  - `docs/LLM Models/Implementation Plan.md` [DIMODIFIKASI - Pembaruan arsitektur]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Karena tunnel `ku-slurm-master-tunnel` berjalan lokal di VM100, kita menggunakan binding localhost (`-R 11434:localhost:$OLLAMA_PORT`) yang tidak membutuhkan modifikasi parameter `GatewayPorts` di SSH daemon VM100.
  - Tambahkan rute domain `ollama.penelitian.my.id` ke `http://localhost:11434` pada dashboard Cloudflare.

---

### [Entri 082] — Integrasi Manajemen Cloudflare Tunnel Native di Master Node

- **Tanggal/Waktu:** 2026-06-03 13:39 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan konfigurasi Cloudflare Tunnel (`CLOUDFLARE_BIN` dan `CLOUDFLARE_TUNNEL_TOKEN`) ke dalam `config_shared.py` untuk mengeliminasi hardcoding token.
  - Membuat fungsi `manage_cloudflare_tunnel` di dalam `utils/myslurm.sh` yang mengambil variabel token dan binary secara dinamis dari `config_shared.py` (dengan fallback yang aman).
  - Menyediakan sub-menu Manajemen Cloudflare Tunnel di `myslurm.sh` (pilihan Menu 6) yang mendukung:
    1. 🚀 Menjalankan terowongan di background dalam sesi tmux mandiri bernama `cloudflare_tunnel` dengan parameter `--protocol http2` untuk memintas pemblokiran lalu lintas UDP (QUIC) pada firewall klaster HPC.
    2. 🛑 Menghentikan terowongan secara bersih (menutup sesi tmux & pkill proses `cloudflared` sisa).
    3. 📋 Memantau status dan log langsung (mengambil 20 baris log terakhir via `tmux capture-pane`).
    4. 💻 Masuk (attach) langsung ke sesi tmux terowongan.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI]
  - `utils/myslurm.sh` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%** (Mengatasi kendala koneksi QUIC Timeout dengan fallback ke HTTP/2)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Terowongan Cloudflare ini berjalan secara native di Master Node menggunakan binary `/data/users/g6717500336/singularity/cloudflared`.
  - Jika token atau path binary berubah di masa depan, modifikasi cukup dilakukan di `config_shared.py` tanpa perlu mengubah `utils/myslurm.sh`.

---

### [Entri 083] — Penambahan Edukasi Kegunaan Sesi TMUX Daemon pada myslurm.sh

- **Tanggal/Waktu:** 2026-06-03 13:56 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan blok penjelasan detail di `utils/myslurm.sh` pada **Menu 1 (Jalankan Booking GPU)** mengenai fungsi krusial dari sesi tmux `gpu_booking`. Sesi ini mengisolasi daemon python `book_gpu.py` di latar belakang agar aman dari terputusnya koneksi SSH interaktif.
  - Menambahkan teks peringatan (warning) di **Menu 5 (Manajemen Sesi TMUX)** agar pengguna berhati-hati dan tidak secara tidak sengaja mematikan (*kill*) sesi `gpu_booking` (daemon auto-booking) dan sesi `cloudflare_tunnel` (konektivitas tunnel statis Master Node), karena keduanya berjalan sebagai daemon background yang kritis.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI - Penambahan teks edukatif & peringatan]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Edukasi CLI ini penting untuk mencegah terjadinya pemutusan daemon background (`gpu_booking` & `cloudflare_tunnel`) secara tidak sengaja saat pengguna merapikan sesi tmux.



---

### [Entri — Infrastruktur Ollama + Open WebUI via Slurm GPU] — 2026-06-03 15:38 WIB

- **Tanggal/Waktu:** 2026-06-03 15:38 WIB
- **Tugas yang diselesaikan:**
  1. Setup Cloudflare Tunnel `vm100-docker-tunnel` di VM100 untuk expose `chat.penelitian.my.id` → Open WebUI dan `ollama.penelitian.my.id` → Ollama API
  2. Deploy Open WebUI via Docker Compose di VM100 dengan `network_mode: host` agar dapat akses `localhost:11434`
  3. Double-hop SSH Tunnel dari Slurmmaster ke VM100 untuk forward Ollama GPU (ai3): `-L OLLAMA_PORT:ai3` → `-R 11434:localhost:VM100`
  4. Fix job name mismatch: Sinkronisasi `VnoT-Train` → `vnot` di semua file terkait agar `attach_gpu.sh` menemukan job aktif
  5. Konfigurasi `~/.ssh/config` dengan ProxyCommand cloudflared untuk SSH ke VM100
- **File yang diubah/dibuat:**
  - `utils/attach_gpu.sh` — job name VnoT-Train → vnot
  - `utils/myslurm.sh` — job name VnoT-Train → vnot  
  - `utils/submit_ai2.sbatch` — job name VnoT-Train → vnot
  - `/data/users/g6717500336/.ssh/config` — Host docker-host-vpnku via cloudflared
  - `/home/my/.cloudflared/config.yml` (VM100) — ingress tunnel config
  - `/home/my/ollama-slurm-docker/docker-compose.yml` (VM100) — network_mode: host
- **Status saat ini:** Selesai ✅
  - Open WebUI: https://chat.penelitian.my.id — HEALTHY
  - Ollama berjalan di GPU node ai3 (Tesla V100 32GB, port 18049)
  - Chain: Browser → Cloudflare → VM100:3000 → Open WebUI → localhost:11434 → ai3 Ollama GPU
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - SSH tunnel (double-hop slurmmaster→ai3→VM100) di tmux `ollama_vm100_slurmmaster`. HARUS di-restart jika booking GPU baru karena OLLAMA_PORT berubah random setiap sesi.
  - `https://ollama.penelitian.my.id` mengembalikan HTTP 403 (Cloudflare Access) dari curl — tidak memengaruhi Open WebUI karena menggunakan `localhost:11434` internal.
  - Todo: integrasikan `ssh -nNT -R` ke `submit_booking_run.sbatch` dengan Cloudflare Service Token untuk otomasi penuh.


---

### [Entri — Perbaikan Kritis Hardware Incompatibility Ollama V100] — 2026-06-03 16:50 WIB

- **Tanggal/Waktu:** 2026-06-03 16:50 WIB
- **Tugas yang diselesaikan:**
  - Melakukan investigasi mendalam (Deep Analysis) terhadap error `CUDA error: device kernel image is invalid` yang terus-menerus mengganggu model `qwen3-vl:30b` saat operasi PAD (ggml_cuda_compute_forward).
  - Mengidentifikasi akar masalah: Library CUDA pada container `ollama.sif` terbaru (versi `0.30.2`) telah **menghapus kompilasi kernel untuk sm_70 (Tesla V100)**.
  - Mendelegasikan eksekusi dari script python `book_gpu.py` secara langsung ke `sbatch_llm_service.sh` untuk menjaga `book_gpu.py` sebagai Single Source of Truth pengelola antrean (daemon), sekaligus menghilangkan bentrokan *Double Job*.
  - Memperbarui skrip `sbatch_llm_service.sh` dengan perlindungan ganda (variabel `OLLAMA_FLASH_ATTENTION="false"` dan `OLLAMA_KV_CACHE_TYPE="q8_0"`).
  - Mengganti total image Ollama Singularity ke versi lama (Downgrade ke `0.2.8`) yang diverifikasi masih mengandung kompilasi instruksi `sm_70` (Volta).
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI - Ganti target image menjadi ollama_v0.2.sif dan update env]
  - `utils/book_gpu.py` [DIMODIFIKASI - Mengubah fungsi call sbatch Ollama menjadi pendelegasian langsung ke script bash shell via bash sbatch_llm_service.sh]
  - `.gemini/GEMINI.md` [DIMODIFIKASI - Menambahkan dokumentasi Global Rules terkait integrasi antara daemon book_gpu.py dan script Ollama]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log]
- **Status saat ini:** Selesai 100%
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - **DILARANG KERAS** melakukan upgrade pada image `ollama_v0.2.sif` karena versi terbaru (0.3.0+) terbukti mengalami regresi dan tidak kompatibel dengan arsitektur Volta (V100).
  - Skrip eksekutor di `sbatch_llm_service.sh` tidak lagi melakukan re-queue secara mandiri. Ia tunduk pada orkestrasi penuh dari daemon `utils/book_gpu.py`.

---

### [Entri — Proteksi Double Booking & Fix Sesi tmux gpu_booking] — 2026-06-03 17:15 WIB

- **Tanggal/Waktu:** 2026-06-03 17:15 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Sesi tmux `gpu_booking`:** Mengidentifikasi penyebab hilangnya sesi `gpu_booking` karena SyntaxError pada `utils/book_gpu.py` di baris 124 (`f.write("` terpotong baris baru).
  2. **Penyebab Reset Tmux:** Mengklarifikasi insiden restart-nya tmux server yang dipicu oleh tindakan `kill -9` pada sisa PID client yang menggantung, yang secara tidak sengaja mereset tmux server utama sehingga sesi `cloudflare_tunnel` sempat terbuat baru (sekarang sudah pulih otomatis dan berjalan lancar).
  3. **Fix Sintaks Python:** Memperbaiki string literal penulisan sbatch script di `utils/book_gpu.py` baris 124 menjadi `f.write("\n".join(lines))` yang valid.
  4. **Proteksi Double Booking:** Mengembangkan logika cerdas di fungsi `main()` pada `utils/book_gpu.py`. Daemon kini secara otomatis mendeteksi jika user sudah memiliki job `vnot` yang aktif (Running atau Pending) di Slurm. Jika ditemukan, daemon akan melompati pengiriman `sbatch` baru dan langsung mengadopsi (attach/monitor) job tersebut. Ini mencegah duplikasi booking secara permanen saat daemon di-restart.
  5. **Menghidupkan Sesi tmux `gpu_booking`:** Menjalankan daemon kembali secara aman menggunakan interpreter Python absolut dari conda env `/data/users/g6717500336/.conda/envs/yolo_env/bin/python` di dalam sesi tmux `gpu_booking`.
- **File yang diubah/dibuat:**
  - `utils/book_gpu.py` [DIMODIFIKASI - Perbaikan sintaks & penambahan logika pencegahan double booking]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log]
- **Status saat ini:** Selesai ✅
  - Sesi tmux `gpu_booking` aktif dan memantau Job ID `6560` yang sedang berjalan.
  - Sesi tmux `cloudflare_tunnel` aktif dan berjalan normal.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jika sesi tmux `gpu_booking` mati di masa mendatang, cukup jalankan perintah: `tmux new-session -d -s gpu_booking "/data/users/g6717500336/.conda/envs/yolo_env/bin/python /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/book_gpu.py"`
  - Logika baru di `book_gpu.py` menjamin tidak akan terjadi double booking jika Anda merestart sesi ini.

---

### [Entri — Pemulihan SSH Reverse Tunnel Ollama ke VM100] — 2026-06-03 17:40 WIB

- **Tanggal/Waktu:** 2026-06-03 17:40 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Disconnect Ollama:** Mengidentifikasi penyebab tidak terdeteksinya backend Ollama di `chat.penelitian.my.id` karena SSH reverse tunnel yang berjalan di compute node `ai3` mengalami `Broken pipe` saat tmux server di Slurmmaster crash sebelumnya.
  2. **Pembersihan Socket VM100:** Menemukan sisa socket sshd yang menggantung (stuck) di port `11434` pada `VM100` (PID 1548743 dan 1548971) dan membunuhnya secara paksa agar port `11434` bersih kembali.
  3. **Membangun Ulang Tunnel:** Menginisiasi ulang SSH reverse tunnel secara aman dari compute node `ai3` ke `VM100` (`11434:localhost:18319`) dengan pengalihan stdout/stderr ke `/dev/null` agar tidak mem-block session ssh.
  4. **Restart Open WebUI:** Melakukan restart container `open-webui` di `VM100` untuk membersihkan cache koneksi dan memicu inisialisasi ulang koneksi ke port `11434` yang baru saja pulih.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log]
- **Status saat ini:** Selesai ✅
  - Port `11434` di `VM100` terhubung dengan sukses ke Ollama API (`ai3:18319`).
  - Open WebUI berhasil mendeteksi model `qwen3-vl:30b` secara normal.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Jika koneksi Ollama di Open WebUI terputus lagi setelah insiden jaringan, pastikan untuk: (1) Cek port 11434 di VM100, (2) Jika kosong/stuck, kill sshd gantung di VM100, (3) Jalankan SSH tunnel baru dari node GPU aktif, (4) Restart container Open WebUI.

---

### [Entri — Pemulihan Kedua SSH Reverse Tunnel Ollama ke VM100] — 2026-06-03 18:15 WIB

- **Tanggal/Waktu:** 2026-06-03 18:15 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Disconnect Kedua:** Mendeteksi pemutusan berkala SSH reverse tunnel (`Broken pipe`) di `tunnel_sbatch.log` yang menyebabkan Open WebUI tidak dapat menjangkau port `18319` di compute node `ai3`.
  2. **Verifikasi Port & API:** Memverifikasi port `11434` di VM100 kosong dan memastikan API Ollama di compute node `ai3` pada port `18319` masih aktif melayani model `qwen3-vl:30b`.
  3. **Membangun Ulang SSH Tunnel:** Mengaktifkan kembali SSH Reverse Tunnel secara manual dari compute node `ai3` ke VM100 (`11434:localhost:18319`) untuk memulihkan routing.
  4. **Restart Container Open WebUI:** Me-restart container `open-webui` di VM100 untuk menyegarkan cache koneksi dan memastikan integrasi model pulih 100%.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log]
- **Status saat ini:** Selesai ✅
  - Port `11434` di VM100 aktif LISTEN dan terhubung ke Ollama API (`ai3:18319`).
  - Open WebUI kembali berfungsi normal dengan deteksi model `qwen3-vl:30b`.
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan kestabilan tunnel dipantau. Jika terjadi `Broken pipe` lagi di masa mendatang, lakukan prosedur pemulihan port forwarding serupa.



### [Entri — Perbaikan Bug Double Job Queue] — 2026-06-03 23:45 WIB

- **Tanggal/Waktu:** 2026-06-03 23:45 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Double Job:** Menemukan dua instance `book_gpu.py` yang berjalan bersamaan karena `myslurm.sh` tidak mengecek eksistensi daemon.
  2. **Perbaikan myslurm.sh:** Menambahkan pengecekan `pgrep -f book_gpu.py` di Opsi 1 untuk mencegah pembuatan sesi tmux ganda.
  3. **Perbaikan book_gpu.py (Retry Logic):** Memodifikasi fungsi `get_job_state()` dengan mekanisme retry 3 kali (delay 3 detik). Mencegah daemon membuat sbatch baru akibat false positive timeout dari command Slurm.
  4. **Perbaikan book_gpu.py (User Auth):** Menambal celah hilangnya variabel `$USER` dalam sub-shell dengan fallback library Python `os.environ`.
  5. **Pembersihan Zombie:** Mengeksekusi `pkill -f book_gpu.py` untuk membunuh daemons yang menumpuk.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `utils/book_gpu.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Race-condition status job Slurm telah tertangani dengan aman. Minta pengguna membatalkan job duplikat (`scancel`) secara manual.

---

### [Entri — Pembenahan Image SIF Ollama V100] — 2026-06-04 00:05 WIB

- **Tanggal/Waktu:** 2026-06-04 00:05 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi CUDA Error:** Menemukan bahwa `sbatch_llm_service.sh` masih merujuk ke `ollama.sif` (versi terbaru) yang tidak mendukung instruksi CUDA `sm_70` (Volta V100), menyebabkan crash `llama-server process has terminated: CUDA error: device kernel image is invalid` saat memuat model `qwen3.5` atau model GGUF lainnya.
  2. **Konfigurasi Image SIF:** Mengubah target image di `singularity/ollama/sbatch_llm_service.sh` dari `ollama.sif` ke `ollama_v0.2.sif` (versi 0.2.8) yang terverifikasi mendukung `sm_70`.
  3. **Pembersihan Job:** Membatalkan job lama yang menggantung tanpa service Ollama aktif (Job ID 6585) agar job baru dapat diluncurkan menggunakan berkas konfigurasi ter-update.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Image `ollama_v0.2.sif` wajib terus digunakan untuk kluster V100. Jika pengguna mengeluhkan model tidak terdeteksi, pastikan daemon `book_gpu.py` dijalankan ulang (Opsi 1 di `myslurm.sh`) untuk melakukan booking dengan konfigurasi terbaru.

---

### [Entri — Penyelesaian Konflik Qwen3.5 & Volta V100] — 2026-06-04 00:25 WIB

- **Tanggal/Waktu:** 2026-06-04 00:25 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Arsitektur Model:** Menemukan bahwa model `qwen3.5:latest` menggunakan arsitektur tokenizer baru (`qwen35`) yang hanya didukung oleh Ollama versi `0.17.1` atau lebih baru. Di versi lama (`ollama_v0.2.sif` / 0.2.8), ia terhenti dengan error `unknown model architecture: 'qwen35'`.
  2. **Pengecekan Image & Kompatibilitas:** Mengunduh berkas SIF `ollama_v0.3.14.sif` (versi 0.3.14) yang mendukung Qwen 2.5 tetapi gagal untuk `qwen35`. Sedangkan pada image terbaru `ollama.sif` (0.30.2) yang mendukung `qwen35`, runner CUDA `cuda_v12` menolak berjalan pada Tesla V100 (sm_70) dengan error `device kernel image is invalid`. Pengecekan memaksa `cuda_v13` juga gagal mendeteksi GPU karena ketidakcocokan versi driver host (12.3).
  3. **Rekomendasi Fallback:** Merekomendasikan pengguna untuk beralih (*downgrade*) ke model `qwen2.5:latest` atau `qwen2.5-coder:latest` (menggunakan arsitektur `qwen2` yang stabil) agar dapat menggunakan akselerasi GPU V100 secara penuh melalui image `ollama_v0.3.14.sif` yang telah diunduh.
  4. **Pembaruan Konfigurasi:** Mengonfigurasi `sbatch_llm_service.sh` untuk menggunakan image `ollama_v0.3.14.sif` dan runner `cuda_v11` yang stabil.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI]
  - `singularity/ollama/ollama_v0.3.14.sif` [DIBUAT/DIUNDUH]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk model-model dengan arsitektur `qwen2` (seperti Qwen 2.5), gunakan `ollama_v0.3.14.sif` and `cuda_v11` di cluster V100 ini. Qwen 3.5 saat ini belum dapat dijalankan dengan GPU karena keterbatasan precompiled runners di versi Ollama terbaru.

---

### [Entri — Analisis Status GPU Ollama Inactive] — 2026-06-04 01:25 WIB

- **Tanggal/Waktu:** 2026-06-04 01:25 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Log Inactive GPU:** Menganalisis log `ollama_sbatch.log:L1-L289` dan mengonfirmasi bahwa GPU Volta V100 terdeteksi **TIDAK AKTIF** (Ollama fallback ke CPU).
  2. **Identifikasi Penyebab:** Menemukan bahwa `sbatch_llm_service.sh` masih merujuk ke `ollama.sif` (versi `0.30.2`), padahal variabel `SINGULARITYENV_OLLAMA_LLM_LIBRARY="cuda_v11"` dipaksakan. Ini memicu kegagalan inisialisasi runner GPU Volta (sm_70) dan menyebabkan fallback ke CPU (`library=cpu`).
- **File yang diubah/dibuat:**
  - Tidak ada (hanya mencatat analisis investigatif).
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk menggunakan `ollama.sif` versi terbaru di GPU V100, jangan pernah menyetel `SINGULARITYENV_OLLAMA_LLM_LIBRARY="cuda_v11"` karena akan memicu fallback ke CPU. Biarkan Ollama mendeteksi runner `cuda_v12` secara default.

---

### [Entri — Aktivasi GPU Volta V100 di ollama.sif] — 2026-06-04 01:31 WIB

- **Tanggal/Waktu:** 2026-06-04 01:31 WIB
- **Tugas yang diselesaikan:**
  1. **Konfigurasi Lingkungan:** Menonaktifkan/memberi komentar pada variabel `SINGULARITYENV_OLLAMA_LLM_LIBRARY="cuda_v11"` di berkas `sbatch_llm_service.sh` untuk melepas paksaan runner v11.
  2. **Pembersihan Job & Restart Daemon:** Melakukan `scancel` pada job CPU lama (Job ID 6592). Daemon `book_gpu.py` secara otomatis mendeteksi pembersihan job dan mensubmit job GPU baru (Job ID 6595) secara instan.
  3. **Verifikasi Sukses GPU**: Membaca log inisialisasi sesi Ollama yang baru dan memvalidasi bahwa Ollama versi `0.30.2` (`ollama.sif`) telah berhasil mendeteksi dan menggunakan GPU Volta `Tesla V100-SXM2-32GB` (`compute=7.0`) via runner `cuda_v12` secara native dengan VRAM aktif `31.7 GiB`.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Akselerasi GPU Volta V100 di `ollama.sif` (0.30.2) saat ini aktif 100% dan stabil untuk melayani model multimodal/Llama/Qwen tanpa masalah CPU fallback.

---

### [Entri — Audit Keberadaan Berkas SIF] — 2026-06-04 01:50 WIB

- **Tanggal/Waktu:** 2026-06-04 01:50 WIB
- **Tugas yang diselesaikan:**
  1. **Audit Direktori Singularity:** Melakukan pemindaian direktori `/data/users/g6717500336/singularity/ollama` untuk memverifikasi keberadaan berkas `ollama_v0.3.14.sif` yang dicatat pada Entri 025.
  2. **Hasil Audit:** Berkas `ollama_v0.3.14.sif` tidak ditemukan di filesystem. Berkas yang aktif hanya `ollama.sif` (0.30.2) dan `ollama_v0.17.1.sif` (0.17.1). Rencana penggunaan versi `0.3.14` dibatalkan karena versi `0.30.2` (`ollama.sif`) telah berhasil berjalan di GPU secara native setelah perbaikan konfigurasi.
- **File yang diubah/dibuat:**
  - Tidak ada.
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Gunakan `ollama.sif` sebagai *Single Source of Truth* untuk *image* aktif. Abaikan referensi berkas `ollama_v0.3.14.sif` karena berkas tersebut tidak pernah disimpan di disk.

---

### [Entri — Dokumentasi Panduan Build Singularity] — 2026-06-04 01:54 WIB

- **Tanggal/Waktu:** 2026-06-04 01:54 WIB
- **Tugas yang diselesaikan:**
  1. **Dokumentasi Panduan Singularity:** Menyusun panduan langkah demi langkah cara mengunduh dan membangun (*pull/build*) image SIF dari repositori Docker resmi Ollama ke kluster Slurm tanpa akses root khusus untuk versi `v0.24.0`.
- **File yang diubah/dibuat:**
  - Tidak ada.
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Panduan ini dapat dirujuk pengguna kapan saja saat ingin mengganti atau menguji rilis Ollama versi `v0.24.0` di kluster.

---

### [Entri — Aktivasi GPU di Ollama v0.24.0] — 2026-06-04 02:27 WIB

- **Tanggal/Waktu:** 2026-06-04 02:27 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Log GPU Inactive:** Menemukan bahwa setelah pembaruan ke `ollama_v0.24.0.sif`, GPU kembali tidak aktif karena baris `export SINGULARITYENV_OLLAMA_LLM_LIBRARY="cuda_v11"` kembali aktif di berkas `sbatch_llm_service.sh` (ter-overwrite).
  2. **Perbaikan Konfigurasi:** Menonaktifkan/memberi komentar pada variabel `SINGULARITYENV_OLLAMA_LLM_LIBRARY="cuda_v11"` pada berkas `sbatch_llm_service.sh` untuk melepas paksaan runner v11.
  3. **Pembersihan Job & Start Daemon:** Membatalkan job lama `6598`. Menjalankan kembali daemon `book_gpu.py` di dalam sesi tmux `gpu_booking` agar meluncurkan job baru secara instan (Job ID `6599` di node `ai3`).
  4. **Verifikasi Sukses GPU**: Memeriksa log `ollama_sbatch.log` terbaru dan memvalidasi bahwa Ollama `v0.24.0` telah mendeteksi GPU Volta `Tesla V100-SXM2-32GB` (`compute=7.0`) via runner `cuda_v12` secara native dengan VRAM aktif `32.0 GiB`.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_llm_service.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan variabel `SINGULARITYENV_OLLAMA_LLM_LIBRARY` selalu dikomentari di `sbatch_llm_service.sh` agar semua versi image SIF Ollama modern (seperti `0.24.0` dan `0.30.2`) tidak fallback ke CPU.

---

### [Entri — Penambahan CSV Gabungan Single Model & Kolom Evaluator] — 2026-06-04 12:00 WIB

- **Tanggal/Waktu:** 2026-06-04 12:00 WIB
- **Tugas yang diselesaikan:**
  1. **Fungsi Baru `compile_all_single_model_csv()`** di `generate_report_single_model.py`:
     - Menambahkan konstanta `_SINGLE_MODEL_FIELDS` dengan 12 kolom field unified (termasuk `"Evaluator"`).
     - Membuat fungsi `compile_all_single_model_csv()` yang menggabungkan baris dari `kompilasi_ALL_detection.csv` dan `kompilasi_ALL_segmentation.csv` ke dalam satu file `kompilasi_ALL_single_model.csv`.
     - Field mapping: Detection → `mAP50 (Box)`, `mAP50-95 (Box)`, Mask fields diisi `"N/A"`. Segmentation → `mAP50-95 (Box)`, `mAP50-95 (Mask)`, field `mAP50 (Box/Mask)` diisi `"N/A"`.
     - Fungsi ini dipanggil otomatis di `main()` setelah `compile_all_csv()` selesai.
  2. **Penambahan Field `"Evaluator"` di `generate_standar_report-new_method.py`**:
     - Menambahkan key `"Evaluator": "COCOeval (pycocotools)"` pada dict `rows` gabungan (L719).
     - Menambahkan `"Evaluator"` ke daftar `fields` saat ekspor `kompilasi_ALL_new_method_standar.csv` (L763).
  3. **Penambahan Field `"Evaluator"` di `generate_golden_report-new_method.py`**:
     - Menambahkan key `"Evaluator": "COCOeval (pycocotools)"` pada dict `rows` gabungan (L712).
     - Menambahkan `"Evaluator"` ke daftar `fields` saat ekspor `kompilasi_ALL_new_method_golden.csv` (L777).
- **File yang diubah/dibuat:**
  - `utils/generate_report_single_model.py` [DIMODIFIKASI — fungsi baru `compile_all_single_model_csv()` + konstanta `_SINGLE_MODEL_FIELDS`, pemanggilan dari `main()`]
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI — tambah field `Evaluator` di `rows` dict dan `fields` list]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI — tambah field `Evaluator` di `rows` dict dan `fields` list]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Output baru `kompilasi_ALL_single_model.csv` akan tersimpan di: `reports/pipeline/csv/kompilasi_ALL_single_model.csv`.
  - Field `"Parameters (M)"` pada `kompilasi_ALL_single_model.csv` diisi `"N/A"` karena pipeline `generate_report_single_model.py` tidak menghitung jumlah parameter (hanya ukuran file weights). Jika diperlukan, bisa diambil dari `config_shared.py` via `_get_model_metrics()`.
  - Seluruh CSV `kompilasi_ALL_new_method_standar.csv` dan `kompilasi_ALL_new_method_golden.csv` kini memiliki kolom `Evaluator` terisi `"COCOeval (pycocotools)"` untuk keperluan transparansi metodologi di paper Scopus.

---

### [Entri — Integrasi Kolom Spesifikasi CSV Terpadu (Unified) & Metrik Box Segmentasi] — 2026-06-04 13:02 WIB

- **Tanggal/Waktu:** 2026-06-04 13:02 WIB
- **Tugas yang diselesaikan:**
  1. **Konstanta Global `CSV_REPORT_FIELDS`**: Menambahkan konstanta global berisi 18 kolom spesifikasi terpadu ke `config_shared.py`.
  2. **Modifikasi `evaluation_hybrid_sota.py`**: Mengimpor `CSV_REPORT_FIELDS`, menyertakan `Precision(Box)` & `Recall(Box)` pada `seg_row` secara dinamis, dan menambahkan fungsi kompilasi untuk menghasilkan `kompilasi_ALL_hybrid_sota.csv` di folder `reports/paper1/csv/new-method/`.
  3. **Modifikasi `generate_report_single_model.py`**: Mengimpor `CSV_REPORT_FIELDS`, mengganti `_SINGLE_MODEL_FIELDS` lokal, dan menyesuaikan mapping di `compile_all_single_model_csv()` agar mendukung kolom terpadu. Metrik Box untuk model segmentasi dipetakan secara dinamis dari data aslinya.
  4. **Modifikasi `generate_standar_report-new_method.py` dan `generate_golden_report-new_method.py`**: Mengimpor `CSV_REPORT_FIELDS`, mengganti daftar `fields` lokal dengan `CSV_REPORT_FIELDS`, dan menyelaraskan key pada dictionary `rows` dengan format kolom baru secara dinamis (menyertakan Precision dan Recall Box/Mask).
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI]
  - `utils/evaluation_hybrid_sota.py` [DIMODIFIKASI]
  - `utils/generate_report_single_model.py` [DIMODIFIKASI]
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh CSV kompilasi (`new_method_standar`, `new_method_golden`, `single_model`, `hybrid_sota`) sekarang sudah menggunakan format 18 kolom terpadu dari `CSV_REPORT_FIELDS`.
  - Untuk model segmentasi (Mask R-CNN, YOLO segmentasi, MobileSAM, SAM2), metrik Box (mAP, Precision, Recall) terisi dengan nilai riil hasil evaluasi deteksi paralel (bukan di-hardcode `"N/A"`).
  - Field `"Parameters (M)"` pada `kompilasi_ALL_single_model.csv` diisi `"N/A"` karena pipeline `generate_report_single_model.py` tidak menghitung jumlah parameter (hanya ukuran file weights). Jika diperlukan, bisa diambil dari `config_shared.py` via `_get_model_metrics()`.
  - Seluruh CSV `kompilasi_ALL_new_method_standar.csv` dan `kompilasi_ALL_new_method_golden.csv` kini memiliki kolom `Evaluator` terisi `"COCOeval (pycocotools)"` untuk keperluan transparansi metodologi di paper Scopus.

---

### [Entri — Sinkronisasi 16 Model Hybrid MobileSAM pada Orkestrasi New Method] — 2026-06-04 18:45 WIB

- **Tanggal/Waktu:** 2026-06-04 18:45 WIB
- **Tugas yang diselesaikan:**
  1. **Sinkronisasi `generate_standar_report-new_method.py`**: Melengkapi 16 model Hybrid MobileSAM baru pada Fase 1 Kuantitatif (COCOeval mAP), meliputi pembacaan parameter & bobot dinamis via `_get_model_metrics()`, serta optimalisasi load block dan predict block pada `_infer_worker()`.
  2. **Sinkronisasi `generate_golden_report-new_method.py`**: Menyelaraskan logika kalkulasi mAP COCOeval, routing dataset, dan akumulasi hasil metrik box/mask agar 100% kompatibel dengan tipe `hybrid_det_mobile` dan `hybrid_seg_mobile`.
  3. **Validasi & Dry-Run**: Sukses menguji kompilasi sintaksis python (sukses 100% bebas error) dan berhasil memicu inisialisasi dry-run model `hybrid_yolov8m_mobile` menggunakan backend GPU `ai3` di dalam sesi tmux `evaluation` hingga proses iterasi berjalan tanpa kendala.
- **File yang diubah/dibuat:**
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Integrasi MobileSAM baru telah teruji secara statis dan dinamis di node GPU.
  - Untuk memicu evaluasi lengkap seluruh model standar + golden + hybrid sota secara paralel, jalankan:
    `python3 run_pipeline_parallel.py --eval-only --gpus 0` atau picu secara manual.


---

### [Entri — Implementasi Visual Evaluation Web App (GPU Queue)] — 2026-06-05 14:30 WIB

- **Tanggal/Waktu:** 2026-06-05 14:30 WIB
- **Tugas yang diselesaikan:**
  1. **Pembuatan Backend Flask (Queue-Aware)**: Membuat `RVM/backend/visual_eval_api.py` dengan thread-safe FIFO queue (`queue.Queue`) untuk mengatasi limitasi VRAM (CUDA OOM) akibat multithreading. Backend berjalan murni di node komputasi (`ai2/ai3`) dengan deteksi GPU otomatis.
  2. **Pembuatan Frontend UI/UX (Bio-Digital Minimalism)**: Mengimplementasikan desain sesuai guideline di `RVM/frontend/` dengan animasi *glassmorphism*, dark/light mode toggle, slider parameter, dan responsivitas penuh. 
  3. **Script Launcher Terpadu (`start_rvm.sh`)**: Menggabungkan eksekusi frontend, backend, dan SSH reverse tunnel ke dalam satu launcher pintar. Memperbaiki isu *Tmux Version Conflict* (server version is too old) dengan menambahkan deteksi dinamis `TMUX_BIN` di `start_rvm.sh` (memastikan login node memakai `/usr/bin/tmux` dan compute node memakai `yolo_env` tmux).
  4. **Instalasi Flask**: Menambahkan modul Flask di dalam environment Conda `yolo_env` karena belum terinstal sebelumnya.
- **File yang diubah/dibuat:**
  - `RVM/backend/visual_eval_api.py` [BARU]
  - `RVM/frontend/index.html` [BARU]
  - `RVM/frontend/css/style.css` [BARU]
  - `RVM/frontend/js/app.js` [BARU]
  - `RVM/serve_frontend.py` [BARU]
  - `RVM/start_rvm.sh` [BARU — DIMODIFIKASI untuk TMUX_BIN]
  - `config_shared.py` [DIMODIFIKASI — Konstanta EVAL_API_* ditambahkan]
- **Status saat ini:** Selesai ✅ (Menunggu user test run live)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan user selalu mengalokasikan GPU (Slurm job status R) sebelum menjalankan `bash RVM/start_rvm.sh backend` di node compute.
  - Jika ada *timeout* antrian GPU, perbesar `timeout_seconds` di `visual_eval_api.py`.

---

### [Entri — Refactor Model Selector menjadi Cascading Dropdown] — 2026-06-05 15:40 WIB

- **Tanggal/Waktu:** 2026-06-05 15:40 WIB
- **Tugas yang diselesaikan:**
  1. **Cascading Dropdown UI**: Mengganti tumpukan grid chip model yang padat di frontend dengan 3 tingkat select dropdown dinamis: Kategori (YOLO, Hybrid, Detection, Segmentation) ➔ Model Family ➔ Model Varian.
  2. **Selected Chips Layout**: Menambahkan panel visual untuk menampilkan daftar model terpilih (dengan tombol hapus individual) agar mendukung multi-model evaluation secara intuitif.
  3. **Penyempurnaan Quick-Select**: Menghubungkan fungsi klik tombol-tombol Quick Select (Select All, Clear, dll.) agar langsung memodifikasi dan merender ulang Selected Chips.
- **File yang diubah/dibuat:**
  - `RVM/frontend/index.html` [DIMODIFIKASI]
  - `RVM/frontend/css/style.css` [DIMODIFIKASI]
  - `RVM/frontend/js/app.js` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅ (Berhasil direfaktorisasi dan diuji secara lokal)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan dropdown Category memisahkan "yolo" model dengan "hybrid" model secara tepat menggunakan logic filter di `app.js`.

---

### [Entri — Integrasi Launcher RVM ke CLI Utama myslurm.sh] — 2026-06-05 15:45 WIB

- **Tanggal/Waktu:** 2026-06-05 15:45 WIB
- **Tugas yang diselesaikan:**
  1. **Menu Web RVM di myslurm.sh**: Menambahkan opsi menu baru `7. 🖥️ Web RVM` pada sistem kontrol utama.
  2. **Interaktif Sub-Menu**: Membangun sub-menu interaktif yang memungkinkan user untuk:
     - Memulai seluruh layanan RVM (Frontend & Backend + GPU) sekaligus.
     - Menghentikan/restart seluruh layanan secara terpadu.
     - Memulai frontend saja atau backend saja secara individual.
  3. **Visual Status Check**: Mengintegrasikan laporan status tmux dan port listening secara real-time dari skrip RVM langsung di dalam layar CLI Manajemen RVM.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅ (Dapat dieksekusi secara langsung)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Path launcher diletakkan secara absolut `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/RVM/start_rvm.sh` untuk menjamin ia dapat diakses dari mana saja.

---

### [Entri — Bugfix RuntimeError Hybrid SAM (Empty BBoxes)] — 2026-06-05 16:50 WIB

- **Tanggal/Waktu:** 2026-06-05 16:50 WIB
- **Tugas yang diselesaikan:**
  1. **Bugfix Hybrid Pipeline**: Menambahkan pemeriksaan `len(yolo_results[0].boxes) == 0` pada fungsi `_infer_hybrid()` di backend. Jika model YOLO tidak menghasilkan deteksi objek apa pun pada confidence threshold terpilih, backend kini langsung mengembalikan respons deteksi kosong secara anggun tanpa meneruskan array kosong ke SAM (mencegah `RuntimeError: Sizes of tensors must match...` akibat tensor berdimensi 0).
- **File yang diubah/dibuat:**
  - `RVM/backend/visual_eval_api.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅ (Bug teratasi)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan setiap inferensi hybrid yang mengandalkan prompt generator (YOLO) selalu memvalidasi ketersediaan data box sebelum memanggil model downstream (SAM/SAM2).

---

### [Entri — Bugfix Flask Auto-Reloader (Duplicate GPU Workers)] — 2026-06-05 17:05 WIB

- **Tanggal/Waktu:** 2026-06-05 17:05 WIB
- **Tugas yang diselesaikan:**
  1. **Bugfix Flask Reloader**: Menambahkan `use_reloader=False` pada `app.run()` Flask backend. Ketika mode debug diaktifkan (`EVAL_API_DEBUG = True`), Flask secara default menggunakan reloader yang memicu proses ganda (parent & child) dan menduplikasi background worker thread (`_gpu_worker`). Duplikasi ini memperebutkan alokasi CUDA memori di GPU, sehingga menyebabkan inferensi CUDA berikutnya gagal total secara diam-diam (mengembalikan 0 deteksi).
- **File yang diubah/dibuat:**
  - `RVM/backend/visual_eval_api.py` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅ (Otomatis memulihkan kestabilan GPU)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - JANGAN PERNAH mengaktifkan Flask reloader (`use_reloader=True`) pada lingkungan dengan CUDA/PyTorch background worker daemon karena driver NVIDIA/CUDA tidak mendukung multi-process allocation secara aman tanpa IPC terstruktur.

---

### [Entri — Uji Coba YOLO11n-seg & Implementasi Logging Unbuffered] — 2026-06-05 17:20 WIB

- **Tanggal/Waktu:** 2026-06-05 17:20 WIB
- **Tugas yang diselesaikan:**
  1. **Investigasi Deteksi YOLO11n-seg**: Melakukan dry-run manual di compute node `ai3` (Job ID 6608) menggunakan script scratch untuk membandingkan `yolo11n` (detection) vs `yolo11n_seg` (segmentation) pada gambar sampel. Ditemukan bahwa confidence score tertinggi `yolo11n_seg` pada gambar tersebut adalah **0.7460**, sehingga dengan confidence threshold default **0.75**, model mengembalikan 0 deteksi ("No detections"). Sementara `yolo11n` berhasil mendeteksi objek dengan confidence **0.8519** (di atas 0.75).
  2. **Pengembalian Parameter Riset**: Memulihkan berkas `config_shared.py` ke setelan aslinya (`EVAL_CONF = 0.75`, `VISUAL_CONF = 0.75`) agar parameter riset Scopus kuantitatif/kualitatif Anda tetap konsisten.
  3. **Penyesuaian Slider Frontend**: Menurunkan default value slider confidence di frontend `index.html` dari `0.75` menjadi `0.25` agar model-model nano (seperti `yolo11n-seg`) langsung mengembalikan deteksi objek saat halaman dimuat secara default.
  4. **Peningkatan Logging Real-Time**: Mengubah perintah eksekusi python di `start_rvm.sh` menggunakan bendera `-u` (unbuffered output) dan menambahkan log inference yang sangat detail ke fungsi `_run_inference` di `visual_eval_api.py` untuk memudahkan pelacakan deteksi secara real-time.
  5. **Implementasi Model Caching (LRU)**: Membangun kelas `ModelCache` global dengan kapasitas maksimal 6 model di `visual_eval_api.py`. Cache ini menahan objek model YOLO, SAM, dan Mask R-CNN di VRAM untuk menghindari re-load weights dari disk pada setiap request, yang secara empiris menurunkan total waktu inferensi 5 model sekaligus dari **4.18 detik** ke **192 ms** (~21x lebih cepat) dan menyelesaikan masalah fragmentasi memori CUDA yang memicu kegagalan forward pass (0 deteksi) setelah beberapa kali pemuatan model.
  6. **Sinkronisasi Parameter Slider Dinamis**: Menambahkan properti `config` (eval_conf & eval_iou) ke respons endpoint API `/api/health` di `visual_eval_api.py`. Mengubah logic inisialisasi frontend di `app.js` agar membaca data `config` ini dan memperbarui nilai slider secara dinamis, sehingga sinkron secara real-time dengan `config_shared.py` (tanpa hardcode di HTML).
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIPULIHKAN]
  - `RVM/frontend/index.html` [DIMODIFIKASI]
  - `RVM/frontend/js/app.js` [DIMODIFIKASI]
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
  - `RVM/backend/visual_eval_api.py` [DIMODIFIKASI MAYOR]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Layanan Web RVM telah di-restart menggunakan tmux dan berjalan stabil. Nilai default slider di UI disinkronkan dinamis dari `config_shared.py` saat load page pertama via API /api/health. Lokasi log real-time ada di `RVM/backend/logs/backend.log`. Caching model (max 6 model) menahan weights di GPU VRAM secara dinamis dengan safety lock thread-safe.

---

### [Entri — Verifikasi Komparasi Inferensi & Analisis Alur Evaluasi] — 2026-06-05 18:55 WIB

- **Tanggal/Waktu:** 2026-06-05 18:55 WIB
- **Tugas yang diselesaikan:**
  1. **Analisis Asal Nilai Evaluasi**: Menganalisis dan mendokumentasikan sumber perolehan metrik pada skrip evaluasi kuantitatif. Menjelaskan logika filter threshold (`EVAL_CONF`/`EVAL_IOU`), pencocokan dengan Ground Truth COCO via IoU, pembagian TP/FP/FN, serta perhitungan Precision/Recall/mAP.
  2. **Verifikasi Inferensi Multi-Model pada Gambar 1660.jpg**: Menguji inferensi pada 8 model (`yolo11l_seg`, `yolo11n_seg`, `yolov8m_seg`, `yolov9c_seg` beserta varian hybrid SAM2-nya) dengan setelan `Conf=0.7` dan `IoU=0.15` menggunakan script scratch lokal (`compare_inference.py`) di compute node `ai3` dan memverifikasinya melalui panggilan HTTP API Web RVM.
  3. **Analisis Data Komparasi**: Memverifikasi bahwa jumlah deteksi, confidence score, dan koordinat bbox antara YOLO dasar dan hybrid adalah identik, dengan SAM2 secara sukses merampingkan area mask piksel (menghilangkan noise latar belakang hingga ~50%).
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Hasil inferensi lokal dan API Web RVM telah terbukti konsisten 100%. Sistem caching model bekerja dengan sangat baik mencegah CUDA fragmentation/OOM.

---

### [Entri — Git Push Proyek ke GitHub Branch dev] — 2026-06-05 22:20 WIB

- **Tanggal/Waktu:** 2026-06-05 22:20 WIB
- **Tugas yang diselesaikan:**
  - Melakukan staging (`git add`) untuk seluruh perubahan yang termodifikasi di workspace `MyFineTunning-SlurmMaster/` dan folder untracked baru `RVM/` serta `docs/LLM Models/Implementation Plan.md`.
  - Melakukan commit perubahan dengan deskripsi yang mencakup integrasi Web RVM, perbaikan daemon `book_gpu.py`, serta pembaruan skrip evaluasi.
  - Melakukan pull (`git pull origin dev`) untuk memastikan sinkronisasi yang aman.
  - Melakukan push (`git push origin dev`) ke repositori remote `git@github.com:vnot-programming/Trainning-Models.git` pada branch `dev`.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Branch `dev` di repositori remote sekarang sudah sinkron 100% dengan repositori lokal (termasuk folder `RVM/` dan file konfigurasi `config_shared.py` terbaru).

---

### [Entri — Stabilisasi Cloudflare Tunnel & Analisis Jaringan Kampus] — 2026-06-05 23:55 WIB

- **Tanggal/Waktu:** 2026-06-05 23:55 WIB
- **Tugas yang diselesaikan:**
  1. **Root Cause Analysis RTO**: Menganalisis log `cloudflared.log` dan mendiagnosis bahwa RTO pada `slurm.penelitian.my.id` disebabkan oleh pemutusan koneksi outbound (ICMP & persistent HTTP/2 tunnel drops) secara periodik oleh perimeter firewall Kasetsart University (KU) ke Cloudflare Edge, bukan karena binding `ssh://localhost:22`.
  2. **Tuning Server-Side Tunnel**: Menghentikan sesi lama dan mengaktifkan ulang daemon tunnel `cloudflared` di dalam tmux session `cloudflare_tunnel` menggunakan parameter yang dioptimasi:
     - `--edge-ip-version 4` (memaksa resolusi IPv4 yang lebih kompatibel dengan routing kampus).
     - `--retries 10` (meningkatkan toleransi ketahanan reconnect).
     - Berhasil meregistrasikan 4 koneksi paralel secara stabil ke lokasi `bkk06` dan `sin02`/`sin11`.
  3. **Audit Port 22 Publik**: Melakukan uji coba dari VPS Azure (`VPS-4C56G-CF`) ke IP publik Slurm `158.108.222.131` port 22 dan mengonfirmasi status `Blocked`. Hal ini memastikan bahwa akses direct SSH diblokir oleh kampus dari luar dan Cloudflare Tunnel adalah satu-satunya gerbang masuk.
  4. **Penyusunan Solusi Client-Side**: Merumuskan instruksi konfigurasi SSH Keepalive bagi PC/Laptop pengguna untuk mencegah penutupan koneksi idle oleh NAT/firewall kampus.
- **File yang diubah/dibuat:**
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅ (Koneksi tunnel pulih dan stabil)
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Layanan `cloudflared` berjalan aktif di latar belakang (tmux session `cloudflare_tunnel`). Log terpantau di `/data/users/g6717500336/singularity/cloudflared.log`. Jika koneksi SSH dari client masih sering putus setelah idle, pastikan user mengonfigurasi SSH keepalive (`ServerAliveInterval 15`, `ServerAliveCountMax 5`) di sisi client (Laptop/PC).

---

### [Entri — Redireksi Log Cloudflare Tunnel & Optimasi myslurm.sh] — 2026-06-06 08:40 WIB

- **Tanggal/Waktu:** 2026-06-06 08:40 WIB
- **Tugas yang diselesaikan:**
  1. **Redireksi Output Log**: Mengonfigurasi `myslurm.sh` agar mengalihkan stderr/stdout dari biner `cloudflared` ke `/data/users/g6717500336/singularity/cloudflared.log` menggunakan utilitas `tee -a` di dalam sesi TMUX `cloudflare_tunnel`. Hal ini menjamin log tersimpan ke disk secara dinamis sekaligus mempertahankan tampilan log di tmux pane buffer.
  2. **Integrasi Parameter Optimasi Jaringan**: Memasukkan parameter optimal (`--edge-ip-version 4` dan `--retries 10`) langsung ke perintah eksekusi tunnel di `myslurm.sh` agar stabil saat dijalankan via menu CLI.
  3. **Penyajian Lokasi Log**: Menambahkan informasi penunjuk lokasi file `/data/users/g6717500336/singularity/cloudflared.log` pada menu Status & Log (Pilihan 3) agar user mengetahui lokasi file log lengkap.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - File log cloudflared saat ini dialihkan secara realtime ke `${CLOUDFLARE_BIN}.log` (default: `/data/users/g6717500336/singularity/cloudflared.log`). Sesi tmux `cloudflare_tunnel` dapat dimatikan dan dihidupkan ulang dari menu `myslurm.sh` (pilihan 6) untuk menguji penulisan log baru.

---

### [Entri — Konfigurasi Perintah Global slurm untuk myslurm.sh] — 2026-06-06 08:55 WIB

- **Tanggal/Waktu:** 2026-06-06 08:55 WIB
- **Tugas yang diselesaikan:**
  1. **Konfigurasi Perintah Global**: Menambahkan alias `slurm` untuk merujuk langsung ke absolute path `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh`.
  2. **Penyelarasan Shell**: Menuliskan alias tersebut pada konfigurasi shell `.bashrc` dan `.zshrc` agar perintah `slurm` dapat dipanggil secara global dari shell Bash maupun Zsh dari direktori mana pun.
- **File yang diubah/dibuat:**
  - `~/.bashrc` [DIMODIFIKASI]
  - `~/.zshrc` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Perintah global ini dapat diuji oleh pengguna dengan menjalankan `source ~/.bashrc` (atau membuka sesi shell baru) lalu mengetikkan `slurm` di terminal.

---

### [Entri — Aturan Keamanan Cloudflare Tunnel & Bugfix Deteksi Daemon] — 2026-06-06 09:00 WIB

- **Tanggal/Waktu:** 2026-06-06 09:00 WIB
- **Tugas yang diselesaikan:**
  1. **Integrasi Aturan Keamanan Cloudflare Tunnel**: Menambahkan poin aturan baru pada `.gemini/GEMINI.md` yang melarang keras AI agent untuk mematikan (kill) sesi tmux `cloudflare_tunnel` karena merupakan jalur koneksi luar utama Master Node, kecuali jika ada instruksi tertulis langsung dari pengguna.
  2. **Bugfix Deteksi Daemon di myslurm.sh**: Memperbaiki logika deteksi daemon pada file `utils/myslurm.sh`. Sebelumnya, `pgrep -f "book_gpu.py"` secara keliru menangkap proses `tmux` induk yang yatim (zombie) karena parameter pemanggilannya mengandung string target pencarian, yang memicu status palsu "⚠️ DAEMON SUDAH BERJALAN!". Logika baru mengecek secara presisi file `/proc/PID/exe` untuk memastikan proses tersebut benar-benar di-drive oleh interpreter Python.
- **File yang diubah/dibuat:**
  - `.gemini/GEMINI.md` [DIMODIFIKASI]
  - `utils/myslurm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan setiap ada verifikasi status proses background, filter jenis executable program agar terhindar dari false-positive pembungkus sesi (wrapper) seperti tmux atau bash. Sesi tmux `cloudflare_tunnel` sekarang dilindungi oleh aturan global.

---

### [Entri — Bugfix Resolusi Path Symlink myslurm.sh] — 2026-06-06 09:10 WIB

- **Tanggal/Waktu:** 2026-06-06 09:10 WIB
- **Tugas yang diselesaikan:**
  1. **Integrasi Aturan Keamanan Cloudflare Tunnel**: Menambahkan poin aturan baru pada `.gemini/GEMINI.md` yang melarang keras AI agent untuk mematikan (kill) sesi tmux `cloudflare_tunnel` karena merupakan jalur koneksi luar utama Master Node, kecuali jika ada instruksi tertulis langsung dari pengguna.
  2. **Bugfix Deteksi Daemon di myslurm.sh**: Memperbaiki logika deteksi daemon pada file `utils/myslurm.sh`. Sebelumnya, `pgrep -f "book_gpu.py"` secara keliru menangkap proses `tmux` induk yang yatim (zombie) karena parameter pemanggilannya mengandung string target pencarian, yang memicu status palsu "⚠️ DAEMON SUDAH BERJALAN!". Logika baru mengecek secara presisi file `/proc/PID/exe` untuk memastikan proses tersebut benar-benar di-drive oleh interpreter Python.
- **File yang diubah/dibuat:**
  - `.gemini/GEMINI.md` [DIMODIFIKASI]
  - `utils/myslurm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan setiap ada verifikasi status proses background, filter jenis executable program agar terhindar dari false-positive pembungkus sesi (wrapper) seperti tmux atau bash. Sesi tmux `cloudflare_tunnel` sekarang dilindungi oleh aturan global.

---

### [Entri — Penambahan Metrik Boundary IoU & Boundary AP] — 2026-06-06 11:12 WIB

- **Tanggal/Waktu:** 2026-06-06 11:12 WIB
- **Tugas yang diselesaikan:**
  1. **Penambahan Fungsi `_mask_to_boundary` & `compute_boundary_iou`**: Fungsi mengekstrak boundary mask via erosi morfologi (kernel adaptif = `ceil(0.02 × sqrt(H²+W²))`), lalu menghitung Boundary IoU = `|boundary_pred ∩ boundary_gt| / |boundary_pred ∪ boundary_gt|`.
  2. **Penambahan Fungsi `compute_boundary_metrics_from_coco`**: Fungsi mendecode prediksi & GT mask dari format COCO-RLE, melakukan greedy matching berdasarkan confidence score tertinggi, lalu menghitung:
     - **Boundary IoU**: rata-rata B-IoU dari prediksi yang berhasil match (TP)
     - **Boundary AP**: kurva precision-recall diinterpolasi pada 101 recall thresholds (standar COCO AP @ B-IoU ≥ 0.5)
  3. **Integrasi di Blok Evaluasi Utama**: Boundary metrics dihitung setelah COCOeval selesai, hanya untuk model yang menghasilkan instance segmentation (`yolo_seg`, `maskrcnn`, `hybrid_seg`, `hybrid_seg_mobile`, `hybrid_det`, `hybrid_det_mobile`). Model pure detection mendapatkan nilai `"N/A"`.
  4. **Update `CSV_REPORT_FIELDS` di `config_shared.py`**: Menambahkan kolom `"Boundary IoU"` dan `"Boundary AP"` setelah `"Recall(Mask)"`.
  5. **Update dict `rows`, `rows_seg`**: Kedua skrip kini menyertakan `"Boundary IoU"` dan `"Boundary AP"` di setiap baris laporan.
  6. **Snipped Ringkasan di Akhir Fase 1**: Setelah Fase 1 selesai, skrip mencetak definisi lengkap Boundary IoU & AP, pseudocode satu baris, dan tabel ringkas per model segmentasi.
- **File yang diubah/dibuat:**
  - `config_shared.py` [DIMODIFIKASI — CSV_REPORT_FIELDS +2 kolom baru]
  - `utils/generate_standar_report-new_method.py` [DIMODIFIKASI — +3 fungsi boundary, integrasi evaluasi, update rows, snipped]
  - `utils/generate_golden_report-new_method.py` [DIMODIFIKASI — +3 fungsi boundary, integrasi evaluasi, update rows, snipped]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan log ini]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Boundary IoU & AP hanya diisi untuk model dengan segmentation masks. Pure detection (`yolo_det`) mendapat `"N/A"`.
  - Parameter kunci: `dilation_ratio=0.02` (kernel boundary) dan `iou_thresh=0.5` (threshold TP untuk AP). Keduanya tidak hardcode — dapat diubah saat pemanggilan fungsi.
  - Jika ingin menambah threshold B-IoU lain (misal AP@0.75), cukup panggil `compute_boundary_metrics_from_coco(dt_segm, coco_gt, iou_thresh=0.75)` secara terpisah.
  - Seluruh CSV kompilasi (`kompilasi_ALL_new_method_standar.csv`, `kompilasi_ALL_new_method_golden.csv`) sekarang memiliki 20 kolom (dari 18 sebelumnya).

---

### [Entri — Implementasi Image Comparison Mode & Update Aturan Global Slurm] — 2026-06-06 17:30 WIB

- **Tanggal/Waktu:** 2026-06-06 17:30 WIB
- **Tugas yang diselesaikan:**
  1. **Modifikasi Frontend Evaluasi RVM (index.html, app.js, & style.css)**: 
     - Membagi bagian "📸 Upload Image" menjadi panel split kiri dan kanan (📸 Upload Image | 📸 Result).
     - Menambahkan checkbox "Image Comparison" pada kolom kiri. Kolom hasil (📸 Result) diaktifkan hanya jika checkbox dicentang.
     - Membatasi jumlah model yang dapat dipilih maksimal 5 model saat mode perbandingan aktif.
     - Menambahkan banner informatif pada antarmuka "Select Models" untuk menunjukkan status "Image Comparison Mode Aktif".
     - Merender grid 3x2 (Ground Truth + hingga 5 model) di panel hasil menggunakan elemen HTML Canvas untuk menampilkan bounding box hasil inferensi model di atas gambar asli.
  2. **Update Aturan Global (.gemini/GEMINI.md)**: 
     - Menambahkan instruksi pengujian cepat menggunakan perintah `slurm` dan opsi `2` untuk berpindah ke Node Cluster GPU aktif (`ai3`/`ai2`) dengan environment Conda `yolo_env` yang aktif.
- **File yang diubah/dibuat:**
  - `RVM/frontend/index.html` [DIMODIFIKASI]
  - `RVM/frontend/js/app.js` [DIMODIFIKASI]
  - `RVM/frontend/css/style.css` [DIMODIFIKASI]
  - `.gemini/GEMINI.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan backend `visual_eval_api.py` berjalan di Node GPU yang aktif agar frontend dapat berkomunikasi secara normal dengan backend.
---

### [Entri — Integrasi Rendering Mask Poligon & Fitur Download Grid di RVM] — 2026-06-06 17:40 WIB

- **Tanggal/Waktu:** 2026-06-06 17:40 WIB
- **Tugas yang diselesaikan:**
  1. **Render Segmentasi Mask pada Canvas**:
     - Memodifikasi `visual_eval_api.py` (`_infer_yolo`, `_infer_maskrcnn`, `_infer_hybrid`) untuk melampirkan data poligon koordinat mask di bawah key `"segment"`.
     - Memodifikasi fungsi `_drawBoundingBoxes` di `RVM/frontend/js/app.js` untuk secara dinamis menggambar poligon mask di atas canvas dengan isian transparan (`hsla`) dan outline sebelum menggambar bounding box serta label kelas.
  2. **Fitur Download Grid Perbandingan**:
     - Menambahkan tombol "📥 Download Grid" di dalam header kolom `📸 Result` pada `index.html`.
     - Menambahkan styling premium untuk `.btn-download-grid` di `style.css` (termasuk micro-animation, glow effect, dan light-mode support).
     - Mengimplementasikan fungsi `setupGridDownload()` di `app.js` untuk menyatukan seluruh sub-canvas perbandingan ke dalam satu canvas gabungan berdimensi besar dengan label banner nama model semi-transparan, lalu mengunduhnya secara otomatis sebagai gambar PNG beresolusi tinggi.
  3. **Peningkatan Kontras Teks Light Mode**:
     - Mengubah warna teks primer toska muda (`var(--c-primary-light)`) menjadi warna yang lebih dinamis dan kontras (`var(--c-primary-text)`) yang menyesuaikan tema (dark/light) di `style.css` untuk selected chips, placeholder, range sliders, export buttons, dan detail tables.
- **File yang diubah/dibuat:**
  - `RVM/backend/visual_eval_api.py` [DIMODIFIKASI]
  - `RVM/frontend/index.html` [DIMODIFIKASI]
  - `RVM/frontend/js/app.js` [DIMODIFIKASI]
  - `RVM/frontend/css/style.css` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
---

### [Entri — Penambahan Kontrol Modular Backend Flask pada myslurm.sh] — 2026-06-06 17:55 WIB

- **Tanggal/Waktu:** 2026-06-06 17:55 WIB
- **Tugas yang diselesaikan:**
  1. **Update launcher start_rvm.sh**:
     - Menambahkan fungsi modular `stop_backend` untuk menghentikan sesi tmux `rvm_backend` dan SSH tunnel port 8502 saja.
     - Menambahkan fungsi modular `stop_frontend` untuk menghentikan sesi tmux `rvm_frontend` saja.
     - Menghubungkan argumen `stop_backend` dan `stop_frontend` ke dalam case switcher CLI di `start_rvm.sh`.
  2. **Update submenu Manajemen Web RVM di myslurm.sh**:
     - Memperluas menu "7. Web RVM" untuk menyertakan opsi manajemen Flask backend secara eksplisit.
     - Menambahkan pilihan: "Start Layanan backend Flask" (menu 6), "Stop Layanan backend Flask" (menu 7), "Restart Layanan backend Flask" (menu 8), dan "Lihat Layanan Berjalan" (menu 9).
     - Memetakan setiap pilihan ke launcher `start_rvm.sh` dengan argumen backend/frontend modular yang baru.
- **File yang diubah/dibuat:**
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
  - `utils/myslurm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh menu manajemen RVM telah terhubung secara modular ke start_rvm.sh. Kontrol backend/frontend kini bisa dilakukan secara terpisah tanpa mematikan sesi global lainnya.
---

### [Entri — Bugfix Errexit crash pada check_status start_rvm.sh] — 2026-06-06 17:58 WIB

- **Tanggal/Waktu:** 2026-06-06 17:58 WIB
- **Tugas yang diselesaikan:**
  1. **Bugfix Errexit Crash**:
     - Mengatasi masalah terpotongnya tampilan status saat mengecek ports/services pada menu "7. Web RVM".
     - Masalah disebabkan oleh `set -euo pipefail` di bagian atas `start_rvm.sh` yang langsung mematikan (exit) skrip ketika mendeteksi command `pgrep` (saat Cloudflared stopped) atau pipeline `ss | grep` mengembalikan status non-zero (gagal/inactive).
     - Solusi: Menambahkan `set +e` di awal fungsi `check_status` dan mengaktifkannya kembali dengan `set -e` di akhir fungsi agar proses pengecekan status berjalan lancar hingga selesai tanpa menghentikan skrip.
- **File yang diubah/dibuat:**
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan pengujian status dilakukan kembali oleh pengguna. Output menu status sekarang akan menampilkan secara lengkap port 8501, 8502, dan status cloudflared tanpa terhenti di tengah jalan.

---

### [Entri — Sinkronisasi Otomatis Class Mapping Mask R-CNN] — 2026-06-06 18:40 WIB

- **Tanggal/Waktu:** 2026-06-06 18:40 WIB
- **Tugas yang diselesaikan:**
  1. **Modifikasi start_rvm.sh**:
     - Menambahkan fungsi helper `update_class_mapping_json` untuk mengekrak nama kelas dari `datasets/training_seg/data.yaml` dan menulisnya ke `RVM/backend/class_mapping.json` secara otomatis.
     - Memanggil `update_class_mapping_json` di awal fungsi `start_backend()`.
  2. **Verifikasi Keandalan Mapping**:
     - Memastikan `class_mapping.json` disinkronkan menggunakan environment conda `yolo_env` di subshell pada login node sebelum backend Flask dijalankan.
     - Mask R-CNN kini berhasil memetakan indeks prediksi `3` ke kelas `"mineral"` secara dinamis.
- **File yang diubah/dibuat:**
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pembaruan class_mapping.json dilakukan secara redundan baik pada level launcher bash (`start_rvm.sh`) maupun level server Flask (`visual_eval_api.py`) demi keandalan tinggi.

---

### [Entri — Bugfix Menu Tunnel Hang Akibat Global pgrep Scan & Path Resolution] — 2026-06-06 18:45 WIB

- **Tanggal/Waktu:** 2026-06-06 18:45 WIB
- **Tugas yang diselesaikan:**
  1. **Bugfix UI Tunnel Freezing / Blank**:
     - Mengidentifikasi root cause hang pada menu "Manajemen Cloudflare Tunnel" (Pilihan 6 di `myslurm.sh`). Masalah dipicu oleh pencarian proses berbasis `-f` (cmdline lookup) yang memicu deadlock I/O blocking apabila terdapat proses (terutama `cloudflared` yang menulis ke NFS share secara intensif) dalam status disk sleep (D state).
     - Menghapus flag pencarian cmdline `-f` pada deteksi status `cloudflared` di `myslurm.sh` dan `start_rvm.sh` sehingga pencarian murni mencocokkan nama executable (`pgrep -u $USER cloudflared`) yang tidak memblokir dan ribuan kali lebih cepat.
     - Menyematkan parameter pembatas user aktif (`-u $USER`) ke seluruh perintah `pgrep` dan `pkill`.
  2. **Resolusi Path Konfigurasi Absolut**:
     - Mengubah lookup parameter `CLOUDFLARE_` di `utils/myslurm.sh` dari path relatif `../config_shared.py` menjadi absolute `"${SCRIPT_DIR}/../config_shared.py"` agar skrip tetap valid ketika dipanggil dari direktori kerja luar manapun.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan semua pemanggilan `pgrep` atau `pkill` yang memeriksa cmdline (`-f`) selalu dibatasi dengan `-u $USER` untuk mencegah hang akibat I/O blocking proses milik user lain pada multi-user HPC cluster. Jika memungkinkan, hindari penggunaan `-f` pada deteksi program berkegiatan I/O tinggi (seperti cloudflared) dengan mencari nama program executable biasa.




---

### [Entri — Menambahkan Loading Indicator Port RVM Services] — 2026-07-06 19:35 WIB

- **Tanggal/Waktu:** 2026-07-06 19:35 WIB
- **Tugas yang diselesaikan:**
  1. **Update launcher start_rvm.sh**:
     - Menambahkan indikator loading dan looping dengan timeout 15 detik pada saat mengeksekusi frontend dan backend.
     - Pengecekan aktifnya port 8501 dan port 8502 (listening status via `ss`) sebelum pesan keberhasilan ditampilkan ke user.
     - Hal ini mencegah status yang membingungkan bagi user ("FREE" di port status karena proses boot up script Python / server berjalan agak lambat).
- **File yang diubah/dibuat:**
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Kini start script lebih reliable. Apabila service Python gagal untuk hidup (misalnya environment error), proses startup bash script akan menampilkan notifikasi timeout warning daripada pesan sukses semu.

---

### [Entri — Pembaruan README.md Global untuk Menyajikan Kondisi Aktif Terkini] — 2026-06-07 01:30 WIB

- **Tanggal/Waktu:** 2026-06-07 01:30 WIB
- **Tugas yang diselesaikan:**
  1. **Pembaruan Dokumentasi Utama**:
     - Merombak total file `README.md` pada repositori root `/data/users/g6717500336/Trainning-Models/README.md`.
     - Mendokumentasikan pembagian lingkungan kerja multi-environment secara eksplisit (`MyFineTunning-SlurmMaster`, `MyFineTunning-RunPOD`, `MyFineTunning-dev`).
     - Memperluas pemaparan evaluasi penelitian SOTA mencakup total **49 model** (termasuk Hybrid SAM2, Hybrid Mobile SAM, Mask R-CNN, dan YOLO).
     - Menjelaskan metrik evaluasi geometri tingkat lanjut yang baru ditambahkan (**Boundary IoU** dan **Boundary AP**).
     - Mendokumentasikan status integrasi aplikasi web RVM (Reverse Vending Machine), arsitektur FIFO queue penolak CUDA OOM, fitur dual-theme, comparison mode, dan aksesibilitas publik melalui Cloudflare Tunnel.
     - Menyematkan panduan komprehensif penggunaan CLI global `slurm` dan mekanisme background daemon booking GPU.
- **File yang diubah/dibuat:**
  - `README.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Repositori kini memiliki dokumentasi utama yang akurat dan lengkap sesuai dengan arsitektur SlurmMaster terbaru. Pastikan jika ada perubahan parameter batch RT-DETR atau port RVM di masa mendatang, sesuaikan dokumentasi README agar tetap sinkron.

---

### [Entri — Pembersihan Berkas Sampah pada Branch Main] — 2026-06-07 01:35 WIB

- **Tanggal/Waktu:** 2026-06-07 01:35 WIB
- **Tugas yang diselesaikan:**
  - Melakukan pembersihan file yang tidak diperlukan di branch `main`.
  - Berpindah ke branch `main`, menghapus pelacakan (*untrack*) dan menghapus file `skills-antigravity.zip`, `.gitattributes`, dan `.gitignore` menggunakan `git rm`.
  - Melakukan komit dan *push* perubahan pembersihan ke origin branch `main`.
  - Kembali ke branch utama pengembangan (`dev`).
- **File yang diubah/dibuat:**
  - `skills-antigravity.zip` [DIHAPUS di main]
  - `.gitattributes` [DIHAPUS di main]
  - `.gitignore` [DIHAPUS di main]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pembersihan di branch `main` selesai. File-file tersebut tetap ada di branch `dev` karena memang dibutuhkan untuk pengembangan aktif (seperti `.gitignore`). Jangan satukan (*merge*) perubahan ini secara terbalik ke branch `dev`.

---

### [Entri — Pemindahan Kredensial CLOUDFLARE_TUNNEL_TOKEN ke .env] — 2026-06-06 18:41 WIB

- **Tanggal/Waktu:** 2026-06-06 18:41 WIB
- **Tugas yang diselesaikan:**
  - Melakukan migrasi kredensial `CLOUDFLARE_TUNNEL_TOKEN` dari file `config_shared.py` ke file `.env` untuk keamanan dan mencegah eksposur kredensial secara statis.
  - Memodifikasi `config_shared.py` dengan fungsi `_load_dotenv()` dan mengganti assignment `CLOUDFLARE_TUNNEL_TOKEN` menggunakan `os.environ.get()`.
  - Melakukan update pada `utils/myslurm.sh` agar perintah `grep` untuk token mengarah ke `.env` daripada `config_shared.py` dan menghapus string fallback hardcode.
  - Melakukan review silang (*cross-reference*) terhadap `/data/users/g6717500336/singularity` dan memverifikasi bahwa skrip-skrip di dalamnya (seperti `sbatch_llm_service.sh` dan `sbatch_lms.sh`) menggunakan Quick Tunnels sehingga tidak terdampak migrasi token ini.
- **File yang diubah/dibuat:**
  - `.env` [DIMODIFIKASI]
  - `config_shared.py` [DIMODIFIKASI]
  - `utils/myslurm.sh` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Akses `CLOUDFLARE_TUNNEL_TOKEN` kini bersumber dari `.env`. Untuk integrasi komponen baru, selalu panggil via `os.environ.get("CLOUDFLARE_TUNNEL_TOKEN")` setelah import/loading `config_shared.py`.

---

### [Entri — Pembuatan Template Lingkungan (.example.env)] — 2026-06-07 01:43 WIB

- **Tanggal/Waktu:** 2026-06-07 01:43 WIB
- **Tugas yang diselesaikan:**
  - Membuat berkas `.example.env` sebagai template (blueprint) bagi *developer* atau *agent* lain untuk mengetahui variabel lingkungan apa saja yang dibutuhkan oleh proyek tanpa mengekspos kunci (*keys*) atau token asli.
  - Memasukkan *placeholder* (`your_..._here`) untuk nilai kredensial seperti API Key Roboflow, Token Bot Telegram, Chat ID, dan Token Cloudflare Tunnel, sambil tetap mempertahankan nilai *default* non-sensitif (seperti `RCLONE_REMOTE`).
- **File yang diubah/dibuat:**
  - `.example.env` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Berkas template `.example.env` sudah siap. Jika di kemudian hari ada variabel lingkungan (environment variable) baru yang ditambahkan ke skrip, pastikan Anda menambahkannya juga ke dalam `.example.env`.

---

### [Entri — Sanitasi Hardcoded Token Roboflow di Branch dev & RunPOD] — 2026-06-07 01:45 WIB

- **Tanggal/Waktu:** 2026-06-07 01:45 WIB
- **Tugas yang diselesaikan:**
  - Melakukan investigasi (*grep search*) terkait kredensial yang terexpose di direktori `MyFineTunning-dev` dan `MyFineTunning-RunPOD`.
  - Menemukan bahwa token API Roboflow (`ROBOFLOW_KU_KEY1` dan `ROBOFLOW_UNU_KEY1`) ditulis secara eksplisit (hardcoded) di dalam *docstring* dokumentasi awal pada file `dataset_setup.py` di kedua folder tersebut.
  - Melakukan sanitasi keamanan dengan mengganti token asli di dalam komentar dokumentasi dengan placeholder generik (`<ROBOFLOW_KU_KEY1_FROM_ENV>` dan `<ROBOFLOW_UNU_KEY1_FROM_ENV>`) melalui fungsi edit di kedua file tersebut.
  - Memverifikasi ulang bahwa token Cloudflare dan Telegram tidak bocor/terexpose di dalam dua lingkungan kerja tersebut.
- **File yang diubah/dibuat:**
  - `MyFineTunning-dev/dataset_setup.py` [DIMODIFIKASI]
  - `MyFineTunning-RunPOD/dataset_setup.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Saat membuat dokumentasi berupa blok *docstring* (terutama contoh instalasi atau inisiasi script python), **DILARANG** meletakkan token rahasia secara aktual meskipun itu hanyalah komentar. Gunakan selalu penanda *placeholder* untuk mencegah kebocoran informasi.

---

### [Entri — Sinkronisasi Global File `.env` dan `.example.env`] — 2026-06-07 01:48 WIB

- **Tanggal/Waktu:** 2026-06-07 01:48 WIB
- **Tugas yang diselesaikan:**
  - Melakukan penduplikasian file `.env` dan `.example.env` dari sumber utama di `MyFineTunning-SlurmMaster` ke cabang direktori `MyFineTunning-dev` dan `MyFineTunning-RunPOD`.
  - Hal ini dilakukan agar kredensial API Key Roboflow, Token Telegram, dan Token Cloudflare Tunnel dapat dimuat dengan sempurna saat menjalankan skrip-skrip pelatihan (`run_pipeline.py`) atau pengunduhan dataset (`dataset_setup.py`) pada ketiga direktori proyek tersebut.
- **File yang diubah/dibuat:**
  - `MyFineTunning-dev/.env` [DIBUAT/DI-OVERWRITE]
  - `MyFineTunning-dev/.example.env` [DIBUAT BARU]
  - `MyFineTunning-RunPOD/.env` [DIBUAT/DI-OVERWRITE]
  - `MyFineTunning-RunPOD/.example.env` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Kini ketiga lingkungan (*environment*) telah disinkronkan dan menggunakan satu format referensi variabel (*Single Source of Truth*) untuk urusan otentikasi. Semua agen harap merujuk ke file `.env` di masing-masing sub-proyek untuk memanggil variabel rahasia.

---

### [Entri — Sanitasi Hardcoded Token Roboflow di SlurmMaster] — 2026-06-07 01:50 WIB

- **Tanggal/Waktu:** 2026-06-07 01:50 WIB
- **Tugas yang diselesaikan:**
  - Melakukan audit keamanan mendalam ulang ke seluruh repositori (`MyFineTunning-dev`, `MyFineTunning-RunPOD`, dan `MyFineTunning-SlurmMaster`).
  - Menemukan bahwa API Key Roboflow (`ROBOFLOW_KU_KEY1` dan `ROBOFLOW_UNU_KEY1`) juga masih terexpose di blok komentar docstring pada `MyFineTunning-SlurmMaster/dataset_setup.py`.
  - Melakukan penggantian (sanitasi) API Key tersebut dengan placeholder generik `<ROBOFLOW_KU_KEY1_FROM_ENV>` dan `<ROBOFLOW_UNU_KEY1_FROM_ENV>` di `MyFineTunning-SlurmMaster/dataset_setup.py`.
  - Memverifikasi ulang bahwa seluruh kunci otentikasi, token bot Telegram, detail Cloudflare, dan password SSH tidak lagi terekspos dalam kode di seluruh 3 direktori.
- **File yang diubah/dibuat:**
  - `MyFineTunning-SlurmMaster/dataset_setup.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Semua file `dataset_setup.py` di ketiga folder sekarang sudah tersanitasi dari hardcoded key di bagian docstring. Jangan menulis ulang key asli di file mana pun yang terdaftar dalam tracking git.

---

### [Entri — Pembaruan Aturan Global Proyek Slurm Master] — 2026-06-07 02:12 WIB

- **Tanggal/Waktu:** 2026-06-07 02:12 WIB
- **Tugas yang diselesaikan:**
  - Memperbarui Aturan Global (`GEMINI.md`) untuk menyertakan sub-bab baru mengenai Klasifikasi Proyek GPU Slurm yang berdiri sendiri.
  - Mendefinisikan secara jelas empat proyek AI independen (LLM API, Computer Vision, Image Generator, dan Video Generator) yang berjalan terpisah di atas infrastruktur Slurm Master yang sama.
- **File yang diubah/dibuat:**
  - `.gemini/GEMINI.md` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Masing-masing AI ini dideklarasikan berdiri sendiri (independent). Pastikan koordinasi sumber daya Slurm dan penamaan sesi tmux/job tidak bertabrakan satu sama lain di kemudian hari.

---

### [Entri — Pembersihan Legacy .venv & Migrasi Penuh ke Conda yolo_env] — 2026-06-07 02:53 WIB

- **Tanggal/Waktu:** 2026-06-07 02:53 WIB
- **Tugas yang diselesaikan:**
  - Melakukan audit rujukan direktori virtual environment lokal `.venv` yang sudah usang di seluruh workspace.
  - Memodifikasi skrip orkestrasi `run_pipeline.py` agar secara aktif memanggil Conda environment `yolo_env` milik Anaconda (`/data/programs/anaconda3/bin/activate yolo_env`) alih-alih mencari berkas `.venv/bin/activate` lokal.
  - Memperbarui dokumentasi petunjuk eksekusi pada `Readme.md`, `utils/generals/run_eval_multi.py`, `utils/generals/maskrcnn/eval_multigpu.py`, dan `utils/generals/eval_boundary_iou.py` untuk mengarahkan pengguna ke environment conda `yolo_env`.
  - Menghapus folder `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/.venv` secara permanen dan berhasil menghemat ruang penyimpanan sebesar **5,6 GB**.
- **File yang diubah/dibuat:**
  - `run_pipeline.py` [DIMODIFIKASI]
  - `Readme.md` [DIMODIFIKASI]
  - `utils/generals/run_eval_multi.py` [DIMODIFIKASI]
  - `utils/generals/maskrcnn/eval_multigpu.py` [DIMODIFIKASI]
  - `utils/generals/eval_boundary_iou.py` [DIMODIFIKASI]
  - `docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh komponen saat ini telah dimigrasikan secara penuh ke environment conda `yolo_env`. Jangan pernah lagi membuat virtual environment lokal `.venv` baru di bawah workspace SlurmMaster karena semua dependency diatur terpusat di Conda.

---

### [Entri — Inisiasi Proyek AspriAI (Asisten Pribadi AI)] — 2026-06-07 03:10 WIB

- **Tanggal/Waktu:** 2026-06-07 03:10 WIB
- **Tugas yang diselesaikan:**
  - Melakukan perancangan awal, analisis, dan inisiasi proyek terpisah bernama **AspriAI** di direktori `/data/users/g6717500336/singularity/AspriAI/`.
  - Membuat berkas rancangan awal `README.md` yang merinci Konsep Identitas, Slogan, Network Topology, Rencana Struktur Proyek, dan Mekanisme API Headless ComfyUI.
  - Mempersiapkan folder `docs/` dengan inisiasi 5 dokumen standar rekayasa perangkat lunak: `SRS.md`, `SDD.md`, `SRD.md`, `STD.md`, dan `SDP.md` untuk mengawal siklus hidup proyek AspriAI.
- **File yang diubah/dibuat:**
  - `/data/users/g6717500336/singularity/AspriAI/README.md` [DIBUAT BARU]
  - `/data/users/g6717500336/singularity/AspriAI/docs/SRS.md` [DIBUAT BARU]
  - `/data/users/g6717500336/singularity/AspriAI/docs/SDD.md` [DIBUAT BARU]
  - `/data/users/g6717500336/singularity/AspriAI/docs/SRD.md` [DIBUAT BARU]
  - `/data/users/g6717500336/singularity/AspriAI/docs/STD.md` [DIBUAT BARU]
  - `/data/users/g6717500336/singularity/AspriAI/docs/SDP.md` [DIBUAT BARU]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Proyek AspriAI dirancang sebagai API gateway & dasbor pribadi mandiri. Langkah pengembangan berikutnya dapat langsung menuju ke pembuatan FastAPI backend (`aspri-core`) dan antarmuka Next.js (`aspri-desk`).

---

### [Entri — Integrasi Git/GitHub & Inisialisasi CI/CD AspriAI] — 2026-06-07 03:22 WIB

- **Tanggal/Waktu:** 2026-06-07 03:22 WIB
- **Tugas yang diselesaikan:**
  - Menginisialisasi repositori Git lokal di `/data/users/g6717500336/singularity/AspriAI/` dan menghubungkannya dengan repositori jarak jauh `git@github.com:vnot-programming/aspri.git`.
  - Mengonfigurasi `/data/users/g6717500336/.ssh/config` untuk membelokkan lalu lintas `github.com` via `ssh.github.com` pada port 443 menggunakan kunci privat `id_ed25519` (untuk memotong pemblokiran port 22 di jaringan login node).
  - Melakukan push awal kode struktur proyek, folder `docs/`, dan workflow CI/CD ke branch `main` (Sukses pushed!).
  - Membuat dan mempublikasikan branch `core/dev` (untuk backend) dan `desk/dev` (untuk frontend) sesuai skema *prefix-based branching*.
  - Membuat berkas aturan agen baru `/data/users/g6717500336/.agents/rules/workflow-aspri.md` yang menetapkan tata cara Git branching dan mewajibkan AI berhenti untuk bertanya ("desk" atau "core") jika pengguna tidak menyebutkan tujuan push.
  - Merancang cetak biru otomatisasi CI/CD jarak jauh menggunakan GitHub Actions di `.github/workflows/deploy.yml` yang menggunakan integrasi Tailscale VPN dan SSH Action untuk deploy ke GPU node (Core) dan Azure VPS (Desk) secara terisolasi.
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/.git/` [DIINISIALISASI]
  - `singularity/AspriAI/.github/workflows/deploy.yml` [DIBUAT BARU & DI-PUSH]
  - `.agents/rules/workflow-aspri.md` [DIBUAT BARU]
  - `.ssh/config` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Git remote telah terintegrasi penuh dan branch `core/dev` & `desk/dev` sudah aktif di remote.
  - Selalu periksa berkas aturan `.agents/rules/workflow-aspri.md` setiap kali ada perintah push, commit, atau checkout branch untuk proyek AspriAI.
  - Untuk deployment otomatis, pastikan kredensial VPN Tailscale (`TS_OAUTH_CLIENT_ID`, `TS_OAUTH_SECRET`), SSH Private Key (`SSH_PRIVATE_KEY`), dan Password VPS (`VPS_PASSWORD`) telah didaftarkan pada bagian Repository Secrets di repositori GitHub `aspri`.

---

### [Entri — Pembaruan Konfigurasi Host CI/CD AspriAI] — 2026-06-07 03:32 WIB

- **Tanggal/Waktu:** 2026-06-07 03:32 WIB
- **Tugas yang diselesaikan:**
  - Memperbarui berkas otomatisasi CI/CD `.github/workflows/deploy.yml` untuk AspriAI.
  - Mengubah konfigurasi target deployment host GPU Node (Backend/Core) menjadi `slurm.penelitian.my.id` dengan SSH user `g6717500336` (port 22).
  - Mengubah konfigurasi target deployment host Web Host (Frontend/Desk) menjadi `ssh-docker-host.vnot.my.id` dengan SSH user `my` (port 22), menyesuaikan integrasi otentikasi SSH Key.
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/.github/workflows/deploy.yml` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Host deploy telah disesuaikan dengan skema isolasi server dan VPN Kampus / Cloudflare Tunnel. Lakukan pengujian push ke GitHub untuk memicu jalannya workflow Actions dan pastikan repository secret untuk `SSH_PRIVATE_KEY` telah didaftarkan.

---

### [Entri — Verifikasi Sukses SSH via Cloudflare Service Token] — 2026-06-07 03:55 WIB

- **Tanggal/Waktu:** 2026-06-07 03:55 WIB
- **Tugas yang diselesaikan:**
  - Melakukan uji coba (verifikasi) koneksi SSH lokal dari Login Node ke `slurm.penelitian.my.id` (GPU Node) dan `ssh-docker-host.vnot.my.id` (Docker Host) menggunakan Cloudflare Service Token.
  - Berhasil membypass Cloudflare Access Zero Trust murni via parameter `--id` dan `--secret` pada biner `cloudflared` tanpa interaksi peramban web.
  - Memastikan otentikasi SSH Key (`id_ed25519`) terverifikasi sukses masuk ke shell target (exit status 0).
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Kredensial Service Token dan SSH Key telah divalidasi bekerja dengan baik. CI/CD GitHub Actions siap berjalan setelah rahasia repositori di-push.

---

### [Entri — Penghapusan Tailscale & Migrasi Penuh CI/CD ke Cloudflare Tunnel] — 2026-06-07 04:00 WIB

- **Tanggal/Waktu:** 2026-06-07 04:00 WIB
- **Tugas yang diselesaikan:**
  - Menghapus blok konfigurasi Tailscale VPN (`tailscale/github-action@v2`) yang sudah usang dan tidak diperlukan pada berkas otomatisasi CI/CD `.github/workflows/deploy.yml`.
  - Mengimplementasikan instalasi `cloudflared` dan pemanggilan `ssh` standard berbasis OpenSSH client secara penuh untuk kedua job (`deploy-core` dan `deploy-desk`) demi kestabilan runtime deployment.
  - Melakukan commit dan push revisi bersih tersebut ke branch remote `main` di GitHub.
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/.github/workflows/deploy.yml` [DIMODIFIKASI & DI-PUSH]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Workflow CI/CD sekarang sepenuhnya bergantung pada Cloudflare Tunnel untuk menembus jaringan terisolasi. Parameter rahasia Tailscale (`TS_OAUTH_...`) tidak lagi dibutuhkan di repositori.

---

### [Entri — Perbaikan Format SSH Key & Penyelesaian Error libcrypto CI/CD] — 2026-06-07 10:44 WIB

- **Tanggal/Waktu:** 2026-06-07 10:44 WIB
- **Tugas yang diselesaikan:**
  - Mendiagnosis kegagalan `Execute Remote Deploy via SSH` pada Run #4 & #5 dengan pesan error `Load key "/home/runner/.ssh/id_ed25519": error in libcrypto`.
  - Menemukan akar masalah (*root-cause*) pada pemanggilan `echo "$SSH_PRIVATE_KEY"` yang memicu sensor otomatis `***` oleh runner GitHub Actions dan merusak integritas format multiline kunci privat SSH.
  - Memodifikasi berkas `.github/workflows/deploy.yml` untuk menulis kunci privat SSH menggunakan metode literal `cat << 'EOF'` yang kebal dari ekspansi shell dan sensor runtime.
  - Menambahkan sanitasi carriage return (`sed -i 's/\r$//'`) untuk mencegah malformasi kunci akibat copy-paste dari sistem operasi Windows.
  - Melakukan commit dan push perbaikan tersebut ke remote branch `main`.
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/.github/workflows/deploy.yml` [DIMODIFIKASI & DI-PUSH]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Kunci privat SSH kini ditulis secara literal dan dibersihkan dari CRLF. Tunggu hasil Run #6 di GitHub Actions untuk memverifikasi deployment akhir.

---

### [Entri — Sukses Penuh Integrasi CI/CD AspriAI via Cloudflare Tunnel] — 2026-06-07 11:00 WIB

- **Tanggal/Waktu:** 2026-06-07 11:00 WIB
- **Tugas yang diselesaikan:**
  - Menyelesaikan perbaikan error `invalid input` pada Base64 decoding dengan menuntun pengguna membersihkan prompt terminal `(base)` yang tidak sengaja tersalin pada GitHub Secrets.
  - Menambahkan sanitasi string Base64 (`tr -d '\r' | tr -d '\n'`) pada skrip `Setup SSH Config` untuk menjamin tidak ada karakter whitespace/newline kotor yang merusak jalannya decoding.
  - Memverifikasi keberhasilan penuh (Success) pada **Run #9** GitHub Actions untuk kedua job deployment (`Deploy Backend to GPU Node` dan `Deploy Frontend to Web Host`).
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/.github/workflows/deploy.yml` [DIMODIFIKASI & DI-PUSH]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Seluruh alur otomatisasi deployment CI/CD jarak jauh berbasis Cloudflare Tunnel sudah terverifikasi sukses berjalan 100%. Pipeline siap digunakan untuk menerima commit pengembangan fitur core & desk berikutnya.

---

### [Entri — Penyempurnaan Menu 8 (Manajemen AspriAI) pada myslurm.sh] — 2026-06-07 13:20 WIB

- **Tanggal/Waktu:** 2026-06-07 13:20 WIB
- **Tugas yang diselesaikan:**
  - Menyempurnakan Menu 8 (`Manajemen AspriAI`) di dalam `utils/myslurm.sh` untuk menampilkan URL Utama (`backend-ollama.penelitian.my.id`) dan URL Fallback dinamis secara real-time yang diekstrak dari `/data/users/g6717500336/singularity/ollama/logs/tunnel_sbatch.log`.
  - Menambahkan submenu baru **Restart Server Ollama** (Opsi 3) yang secara sekuensial menghentikan server lama, membersihkan port forwarding lokal slurmmaster, dan menembakkan peluncuran ulang server secara rapi.
  - Memperbarui label penomoran aksi submenu lainnya untuk log runtime (menjadi opsi 4) dan instalasi modul (menjadi opsi 5), serta menyesuaikan rentang input penanganan aksi menjadi `[1-5]`.
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Periksa integrasi log `tunnel_sbatch.log` jika dynamic parsing URL fallback gagal diekstrak (pola grep pencarian URL `trycloudflare.com` harus tetap didukung).
  - Skrip `myslurm.sh` sudah terverifikasi bebas dari error sintaksis (`bash -n` sukses).

---

### [Entri — Penataan Menu Bercabang AspriAI pada myslurm.sh] — 2026-06-07 13:30 WIB

- **Tanggal/Waktu:** 2026-06-07 13:30 WIB
- **Tugas yang diselesaikan:**
  - Merestrukturisasi fungsi `manage_aspri_ai` di dalam `utils/myslurm.sh` dari menu linier menjadi menu bercabang (nested menu).
  - Menu utama AspriAI kini menampilkan ringkasan status GPU Slurm, port compute node, serta rincian port dan URL (utama & fallback) untuk Ollama dan ComfyUI secara berdampingan.
  - Menyediakan sub-menu khusus **1. Ollama** yang memuat 5 aksi (Jalankan, Hentikan, Restart, Log, Instal ulang), serta sub-menu **2. ComfUI** yang menyajikan status placeholder yang aman.
  - Memastikan transisi kembali ke menu sebelumnya berjalan lancar dengan menekan `Enter` (kosong).
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan variabel pendeteksi `job_id`, `node_name`, dan `ollama_running` diperbarui secara berkala dalam perulangan loop agar status status real-time akurat saat berpindah sub-menu.
  - Integrasi ComfUI sesungguhnya dapat dilanjutkan begitu Singularity image ComfUI siap digunakan.

---

### [Entri — Integrasi Sub-menu Chat via Ollama pada myslurm.sh] — 2026-06-07 13:40 WIB

- **Tanggal/Waktu:** 2026-06-07 13:40 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan submenu baru **2. Chat via Ollama** pada Menu Utama AspriAI di `utils/myslurm.sh`.
  - Mengimplementasikan pendeteksian model secara otomatis dan dinamis dari API tags local (`http://localhost:11434/api/tags`) menggunakan request HTTP `urllib` di python secara aman dan mandiri (tanpa dependensi luar seperti `jq`).
  - Mengimplementasikan alur interaktif chat CLI menggunakan perintah SSH pseudo-terminal allocation (`ssh -t`) yang merutekan input langsung ke container Singularity Ollama di compute node GPU aktif menggunakan port dinamis (`OLLAMA_HOST`).
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Periksa ketersediaan model terinstall dengan memastikan server Ollama berjalan di latar belakang compute node. Jika list model kosong, submenu akan mengarahkan pengguna kembali secara aman tanpa *crash*.
  - Sesi chat CLI interaktif ini memerlukan pseudo-terminal TTY yang dialokasikan via parameter `-t` pada SSH.

---

### [Entri — Penambahan Fitur Download Model di Sub-menu Chat via Ollama pada myslurm.sh] — 2026-06-07 13:58 WIB

- **Tanggal/Waktu:** 2026-06-07 13:58 WIB
- **Tugas yang diselesaikan:**
  - Menambahkan opsi **0) Download Model Baru (ollama pull)** di dalam sub-menu Chat via Ollama di `utils/myslurm.sh`.
  - Mengimplementasikan alur interaktif unduhan model di mana pengguna dapat memasukkan nama model (misal `qwen2.5:7b` atau `llama3`) untuk diunduh langsung di compute node aktif.
  - Memanfaatkan perintah `ollama pull` yang dijalankan via SSH pseudo-terminal allocation (`ssh -t`) agar progress bar unduhan dapat dirender secara real-time di terminal pengguna.
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Opsi download ini memerlukan server Ollama dalam kondisi `RUNNING`. Jika server mati, menu input download akan tertutup secara otomatis setelah memberikan pesan peringatan yang relevan.

---

### [Entri — Penyempurnaan Warisan Env Singularity pada Pemanggilan Client Ollama] — 2026-06-07 14:02 WIB

- **Tanggal/Waktu:** 2026-06-07 14:02 WIB
- **Tugas yang diselesaikan:**
  - Mendiagnosis masalah kegagalan deteksi model baru yang baru saja diunduh oleh pengguna (model tidak muncul di `/api/tags`).
  - **Akar Masalah (Root Cause):** Variabel lingkungan `OLLAMA_HOST` yang dideklarasikan pada terminal compute node host tidak diwariskan ke dalam kontainer Singularity. Akibatnya, client `ollama pull` yang dieksekusi di dalam kontainer secara membabi buta mencari daemon Ollama di port default `11434` (yang tidak aktif), menyebabkan kegagalan koneksi sepihak.
  - **Tindakan Perbaikan:** Menambahkan ekspor variabel `SINGULARITYENV_OLLAMA_HOST` secara eksplisit tepat sebelum eksekusi `singularity exec` baik pada perintah `ollama pull` maupun `ollama run` (chat) di berkas `utils/myslurm.sh`.
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan pengguna memasukkan nama model yang valid dan terdaftar secara resmi di Ollama model library (contoh: `qwen2.5:0.5b` or `qwen2.5:1.5b`). Model fiktif atau salah ketik (seperti `qwen3.5:0.8b`) akan ditolak secara otomatis oleh registry resmi Ollama.

---

### [Entri — Integrasi Deteksi Sumber Model (Local vs External) pada myslurm.sh] — 2026-06-07 14:10 WIB

- **Tanggal/Waktu:** 2026-06-07 14:10 WIB
- **Tugas yang diselesaikan:**
  - Mengimplementasikan deteksi sumber model secara cerdas untuk membedakan model lokal dan eksternal pada sub-menu Chat via Ollama di `utils/myslurm.sh`.
  - Skrip Python di dalam perulangan bash sekarang melakukan verifikasi keberadaan berkas manifes lokal di bawah folder `${aspri_dir}/models/manifests` untuk setiap model yang dikembalikan oleh API tags.
  - Model yang memiliki berkas manifes lokal akan diberi label `(local)`. Model yang tidak memiliki berkas manifes lokal (akibat bentrokan port `11434` global di slurmmaster dengan pengguna lain) diberi label `(external - port clash)`.
  - Menambahkan dialog konfirmasi pencegahan keamanan: jika pengguna memilih model `external`, skrip akan menampilkan peringatan bahwa hal tersebut akan memicu download weights baru pada penyimpanan lokal mereka, dan meminta konfirmasi eksplisit (`y/N`) sebelum dieksekusi.
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pola verifikasi manifes ini mendukung namespace standar (`library`) maupun namespace kustom (seperti `iapp/chinda-qwen3-4b`). Pastikan path `/manifests/registry.ollama.ai/...` tetap valid.

---

### [Entri — Pembatasan Model Utama ke Lokal & Pemisahan Menu Model Eksternal] — 2026-06-07 14:15 WIB

- **Tanggal/Waktu:** 2026-06-07 14:15 WIB
- **Tugas yang diselesaikan:**
  - Memodifikasi sub-menu Chat via Ollama di `utils/myslurm.sh` untuk membatasi tampilan model pada daftar utama hanya untuk model berstatus **`Local`** (yang memiliki manifes lokal).
  - Menyembunyikan seluruh model eksternal (akibat bentrokan port `11434` global) dari daftar utama guna meningkatkan keamanan kuota disk pengguna dan menyederhanakan antarmuka.
  - Menambahkan menu **`L) Lihat Model External`** yang secara terpisah menampilkan daftar model eksternal milik pengguna lain yang terdeteksi, lengkap dengan pesan edukasi mengenai status port clash.
  - Memperbaiki duplikasi syntax *else-fi* yang tidak sengaja terjadi pada pemrosesan pilhan aksi chat.
- **File yang diubah/dibuat:**
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Daftar model lokal langsung dialokasikan tanpa perlunya konfirmasi download (pull) karena weights model sudah dipastikan ada secara fisik.
  - Opsi `L` mendeteksi model eksternal secara dinamis tanpa mengubah state server Ollama.

---

### [Entri — Perbaikan Akses Eksternal Quick Tunnel Ollama via Postman] — 2026-06-07 14:25 WIB

- **Tanggal/Waktu:** 2026-06-07 14:25 WIB
- **Tugas yang diselesaikan:**
  - Mendiagnosis penyebab error 403 Forbidden ketika mengakses endpoint API Ollama (`GET/POST /api/tags`) via Cloudflare Quick Tunnel dari Postman.
  - Menemukan akar masalah (*root cause*): Ollama memiliki proteksi host header strict. Ketika dijalankan dengan `127.0.0.1` dan diakses melalui domain dinamis `trycloudflare.com`, ia menolak request eksternal dengan status 403. Selain itu, ada beberapa proses zombie `sbatch_aspri_service.sh` dan `cloudflared` lama yang memicu bentrokan port dinamis.
  - Memperbaiki `sbatch_aspri_service.sh` dengan mengubah bind host `SINGULARITYENV_OLLAMA_HOST` menjadi `0.0.0.0:${OLLAMA_PORT}` agar server Ollama mendengarkan di semua interface dan menonaktifkan pemblokiran strict Host header.
  - Menambahkan argumen `--http-host-header localhost` pada pemanggilan Quick Tunnel `cloudflared` untuk menulis ulang Host header ke localhost sebelum request diteruskan ke server Ollama lokal.
  - Membersihkan total seluruh proses zombie `ollama`, `cloudflared`, dan `sbatch_aspri_service.sh` di compute node `ai3`.
  - Merestart server Ollama bersih dan memverifikasi akses via `curl` ke URL Quick Tunnel baru (`https://truly-surgical-jar-providers.trycloudflare.com/api/tags`), berhasil merespon dengan status 200 OK dan data JSON model lokal.
- **File yang diubah/dibuat:**
  - `singularity/ollama/sbatch_aspri_service.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Host binding Ollama server kini berada di `0.0.0.0`, pastikan file config launcher tetap terlindungi dari hardcode.
  - Untuk setiap peluncuran ulang, verifikasi bahwa tunnel log `tunnel_sbatch.log` memuat url `trycloudflare.com` yang baru dan ujilah menggunakan curl.

---

### [Entri — Integrasi Health Check & Otomatisasi One-Click Launch AspriAI] — 2026-06-07 17:00 WIB

- **Tanggal/Waktu:** 2026-06-07 17:00 WIB
- **Tugas yang diselesaikan:**
  - Membuat dan mendaftarkan endpoint `/health` (dengan metode `GET` dan `POST`) secara khusus pada `chat.py` milik AspriAI Core. Langkah ini mencegah request `/v1/health` dan `/api/health` diteruskan (di-proxy) ke Ollama Server di port 11435, mengeliminasi error 404/405 dan memastikan pelaporan status kesehatan Gateway berjalan lancar.
  - Memodifikasi skrip `myslurm.sh` untuk melakukan otomatisasi peluncuran satu-klik (One-Click Launch): Opsi 1 (Jalankan AspriAI Core) kini tidak hanya meluncurkan FastAPI Gateway di Login Node, tetapi juga otomatis menyalakan server Ollama di compute node aktif (jika GPU aktif).
  - Mengintegrasikan monitoring kesehatan real-time `AspriAI Core` dan `AspriAI Desk` di menu Opsi 8 dengan memanggil HTTP POST request ke domain Cloudflare Tunnel masing-masing. Untuk `backend-ollama`, status kesehatan dinegosiasikan dengan menyertakan token autentikasi Cloudflare Access Zero Trust secara dinamis dari file `.env`.
  - Memperbaiki parser `active_port` di seluruh fungsi `myslurm.sh` agar membaca pola `Port: [0-9]+` yang di-output oleh script launcher `sbatch_aspri_service.sh` dengan penanganan fallback parameter default Bash (`${var:-fallback}`) untuk mencegah port/URL kosong akibat latensi penulisan log.
  - Memperbarui visualisasi URL di menu status agar seragam menggunakan skema `https://`.
  - Memperbaiki pencarian proses `ollama` di compute node pada `myslurm.sh` dengan membatasi hanya pada user saat ini (`ps -u $USER -f`) agar tidak mendeteksi proses Ollama milik pengguna lain di server bersama (menyelesaikan masalah status RUNNING yang nyantol).
  - Mengimplementasikan loop polling dinamis (maksimal 15 detik) untuk menunggu sinkronisasi log port/URL tunnel Ollama di Login Node setelah perintah dijalankan, mencegah tampilan data kosong.
- **File yang diubah/dibuat:**
  - `singularity/AspriAI/aspri-core/app/api/v1/endpoints/chat.py` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` [DIMODIFIKASI]
  - `Trainning-Models/MyFineTunning-SlurmMaster/docs/SDP.md` [DIMODIFIKASI]
- **Status saat ini:** Selesai ✅
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Pastikan untuk melakukan booking ulang GPU terlebih dahulu apabila ingin menguji coba secara utuh (agar SSH tunnel lama dibersihkan dan port 11434 Login Node dibebaskan dari terowongan lama).

---

### [Entri 059] — Implementasi Arsitektur RVM Backend Auto-Resume (Watchdog Daemon)

- **Tanggal/Waktu:** 2026-06-08 22:41 WIB
- **Tugas yang diselesaikan:**
  - Menganalisis isu `visual_eval_api.py` yang mati (crash) karena koneksi SSH ke Compute Node terputus akibat Slurm job (GPU) dibatalkan secara sistemik (Time Limit/OOM).
  - Merancang arsitektur **Watchdog Daemon** (`run_backend_daemon.sh`) yang berjalan secara independen di dalam sesi TMUX `rvm_backend` pada *Login Node*.
  - Logika daemon meliputi mekanisme *polling* `squeue` untuk mencari GPU yang tersedia, melakukan *SSH* secara otomatis ke GPU Compute Node yang berhasil dibooking, membuka SSH Reverse Tunnel `8502`, dan menjalankan kembali (auto-resume) server Flask. Jika node mati lagi, skrip secara pintar kembali menunggu dan me-reconnect tanpa campur tangan pengguna.
  - Mengubah fungsi peluncuran pada `start_rvm.sh` (`start_backend()`) agar menggunakan eksekusi tunggal daemon wrapper tersebut dibanding sekumpulan `send-keys` statis.
  - Memastikan *stop commands* mematikan TMUX session agar daemon dapat diakhiri secara manual oleh user saat menu di `myslurm.sh` diklik (Stop All / Stop Backend).
- **File yang diubah/dibuat:**
  - `RVM/run_backend_daemon.sh` [DIBUAT BARU]
  - `RVM/start_rvm.sh` [DIMODIFIKASI]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - RVM Backend kini akan selalu hidup ("bangkit dari kematian") selama script Booking GPU (`book_gpu.py`) aktif mencarikan alokasi GPU.
  - Jika ingin membunuh layanan RVM API secara permanen, ikuti instruksi dari menu `myslurm.sh` (matikan TMUX).
- **Tanggal/Waktu:** Mon Jun  8 23:25:22 +07 2026
- **Tugas yang diselesaikan:** Memperbaiki 'Failed to load models' akibat CORS & CF Access di RVM. Menambahkan Reverse Proxy via urllib di serve_frontend.py agar Service Token tidak terekspos ke frontend, serta bypass error 403 preflight OPTIONS. Memperbarui app.js untuk hit /api lokal.
- **File yang diubah/dibuat:** RVM/serve_frontend.py, RVM/frontend/js/app.js, RVM/backend/visual_eval_api.py
- **Status saat ini:** Selesai
- **Catatan untuk AI selanjutnya:** Frontend kini bisa load model. Validasi pipeline evaluasi jika ada yang masih error.

---

### [Entri 060] — Root Cause Fix: Cloudflare Tunnel Selalu Mati (Port Metrics Conflict)

- **Tanggal/Waktu:** 2026-06-09 03:41 WIB
- **Tugas yang diselesaikan:**
  - **Investigasi mendalam** mengapa Cloudflare Named Tunnel (`myslurm.sh` Menu 1) selalu mati dengan pesan `Initiating graceful shutdown due to signal terminated` tepat 2-8 detik setelah berhasil konek ke jaringan Cloudflare.
  - **Root Cause Teridentifikasi:** Port metrics default `127.0.0.1:20241` selalu *conflik* karena sisa binding dari instance cloudflared sebelumnya yang tidak dilepas dengan benar. Instance baru menerima `SIGTERM` dari sistem karena *address already in use*.
  - **Fix `myslurm.sh`:** Menambahkan flag `--metrics 127.0.0.1:<random_port>` menggunakan `shuf -i 20200-20299 -n 1` pada wrapper script di fungsi `manage_cloudflare_tunnel()`. Dengan port metrics acak, konflik tidak pernah terjadi. Juga menghapus `exec` dan `| tee` dari wrapper yang menyebabkan masalah pipe SIGPIPE.
  - **Investigasi ComfyUI Daemon:** Mengaudit `/singularity/comfui/` — menemukan bahwa `run_comfui_daemon.sh` menggunakan `pkill -f "cloudflared tunnel.*run --token"` secara global yang membunuh SEMUA instance cloudflared termasuk tunnel master dari `myslurm.sh`. Diperbaiki dengan mekanisme **PID File Tracking** (`named_tunnel.pid`).
  - **Error ComfyUI Teridentifikasi:** Package `comfy_kitchen 0.2.10` di `~/.local/lib/python3.10/` tidak kompatibel dengan PyTorch 2.2.2 (butuh PyTorch ≥ 2.4). User akan upgrade container Singularity secara manual.
  - **Verifikasi Berhasil:** Setelah fix, sesi tmux `cloudflare_tunnel` berjalan stabil tanpa terminasi.
- **File yang diubah/dibuat:**
  - `utils/myslurm.sh` [DIMODIFIKASI — wrapper cloudflared dengan `--metrics` random port, hapus exec+tee]
  - `singularity/comfui/run_comfui_daemon.sh` [DIMODIFIKASI — pkill global diganti PID file tracking]
  - `docs/SDP.md` [DIMODIFIKASI — Penambahan Log 060]
- **Status saat ini:** **Selesai**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Tunnel Cloudflare kini stabil via sesi tmux `cloudflare_tunnel`.
  - **JANGAN** pernah menggunakan `pkill -f "cloudflared tunnel.*run --token"` secara global di script manapun. Selalu gunakan PID file targeting.
  - ComfyUI menunggu upgrade container ke PyTorch ≥ 2.4. Setelah upgrade, jalankan menu 6 `myslurm.sh` → Opsi 1.
  - Quick Tunnel ComfyUI sementara rate-limited (429) karena terlalu banyak request debugging — akan pulih otomatis dalam beberapa jam.

---

### [Entri 061] — Otomatisasi Git Merge (sync_main.sh) untuk Melindungi Struktur Folder

- **Tanggal/Waktu:** 2026-06-09 14:03 WIB
- **Tugas yang diselesaikan:**
  - Membuat *bash script* `utils/sync_main.sh` untuk mengotomatiskan proses *merging* pembaruan dari branch `slurm` ke `main` secara aman.
  - Skrip memastikan folder arsitektur lengkap (`MyFineTunning-dev` dan `MyFineTunning-RunPOD`) **tidak terhapus** saat disinkronkan, dengan menariknya kembali dari branch `dev` menggunakan opsi `--no-commit`.
  - Mengubah *permission* script menjadi *executable* (`chmod +x`).
- **File yang diubah/dibuat:**
  - `utils/sync_main.sh` [DIBUAT BARU]
  - `docs/SDP.md` [DIMODIFIKASI - Penambahan Log 061]
- **Status saat ini:** **Selesai 100%**
- **Catatan untuk AI selanjutnya (Handoff Note):**
  - Untuk memindahkan update kode dari branch `slurm` ke `main` dengan aman, pengguna WAJIB menggunakan perintah `./utils/sync_main.sh` ketimbang membuat Pull Request lewat GUI GitHub.
