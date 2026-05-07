Script `hybrid/main.py` ini berfungsi sebagai **tahap evaluasi final dan perbandingan** dari seluruh model yang telah dilatih sebelumnya (seperti YOLOv8, v9, v11, dan Mask R-CNN), sekaligus mengimplementasikan dan menguji sebuah model **"Hybrid" (kombinasi YOLO11m dan SAM2)**.

Secara garis besar, berikut adalah langkah demi langkah cara kerja pipeline di dalam script ini:

### 1. Memuat Semua Model Terbaik (Best Weights)
Script pertama-tama mendata dan menyiapkan *path* (lokasi file) menuju bobot terbaik (`best.pt`) dari semua model yang telah selesai dilatih pada tahap sebelumnya:
- **Detection**: YOLOv8m, YOLOv9m, YOLO11m.
- **Segmentation**: YOLOv8m-seg, YOLOv9c-seg, YOLO11m-seg, dan Mask R-CNN.
- **Model SAM2**: Memuat arsitektur *Segment Anything Model 2* (`sam2.1_b.pt`).

### 2. Memilih Gambar Sampel
Script mengambil 10 gambar sampel secara berurutan/acak dari folder dataset testing untuk digunakan sebagai bahan visualisasi perbandingan (meskipun di komentar tertulis 5 gambar, namun di kodenya diset `all_imgs[:10]`).

### 3. Menjalankan Pipeline Hybrid (YOLO11m + SAM2)
Untuk setiap gambar sampel, script menjalankan sebuah proses yang disebut "Hybrid Pipeline":
- **Langkah 1**: Gambar dilemparkan ke model **YOLO11m** murni (Detection) untuk mendeteksi letak objek berupa *bounding box* (kotak).
- **Langkah 2**: *Bounding box* yang didapatkan dari YOLO11m kemudian dilempar ke model **SAM2** (*Segment Anything Model*) sebagai *prompt* (petunjuk). SAM2 kemudian bertugas menggambar *masking* (segmentasi) yang sangat akurat berdasarkan kotak tersebut.
- **Langkah 3**: Script mencatat *latency* (waktu proses/FPS) dari proses ini dan menyimpan gambar visualisasi yang membandingkan hasil deteksi YOLO11m di kiri dan hasil segmentasi SAM2 di kanan.

### 4. Evaluasi & Benchmark Seluruh Model
Script kemudian mengevaluasi semua model menggunakan fungsi `_benchmark_yolo` untuk mencari nilai akurasi dan kecepatan:
- Mengukur metrik seperti **mAP50-95, mAP50, Precision, dan Recall**.
- Mengukur waktu **Preprocess, Inference, Postprocess, dan FPS** (kecepatan).
- Khusus untuk Mask R-CNN, nilai kecepatannya dimuat (*load*) dari file CSV hasil *report* yang mungkin sudah di-generate secara terpisah.

### 5. Membuat Laporan (CSV)
Hasil metrik dari langkah ke-4 disimpan menjadi tiga file laporan berformat `.csv` agar mudah dianalisis:
- `report_det_comparison.csv`: Membandingkan semua model deteksi (*object detection*).
- `report_seg_comparison.csv`: Membandingkan semua model segmentasi, termasuk performa model Hybrid.
- `report_hybrid_latency.csv`: Rincian waktu proses per gambar untuk model Hybrid.

### 6. Visualisasi Grid Perbandingan (1x5)
Untuk memudahkan manusia melihat perbedaannya, script merender sebuah gambar kisi (grid) untuk setiap gambar sampel. Grid ini menyandingkan hasil prediksi 5 model segmentasi berbeda secara bersebelahan:
1. YOLOv8m-Seg
2. YOLOv9c-Seg
3. YOLO11m-Seg
4. Mask R-CNN
5. Hybrid (YOLO11m + SAM2)

Setiap *class* objek diberikan warna yang konsisten, dan hasil gambar akhirnya disimpan ke dalam folder `runs/visuals/hybrid/`.

### 7. Kompresi dan Notifikasi
Di bagian akhir, script mengompresi semua hasil gambar menjadi satu file `visuals.tar.gz` agar mudah diunduh, lalu mengirimkan pesan pemberitahuan ke **Telegram** yang menginformasikan bahwa seluruh pipeline (*training* dan *evaluation*) telah resmi selesai.

=================
# KLARIFIKASI 1 #
=================

**1. Sumber 10 Gambar Sampel**
Ya, 10 gambar sampel tersebut otomatis diambil dari variabel `IMAGE_SAMPLES_DIR` yang ada di `config_shared.py`. Mengingat struktur direktori Anda, path ini memang merujuk ke folder `Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260502_134510/image_samples`.

**2. YOLO sebagai Detektor dan SAM2 sebagai Validator Segmentasi**
Tepat sekali! YOLO11m hanya berfungsi untuk mencari titik koordinat kotak (bounding box) dan mengenali kelas (nama) objeknya. Kotak tersebut kemudian diserahkan kepada SAM2. SAM2 tidak memprediksi nama objek, ia hanya menerima "petunjuk" berupa kotak tersebut dan membuatkan *masking* (potongan piksel) yang seakurat mungkin mengikuti tepi objek.

**3. Bentuk Visualisasi Hybrid (Langkah 3)**
File yang dihasilkan di Langkah 3 bukan berbentuk tabel, melainkan gambar bersebelahan (1 baris, 2 kolom):
- **Kiri**: Menampilkan gambar asli dengan **kotak** (bounding box) deteksi YOLO11m.
- **Kanan**: Menampilkan gambar asli dengan **masking** (segmentasi warna) buatan SAM2.
File gambar dinamai `hybrid_01_namagambar.png` hingga `hybrid_10_namagambar.png` dan disimpan di `runs/visuals/hybrid/`.

**4 & 5. Isi Tabel Benchmark dan Laporan CSV**
Benchmark mengevaluasi metrik deteksi (mAP, Precision), segmentasi (mAP Mask), dan kecepatan (Latency/FPS). Berikut adalah detail isi dari ketiga file `.csv` yang dibuat:
- **`report_det_comparison.csv`**: Membandingkan model khusus deteksi.
  *Isi kolom*: Model, Model Size (MB), mAP50-95, mAP50, Precision, Recall, Preprocess (ms), Inference (ms), Postprocess (ms).
- **`report_seg_comparison.csv`**: Membandingkan semua model segmentasi **DITAMBAH** model Hybrid.
  *Isi kolom*: Model, Model Size (MB), mAP50-95(Box), mAP50-95(Mask), Latency(ms), FPS.
- **`report_hybrid_latency.csv`**: Tabel ini **hanya untuk model Hybrid**. Memecah waktu yang dibutuhkan per gambar.
  *Isi kolom*: Image (nama gambar), YOLO_ms (waktu deteksi yolo), SAM2_ms (waktu segmentasi sam2), Total_ms (waktu gabungan), FPS, Detections (jumlah objek yang dideteksi).

**6. Grid Perbandingan 1x5 (Langkah 6)**
Ya, kisi (grid) 1x5 ini **fokus untuk membandingkan hasil segmentasi (masking)** antar model. Model YOLO versi deteksi murni tidak dimasukkan di sini karena tujuan utamanya adalah melihat model mana yang potongan/masking objeknya paling rapi: YOLO-Seg (v8, v9, v11) vs Mask R-CNN vs SAM2 (Hybrid).

**7. Kompresi Visuals**
Betul, fungsi `compress_visuals()` secara spesifik hanya membungkus folder `runs/visuals/` beserta isinya (semua gambar hasil perbandingan) menjadi file arsip `runs/visuals.tar.gz`. File `.csv` (laporan tabel) tetap dibiarkan utuh di folder `runs/reports/` dan tidak ikut dikompres.

=================
# KLARIFIKASI 2 #
=================

**4 & 5. Isi Tabel Benchmark dan Laporan CSV (Terkait Latency)**
- **Apakah Latency Hybrid sama dengan model lain?**
  Tidak. Tabel `report_hybrid_latency.csv` **hanya** menghitung waktu spesifik untuk proses Hybrid (waktu YOLO mendeteksi kotak + waktu SAM2 melakukan segmentasi) per gambar. 
  Sedangkan waktu (*latency*) untuk model-model tunggal seperti `report_yolov8m_seg.csv` atau `report_maskrcnn.csv` dihitung secara terpisah. 
  Namun, Anda bisa melihat perbandingan *latency* keseluruhan model-model tersebut (termasuk Hybrid) di dalam tabel **`report_seg_comparison.csv`** yang menggabungkan semua hasil evaluasinya.

**6. Grid Perbandingan 1x5 (Langkah 6) & Menambahkan YOLO Detection**
- **Bisa ditambahkan YOLO detection?**
  **Sangat bisa.** Saat ini kode di `main.py` dikonfigurasi untuk membandingkan model-model berjenis segmentasi (YOLO-Seg vs Mask R-CNN vs Hybrid). Jika Anda merasa perbandingan antar model segmentasi tidak perlu dan ingin diganti menjadi perbandingan **YOLO Detection (v8, v9, v11) vs Pipeline Hybrid**, kita hanya perlu mengubah sedikit variabel `PANELS` dan fungsi render di dalam `main.py`. Beri tahu saya jika Anda ingin saya langsung mengubah kode `main.py` tersebut.

**7 & 8. Laporan Kompresi & Perkiraan Ukuran File (Size)**
- **Total Ukuran Folder `MyFineTunning-20260502_134510`**: Saat ini kurang lebih sekitar **1.87 GB (1869 MB)**.
- **Ukuran File `.tar.gz` Masing-Masing Model**:
  - `yolo11m_seg.tar.gz`: ~86 MB
  - `yolo11m.tar.gz`: ~78 MB
  - `yolov8m_seg.tar.gz`: ~103 MB
  - `yolov8m.tar.gz`: ~99 MB
  - `yolov9c_seg.tar.gz`: ~105 MB
  - `yolov9m.tar.gz`: ~79 MB
  *Total ukuran ke-6 file di atas*: **~550 MB**.
- **Perkiraan Jika Kompresi Tanpa File `.tar.gz` di Atas**:
  Jika kita mengompresi seluruh folder `MyFineTunning-20260502_134510` namun *mengecualikan* (exclude) keenam file `.tar.gz` dari hasil model tunggal tersebut, ukuran akhirnya akan turun menjadi sekitar **1.32 GB (1319 MB)** (atau bisa sedikit lebih kecil tergantung rasio kompresi).

=================
# KLARIFIKASI 3 #
=================

**9. Menambahkan Grid YOLO Detection**
Sesuai permintaan Anda, saya sudah memodifikasi script `main.py` sehingga sekarang akan me-render *keduanya*! Saat pipeline dijalankan, ia akan menghasilkan dua gambar *grid* untuk setiap sampel:
1. `comparison_seg_...`: Grid 1x5 untuk membandingkan Mask R-CNN, varian YOLO-Seg, dan Hybrid.
2. `comparison_det_...`: Grid 2x2 baru untuk membandingkan murni YOLOv8m (Det), YOLOv9m (Det), YOLO11m (Det), vs Pipeline Hybrid.

**10. Perkiraan Kompresi Folder Tanpa .tar.gz Tunggal**
Berdasarkan kalkulasi sebelumnya, jika sewaktu-waktu nanti Anda ingin mengompresi folder `MyFineTunning-20260502_134510` namun *mengabaikan* file hasil `runs/*.tar.gz`, maka hasil kompresi arsip tersebut (`MyFineTunning-20260502_134510.tar.gz`) diperkirakan hanya akan memakan ruang sekitar **1.32 GB**.

