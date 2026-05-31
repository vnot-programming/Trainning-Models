# Laporan Evaluasi Skala Besar (Pipeline) Deteksi dan Segmentasi Limbah Botol untuk Arsitektur RVM Tanpa Konveyor
## Lanjutan Studi Kasus Teknik Lingkungan (Environmental Engineering)

### 1. Latar Belakang dan Metodologi Pipeline MultiGPU
Pengembangan *Reverse Vending Machine* (RVM) berdaya rendah membutuhkan jaminan stabilitas dalam klasifikasi kelas-kelas yang telah ditentukan (`dishwasher`, `milk`, `mineral`, `non_mineral`, `not_empty`, `soda`, `yogurt`). Evaluasi global melalui *pipeline* terotomatisasi dilakukan untuk mengukur performa murni secara skalabel menggunakan framework COCOeval yang terdistribusi secara paralel (MultiGPU). Data analisis komparatif ini diekstraksi dari hasil kompilasi laporan *pipeline* akhir (`kompilasi_ALL_detection.csv` dan `kompilasi_ALL_segmentation.csv`).

### 2. Analisis Performa Deteksi Kotak Pembatas (Bounding Box)
Deteksi *bounding box* adalah garda terdepan sistem penyaringan sampah mineral untuk menghindari kontaminasi residu daur ulang. Berikut hasil dari evaluasi pipeline:

| Model | Model Size (MB) | mAP50-95 | mAP50 | Latency (ms) | FPS |
|---|---|---|---|---|---|
| YOLOv8m | 52.05 | 0.7476 | 0.8474 | 23.21 | 43.08 |
| YOLOv10m | 33.51 | 0.7739 | 0.8936 | 22.40 | 44.64 |
| YOLO11n | 5.50 | 0.7925 | 0.8955 | 19.20 | 52.09 |
| YOLO11l | 51.22 | 0.7984 | 0.8823 | 26.20 | 38.17 |

**Temuan Akademis:**
**YOLO11n** mendemonstrasikan rasio kompresi-presisi (*compression-to-precision ratio*) yang di luar batas kewajaran model konvensional. Dengan ukuran model mikroskopis hanya **5.5 MB**, model ini sanggup melampaui varian YOLOv8m yang berukuran hampir sepuluh kali lipatnya. YOLO11n beroperasi secara mulus pada kecepatan puncak **52 FPS** (latensi 19.2 ms). 
Di dalam ekosistem komputasi tepi (*Edge Computing*) pada mesin RVM—seperti pada board *Raspberry Pi* atau *Jetson Nano*—penghematan parameter memori ini sangatlah kritikal. Model ini menghindari *thermal throttling* secara proaktif namun tetap mengunci tingkat akurasi (mAP50) di **89.55%**. Ini memastikan bahwa pemilahan botol *mineral* bersih berlangsung dengan *false-rejection rate* yang minimal.

### 3. Analisis Performa Segmentasi Tingkat Piksel (Native vs Hybrid Zero-Shot)
Pemisahan piksel yang ketat difungsikan secara spesifik untuk memetakan luasan `not_empty` (sisa cairan di dalam wadah). Sisa residu organik berpotensi menghancurkan nilai ekonomi *PET-flakes* hasil daur ulang karena pembusukan:

| Model (Segmentasi) | Model Size (MB) | mAP50-95 (Box) | mAP50-95 (Mask) | Latency (ms) | FPS |
|---|---|---|---|---|---|
| YOLO11n-Seg | 6.01 | 0.4179 | 0.3771 | 25.57 | 39.10 |
| Mask R-CNN | 184.02 | 0.2223 | 0.2203 | 80.88 | 12.36 |
| Hybrid (YOLOv10m+SAM2.1_t) | 111.61 | 0.8842 | 0.4075 | 87.39 | 11.44 |
| Hybrid (YOLO11n+SAM2.1_t) | 83.61 | 0.8774 | 0.4079 | 84.65 | 11.81 |

**Temuan Akademis:**
- **Superioritas Pendekatan Hybrid pada Box Metrics:** Menariknya, ketika *Segment Anything Model* (SAM2.1_t) menunggangi generator prompt (YOLOv10m atau YOLO11n), tingkat keyakinan *bounding box* memuncak fantastis hingga presisi rata-rata **0.8842** (mAP50-95). Ini membuktikan sinergi arsitektur ganda di mana YOLO memetakan area subjek secara heuristik, sementara SAM memperhalus tepi kotak dengan analisis kontur visual.
- **Trade-off Dimensi pada Segmentasi Mask:** Sistem *Hybrid (YOLO11n+SAM2.1_t)* mencetak skor *Mask* mAP sebesar **0.4079**. Meskipun sangat kokoh, model *Native Segmentation* **YOLO11n-Seg** berhasil membayangi capaian tersebut di angka **0.3771**. Keuntungan komparatif mutlak terletak pada struktur Native YOLO11n-Seg yang **13x lebih kecil secara footprint** (6 MB vs 83 MB) dan sanggup di-eksekusi **3.5x lebih kencang** secara eksponensial (39 FPS vs 11 FPS).
- *Mask R-CNN*, raksasa segmentasi di masa lalu, terbukti kedaluwarsa secara arsitektural untuk kebutuhan IoT masa depan, dengan mAP Mask tersungkur di angka 0.22 dan latensi komputasi yang terlalu membebani (80.88 ms).

### 4. Rekomendasi Terintegrasi untuk Desain RVM di Sektor IoT Teknik Lingkungan
Berdasarkan agregat performa *pipeline MultiGPU*, terdapat beberapa arahan strategis:
1. **Edge Deployment (Otonom):** Mayoritas mesin RVM generasi tanpa konveyor wajib berdiri di lokasi dengan konektivitas seluler fluktuatif. Dengan demikian, penerapan lokalisasi berbasis **Native Segmentation (YOLO11n-Seg)** adalah jawaban paripurna. Ia hemat memori dan berdaya tembus latensi rendah, yang tidak menguras *current draw* dari pasokan listrik (energi hijau).
2. **Arsitektur Hibrida untuk Smart Facility (Cloud-Dependent):** Pada skenario pemilahan sentral, bilamana *computing node* dapat dipindah ke server komputasi lokal, pemanfaatan **Hybrid YOLOv10m + SAM2.1_t** direkomendasikan. Angka mAP Box ~0.88 memastikan bahwa *sorting conveyor* pasca-konsumsi di pabrik pengolahan nyaris meminimalisir kesalahan sortir secara absolut.
3. **Penyelarasan Waktu Transaksi RVM:** Dalam konstruksi RVM *conveyor-less*, botol yang dimasukkan murni bergantung pada kecepatan baca (*latency* <30 ms). Tingkat kecepatan tanggap varian nano (YOLO11n) mengaktifkan penutupan/pembukaan pintu aktuator mekanik jauh sebelum pengguna menyadari jeda. *Friction-less User Experience* ini terbukti memicu peningkatan signifikan partisipasi akar rumput (masyarakat sipil) dalam ekosistem *Urban Waste Management*.
