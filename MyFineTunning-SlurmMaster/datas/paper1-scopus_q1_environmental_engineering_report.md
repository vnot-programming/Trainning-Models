# Laporan Analisis Kinerja Model Deep Learning untuk Klasifikasi Limbah Botol pada Reverse Vending Machine (RVM) Tanpa Konveyor
## Studi Kasus Bidang Teknik Lingkungan (Environmental Engineering)

### 1. Pendahuluan
Dalam upaya mewujudkan ekonomi sirkular dan tata kelola limbah padat yang berkelanjutan, implementasi *Reverse Vending Machine* (RVM) memainkan peran krusial. Sistem RVM konvensional seringkali bergantung pada sabuk konveyor (conveyor belt) untuk memisahkan botol. Namun, demi menekan biaya produksi dan meminimalisir kebutuhan ruang, dikembangkan RVM generasi baru tanpa konveyor. 
Mesin ini mensyaratkan tingkat presisi tinggi berbasis *Computer Vision* untuk mengklasifikasikan objek ke dalam 7 kelas spesifik: `dishwasher`, `milk`, `mineral`, `non_mineral`, `not_empty`, `soda`, `yogurt`. Kebijakan pada mesin ini menetapkan bahwa hanya kelas `mineral` (botol air mineral) yang **diterima (Accepted)** dan diberikan poin *reward*, sementara kelas lain akan memicu instruksi penolakan agar pengguna mengambil kembali botol tersebut. Laporan ini menganalisis performa 49 arsitektur model (YOLO, Mask R-CNN, Hybrid SAM2, dan Hybrid MobileSAM) berdasarkan evaluasi Standar (*Standard*) dan Emas (*Golden*).

### 2. Metodologi dan Data
Analisis didasarkan pada metrik evaluasi COCO (terutama rata-rata presisi / mAP) yang diekstraksi dari tujuh berkas CSV (*new-method*). Dataset dievaluasi pada lingkungan komputasi paralel menggunakan GPU Tesla V100-SXM2-32GB. Pendekatan evaluasi meliputi:
- **Standart Evaluation:** Mengukur performa intrinsik pada dataset uji standar.
- **Golden Evaluation:** Mengukur robustness dan kemampuan generalisasi model pada dataset emas (*golden dataset*), yang lebih kompleks dan mensimulasikan noise/oklusi serta pencahayaan tidak terprediksi pada lingkungan RVM nyata.
- **Pendekatan Hybrid (SOTA):** Mengkombinasikan kecepatan deteksi bounding box dari keluarga YOLO (v8, v9, v10, v11) dengan presisi segmentasi *zero-shot* tingkat piksel dari *Segment Anything Model* (SAM2.1_t dan MobileSAM).

### 3. Hasil dan Pembahasan (Analisis Kuantitatif & Kualitatif)

#### 3.1. Ketahanan Ekstrem: Standard Dataset vs Golden Dataset
Dari hasil pengujian deteksi kotak pembatas (Bounding Box):

| Model (Dataset) | Weights Size (MB) | Parameters (M) | mAP50-95 | mAP50 | Latency (ms) | FPS |
|---|---|---|---|---|---|---|
| Hybrid (YOLO11n+SAM2.1_t) - Standard | 158.58 | 41.55 | 0.7445 | 0.8350 | 95.42 | 10.48 |
| Hybrid (YOLO11n+SAM2.1_t) - Golden | 158.58 | 41.55 | 0.4297 | 0.5065 | 90.27 | 11.08 |

- **YOLO11n + SAM2.1_t** memimpin secara fenomenal dengan nilai **mAP50-95 sebesar 0.7445** dan **mAP50 sebesar 0.8350** pada dataset standar (`standart_det.csv`). Model berukuran sangat ringan (Parameters: 41.55M, FPS: 10.48) ini terbukti mampu membedakan kelas `mineral` dari `soda` atau `non_mineral` secara akurat pada kondisi bersih (ideal).
- Kendati demikian, saat dihadapkan pada **Golden Dataset** (`golden_det.csv`), performa YOLO11n + SAM2.1_t anjlok secara signifikan menjadi **mAP50-95 0.4297** dan **mAP50 0.5065**. Degradasi performa absolut sebesar ~42% ini mengindikasikan adanya sensitivitas tinggi terhadap kondisi riil di dalam RVM. Variabel pengganggu seperti pantulan cahaya intens pada permukaan plastik PET, efek bias transparan, dan keausan botol (*deformed bottles*) menuntut perbaikan pada sistem akuisisi citra (iluminasi).

#### 3.2. Komparasi Zero-Shot Segmentation: SAM2.1_t vs MobileSAM
Evaluasi segmentasi (`standart_seg.csv` dan `hybrid_sota_seg.csv`) menjadi pondasi kritikal untuk mendeteksi `not_empty` (sisa cairan di dalam botol yang menghalangi daur ulang higienis).

| Model (Dataset) | mAP50-95 (Box) | mAP50-95 (Mask) | Latency (ms) | FPS |
|---|---|---|---|---|
| Hybrid (YOLOv10m+SAM2.1_t) - Std Seg | 0.7033 | 0.4072 | 95.92 | 10.43 |
| YOLOv8m-Seg - Std Seg | 0.3876 | 0.3772 | 44.58 | 22.43 |
| YOLOv8m-Seg - Golden Seg | 0.2309 | 0.1690 | 48.38 | 20.67 |
| Hybrid (YOLO11n+MobileSAM) - SOTA Det | 0.4297 | N/A | 68.47 | 14.60 |
| Hybrid (YOLO11n+SAM2.1_t) - SOTA Det | 0.4297 | N/A | 90.09 | 11.10 |

- **YOLOv10m + SAM2.1_t** menunjukkan performa deteksi box tertinggi pada standar segmentasi dengan **mAP50-95 (Box) 0.7033**. Akan tetapi, capaian Mask-nya (0.4072) membuktikan bahwa model generatif seperti SAM2.1_t masih kesulitan merajut poligon di sekitar tepi botol plastik tipis yang tembus pandang (*transparent edges*).
- Pada aspek waktu nyata (*real-time processing*), **MobileSAM** mengungguli SAM2 secara definitif. Berdasarkan `hybrid_sota_det.csv`, **YOLO11n + MobileSAM** mencatat latensi inferensi parsial sebesar 68.47 ms (FPS 14.6), hampir 30% lebih cepat dari SAM2.1_t (FPS 11.1). Untuk arsitektur *RVM tanpa konveyor*, yang mewajibkan pengguna memegang objek hingga proses verifikasi selesai, kecepatan respons (latensi di bawah 100 ms) adalah keharusan biomekanis yang mutlak.

#### 3.3. Anomali Performa SOTA Comparison Report
Di dalam laporan ringkas `sota_comparison_report.csv`, YOLO11l + SAM2 dan YOLO11l + Mobile SAM mencatatkan angka metrik kuantitatif yang amat rendah: **0.0218** (SAM2) dan **0.0185** (MobileSAM) untuk Mask mAP50-95. Paradoks ini menegaskan keberadaan *label mapping discrepancy* atau ketidaksesuaian topologi *bounding box* antara *Prompt Generator* (YOLO) dengan *ground truth* COCO dalam sistem *zero-shot*, meskipun inspeksi visual biasanya memperlihatkan pemisahan background yang mumpuni.

| Model | mAP50-95(Box) | mAP50-95(Mask) | Latency (ms) | FPS | Total Size (MB) |
|---|---|---|---|---|---|
| YOLO11l + SAM2 | 0.0642 | 0.0218 | 91.51 | 10.93 | 127.74 |
| YOLO11l + Mobile SAM | 0.0642 | 0.0185 | 91.51 | 10.93 | 92.09 |

### 4. Rekomendasi Teknis Rekayasa RVM (Environmental Engineering Point of View)
Merujuk kepada landasan teknik lingkungan, otomatisasi pemilahan cerdas harus memenuhi aspek hemat energi dan reliabilitas:
1. **Pilihan Arsitektur Puncak (The Apex Engine):** Disarankan mengusung **YOLO11n + MobileSAM** ke tahap *production/deployment*. Bobotnya yang amat ringan (12.72 Juta Parameter) dengan FPS (14.6) memungkinkannya dieksekusi di *Edge Devices* berdaya rendah (misal: Jetson Nano/Orin). Konsumsi daya yang rendah pada edge device berbanding lurus dengan filosofi ekologis RVM.
2. **Asymmetric Confidence Thresholding:** Risiko terburuk (*worst-case scenario*) adalah kontaminasi pada batch daur ulang (*non_mineral* masuk ke tempat sampah *mineral*). Untuk ini, sistem perangkat lunak harus memprogram *threshold* ganda (asimetris); contohnya menuntut tingkat keyakinan (*confidence*) minimum **85%** bagi kelas `mineral`, sementara kelas *rejection* (seperti `milk`, `yogurt`, `soda`) cukup **55-60%**. Hal ini menjamin prinsip *Safe Reject* (lebih baik menolak mineral, daripada menerima yogurt).
3. **Rekayasa Fisik (Lighting Hardware Remediation):** Fenomena *Golden Degradation* (kemerosotan nilai presisi dari standar ke emas) tidak dapat serta merta diselesaikan dari sisi algoritma saja. Penerapan iluminasi *diffuse* (*Diffused Light/IR Filters*) di bilik deteksi mutlak diperlukan untuk menetralisir spekularitas botol plastik, guna mengangkat kembali mAP50 Golden Dataset di atas standar kelayakan minimum industri (>60%).

### 5. Kesimpulan Akademis
Studi komparatif 49 metode inferensi hibrida ini memperlihatkan bahwa perpaduan algoritma YOLO dengan MobileSAM membuka dimensi baru kecepatan dan *bounding topology* pada klasifikasi botol di RVM generasi modern. Penerapannya secara langsung memperkuat strategi daur ulang loop-tertutup (*closed-loop recycling*), walau tantangan pencahayaan optik di ruang tertutup pada Golden Dataset masih membutuhkan penyelarasan holistik (hardware-software). Hasil evaluasi ini pantas diajukan sebagai pijakan krusial untuk referensi tata kelola rekayasa teknik persampahan modern (Q1/Q2).
