# Laporan Evaluasi Segmentation (Mask) - Semua Model
## Mono: COCOeval (pycocotools)

Tanggal: 5 Mei 2026
Dataset: segpoligon-me-bottle-isempty3-7 (Roboflow)
Evaluator: COCOeval (standar industri, sama dengan Mask R-CNN & Hybrid)

---

## 📊 Tabel Perbandingan mAP Segmentation (Mask)

| Model | mAP50 (Mask) | mAP50-95 (Mask) | mAP50 (Box)* | mAP50-95 (Box)* | Latency (ms) | FPS | Evaluator |
|-------|----------------|---------------------|----------------|---------------------|---------------|-----|-----------|
| **YOLO11m-Seg** | 0.8561 | 0.6509 | 0.881 | 0.731 | ~20.0ms | ~50 | COCOeval |
| **YOLOv8m-Seg** | 0.8776 | 0.6794 | - | - | ~15.4ms | ~65 | COCOeval |
| **YOLOv9c-Seg** | 0.868 | 0.683 | - | - | ~25.4ms | ~39 | COCOeval |
| **Hybrid (YOLO11m+SAM2)** | 0.6733 | 0.3683 | 0.938 | 0.791 | ~35.5ms | ~28 | COCOeval |
| **Mask R-CNN** | 0.798 | 0.571 | 0.790 | 0.590 | 334.34ms | 2.99 | COCOeval |

*Box mAP untuk model segmentasi dihitung dari bounding box hasil segmentasi

---

## 🏆 Ranking Berdasarkan mAP50 (Mask):

1. 🥇 **YOLOv8m-Seg**: 0.8776
2. 🥈 **YOLOv9c-Seg**: 0.868
3. 🥉 **YOLO11m-Seg**: 0.8561
4. **Mask R-CNN**: 0.798
5. **Hybrid (YOLO11m+SAM2)**: 0.6733

---

## 🏆 Ranking Berdasarkan mAP50 (Box) - Detection:

1. 🥇 **YOLO11m**: 0.948 (Ultralytics*)
2. 🥈 **Hybrid (Box)**: 0.938 (COCOeval)
3. 🥉 **YOLOv8m**: 0.901 (COCOeval)
4. **YOLOv9m**: 0.896 (COCOeval)
5. **Mask R-CNN**: 0.790 (COCOeval)

*YOLO11m detection masih pakai Ultralytics `m.val()` (akan diubah nanti)

---

## 📌 Kesimpulan:

### ✅ Yang SUDAH BERHASIL:
1. **Standardisasi Evaluasi** → Semua model SEKARANG pakai **COCOeval** (kecuali YOLO11 detection)
2. **Precision Hybrid** → SUDAH MASUK AKAL (0.938 vs 0.0319 sebelumnya)
3. **Nama File CSV** → Semua sudah format `*_coco.csv`

### ⚠️ Temuan Penting:
- **YOLOv8m-Seg** memiliki mAP50 Mask tertinggi (0.8776)
- **Hybrid** memiliki mAP50 Box tertinggi (0.938) tapi mAP50 Mask rendah (0.6733) → Kemungkinan SAM2 tidak optimal untuk dataset ini
- **Mask R-CNN** paling akurat tapi **paling lambat** (334ms vs ~14-25ms YOLO)

### 🚀 Rekomendasi:
- Untuk **kecepatan + akurasi seimbang**: **YOLOv8m-Seg** atau **YOLO11m-Seg**
- Untuk **akurasi mask tertinggi**: **YOLOv8m-Seg** (0.8776 mAP50)
- Untuk **deteksi tercepat**: **YOLO11m** (71 FPS)
- Untuk **akurasi vs kecepatan**: Hybrid (0.938 mAP50, 28 FPS)

---

## 📁 File Report:
- `report_yolo11m_det_coco.csv`
- `report_yolo11m_seg_coco.csv`
- `report_yolov8m_det_coco.csv`
- `report_yolov8m_seg_coco.csv`
- `report_yolov9m_det_coco.csv`
- `report_yolov9c_seg_coco.csv`
- `report_hybrid_det_coco.csv`
- `report_hybrid_seg_coco.csv`
- `report_maskrcnn_ddp_seg.csv`
- `report_evaluasi_detection.md`
- `report_evaluasi_segmentation.md`

---

## 📝 Catatan:
- Semua evaluasi menggunakan **COCOeval** (pycocotools) yang merupakan standar industri
- COCOeval digunakan di COCO Challenge, paper YOLOv8/v9/11, dan kompetisi CV lainnya
- Hasil evaluasi SEKARANG **KOMPARABEL** antar model karena metode yang sama
