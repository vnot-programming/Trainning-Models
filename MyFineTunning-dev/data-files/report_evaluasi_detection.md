# Laporan Evaluasi Detection (Box) - Semua Model
## Mono: COCOeval (pycocotools)

Tanggal: 5 Mei 2026
Dataset: me-bottle-isempty-ku3-8 (Roboflow)
Evaluator: COCOeval (standar industri, sama dengan Mask R-CNN & Hybrid)

---

## 📊 Tabel Perbandingan mAP Detection (Box)

| Model | mAP50 | mAP50-95 (Box) | Precision | Recall | Latency (ms) | FPS | Evaluator |
|-------|--------|-------------------|-----------|--------|---------------|-----|-----------|
| **YOLO11m** | 0.948 | 0.840 | 0.919 | 0.924 | ~14.1ms | ~71 | Ultralytics* |
| **YOLOv8m** | 0.901 | 0.758 | - | - | ~14.6ms | ~68 | COCOeval** |
| **YOLOv9m** | 0.896 | 0.766 | - | - | ~25.4ms | ~39 | COCOeval** |
| **Hybrid (YOLO11m+SAM2)** | 0.938 | 0.791 | 0.938 | 0.938 | ~35.5ms | ~28 | COCOeval |
| **Mask R-CNN** | 0.790 | 0.590 | - | - | 334.34ms | 2.99 | COCOeval |

*YOLO11m detection masih pakai Ultralytics `m.val()` (akan diubah ke COCOeval nanti)
**YOLOv8m & YOLOv9m detection SUDAH pakai COCOeval (modifikasi terbaru)

---

## 🏆 Ranking Berdasarkan mAP50 (Box)

1. 🥇 **YOLO11m** - 0.948 (Ultralytics)
2. 🥈 **Hybrid (YOLO11m+SAM2)** - 0.938 (COCOeval)
3. 🥉 **YOLOv8m** - 0.901 (COCOeval)
4. **YOLOv9m** - 0.896 (COCOeval)
5. **Mask R-CNN** - 0.790 (COCOeval)

---

## 📌 Keterangan:
- **mAP50**: mAP @ IoU=0.50
- **mAP50-95**: mAP @ IoU=0.50:0.95 (rata-rata seluruh threshold)
- **COCOeval**: Standar evaluasi COCO (digunakan di COCO Challenge, paper YOLOv8/v9/11)
- **Ultralytics**: Evaluasi internal YOLO (`m.val()`) - tidak sepenuhnya komparabel

---

## 💡 Rekomendasi:
- Untuk **akurasi tertinggi**: YOLO11m (0.948 mAP50)
- Untuk **kecepatan + akurasi seimbang**: YOLOv8m (0.901 mAP50, 68 FPS)
- Untuk **real-time application**: YOLO11m atau YOLOv8m
- Untuk **akurasi vs kecepatan**: Hybrid (0.938 mAP50, 28 FPS)

---

## 📁 File Report:
- `report_yolo11m_det_coco.csv`
- `report_yolov8m_det_coco.csv`
- `report_yolov9m_det_coco.csv`
- `report_hybrid_det_coco.csv`
- `report_maskrcnn_ddp_seg.csv` (mengandung juga mAP Box)
