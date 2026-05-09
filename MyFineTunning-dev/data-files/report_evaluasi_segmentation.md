# Laporan Evaluasi Segmentation (Mask) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Segmentation (Mask)

| Model | mAP50-95 (Box) | mAP50-95 (Mask) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|----------------|-----------------|--------------|-----|------|-----------|
| **YOLO11m-Seg** | 0.7315 | 0.6509 | 20.86ms | 47.94 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv9c-Seg** | 0.7663 | 0.683 | 26.43ms | 37.84 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv8m-Seg** | 0.7577 | 0.6794 | 20.03ms | 49.93 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Hybrid (YOLO11m+SAM2)** | 0.7905 | 0.3683 | 226.09ms | 4.42 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Mask R-CNN ResNet-50 FPN-v2** | 0.5904 | 0.5714 | 235.36ms | 4.25 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
