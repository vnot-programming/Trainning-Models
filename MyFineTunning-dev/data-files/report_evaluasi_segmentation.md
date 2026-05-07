# Laporan Evaluasi Segmentation (Mask) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Segmentation (Mask)

| Model | mAP50-95 (Box) | mAP50-95 (Mask) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|----------------|-----------------|--------------|-----|------|-----------|
| **YOLO11m-Seg (Fine-tuned COCOeval)** | 0.7315 | 0.6509 | 20.69ms | 48.33 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv9c-Seg (Fine-tuned COCOeval)** | 0.7663 | 0.683 | 26.35ms | 37.95 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv8m-Seg (Fine-tuned COCOeval)** | 0.7577 | 0.6794 | 19.99ms | 50.03 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Hybrid (YOLO11m+SAM2)** | 0.7905 | 0.3683 | 223.31ms | 4.48 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Mask R-CNN ResNet-50 FPN-v2 (DDP Fine-tuned)** | 0.5904 | 0.5714 | 209.3ms | 4.78 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
