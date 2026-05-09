# Laporan Evaluasi Detection (Box) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Detection (Box)

| Model | mAP50 | mAP50-95 (Box) | Precision | Recall | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|--------|-------------------|-----------|--------|-----------------|----------------|------------------|---------------|-----|------|-----------|
| **YOLO11m** | 0.9284 | 0.7827 | 0.9594 | 0.929 | 0.88ms | 13.19ms | 1.0ms | 15.07ms | 66.36 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv9m** | 0.9252 | 0.8146 | 0.9516 | 0.9214 | 0.88ms | 17.32ms | 1.11ms | 19.31ms | 51.79 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv8m** | 0.9276 | 0.8089 | 0.9505 | 0.9252 | 0.73ms | 13.91ms | 0.92ms | 15.56ms | 64.27 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Hybrid (YOLO11m+SAM2)** | 0.938 | 0.7905 | 0.9697 | 0.9352 | 8.08ms | 214.89ms | 3.12ms | 226.09ms | 4.42 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
