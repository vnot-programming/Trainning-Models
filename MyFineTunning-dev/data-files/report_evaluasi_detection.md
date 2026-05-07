# Laporan Evaluasi Detection (Box) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Detection (Box)

| Model | mAP50 | mAP50-95 (Box) | Precision | Recall | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|--------|-------------------|-----------|--------|-----------------|----------------|------------------|---------------|-----|------|-----------|
| **YOLO11m (Fine-tuned COCOeval)** | 0.9284 | 0.7827 | 0.9594 | 0.929 | 0.62ms | 13.14ms | 0.93ms | 14.69ms | 68.07 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv9m (Fine-tuned COCOeval)** | 0.9252 | 0.8146 | 0.9516 | 0.9214 | 0.63ms | 17.28ms | 1.0ms | 18.91ms | 52.88 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **YOLOv8m (Fine-tuned COCOeval)** | 0.9276 | 0.8089 | 0.9505 | 0.9252 | 0.66ms | 13.9ms | 0.94ms | 15.5ms | 64.52 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
| **Hybrid (YOLO11m+SAM2)** | 0.938 | 0.7905 | 0.9697 | 0.9352 | 7.38ms | 212.98ms | 2.95ms | 223.31ms | 4.48 | 1x NVIDIA GeForce RTX 3060 | COCOeval |
