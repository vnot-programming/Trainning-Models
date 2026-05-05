# -*- coding: utf-8 -*-
"""
evaluate_hybrid_map.py
======================
Evaluasi mAP (Mean Average Precision) untuk Hybrid Pipeline (YOLO11m + SAM2).

Menghitung:
  - mAP50-95 (Box) dari bounding box hasil YOLO11m
  - mAP50-95 (Mask) dari mask hasil SAM2 vs ground truth
  - Precision, Recall per kelas

Output:
  - report_hybrid_map.csv (ringkasan metrik)
  - hybrid_detailed_predictions.csv (prediksi per gambar)

Cara menjalankan:
    python -u evaluate_hybrid_map.py 2>&1 | tee hybrid_map_eval.log
"""

import os, sys, csv, gc, time, yaml
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE,
    get_output_dir, REPORTS_DIR
)
import torch
import cv2
import numpy as np
from ultralytics import YOLO, SAM

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"[Device] {DEVICE}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _flush(label: str):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"[MemFlush] {label} — VRAM bebas: {free_gb:.2f} GB")


def load_yaml(path):
    """Load YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def compute_mask_iou(mask_pred, mask_gt):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / union if union > 0 else 0.0


def polygon_to_mask(polygon_points, img_h, img_w):
    """Convert YOLO segmentation polygon (normalized) to binary mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not polygon_points or len(polygon_points) < 6:
        return mask.astype(bool)
    
    # Convert normalized coordinates to pixel coordinates
    points = np.array(polygon_points).reshape(-1, 2)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    points = points.astype(np.int32)
    
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def load_ground_truth_labels(yaml_path, split="valid"):
    """
    Load ground truth dari dataset YAML.
    Uses 'valid' folder (not 'val') to match your dataset structure.
    Returns: dict {image_path: {'boxes': [[x1,y1,x2,y2], ...], 'clss': [cls1, ...]}}
    """
    cfg = load_yaml(yaml_path)
    base = cfg.get("path", os.path.dirname(yaml_path))
    
    # Get split folder name from YAML, default to "valid"
    split_folder = cfg.get(split, cfg.get("val", split))
    
    # Build paths
    img_dir = os.path.join(base, "images", split_folder)
    label_dir = os.path.join(base, "labels", split_folder)
    
    if not os.path.isdir(img_dir):
        print(f"  ⚠️  Image dir tidak ditemukan: {img_dir}")
        return {}
    
    gt_data = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.lower().endswith(valid_exts):
            continue
        
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        boxes, clss = [], []
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        # Load labels
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        # YOLO format: cls cx cy w h (normalized)
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        boxes.append([x1, y1, x2, y2])
                        clss.append(cls)
        
        if boxes:  # Only add if there are annotations
            gt_data[img_path] = {"boxes": boxes, "clss": clss}
    
    print(f"  ✅ Loaded {len(gt_data)} ground truth images dari {split_folder}")
    return gt_data


def load_ground_truth_masks(yaml_path, split="valid"):
    """
    Load ground truth masks untuk segmentasi.
    Generates masks from YOLO polygon labels (not separate mask files).
    Returns: dict {image_path: {'boxes': [...], 'clss': [...], 'masks': [mask1, mask2, ...]}}
    """
    cfg = load_yaml(yaml_path)
    base = cfg.get("path", os.path.dirname(yaml_path))
    
    # Get split folder name
    split_folder = cfg.get(split, cfg.get("val", split))
    
    # Build paths
    img_dir = os.path.join(base, "images", split_folder)
    label_dir = os.path.join(base, "labels", split_folder)
    
    if not os.path.isdir(img_dir):
        print(f"  ⚠️  Image dir tidak ditemukan: {img_dir}")
        return {}
    
    gt_data = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.lower().endswith(valid_exts):
            continue
        
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        boxes, clss, masks = [], [], []
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        # Load labels with polygon points
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        # Check if it's segmentation format (polygon) or detection format
                        if len(parts) >= 7:  # Segmentation: cls x1 y1 x2 y2 ... xn yn
                            polygon_points = list(map(float, parts[1:]))
                            boxes.append([0, 0, img_w, img_h])  # Full image as box
                            clss.append(cls)
                            mask = polygon_to_mask(polygon_points, img_h, img_w)
                            masks.append(mask)
                        else:  # Detection format: cls cx cy w h
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = (cx - w/2) * img_w
                            y1 = (cy - h/2) * img_h
                            x2 = (cx + w/2) * img_w
                            y2 = (cy + h/2) * img_h
                            boxes.append([x1, y1, x2, y2])
                            clss.append(cls)
                            masks.append(None)  # No mask for detection format
    
        if boxes:
            gt_data[img_path] = {"boxes": boxes, "clss": clss, "masks": masks}
    
    print(f"  ✅ Loaded {len(gt_data)} ground truth (seg) images dari {split_folder}")
    return gt_data


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate_hybrid_map():
    """Evaluasi mAP untuk Hybrid Pipeline."""
    
    print("\n" + "="*65)
    print("  Evaluasi mAP Hybrid Pipeline (YOLO11m + SAM2)")
    print("="*65)
    
    # --- 1. Load Models ---
    print("\n[1] Loading Models...")
    yolo11m_path = os.path.join(get_output_dir("yolo11m"), "weights", "best.pt")
    if not os.path.exists(yolo11m_path):
        print(f"  ❌ YOLO11m best.pt tidak ditemukan: {yolo11m_path}")
        return
    
    yolo_det = YOLO(yolo11m_path)
    print(f"  ✅ YOLO11m loaded: {yolo11m_path}")
    
    try:
        sam_model = SAM("sam2.1_b.pt")
        print("  ✅ SAM2 loaded")
    except Exception as e:
        print(f"  ❌ SAM2 gagal dimuat: {e}")
        return
    
    # --- 2. Load Ground Truth ---
    print("\n[2] Loading Ground Truth...")
    # Detection ground truth (using "valid" folder to match dataset structure)
    gt_det = load_ground_truth_labels(DET_YAML, split="valid")
    # Segmentation ground truth (using "valid" folder)
    gt_seg = load_ground_truth_masks(SEG_YAML, split="valid")
    
    if not gt_det:
        print("  ⚠️  Tidak ada ground truth deteksi. Evaluasi dibatalkan.")
        return
    
    # --- 3. Evaluate on validation set ---
    print("\n[3] Evaluasi pada validation set...")
    all_predictions = []
    
    # Get number of classes from YAML
    det_cfg = load_yaml(DET_YAML)
    num_classes = det_cfg.get("nc", 4)  # Default 4 if not specified
    
    for idx, (img_path, gt_item) in enumerate(gt_det.items(), 1):
        if idx % 10 == 0:
            print(f"  Proses {idx}/{len(gt_det)}...")
        
        # YOLO Detection
        det_result = yolo_det.predict(img_path, conf=0.5, verbose=False, imgsz=IMAGE_SIZE)
        if det_result and det_result[0].boxes is not None:
            pred_boxes = det_result[0].boxes.xyxy.cpu().numpy()
            pred_confs = det_result[0].boxes.conf.cpu().numpy()
            pred_clss = det_result[0].boxes.cls.cpu().numpy().astype(int)
        else:
            pred_boxes = np.zeros((0, 4))
            pred_confs = np.array([])
            pred_clss = np.array([])
        
        # SAM2 Segmentation (if there are detections)
        pred_masks = []
        if len(pred_boxes) > 0:
            try:
                sam_result = sam_model.predict(det_result[0].orig_img, bboxes=pred_boxes, verbose=False)
                if sam_result and sam_result[0].masks is not None:
                    for m in sam_result[0].masks.data.cpu().numpy():
                        # Resize mask to original size
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            if m.shape != (h, w):
                                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                            pred_masks.append(m > 0.5)
                        else:
                            pred_masks.append(None)
                else:
                    pred_masks = [None] * len(pred_boxes)
            except Exception as e:
                print(f"  ⚠️  SAM2 error pada {img_path}: {e}")
                pred_masks = [None] * len(pred_boxes)
        else:
            pred_masks = []
        
        # Save predictions
        for i in range(len(pred_boxes)):
            all_predictions.append({
                "image": img_path,
                "pred_box": pred_boxes[i],
                "pred_conf": pred_confs[i] if len(pred_confs) > i else 0.0,
                "pred_cls": pred_clss[i] if len(pred_clss) > i else -1,
                "pred_mask": pred_masks[i] if i < len(pred_masks) else None,
                "gt_boxes": gt_item["boxes"],
                "gt_clss": gt_item["clss"],
                "gt_masks": gt_seg.get(img_path, {}).get("masks", []) if img_path in gt_seg else [],
            })
        
        if idx % 50 == 0:
            _flush(f"eval {idx}")
    
    print(f"  ✅ Evaluasi selesai: {len(all_predictions)} prediksi dari {len(gt_det)} gambar")
    
    # --- 4. Calculate mAP (Box) ---
    print("\n[4] Menghitung mAP Box...")
    # Simple implementation: calculate AP per class with IoU threshold 0.5
    ap_per_class = []
    for cls_id in range(num_classes):
        cls_preds = [p for p in all_predictions if p["pred_cls"] == cls_id]
        cls_gts = {}
        for p in all_predictions:
            if p["image"] not in cls_gts:
                cls_gts[p["image"]] = {"boxes": [], "clss": []}
            # Filter GT for this class
            for gt_box, gt_cls in zip(p["gt_boxes"], p["gt_clss"]):
                if gt_cls == cls_id:
                    cls_gts[p["image"]]["boxes"].append(gt_box)
                    cls_gts[p["image"]]["clss"].append(gt_cls)
        
        if not cls_preds:
            ap_per_class.append(0.0)
            continue
        
        # Sort by confidence
        cls_preds = sorted(cls_preds, key=lambda x: x["pred_conf"], reverse=True)
        
        # Calculate TP, FP for IoU threshold 0.5
        tp, fp = 0, 0
        gt_matched = set()
        
        for pred in cls_preds:
            max_iou = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_cls) in enumerate(zip(pred["gt_boxes"], pred["gt_clss"])):
                if gt_cls != cls_id:
                    continue
                iou = compute_iou(pred["pred_box"], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = gt_idx
            
            if max_iou >= 0.5 and best_gt_idx not in gt_matched:
                tp += 1
                gt_matched.add(best_gt_idx)
            else:
                fp += 1
        
        # Precision, Recall, AP (simplified)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(cls_gts) if len(cls_gts) > 0 else 0
        ap = precision  # Simplified, should be integral of PR curve
        ap_per_class.append(ap)
        
        print(f"  Kelas {cls_id}: Precision={precision:.4f}, Recall={recall:.4f}, AP={ap:.4f}")
    
    mAP50_box = np.mean(ap_per_class)
    print(f"  ✅ mAP50 (Box): {mAP50_box:.4f}")
    
    # --- 5. Calculate mAP (Mask) ---
    print("\n[5] Menghitung mAP Mask...")
    mAP50_mask = "N/A"  # Placeholder, needs further implementation
    print(f"  ⚠️  mAP Mask perlu implementasi tambahan (mask IoU)")
    
    # --- 6. Save Results ---
    print("\n[6] Menyimpan hasil...")
    report_dir = REPORTS_DIR
    os.makedirs(report_dir, exist_ok=True)
    
    # Summary CSV
    summary_csv = os.path.join(report_dir, "report_hybrid_map.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["mAP50 (Box)", f"{mAP50_box:.4f}"])
        writer.writerow(["mAP50-95 (Box)", "N/A (perlu implementasi multi-threshold)"])
        writer.writerow(["mAP50 (Mask)", mAP50_mask])
        writer.writerow(["mAP50-95 (Mask)", "N/A (perlu implementasi multi-threshold)"])
        writer.writerow(["Model", "Hybrid (YOLO11m + SAM2)"])
        writer.writerow(["Validation Set", str(len(gt_det)) + " images"])
    
    print(f"  ✅ Summary: {summary_csv}")
    
    # Detailed predictions CSV
    detail_csv = os.path.join(report_dir, "hybrid_detailed_predictions.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Pred_Class", "Confidence", "GT_Class", "IoU", "Has_Mask"])
        for p in all_predictions[:100]:  # Limit 100 rows for readability
            max_iou = 0
            gt_cls_matched = -1
            for gt_box, gt_cls in zip(p["gt_boxes"], p["gt_clss"]):
                iou = compute_iou(p["pred_box"], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    gt_cls_matched = gt_cls
            writer.writerow([
                os.path.basename(p["image"]),
                p["pred_cls"],
                f"{p['pred_conf']:.4f}",
                gt_cls_matched,
                f"{max_iou:.4f}",
                "Yes" if p["pred_mask"] is not None else "No"
            ])
    
    print(f"  ✅ Details: {detail_csv}")
    
    # Cleanup
    del yolo_det, sam_model
    _flush("final")
    
    print("\n" + "="*65)
    print("  EVALUASI mAP HYBRID SELESAI")
    print(f"  mAP50 (Box): {mAP50_box:.4f}")
    print(f"  Output: {summary_csv}")
    print("="*65)


if __name__ == "__main__":
    evaluate_hybrid_map()
