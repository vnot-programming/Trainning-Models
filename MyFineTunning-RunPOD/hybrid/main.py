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
    cd hybrid && python3 -u main.py 2>&1 | tee hybrid_map_eval.log
    
  tmux new-session -d -s hybrideval "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Trainning-Models/MyFineTunning-dev/hybrid && python3 -u main.py 2>&1 | tee hybrideval.log"
    python -u Trainning-Models/MyFineTunning-dev/main.py 2>&1 | tee hybrid_map_eval.log
"""

import os, sys, csv, gc, time, yaml
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE, MODELS_DIR,
    get_output_dir, REPORTS_DIR, DATA_FILES_DIR
)
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools
)
import torch
import cv2
import numpy as np
from ultralytics import YOLO, SAM

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"[Device] {DEVICE}")
def get_gpu_report_str():
    """Get GPU report string (e.g., '1x NVIDIA RTX 3060' or '2x NVIDIA RTX 3060')."""
    if not torch.cuda.is_available():
        return "1x CPU"
    from collections import Counter
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    counts = Counter(gpu_names)
    return ", ".join([f"{count}x {name}" for name, count in counts.items()])
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


def resolve_dataset_path(base, relative_path):
    """Resolve a dataset path from YAML, normalizing relative and absolute entries."""
    if os.path.isabs(relative_path):
        return os.path.normpath(relative_path)

    normalized = os.path.normpath(os.path.join(base, relative_path))
    if os.path.isdir(normalized) or not relative_path.startswith(".."):
        return normalized

    # Fallback for Roboflow-style YAML entries like '../valid/images'
    parts = list(Path(relative_path).parts)
    while parts and parts[0] == "..":
        parts = parts[1:]
        candidate = os.path.normpath(os.path.join(base, *parts))
        if os.path.isdir(candidate):
            return candidate

    return normalized


def resolve_dataset_base(yaml_path, cfg):
    """Resolve the dataset base path from YAML file location and optional 'path' entry."""
    yaml_dir = os.path.abspath(os.path.dirname(yaml_path))
    base = cfg.get("path")
    if base is None:
        return yaml_dir
    return resolve_dataset_path(yaml_dir, base)


def infer_label_path_from_image_path(image_dir):
    """Infer the corresponding label directory from an image directory."""
    image_path = Path(image_dir)
    if image_path.name == "images":
        return str(image_path.with_name("labels"))

    if "images" in image_path.parts:
        parts = list(image_path.parts)
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx] == "images":
                parts[idx] = "labels"
                return str(Path(*parts))

    return str(image_path.parent / "labels" / image_path.name)


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


def mask_to_rle(mask):
    """Encode a binary mask into COCO RLE format."""
    import pycocotools.mask as maskUtils

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    rle = maskUtils.encode(np.asfortranarray(mask))
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def build_coco_categories(names):
    return [{"id": idx + 1, "name": name} for idx, name in enumerate(names)]


def build_coco_gt(gt_seg, category_names):
    images = []
    annotations = []
    image_ids = {}
    ann_id = 1

    for idx, image_path in enumerate(sorted(gt_seg.keys()), start=1):
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        image_ids[image_path] = idx
        images.append({
            "id": idx,
            "file_name": os.path.basename(image_path),
            "height": h,
            "width": w,
        })

        for box, cls, mask in zip(gt_seg[image_path]["boxes"], gt_seg[image_path]["clss"], gt_seg[image_path]["masks"]):
            if mask is None:
                continue

            x1, y1, x2, y2 = box
            bbox = [x1, y1, x2 - x1, y2 - y1]
            rle = mask_to_rle(mask)
            area = float(mask.sum())

            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": cls + 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": rle,
            })
            ann_id += 1

    categories = build_coco_categories(category_names)
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "type": "instances",
    }, image_ids


def build_coco_mask_predictions(all_predictions, image_ids):
    results = []
    skipped_no_mask = 0
    skipped_no_image = 0
    
    for p in all_predictions:
        if p["image"] not in image_ids:
            skipped_no_image += 1
            continue
        if p["pred_mask"] is None:
            skipped_no_mask += 1
            continue

        x1, y1, x2, y2 = p["pred_box"]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        results.append({
            "image_id": image_ids[p["image"]],
            "category_id": int(p["pred_cls"]) + 1,
            "bbox": bbox,
            "score": float(p["pred_conf"]),
            "segmentation": mask_to_rle(p["pred_mask"]),
        })
    
    print(f"  ℹ️  Debug: {len(all_predictions)} total prediksi, {skipped_no_image} skipped (image not in GT), {skipped_no_mask} skipped (no mask), {len(results)} valid")
    return results


def evaluate_coco_mask(gt_seg, all_predictions, category_names):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  ⚠️  pycocotools tidak tersedia; mAP Mask tidak dapat dihitung.")
        return "N/A", "N/A"

    coco_gt_data, image_ids = build_coco_gt(gt_seg, category_names)
    if not coco_gt_data["annotations"]:
        print("  ⚠️  Ground truth mask kosong; mAP Mask tidak dapat dihitung.")
        return "N/A", "N/A"

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    coco_preds = build_coco_mask_predictions(all_predictions, image_ids)
    if not coco_preds:
        print("  ⚠️  Tidak ada prediksi mask untuk dievaluasi.")
        return "N/A", "N/A"

    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.imgIds = list(image_ids.values())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = float(coco_eval.stats[1]) if coco_eval.stats.size > 1 else "N/A"
    ap = float(coco_eval.stats[0]) if coco_eval.stats.size > 0 else "N/A"
    return ap50, ap


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
    Supports both detection format (cls cx cy w h) and segmentation format (cls x1 y1 x2 y2 ...).
    Returns: dict {image_path: {'boxes': [[x1,y1,x2,y2], ...], 'clss': [cls1, ...]}}
    """
    cfg = load_yaml(yaml_path)
    base = resolve_dataset_base(yaml_path, cfg)
    
    # Get split folder name from YAML, default to "valid"
    split_folder = cfg.get(split, cfg.get("val", split))
    
    # Build paths from YAML, normalizing relative paths
    img_dir = resolve_dataset_path(base, split_folder)
    label_dir = infer_label_path_from_image_path(img_dir)
    
    if not os.path.isdir(img_dir):
        print(f"  ⚠️  Image dir tidak ditemukan: {img_dir}")
        return {}
    if not os.path.isdir(label_dir):
        print(f"  ⚠️  Label dir tidak ditemukan: {label_dir}")
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
                        
                        # Check if segmentation format (polygon with many points)
                        if len(parts) >= 7:
                            # Segmentation format: cls x1 y1 x2 y2 ... xn yn (normalized)
                            polygon_points = list(map(float, parts[1:]))
                            points = np.array(polygon_points).reshape(-1, 2)
                            points[:, 0] *= img_w
                            points[:, 1] *= img_h
                            points = points.astype(np.int32)
                            
                            # Compute bounding box from polygon
                            x1 = float(points[:, 0].min())
                            y1 = float(points[:, 1].min())
                            x2 = float(points[:, 0].max())
                            y2 = float(points[:, 1].max())
                            boxes.append([x1, y1, x2, y2])
                        else:
                            # Detection format: cls cx cy w h (normalized)
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
    base = resolve_dataset_base(yaml_path, cfg)
    
    # Get split folder name
    split_folder = cfg.get(split, cfg.get("val", split))
    
    # Build paths from YAML, normalizing relative paths
    img_dir = resolve_dataset_path(base, split_folder)
    label_dir = infer_label_path_from_image_path(img_dir)
    
    if not os.path.isdir(img_dir):
        print(f"  ⚠️  Image dir tidak ditemukan: {img_dir}")
        return {}
    if not os.path.isdir(label_dir):
        print(f"  ⚠️  Label dir tidak ditemukan: {label_dir}")
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
    print("  Evaluasi mAP Hybrid Pipeline (YOLO11l + SAM2)")
    print("="*65)
    
    # --- 1. Load Models ---
    print("\n[1] Loading Models...")
    yolo11l_path = os.path.join(get_output_dir("yolo11l"), "weights", "best.pt")
    if not os.path.exists(yolo11l_path):
        print(f"  ❌ YOLO11l best.pt tidak ditemukan: {yolo11l_path}")
        return
    
    yolo_det = YOLO(yolo11l_path)
    print(f"  ✅ YOLO11l loaded: {yolo11l_path}")
    
    try:
        sam_path = os.path.join(MODELS_DIR, "sam2.1_t.pt")
        sam_model = SAM(sam_path)
        print("  ✅ SAM2 loaded")
    except Exception as e:
        print(f"  ❌ Gagal load SAM2: {e}")
        return
        
    # Kalkulasi ukuran model dinamis
    try:
        yolo_mb = os.path.getsize(yolo11l_path) / 1e6
        sam_mb = os.path.getsize(os.path.join(MODELS_DIR, "sam2.1_t.pt")) / 1e6
        total_mb = yolo_mb + sam_mb
        model_size_str = f"{total_mb:.2f}"
    except Exception:
        model_size_str = "N/A (Multi)"
        
    # --- 2. Load Ground Truth ---
    print("\n[2] Loading Ground Truth...")
    
    # Load BOTH from SEG_YAML to ensure consistency
    # Segmentation ground truth (with masks generated from polygons)
    gt_seg = load_ground_truth_masks(SEG_YAML, split="valid")
    if not gt_seg:
        print("  ⚠️  Tidak ada ground truth segmentasi. Evaluasi dibatalkan.")
        return
        
    # Detection ground truth (using the exact same images as segmentation)
    # This ensures we evaluate on the same images
    gt_det = load_ground_truth_labels(SEG_YAML, split="valid")
    
    if not gt_det:
        print("  ⚠️  Tidak ada ground truth deteksi. Evaluasi dibatalkan.")
        return
        
    print(f"  ✅ Using same dataset for detection & segmentation: {len(gt_det)} images")
    
    # --- 3. Evaluate on validation set ---
    print("\n[3] Evaluasi pada validation set...")
    all_predictions = []
    
    # Get number of classes from YAML
    det_cfg = load_yaml(DET_YAML)
    num_classes = det_cfg.get("nc", 4)  # Default 4 if not specified
    mask_count = 0
    
    # For dynamic latency measurement
    total_yolo_pre, total_yolo_inf, total_yolo_post = 0.0, 0.0, 0.0
    total_sam_pre, total_sam_inf, total_sam_post = 0.0, 0.0, 0.0
    latency_measured_images = 0
    
    for idx, (img_path, gt_item) in enumerate(gt_det.items(), 1):
        if idx % 10 == 0:
            print(f"  Proses {idx}/{len(gt_seg)}...")
        
        gt_item = gt_det.get(img_path, {"boxes": [], "clss": []})
        
        # YOLO Detection
        det_result = yolo_det.predict(img_path, conf=0.5, verbose=False, imgsz=IMAGE_SIZE)
        
        # Accumulate YOLO speeds
        yolo_speed = getattr(det_result[0], 'speed', {})
        total_yolo_pre += yolo_speed.get('preprocess', 0.0)
        total_yolo_inf += yolo_speed.get('inference', 0.0)
        total_yolo_post += yolo_speed.get('postprocess', 0.0)
        
        if det_result and det_result[0].boxes is not None and len(det_result[0].boxes) > 0:
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
                
                # Accumulate SAM speeds
                sam_speed = getattr(sam_result[0], 'speed', {})
                total_sam_pre += sam_speed.get('preprocess', 0.0)
                total_sam_inf += sam_speed.get('inference', 0.0)
                total_sam_post += sam_speed.get('postprocess', 0.0)
                
                if sam_result and sam_result[0].masks is not None:
                    for m in sam_result[0].masks.data.cpu().numpy():
                        # Resize mask to original size
                        img = cv2.imread(img_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            if m.shape != (h, w):
                                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                            pred_masks.append(m > 0.5)
                            mask_count += 1
                        else:
                            pred_masks.append(None)
                else:
                    pred_masks = [None] * len(pred_boxes)
            except Exception as e:
                print(f"  ⚠️  SAM2 error pada {img_path}: {e}")
                pred_masks = [None] * len(pred_boxes)
                # Ensure we add something for speed if predict failed but we still want average
        else:
            pred_masks = []
        
        latency_measured_images += 1
        
        # Save predictions
        for i in range(len(pred_boxes)):
            all_predictions.append({
                "image": img_path,
                "pred_box": pred_boxes[i].tolist(),
                "pred_conf": float(pred_confs[i]) if len(pred_confs) > i else 0.0,
                "pred_cls": int(pred_clss[i]) if len(pred_clss) > i else -1,
                "pred_mask": pred_masks[i] if i < len(pred_masks) else None,
                "gt_boxes": gt_item["boxes"],
                "gt_clss": gt_item["clss"],
                "gt_masks": gt_seg.get(img_path, {}).get("masks", []),
            })
        
        if idx % 50 == 0:
            _flush(f"eval {idx}")
            
    print(f"  ✅ Evaluasi selesai: {len(all_predictions)} prediksi, {mask_count} masks dari {len(gt_seg)} gambar")
    
    # Calculate Latency & FPS Dynamically
    if latency_measured_images > 0:
        avg_yolo_pre = total_yolo_pre / latency_measured_images
        avg_yolo_inf = total_yolo_inf / latency_measured_images
        avg_yolo_post = total_yolo_post / latency_measured_images
        
        avg_sam_pre = total_sam_pre / latency_measured_images
        avg_sam_inf = total_sam_inf / latency_measured_images
        avg_sam_post = total_sam_post / latency_measured_images
        
        final_pre = avg_yolo_pre + avg_sam_pre
        final_inf = avg_yolo_inf + avg_sam_inf
        final_post = avg_yolo_post + avg_sam_post
        
        latency_ms = final_pre + final_inf + final_post
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    else:
        final_pre, final_inf, final_post, latency_ms, fps = 0.0, 0.0, 0.0, 0.0, 0.0
    
    print(f"  ✅ Kecepatan Dinamis (YOLO+SAM): Pre={final_pre:.2f}ms, Inf={final_inf:.2f}ms, Post={final_post:.2f}ms")
    print(f"  ✅ Latency Total={latency_ms:.2f}ms, FPS={fps:.2f}")

    # --- 4. Calculate mAP (Box) and mAP (Mask) using COCOeval ---
    print("\n[4] Menghitung mAP COCOeval...")
    if not check_pycocotools():
        print("  ⚠️  pycocotools tidak tersedia. Gagal menghitung mAP dengan akurat.")
        mAP50_box, mAP50_95_box, mAP50_mask, mAP50_95_mask = "N/A", "N/A", "N/A", "N/A"
        precision_box, recall_box = "N/A", "N/A"
    else:
        # We need to construct predictions for `evaluate_coco_predictions`
        # which expects: [{"image": path, "pred_box": [x,y,x,y], "pred_cls": cls, "pred_conf": conf, "pred_mask": mask}, ...]
        coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
        
        if coco_gt_dict is None:
            print("  ⚠️  Gagal memuat ground truth COCO. mAP tidak dapat dihitung.")
            mAP50_box, mAP50_95_box, mAP50_mask, mAP50_95_mask = "N/A", "N/A", "N/A", "N/A"
            precision_box, recall_box = "N/A", "N/A"
        else:
            print("  📊 COCOeval Deteksi (Box)...")
            mAP50_box, mAP50_95_box = evaluate_coco_predictions(coco_gt_dict, image_ids, all_predictions, iou_type="bbox")
            
            # Simple Precision / Recall calc
            tp, fp, fn = 0, 0, 0
            for pred in sorted(all_predictions, key=lambda x: x["pred_conf"], reverse=True):
                if pred["pred_conf"] < 0.5:
                    continue # only count high conf for P/R or we use all? Usually confidence threshold.
                matched = False
                for gt_b, gt_c in zip(pred["gt_boxes"], pred["gt_clss"]):
                    if gt_c == pred["pred_cls"] and compute_iou(pred["pred_box"], gt_b) >= 0.5:
                        tp += 1
                        matched = True
                        break
                if not matched:
                    fp += 1
            # GT total for FN
            total_gt = sum(len(g["clss"]) for g in gt_det.values())
            precision_box = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_box = tp / total_gt if total_gt > 0 else 0.0

            print("  📊 COCOeval Segmentasi (Mask)...")
            mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(coco_gt_dict, image_ids, all_predictions, iou_type="segm")

    # --- 6. Save Results ---
    print("\n[6] Menyimpan hasil CSV Hybrid (Evaluasi Mandiri)...")
    report_dir = REPORTS_DIR
    os.makedirs(report_dir, exist_ok=True)
    
    det_fields = ["Model", "Model Size (MB)", "mAP50-95", "mAP50", "Precision", "Recall", 
                  "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"]
    det_csv = os.path.join(report_dir, "report_hybrid_det_coco.csv")
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=det_fields)
        writer.writeheader()
        writer.writerow({
            "Model": "Hybrid (YOLO11m+SAM2)",
            "Model Size (MB)": model_size_str,
            "mAP50-95": mAP50_95_box if mAP50_95_box != "N/A" else "ERR",
            "mAP50": mAP50_box if mAP50_box != "N/A" else "ERR",
            "Precision": f"{precision_box:.4f}" if isinstance(precision_box, float) else precision_box,
            "Recall": f"{recall_box:.4f}" if isinstance(recall_box, float) else recall_box,
            "Preprocess (ms)": f"{final_pre:.2f}",
            "Inference (ms)": f"{final_inf:.2f}",
            "Postprocess (ms)": f"{final_post:.2f}",
            "Latency (ms)": f"{latency_ms:.2f}",
            "FPS": f"{fps:.2f}",
            "GPUs": get_gpu_report_str(),
            "Evaluator": "COCOeval"
        })
    print(f"  ✅ Detection report: {det_csv}")
    
    seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)", "Latency (ms)", "FPS", "GPUs", "Evaluator"]
    seg_csv = os.path.join(report_dir, "report_hybrid_seg_coco.csv")
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seg_fields)
        writer.writeheader()
        writer.writerow({
            "Model": "Hybrid (YOLO11m+SAM2)",
            "Model Size (MB)": model_size_str,
            "mAP50-95(Box)": mAP50_95_box if mAP50_95_box != "N/A" else "ERR",
            "mAP50-95(Mask)": mAP50_95_mask if mAP50_95_mask != "N/A" else "ERR",
            "Latency (ms)": f"{latency_ms:.2f}",
            "FPS": f"{fps:.2f}",
            "GPUs": get_gpu_report_str(),
            "Evaluator": "COCOeval"
        })
    print(f"  ✅ Segmentation report: {seg_csv}")
    
    detail_csv = os.path.join(report_dir, "hybrid_detailed_predictions.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Pred_Class", "Confidence", "GT_Class", "IoU", "Has_Mask"])
        for p in all_predictions[:100]:
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
    
    # --- 7. Gabungkan Laporan ---
    print("\n[7] Menggabungkan Laporan (Aggregator)...")
    
    det_available = [
        ("report_yolo11l_det_multigpu.csv", "YOLO11l (MultiGPU)"),
        ("report_yolov9m_det_multigpu.csv", "YOLOv9m (MultiGPU)"),
        ("report_yolov8m_det_multigpu.csv", "YOLOv8m (MultiGPU)"),
        ("report_hybrid_det_coco.csv", "Hybrid (YOLO11l+SAM2)"),
    ]
    
    seg_available = [
        ("report_yolo11l_seg_multigpu.csv", "YOLO11l-Seg (MultiGPU)"),
        ("report_yolov9c_seg_multigpu.csv", "YOLOv9c-Seg (MultiGPU)"),
        ("report_yolov8m_seg_multigpu.csv", "YOLOv8m-Seg (MultiGPU)"),
        ("report_hybrid_seg_coco.csv", "Hybrid (YOLO11l+SAM2)"),
        ("report_maskrcnn_ddp_seg_multigpu.csv", "Mask R-CNN (MultiGPU)"),
        ("report_maskrcnn_ddp_seg.csv", "Mask R-CNN (Local)"),
    ]
    
    combined_det_csv = os.path.join(report_dir, "report_evaluasi_detection.csv")
    det_missing = []
    det_rows_all = []
    for fname, model_label in det_available:
        csv_path = os.path.join(report_dir, fname)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    det_rows_all.append(row)
        else:
            det_missing.append((fname, model_label))
            
    if det_rows_all:
        with open(combined_det_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=det_fields)
            writer.writeheader()
            writer.writerows(det_rows_all)
        print(f"  ✅ Combined Detection: {combined_det_csv}")
    
    combined_seg_csv = os.path.join(report_dir, "report_evaluasi_segmentation.csv")
    seg_missing = []
    seg_rows_all = []
    for fname, model_label in seg_available:
        csv_path = os.path.join(report_dir, fname)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    seg_rows_all.append(row)
        else:
            seg_missing.append((fname, model_label))
            
    if seg_rows_all:
        with open(combined_seg_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=seg_fields)
            writer.writeheader()
            writer.writerows(seg_rows_all)
        print(f"  ✅ Combined Segmentation: {combined_seg_csv}")

    # --- 8. Narrative Reports ---
    print("\n[8] Update Laporan Naratif (Markdown)...")
    
    # Deteksi Markdown (Global)
    md_det_global = os.path.join(DATA_FILES_DIR, "report_evaluasi_detection.md")
    
    # Deteksi Markdown (Narrative)
    narrative_dir = os.path.join(report_dir, "narrative_reports")
    os.makedirs(narrative_dir, exist_ok=True)
    md_det_narr = os.path.join(narrative_dir, "Laporan_Evaluasi_Detection_(Box)_-_Semua_Models.md")
    
    det_md_content = """# Laporan Evaluasi Detection (Box) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Detection (Box)

| Model | mAP50 | mAP50-95 (Box) | Precision | Recall | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|--------|-------------------|-----------|--------|-----------------|----------------|------------------|---------------|-----|------|-----------|
"""
    if det_rows_all:
        for r in det_rows_all:
            det_md_content += f"| **{r.get('Model', '')}** | {r.get('mAP50', '')} | {r.get('mAP50-95', '')} | {r.get('Precision', '')} | {r.get('Recall', '')} | {r.get('Preprocess (ms)', '')}ms | {r.get('Inference (ms)', '')}ms | {r.get('Postprocess (ms)', '')}ms | {r.get('Latency (ms)', '')}ms | {r.get('FPS', '')} | {r.get('GPUs', '')} | {r.get('Evaluator', '')} |\n"
    
    with open(md_det_global, "w") as f: f.write(det_md_content)
    with open(md_det_narr, "w") as f: f.write(det_md_content)
    
    # Segmentasi Markdown (Global)
    md_seg_global = os.path.join(DATA_FILES_DIR, "report_evaluasi_segmentation.md")
    
    # Segmentasi Markdown (Narrative)
    md_seg_narr = os.path.join(narrative_dir, "Laporan_Evaluasi_Segmentation_(Mask)_-_Semua_Models.md")
    
    seg_md_content = """# Laporan Evaluasi Segmentation (Mask) - Semua Model
## Mono: COCOeval (pycocotools)

Evaluator: COCOeval (standar industri)

---

## 📊 Tabel Perbandingan mAP Segmentation (Mask)

| Model | mAP50-95 (Box) | mAP50-95 (Mask) | Latency (ms) | FPS | GPUs | Evaluator |
|-------|----------------|-----------------|--------------|-----|------|-----------|
"""
    if seg_rows_all:
        for r in seg_rows_all:
            seg_md_content += f"| **{r.get('Model', '')}** | {r.get('mAP50-95(Box)', '')} | {r.get('mAP50-95(Mask)', '')} | {r.get('Latency (ms)', '')}ms | {r.get('FPS', '')} | {r.get('GPUs', '')} | {r.get('Evaluator', '')} |\n"
            
    with open(md_seg_global, "w") as f: f.write(seg_md_content)
    with open(md_seg_narr, "w") as f: f.write(seg_md_content)
    
    print("\n" + "="*65)
    print("  EVALUASI mAP HYBRID SELESAI")
    print("="*65)
    
    # Cleanup
    del yolo_det, sam_model
    _flush("final")
    
if __name__ == "__main__":
    evaluate_hybrid_map()

    # --- 9. Generate Visual Comparisons ---
    print("\n[9] Generate Visual Comparisons...")
    try:
        sys.path.insert(0, ROOT)
        from visual_utils import generate_single_hybrid, generate_hybrid_grids
        generate_single_hybrid(is_multigpu=False)
        generate_hybrid_grids(is_multigpu=False)
    except Exception as _ve:
        print(f"[Visual] ⚠️  Gagal generate visual: {_ve}")
