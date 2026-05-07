# -*- coding: utf-8 -*-
"""
coco_eval_utils.py
==================
Utility untuk evaluasi mAP menggunakan COCOeval (pycocotools).
Bisa digunakan oleh: YOLOv8, YOLOv9, YOLO11, Mask R-CNN, dan Hybrid.
Semua model WAJIB menggunakan COCOeval agar hasil evaluasi KOMPARABEL.
"""

import os
import sys
import csv
import gc
import numpy as np
from pathlib import Path


def check_pycocotools():
    """Cek apakah pycocotools terinstall."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return True
    except ImportError:
        print("[COCO] ⚠️  pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return False


def build_coco_ground_truth(yaml_path, split="valid", category_names=None):
    """
    Build COCO ground truth dari dataset YAML (YOLO format).
    
    Returns:
        coco_gt_dict: dict dalam format COCO
        image_ids: dict {image_path: coco_image_id}
    """
    from pycocotools.coco import COCO
    import yaml
    
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Resolve paths
    yaml_dir = os.path.abspath(os.path.dirname(yaml_path))
    
    # Get the split folder from YAML
    if split == "valid":
        split_folder = cfg.get("val", cfg.get("valid", split))
    elif split == "test":
        split_folder = cfg.get("test", split)
    else:
        split_folder = cfg.get(split, split)
    
    # SIMPLIFIED: For Roboflow, ignore "../" prefix
    # Roboflow YAML has "../valid/images" but YAML is in dataset root
    # So we should treat it as "valid/images" relative to yaml_dir
    
    # Remove "../" prefixes
    clean_path = split_folder
    while clean_path.startswith("../"):
        clean_path = clean_path[3:]
    
    # Now clean_path should be like "valid/images" or "test/images"
    # Try joining with yaml_dir
    img_dir = os.path.normpath(os.path.join(yaml_dir, clean_path))
    
    # If not found, try extracting just the split name
    if not os.path.isdir(img_dir):
        parts = clean_path.split("/")
        if len(parts) >= 2 and parts[1] == "images":
            split_name = parts[0]
            img_dir = os.path.join(yaml_dir, split_name, "images")
    
    # If still not found, try parent directory (original Roboflow intent)
    if not os.path.isdir(img_dir):
        parent_dir = os.path.dirname(yaml_dir)
        img_dir = os.path.normpath(os.path.join(parent_dir, clean_path))
    
    if not os.path.isdir(img_dir):
        print(f"[COCO] ⚠️  Image dir tidak ditemukan: {img_dir}")
        print(f"[COCO]    yaml_path: {yaml_path}")
        print(f"[COCO]    yaml_dir: {yaml_dir}")
        print(f"[COCO]    split_folder: {split_folder}")
        print(f"[COCO]    clean_path: {clean_path}")
        return None, None
    
    print(f"[COCO] ✅ Image dir ditemukan: {img_dir}")
    
    # Infer label dir
    image_path = Path(img_dir)
    if image_path.name == "images":
        label_dir = str(image_path.with_name("labels"))
    elif "images" in image_path.parts:
        parts = list(image_path.parts)
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx] == "images":
                parts[idx] = "labels"
                label_dir = str(Path(*parts))
                break
    else:
        label_dir = str(image_path.parent / "labels" / image_path.name)
    
    if not os.path.isdir(label_dir):
        print(f"[COCO] ⚠️  Label dir tidak ditemukan: {label_dir}")
        return None, None
    
    print(f"[COCO] ✅ Label dir ditemukan: {label_dir}")
    
    # Category names
    if category_names is None:
        category_names = cfg.get("names", {})
        if isinstance(category_names, dict):
            category_names = [category_names[i] for i in sorted(category_names.keys())]
    
    coco_gt_dict = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(category_names)],
        "type": "instances",
    }
    
    image_ids = {}
    ann_id = 1
    
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.lower().endswith(valid_exts):
            continue
        
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Get image dimensions
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Add image
        img_id = len(coco_gt_dict["images"]) + 1
        image_ids[img_path] = img_id
        coco_gt_dict["images"].append({
            "id": img_id,
            "file_name": img_file,
            "height": h,
            "width": w,
        })
        
        # Parse labels
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    cls = int(parts[0])
                    
                    # Check if segmentation format (polygon) or detection format
                    if len(parts) >= 7:  # Segmentation: cls x1 y1 x2 y2 ... xn yn
                        polygon_points = list(map(float, parts[1:]))
                        points = np.array(polygon_points).reshape(-1, 2)
                        points[:, 0] *= w
                        points[:, 1] *= h
                        points = points.astype(np.int32)
                        
                        # Compute bounding box from polygon
                        x1 = float(points[:, 0].min())
                        y1 = float(points[:, 1].min())
                        x2 = float(points[:, 0].max())
                        y2 = float(points[:, 1].max())
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        
                        # Convert mask to RLE
                        from pycocotools import mask as maskUtils
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 1)
                        rle = maskUtils.encode(np.asfortranarray(mask))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        
                        coco_gt_dict["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls + 1,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "segmentation": rle,
                            "iscrowd": 0,
                        })
                        ann_id += 1
                    else:  # Detection format: cls x_center y_center width height
                        xc, yc, bw, bh = map(float, parts[1:5])
                        bbox = [(xc - bw/2) * w, (yc - bh/2) * h, bw * w, bh * h]
                        coco_gt_dict["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls + 1,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "segmentation": [],
                            "iscrowd": 0,
                        })
                        ann_id += 1
    
    print(f"[COCO] ✅ Ground truth dibuat: {len(coco_gt_dict['images'])} images, {len(coco_gt_dict['annotations'])} annotations")
    return coco_gt_dict, image_ids


def evaluate_coco_predictions(coco_gt_dict, image_ids, predictions, iou_type="bbox"):
    """
    Evaluate predictions using COCOeval.
    
    Args:
        coco_gt_dict: COCO ground truth dict
        image_ids: dict {image_path: coco_image_id}
        predictions: list of dict with keys: image, pred_box/pred_mask, pred_cls, pred_conf
        iou_type: "bbox" or "segm"
    
    Returns:
        mAP50, mAP50_95
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    # Create COCO GT object from dict
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    
    # Build predictions in COCO format
    coco_preds = []
    for pred in predictions:
        img_path = pred["image"]
        if img_path not in image_ids:
            continue
        
        img_id = image_ids[img_path]
        cls = pred["pred_cls"]
        conf = pred["pred_conf"]
        
        if iou_type == "bbox":
            bbox = pred["pred_box"]
            # COCO format: [x1, y1, width, height]
            x1, y1, x2, y2 = bbox
            coco_preds.append({
                "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
            })
        elif iou_type == "segm":
            if "pred_mask" not in pred:
                continue
            mask = pred["pred_mask"]
            # Convert mask to RLE
            from pycocotools import mask as maskUtils
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            
            bbox = pred["pred_box"]
            x1, y1, x2, y2 = bbox
            coco_preds.append({
                "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle,
                "score": conf,
            })
    
    # Create COCO preds object
    coco_dt = coco_gt.loadRes(coco_preds)
    
    # Run COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract mAP values
    mAP50 = coco_eval.stats[1]  # mAP @ IoU=0.50
    mAP50_95 = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
    
    return round(float(mAP50), 4), round(float(mAP50_95), 4)


def save_detection_report_csv(model_name, mAP50, mAP50_95, precision, recall, model_size_mb, preprocess_ms, inference_ms, postprocess_ms, output_dir):
    """Save detection report to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"report_{model_name}_det.csv")
    
    fields = ["Model", "Model Size (MB)", "mAP50-95", "mAP50", "Precision", "Recall",
              "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]
    
    row = {
        "Model": model_name,
        "Model Size (MB)": model_size_mb,
        "mAP50-95": mAP50_95,
        "mAP50": mAP50,
        "Precision": precision,
        "Recall": recall,
        "Preprocess (ms)": preprocess_ms,
        "Inference (ms)": inference_ms,
        "Postprocess (ms)": postprocess_ms,
    }
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    
    print(f"[COCO] ✅ Detection report disimpan: {csv_path}")
    return csv_path


def save_segmentation_report_csv(model_name, box_mAP50, box_mAP50_95, mask_mAP50, mask_mAP50_95, latency_ms, fps, gpu_report, output_dir, evaluator="COCOeval"):
    """Save segmentation report to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"report_{model_name}_seg.csv")
    
    fields = ["Model", "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)",
              "Latency(ms)", "FPS", "GPUs", "Evaluator"]
    
    # Get model size
    model_size_mb = "N/A"
    if os.path.exists(model_name):
        model_size_mb = round(os.path.getsize(model_name) / 1e6, 2)
    
    row = {
        "Model": model_name,
        "Model Size (MB)": model_size_mb,
        "mAP50-95(Box)": box_mAP50_95,
        "mAP50-95(Mask)": mask_mAP50_95,
        "Latency(ms)": latency_ms,
        "FPS": fps,
        "GPUs": gpu_report,
        "Evaluator": evaluator,
    }
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    
    print(f"[COCO] ✅ Segmentation report disimpan: {csv_path}")
    return csv_path
