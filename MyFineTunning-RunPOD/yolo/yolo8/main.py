# -*- coding: utf-8 -*-
"""
yolo/yolo8/main.py
==================
Fine-tuning YOLOv8 Medium pada dataset botol plastik (RVM).

Model:
  - yolov8m.pt      → Detection
  - yolov8m-seg.pt  → Instance Segmentation

Cara menjalankan:
    python -u main.py 2>&1 | tee yolov8_train.log
"""

import os, sys, csv, gc
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# GPU Fan Manager
try:
    from gpu_fan_manager import start_fan_manager
    start_fan_manager()
except ImportError:
    print("[Warning] gpu_fan_manager.py not found in ROOT.")

from config_shared import (
    WORKSPACE_DIR, DET_YAML, SEG_YAML, MODELS_DIR,
    EPOCHS, IMAGE_SIZE, YOLO_BATCH_SIZE, get_output_dir, compress_run,
    save_yolo_visual_samples, parse_device, download_and_move_model, REPORTS_DIR
)
from telegram_utils import get_yolo_callbacks, send_telegram_msg
import argparse
import torch
from ultralytics import YOLO, settings

settings.update({'weights_dir': MODELS_DIR})

# Import COCO eval utils
sys.path.insert(0, os.path.join(ROOT, ".."))  # Agar bisa import coco_eval_utils
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, 
    save_detection_report_csv, save_segmentation_report_csv,
    check_pycocotools
)

# --- Argumen CLI ---
parser = argparse.ArgumentParser(description="YOLOv8m Fine-tuning")
parser.add_argument(
    "--device", type=str, default=None,
    help="GPU yang digunakan. Contoh: '0', '1,2', '0,1,2', 'cpu'. "
         "Default: semua GPU tersedia."
)
args = parser.parse_args()

if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    n = torch.cuda.device_count()
    DEVICE = list(range(n)) if n > 1 else (0 if n == 1 else "cpu")

print(f"[Device] YOLOv8m → {DEVICE}")


def get_gpu_report_str(device):
    if device == "cpu":
        return "1x CPU"
    from collections import Counter
    ids = [device] if isinstance(device, int) else device
    gpu_names = [torch.cuda.get_device_name(i) for i in ids]
    counts = Counter(gpu_names)
    return ", ".join([f"{count}x {name}" for name, count in counts.items()])


def _flush(label):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"[MemFlush] {label} — VRAM bebas: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")


def _train(model_pt, yaml_path, run_name, label):
    out_dir = get_output_dir(run_name)
    best_pt = os.path.join(out_dir, "weights", "best.pt")
    last_pt = os.path.join(out_dir, "weights", "last.pt")

    if os.path.exists(best_pt):
        print(f"\n[SKIP] {label}: training sudah selesai.\n  best.pt: {best_pt}")
        return best_pt

    if os.path.exists(last_pt):
        print(f"\n[RESUME] {label}: melanjutkan dari last.pt\n  {last_pt}")
        model = YOLO(last_pt)
        # Tambahkan Telegram Callbacks (meskipun resume)
        for k, v in get_yolo_callbacks(label).items():
            model.add_callback(k, v)
        model.train(resume=True)
    else:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        model = YOLO(model_pt)
        # Tambahkan Telegram Callbacks
        for k, v in get_yolo_callbacks(label).items():
            model.add_callback(k, v)
            
        model.train(data=yaml_path, epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=YOLO_BATCH_SIZE,
                    project=os.path.dirname(out_dir), name=os.path.basename(out_dir),
                    exist_ok=True, device=DEVICE)

    result = str(model.trainer.best)
    del model; _flush(label)
    return result


def _eval_det(label, pt, yaml):
    """Detection — Ultralytics native evaluation."""
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        box  = met.box
        del m; _flush(f"eval {label}")
        return {
            "Model":            label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95":        round(float(box.map),   4),
            "mAP50":           round(float(box.map50), 4),
            "Precision":       round(float(box.mp),    4),
            "Recall":          round(float(box.mr),    4),
            "Preprocess (ms)": pre,
            "Inference (ms)":  inf,
            "Postprocess (ms)":post,
            "Latency (ms)": round(pre+inf+post, 2),
            "FPS": round(1000/(pre+inf+post), 2) if pre+inf+post>0 else "N/A",
            "GPUs": get_gpu_report_str(DEVICE),
            "Evaluator":       "Ultralytics",
        }
    except Exception as e:
        print(f"  ⚠️ {label}: {e}")
        return {"Model": label, **{k: "ERR" for k in [
            "Model Size (MB)", "mAP50-95", "mAP50", "Precision", "Recall",
            "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"]}}


def _eval_seg(label, pt, yaml):
    """Segmentation — format reporter.py seg_performance."""
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        total_ms = round(pre + inf + post, 2)
        fps = round(1000 / total_ms, 2) if total_ms > 0 else "N/A"
        try:
            box_map  = round(float(met.box.map), 4)
            mask_map = round(float(met.seg.map), 4)
        except Exception:
            box_map = mask_map = "N/A"
        del m; _flush(f"eval {label}")
        return {
            "Model":           label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95(Box)":   box_map,
            "mAP50-95(Mask)":  mask_map,
            "Latency (ms)":     total_ms,
            "FPS":             fps,
            "GPUs":            get_gpu_report_str(DEVICE),
        "Evaluator":       "Ultralytics",
        }
    except Exception as e:
        print(f"  ⚠️ {label}: {e}")
        return {"Model": label, **{k: "ERR" for k in [
            "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)",
            "Latency (ms)", "FPS", "GPUs", "Evaluator"]}}


def _coco_eval_det(label, pt, yaml):
    """Detection — COCOeval dengan Precision & Recall dari prediksi nyata."""
    if not check_pycocotools():
        print(f"  ⚠️ {label}: pycocotools tidak tersedia, fallback ke _eval_det()")
        return None  # Signal to use fallback
    
    try:
        # Build ground truth
        coco_gt_dict, image_ids = build_coco_ground_truth(yaml, split="valid")
        if coco_gt_dict is None:
            print(f"  ⚠️ {label}: Gagal build COCO GT, fallback ke _eval_det()")
            return None
        
        # Run inference & collect predictions in COCO format
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        
        # Collect predictions in COCO format
        predictions = []
        img_dir = os.path.join(os.path.dirname(yaml), "valid", "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(os.path.dirname(yaml), "test", "images")
        
        # Also build GT for P/R calculation
        gt_det = {}
        for img_path, img_id in image_ids.items():
            gt_det[img_path] = {"boxes": [], "clss": []}
            for ann in coco_gt_dict["annotations"]:
                if ann["image_id"] == img_id:
                    bbox = ann["bbox"]
                    # Convert [x, y, w, h] to [x1, y1, x2, y2]
                    gt_det[img_path]["boxes"].append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    gt_det[img_path]["clss"].append(ann["category_id"] - 1)  # Convert to 0-indexed
        
        print(f"  🔍 Mengumpulkan prediksi deteksi untuk COCOeval...")
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, img_file)
            if img_path not in image_ids:
                continue
            
            result = m.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, verbose=False)
            if result and result[0].boxes is not None:
                boxes = result[0].boxes.xyxy.cpu().numpy()
                confs = result[0].boxes.conf.cpu().numpy()
                clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, clss):
                    predictions.append({
                        "image": img_path,
                        "pred_box": box.tolist(),
                        "pred_cls": int(cls),
                        "pred_conf": float(conf),
                    })
        
        print(f"  ✅ Total prediksi deteksi: {len(predictions)}")
        
        # Evaluate with COCOeval
        print(f"  📊 Menjalankan COCOeval untuk deteksi...")
        mAP50, mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, predictions, iou_type="bbox")
        
        # Calculate Precision & Recall manually from predictions
        print(f"  🔍 Menghitung Precision & Recall...")
        
        # Count GT per class
        gt_per_class = {}
        for img_path, gt_item in gt_det.items():
            for cls in gt_item["clss"]:
                gt_per_class[cls] = gt_per_class.get(cls, 0) + 1
        
        total_tp, total_fp, total_gt = 0, 0, 0
        num_classes = max(gt_per_class.keys()) + 1 if gt_per_class else 1
        
        for cls_id in range(num_classes):
            cls_preds = [p for p in predictions if p["pred_cls"] == cls_id]
            cls_gt_count = gt_per_class.get(cls_id, 0)
            
            if not cls_preds:
                total_gt += cls_gt_count
                continue
            
            tp, fp = 0, 0
            gt_matched = set()
            
            # Get all GT boxes for this class
            gt_boxes_for_class = []
            for img_path, gt_item in gt_det.items():
                for idx, (box, cls) in enumerate(zip(gt_item["boxes"], gt_item["clss"])):
                    if cls == cls_id:
                        gt_boxes_for_class.append((img_path, idx, box))
            
            for pred in sorted(cls_preds, key=lambda x: x["pred_conf"], reverse=True):
                max_iou = 0
                best_gt_key = None
                
                for gt_img_path, gt_idx, gt_box in gt_boxes_for_class:
                    # Only match predictions with GT from same image
                    if pred["image"] != gt_img_path:
                        continue
                    # IoU calculation
                    x1 = max(pred["pred_box"][0], gt_box[0])
                    y1 = max(pred["pred_box"][1], gt_box[1])
                    x2 = min(pred["pred_box"][2], gt_box[2])
                    y2 = min(pred["pred_box"][3], gt_box[3])
                    
                    inter_w = max(0, x2 - x1)
                    inter_h = max(0, y2 - y1)
                    inter_area = inter_w * inter_h
                    
                    pred_area = (pred["pred_box"][2] - pred["pred_box"][0]) * (pred["pred_box"][3] - pred["pred_box"][1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union_area = pred_area + gt_area - inter_area
                    
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_key = (gt_img_path, gt_idx)
                
                if max_iou >= 0.5 and best_gt_key not in gt_matched:
                    tp += 1
                    gt_matched.add(best_gt_key)
                else:
                    fp += 1
            
            total_tp += tp
            total_fp += fp
            total_gt += cls_gt_count
        
        precision_val = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall_val = total_tp / total_gt if total_gt > 0 else 0.0
        
        del m; _flush(f"eval {label} (COCOeval)")
        
        print(f"  ✅ COCOeval selesai: mAP50={mAP50}, mAP50-95={mAP50_95}")
        print(f"  ✅ Precision: {precision_val:.4f}, Recall: {recall_val:.4f}")
        
        return {
            "Model":            label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95":        mAP50_95,
            "mAP50":           mAP50,
            "Precision":       round(precision_val, 4),
            "Recall":          round(recall_val, 4),
            "Preprocess (ms)": pre,
            "Inference (ms)":  inf,
            "Postprocess (ms)":post,
            "Latency (ms)": round(pre+inf+post, 2),
            "FPS": round(1000/(pre+inf+post), 2) if pre+inf+post>0 else "N/A",
            "GPUs": get_gpu_report_str(DEVICE),
            "Evaluator":       "COCOeval",
        }
    except Exception as e:
        print(f"  ⚠️ {label} (COCOeval): {e}")
        return None  # Signal to use fallback


def _coco_eval_seg(label, pt, yaml):
    """Segmentation — COCOeval (consistent with Hybrid & Mask R-CNN)."""
    if not check_pycocotools():
        print(f"  ⚠️ {label}: pycocotools tidak tersedia, fallback ke _eval_seg()")
        return None  # Signal to use fallback
    
    try:
        # Build ground truth
        coco_gt_dict, image_ids = build_coco_ground_truth(yaml, split="valid")
        if coco_gt_dict is None:
            print(f"  ⚠️ {label}: Gagal build COCO GT, fallback ke _eval_seg()")
            return None
        
        # Run inference
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        total_ms = round(pre + inf + post, 2)
        fps = round(1000 / total_ms, 2) if total_ms > 0 else "N/A"
        
        # Collect segmentation predictions in COCO format
        predictions = []
        img_dir = os.path.join(os.path.dirname(yaml), "valid", "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(os.path.dirname(yaml), "test", "images")
        
        print(f"  🔍 Mengumpulkan prediksi segmentasi untuk COCOeval...")
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, img_file)
            if img_path not in image_ids:
                continue
            
            result = m.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, verbose=False)
            if result and result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy()
                boxes = result[0].boxes.xyxy.cpu().numpy()
                confs = result[0].boxes.conf.cpu().numpy()
                clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, (mask, box, conf, cls) in enumerate(zip(masks, boxes, confs, clss)):
                    predictions.append({
                        "image": img_path,
                        "pred_mask": mask,
                        "pred_box": box.tolist(),
                        "pred_cls": int(cls),
                        "pred_conf": float(conf),
                    })
        
        print(f"  ✅ Total prediksi segmentasi: {len(predictions)}")
        
        # Evaluate with COCOeval
        print(f"  📊 Menjalankan COCOeval untuk segmentasi...")
        mAP50, mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, predictions, iou_type="segm")
        
        # Also get box mAP for comparison
        box_mAP50, box_mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, predictions, iou_type="bbox")
        
        del m; _flush(f"eval {label} (COCOeval)")
        
        print(f"  ✅ COCOeval selesai: mAP50(Mask)={mAP50}, mAP50-95(Mask)={mAP50_95}")
        
        return {
            "Model":           label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95(Box)":   box_mAP50_95,
            "mAP50-95(Mask)":  mAP50_95,
            "Latency (ms)":     total_ms,
            "FPS":             fps,
            "GPUs":            get_gpu_report_str(DEVICE),
            "Evaluator":       "COCOeval",
        }
    except Exception as e:
        print(f"  ⚠️ {label} (COCOeval): {e}")
        return None  # Signal to use fallback


print("\n" + "="*65 + "\n  YOLOv8m Fine-tuning\n" + "="*65)
# Download & Pindahkan model dasar terlebih dahulu
det_model_path = download_and_move_model("yolov8m.pt")
seg_model_path = download_and_move_model("yolov8m-seg.pt")

best_det = _train(det_model_path,     DET_YAML, "yolov8m",    "YOLOv8m Detection")
best_seg = _train(seg_model_path, SEG_YAML, "yolov8m_seg","YOLOv8m-Seg Segmentation")

report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)

# Evaluasi Multi-GPU menggunakan eval_multigpu.py
print("\n" + "="*65 + "\n  Menjalankan Evaluasi Multi-GPU YOLOv8m & YOLOv8m-Seg\n" + "="*65)
import subprocess
import sys
try:
    subprocess.run([sys.executable, "-u", "eval_multigpu.py", "--gpus", "all"], check=True)
except subprocess.CalledProcessError as e:
    print(f"\n⚠️ Evaluasi Multi-GPU gagal: {e}")

# ------ Kompres folder hasil training ------
try:
    compress_run("yolov8m")
    compress_run("yolov8m_seg")
except Exception as e:
    print(f"⚠️ Gagal kompres: {e}")

print("\n✅ YOLOv8m selesai.")
send_telegram_msg(f"✅ <b>YOLOv8m Pipeline Finished</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>")
