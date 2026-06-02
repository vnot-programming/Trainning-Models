# -*- coding: utf-8 -*-
"""
utils/eval_paper.py
===================
Skrip evaluasi komprehensif untuk Paper (Dual-Evaluation Experimental Design).
Dipisah secara logis: Model Deteksi murni untuk dataset Deteksi, 
dan Model Segmentasi untuk dataset Segmentasi.

Menghitung:
- Wall-clock Latency & FPS
- Ultralytics internal Latency & FPS (N/A untuk Mask R-CNN)
- P1 Box mAP, P2 Box mAP
- P1 Mask mAP, P2 Mask mAP
- P1 Boundary AP, P2 Boundary AP
"""

import os
import sys
import gc
import csv
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
import cv2

from config_shared import (
    DATASETS_DIR, get_output_dir, REPORTS_DIR, NUM_CLASSES, IMAGE_SIZE
)
from coco_eval_utils import build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools, load_native_coco_gt
from eval_unu_helpers import (
    ensure_boundary_iou_installed,
    flush_gpu,
    gpu_str,
    mask_to_rle
)

# Direktori Datasets
P1_DET_DIR = os.path.join(DATASETS_DIR, "standard_datasets_det")
P1_SEG_DIR = os.path.join(DATASETS_DIR, "standard_datasets_seg")
P2_DET_DIR = os.path.join(DATASETS_DIR, "golden_dataset_det")
P2_SEG_DIR = os.path.join(DATASETS_DIR, "golden_dataset_seg")

PAPER_REPORTS_DIR = os.path.join(REPORTS_DIR, "paper")

DET_MODELS = [
    {"label": "YOLOv8m", "type": "yolo_det", "key": "yolov8m"},
    {"label": "YOLOv9m", "type": "yolo_det", "key": "yolov9m"},
    {"label": "YOLO11l", "type": "yolo_det", "key": "yolo11l"},
    {"label": "Mask R-CNN ResNet-50 FPN-v2", "type": "maskrcnn", "key": "maskrcnn"},
    {"label": "Hybrid (YOLO11l+SAM2)", "type": "hybrid", "key": "hybrid", "yolo_key": "yolo11l"},
]

SEG_MODELS = [
    {"label": "YOLOv8m-Seg", "type": "yolo_seg", "key": "yolov8m_seg"},
    {"label": "YOLOv9c-Seg", "type": "yolo_seg", "key": "yolov9c_seg"},
    {"label": "YOLO11l-Seg", "type": "yolo_seg", "key": "yolo11l_seg"},
    {"label": "Mask R-CNN ResNet-50 FPN-v2", "type": "maskrcnn", "key": "maskrcnn"},
    {"label": "Hybrid (YOLO11l+SAM2)", "type": "hybrid", "key": "hybrid", "yolo_key": "yolo11l"},
]



def run_boundary_eval_paper(coco_gt_dict, image_ids, predictions):
    from pycocotools.coco import COCO
    from boundary_iou.coco_instance_api.coco import COCO as BCOCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_preds = []
    for pred in predictions:
        img_path = pred["image"]
        if img_path not in image_ids:
            continue
        img_id = image_ids[img_path]
        if "pred_mask" in pred:
            coco_preds.append({
                "image_id": img_id,
                "category_id": pred["pred_cls"] + 1,
                "segmentation": pred["pred_mask"],
                "score": pred["pred_conf"]
            })

    if not coco_preds:
        return 0.0

    coco_dt = coco_gt.loadRes(coco_preds)
    bcoco_gt = BCOCO()
    bcoco_gt.dataset = coco_gt.dataset
    bcoco_gt.createIndex()
    
    bcoco_dt = BCOCO()
    bcoco_dt.dataset = coco_dt.dataset
    bcoco_dt.createIndex()

    evaluator = BCOCOeval(bcoco_gt, bcoco_dt, 'boundary')
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return evaluator.stats[0]

def get_model_size_mb(model_type, model_obj, sam_obj=None):
    if model_type in ["yolo_seg", "yolo_det", "hybrid"]:
        if hasattr(model_obj, "model"):
            param_size = sum(p.nelement() * p.element_size() for p in model_obj.model.parameters())
            buffer_size = sum(b.nelement() * b.element_size() for b in model_obj.model.buffers())
            size_mb = (param_size + buffer_size) / 1024**2
            if model_type == "hybrid" and sam_obj is not None:
                sparam_size = sum(p.nelement() * p.element_size() for p in sam_obj.model.parameters())
                sbuffer_size = sum(b.nelement() * b.element_size() for b in sam_obj.model.buffers())
                size_mb += (sparam_size + sbuffer_size) / 1024**2
            return size_mb
    elif model_type == "maskrcnn":
        param_size = sum(p.nelement() * p.element_size() for p in model_obj.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model_obj.buffers())
        return (param_size + buffer_size) / 1024**2
    return 0.0

def load_maskrcnn(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
    pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
    if os.path.exists(pt):
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
        model.to(device).eval()
        return model
    return None

def evaluate_models_for_task(task_name, models_list, datasets, device):
    from ultralytics import YOLO, SAM
    results = []

    for mcfg in models_list:
        print(f"\n{'='*50}\nMemproses Model {task_name}: {mcfg['label']}\n{'='*50}")
        flush_gpu(mcfg['label'])

        mtype = mcfg["type"]
        mkey = mcfg["key"]
        
        # Load Model
        model_obj = None
        sam_obj = None
        size_mb = 0.0

        if mtype in ["yolo_seg", "yolo_det"]:
            pt = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            if os.path.exists(pt):
                model_obj = YOLO(pt)
                size_mb = get_model_size_mb(mtype, model_obj)
        elif mtype == "maskrcnn":
            model_obj = load_maskrcnn(device)
            size_mb = get_model_size_mb(mtype, model_obj)
        elif mtype == "hybrid":
            yolo_pt = os.path.join(get_output_dir(mcfg["yolo_key"]), "weights", "best.pt")
            sam_pt = os.path.join(ROOT, "models", "sam2.1_t.pt")
            if os.path.exists(yolo_pt) and os.path.exists(sam_pt):
                model_obj = YOLO(yolo_pt)
                sam_obj = SAM(sam_pt)
                size_mb = get_model_size_mb(mtype, model_obj, sam_obj)

        if model_obj is None:
            print(f"❌ Model weights untuk {mcfg['label']} tidak ditemukan, skip.")
            continue

        model_metrics = {
            "Model": mcfg['label'],
            "Model Size (MB)": f"{size_mb:.2f}",
            "GPUs": gpu_str(),
            "Latency (ms) [Wall-Clock]": "N/A",
            "FPS [Wall-Clock]": "N/A",
            "Latency (ms) [Ultralytics]": "N/A",
            "FPS [Ultralytics]": "N/A",
        }
        
        if task_name == "Detection":
            model_metrics["P1 Box mAP"] = "N/A"
            model_metrics["P2 Box mAP"] = "N/A"
        else:
            model_metrics["P1 Mask mAP"] = "N/A"
            model_metrics["P2 Mask mAP"] = "N/A"
            model_metrics["P1 BoundAP"] = "N/A"
            model_metrics["P2 BoundAP"] = "N/A"

        total_wall_time = 0.0
        total_images = 0
        total_ultra_time_ms = 0.0

        for dname, dinfo in datasets.items():
            if not dinfo["gt"] or not dinfo["img_dir"]:
                continue
            
            is_seg_task = "Seg" in dname
            img_dir = dinfo["img_dir"]
            
            predictions = []
            print(f"  -> Evaluasi pada {dname} ({len(dinfo['ids'])} gambar)...")
            
            start_wall = time.perf_counter()

            if mtype in ["yolo_seg", "yolo_det"]:
                for img_key in dinfo["ids"].keys():
                    img_path = img_key if os.path.isabs(img_key) else os.path.join(img_dir, img_key)
                    if not os.path.exists(img_path): continue
                    res = model_obj.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
                    total_ultra_time_ms += sum(res.speed.values())
                    total_images += 1
                    
                    if res.boxes is not None:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy().astype(int)
                        
                        masks = None
                        if res.masks is not None and is_seg_task and mtype == "yolo_seg":
                            masks = res.masks.data.cpu().numpy()

                        for i in range(len(boxes)):
                            pred_dict = {
                                "image": img_key,
                                "pred_cls": clss[i],
                                "pred_conf": confs[i],
                                "pred_box": [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                            }
                            if masks is not None:
                                m = masks[i]
                                h, w = res.orig_img.shape[:2]
                                if m.shape != (h, w):
                                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                pred_dict["pred_mask"] = mask_to_rle((m > 0.5).astype(np.uint8))
                            predictions.append(pred_dict)

            elif mtype == "maskrcnn":
                import torchvision.transforms.functional as TF
                from PIL import Image as PILImage
                for img_key in dinfo["ids"].keys():
                    img_path = img_key if os.path.isabs(img_key) else os.path.join(img_dir, img_key)
                    if not os.path.exists(img_path): continue
                    pil = PILImage.open(img_path).convert("RGB")
                    t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
                    total_images += 1
                    with torch.no_grad():
                        preds = model_obj(t_img)[0]
                    
                    scores = preds["scores"].cpu().numpy()
                    labels = preds["labels"].cpu().numpy()
                    boxes = preds["boxes"].cpu().numpy()
                    masks = preds["masks"].cpu().numpy()

                    for i in range(len(scores)):
                        if scores[i] < 0.001: continue
                        pred_dict = {
                            "image": img_key,
                            "pred_cls": int(labels[i]) - 1,
                            "pred_conf": scores[i],
                            "pred_box": [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                        }
                        if is_seg_task:
                            m = masks[i, 0]
                            h, w = pil.size[1], pil.size[0]
                            if m.shape != (h, w):
                                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                            pred_dict["pred_mask"] = mask_to_rle((m > 0.5).astype(np.uint8))
                        predictions.append(pred_dict)

            elif mtype == "hybrid":
                for img_key in dinfo["ids"].keys():
                    img_path = img_key if os.path.isabs(img_key) else os.path.join(img_dir, img_key)
                    if not os.path.exists(img_path): continue
                    yolo_res = model_obj.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
                    total_ultra_time_ms += sum(yolo_res.speed.values())
                    total_images += 1

                    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                        boxes = yolo_res.boxes.xyxy.cpu().numpy()
                        confs = yolo_res.boxes.conf.cpu().numpy()
                        clss = yolo_res.boxes.cls.cpu().numpy().astype(int)
                        
                        sam_masks = None
                        if is_seg_task:
                            sam_res = sam_obj.predict(yolo_res.orig_img, bboxes=yolo_res.boxes.xyxy, verbose=False)[0]
                            total_ultra_time_ms += sum(sam_res.speed.values())
                            if sam_res.masks is not None:
                                sam_masks = sam_res.masks.data.cpu().numpy()

                        for i in range(len(boxes)):
                            pred_dict = {
                                "image": img_key,
                                "pred_cls": clss[i],
                                "pred_conf": confs[i],
                                "pred_box": [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                            }
                            if sam_masks is not None and i < len(sam_masks):
                                m = sam_masks[i]
                                h, w = yolo_res.orig_img.shape[:2]
                                if m.shape != (h, w):
                                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                pred_dict["pred_mask"] = mask_to_rle((m > 0.5).astype(np.uint8))
                            predictions.append(pred_dict)

            end_wall = time.perf_counter()
            total_wall_time += (end_wall - start_wall)

            # Evaluate Metrics
            if "Det" in dname:
                map50, map50_95 = evaluate_coco_predictions(dinfo["gt"], dinfo["ids"], predictions, "bbox")
                if map50_95 is not None:
                    model_metrics[f"{dname.split('_')[0]} Box mAP"] = f"{map50_95:.4f}"
            else:
                map50_m, map50_95_m = evaluate_coco_predictions(dinfo["gt"], dinfo["ids"], predictions, "segm")
                if map50_95_m is not None:
                    model_metrics[f"{dname.split('_')[0]} Mask mAP"] = f"{map50_95_m:.4f}"
                
                bound_ap = run_boundary_eval_paper(dinfo["gt"], dinfo["ids"], predictions)
                model_metrics[f"{dname.split('_')[0]} BoundAP"] = f"{bound_ap:.4f}"

        # Calculate final speeds
        if total_images > 0:
            avg_wall_ms = (total_wall_time / total_images) * 1000
            fps_wall = 1000 / avg_wall_ms if avg_wall_ms > 0 else 0
            model_metrics["Latency (ms) [Wall-Clock]"] = f"{avg_wall_ms:.2f}"
            model_metrics["FPS [Wall-Clock]"] = f"{fps_wall:.2f}"
            
            if mtype != "maskrcnn":
                avg_ultra_ms = total_ultra_time_ms / total_images
                fps_ultra = 1000 / avg_ultra_ms if avg_ultra_ms > 0 else 0
                model_metrics["Latency (ms) [Ultralytics]"] = f"{avg_ultra_ms:.2f}"
                model_metrics["FPS [Ultralytics]"] = f"{fps_ultra:.2f}"

        gpu_val = model_metrics.pop("GPUs")
        model_metrics["GPUs"] = gpu_val
        results.append(model_metrics)
        
        del model_obj
        del sam_obj
        flush_gpu("Post-Model")
        
    return results

def run_evaluation():
    if not check_pycocotools() or not ensure_boundary_iou_installed():
        print("❌ Persyaratan evaluasi tidak terpenuhi.")
        return

    os.makedirs(PAPER_REPORTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def auto_load_gt(ds_dir):
        yaml_path = os.path.join(ds_dir, "data.yaml")
        coco_json = os.path.join(ds_dir, "valid", "_annotations.coco.json")
        if os.path.exists(coco_json):
            gt, ids = load_native_coco_gt(os.path.join(ds_dir, "valid"))
            img_dir = os.path.join(ds_dir, "valid")
            return gt, ids, img_dir
        elif os.path.exists(yaml_path):
            gt, ids = build_coco_ground_truth(yaml_path, split="valid")
            # Resolve image dir based on yaml
            img_dir = os.path.join(ds_dir, "valid", "images")
            if not os.path.exists(img_dir):
                img_dir = os.path.join(ds_dir, "valid")
            return gt, ids, img_dir
        else:
            print(f"❌ Format dataset tidak dikenali di: {ds_dir}")
            return None, None, None

    print("\n[Data] Memuat Ground Truths...")
    p1_det_gt, p1_det_ids, p1_det_img = auto_load_gt(P1_DET_DIR)
    p1_seg_gt, p1_seg_ids, p1_seg_img = auto_load_gt(P1_SEG_DIR)
    p2_det_gt, p2_det_ids, p2_det_img = auto_load_gt(P2_DET_DIR)
    p2_seg_gt, p2_seg_ids, p2_seg_img = auto_load_gt(P2_SEG_DIR)

    datasets_det = {
        "P1_Det": {"gt": p1_det_gt, "ids": p1_det_ids, "img_dir": p1_det_img},
        "P2_Det": {"gt": p2_det_gt, "ids": p2_det_ids, "img_dir": p2_det_img},
    }
    
    datasets_seg = {
        "P1_Seg": {"gt": p1_seg_gt, "ids": p1_seg_ids, "img_dir": p1_seg_img},
        "P2_Seg": {"gt": p2_seg_gt, "ids": p2_seg_ids, "img_dir": p2_seg_img},
    }

    print("\n" + "#"*60)
    print("🚀 MULAI EVALUASI DETECTION TASK")
    print("#"*60)
    results_det = evaluate_models_for_task("Detection", DET_MODELS, datasets_det, device)

    print("\n" + "#"*60)
    print("🚀 MULAI EVALUASI SEGMENTATION TASK")
    print("#"*60)
    results_seg = evaluate_models_for_task("Segmentation", SEG_MODELS, datasets_seg, device)

    # Write CSVs
    if results_det:
        det_csv_path = os.path.join(PAPER_REPORTS_DIR, "paper_eval_detection.csv")
        with open(det_csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results_det[0].keys())
            w.writeheader()
            w.writerows(results_det)
        print(f"\n✅ Laporan Detection disimpan di: {det_csv_path}")

    if results_seg:
        seg_csv_path = os.path.join(PAPER_REPORTS_DIR, "paper_eval_segmentation.csv")
        with open(seg_csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results_seg[0].keys())
            w.writeheader()
            w.writerows(results_seg)
        print(f"✅ Laporan Segmentation disimpan di: {seg_csv_path}")

if __name__ == "__main__":
    run_evaluation()
