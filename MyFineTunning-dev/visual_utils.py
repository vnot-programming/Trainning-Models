# -*- coding: utf-8 -*-
"""
visual_utils.py
===============
Fungsi utilitas untuk men-generate visualisasi gambar tunggal (single image) 
dan grid perbandingan (comparison grids) untuk evaluasi pipeline CV.
"""

import os, sys, gc, csv
import cv2
import numpy as np
import torch

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_MASKRCNN_DIR = os.path.join(_THIS_DIR, "mask-r-cnn")
if _MASKRCNN_DIR not in sys.path:
    sys.path.insert(0, _MASKRCNN_DIR)

from config_shared import (
    IMAGE_SAMPLES_DIR, VISUALS_DIR, REPORTS_DIR, MODEL_COLORS,
    get_output_dir, IMAGE_SIZE, CLASS_NAMES
)

SAM2_PT = os.path.join(_THIS_DIR, "hybrid", "sam2.1_b.pt")
DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"

PANEL_W  = 640
PANEL_H  = 480
HEADER_H = 52
GAP      = 6
BG_COLOR = (15, 15, 15)


def _read_map(model_key: str, task: str = "det") -> str:
    candidates = [
        f"report_{model_key}_{task}_multigpu.csv",
        f"report_{model_key}_{task}_coco.csv",
        f"report_{model_key}_{task}.csv",
    ]
    if model_key == "maskrcnn":
        candidates += ["report_maskrcnn_ddp_seg.csv"]
    for fname in candidates:
        p = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        # Pakai mAP50 jika ada, sesuai permintaan USER
                        for key in ["mAP50(Box)", "mAP50", "mAP50_95", "mAP50-95"]:
                            v = row.get(key, "")
                            if v and v not in ("N/A", "ERR", ""):
                                return str(round(float(v), 4))
            except Exception:
                pass
    return "N/A"

def _placeholder(msg: str = "Model not found") -> np.ndarray:
    img = np.full((PANEL_H, PANEL_W, 3), 30, dtype=np.uint8)
    cv2.putText(img, msg, (20, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (120, 120, 120), 2, cv2.LINE_AA)
    return img

def _add_header(panel: np.ndarray, label: str, color_bgr: tuple, map_str: str) -> np.ndarray:
    bar = np.full((HEADER_H, panel.shape[1], 3), color_bgr, dtype=np.uint8)
    cv2.putText(bar, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if map_str != "N/A":
        cv2.putText(bar, f"mAP50: {map_str}", (panel.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 180), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])

def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

def _draw_yolo_result(img_bgr: np.ndarray, result, color: tuple, draw_mask: bool) -> np.ndarray:
    out   = img_bgr.copy()
    names = result.names
    if result.masks is not None and draw_mask:
        H, W = out.shape[:2]
        for m in result.masks.data.cpu().numpy():
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            alpha_mask = (m > 0.5).astype(np.uint8)
            colored = np.zeros_like(out)
            colored[alpha_mask == 1] = color
            cv2.addWeighted(colored, 0.40, out, 1.0, 0, out)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls_name = names.get(clss[i], f"cls{clss[i]}")
            label_txt = f"{cls_name} {float(confs[i]):.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label_txt, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _get_sample_images():
    if not os.path.isdir(IMAGE_SAMPLES_DIR): return []
    return sorted([os.path.join(IMAGE_SAMPLES_DIR, f) for f in os.listdir(IMAGE_SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

# ==============================================================================
# YOLO (v8, v9, 11) SINGLE VISUALS
# ==============================================================================
def generate_single_yolo(model_key: str, label: str, is_multigpu: bool, task: str):
    """
    Generate single image visual untuk YOLO (Det atau Seg) untuk semua gambar di sampel.
    task: "det" atau "seg"
    """
    print(f"\n[Visual] Generate Single YOLO ({model_key} | {task} | multi={is_multigpu})...")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    pt_path = os.path.join(get_output_dir(model_key), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ File weights tidak ditemukan: {pt_path}")
        return
    
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
    except Exception as e:
        print(f"  ❌ Gagal load YOLO: {e}")
        return

    color = MODEL_COLORS.get(model_key, (0, 255, 0))
    map_v = _read_map(model_key, task)
    suffix = "_multigpu" if is_multigpu else ""
    draw_mask = (task == "seg")
    
    samples = _get_sample_images()
    for img_path in samples:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        # Misal: sample_01_yolov8m_det_multigpu.jpg
        # Tapi hapus akhiran _seg dari model_key jika ada untuk nama file agar lebih rapi
        clean_key = model_key.replace("_seg", "")
        out_name = f"{fname}_{clean_key}_{task}{suffix}.jpg"
        out_path = os.path.join(VISUALS_DIR, out_name)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        try:
            res = model.predict(img_bgr, conf=0.5, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
            drawn = _draw_yolo_result(img_bgr, res, color, draw_mask)
            # Add header
            final_img = _add_header(drawn, label, color, map_v)
            cv2.imwrite(out_path, final_img)
        except Exception as e:
            print(f"  ⚠️ Error memproses {fname}: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  ✅ Selesai generate visual untuk {model_key}")


# ==============================================================================
# MASK R-CNN SINGLE VISUALS
# ==============================================================================
def generate_single_maskrcnn(is_multigpu: bool):
    print(f"\n[Visual] Generate Single Mask R-CNN (multi={is_multigpu})...")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    pt_path = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ File weights tidak ditemukan: {pt_path}")
        return
    
    try:
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
        from maskrcnn_builder import build_model
        
        device = torch.device(DEVICE)
        model = build_model(device=device)
        ckpt = torch.load(pt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
    except Exception as e:
        print(f"  ❌ Gagal load Mask R-CNN: {e}")
        return

    color = MODEL_COLORS.get("maskrcnn", (255, 0, 255))
    map_v = _read_map("maskrcnn", "seg")
    suffix = "_multigpu" if is_multigpu else ""
    
    samples = _get_sample_images()
    for img_path in samples:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{fname}_maskrcnn_seg{suffix}.jpg"
        out_path = os.path.join(VISUALS_DIR, out_name)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        out = img_bgr.copy()
        H, W = out.shape[:2]
        img_tensor = TF.to_tensor(PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))).to(device)
        
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        
        boxes  = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks  = outputs["masks"].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] < 0.5: continue
            m = masks[i, 0]
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            alpha = (m > 0.5).astype(np.uint8)
            colored = np.zeros_like(out)
            colored[alpha == 1] = color
            cv2.addWeighted(colored, 0.40, out, 1.0, 0, out)

            x1, y1, x2, y2 = map(int, boxes[i])
            cls_idx  = int(labels[i]) - 1
            cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"cls{cls_idx}"
            label_txt = f"{cls_name} {scores[i]:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label_txt, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            
        final_img = _add_header(out, "Mask R-CNN ResNet-50", color, map_v)
        cv2.imwrite(out_path, final_img)

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  ✅ Selesai generate visual untuk Mask R-CNN")


# ==============================================================================
# HYBRID SINGLE VISUALS (DET & SEG)
# ==============================================================================
def generate_single_hybrid(is_multigpu: bool):
    print(f"\n[Visual] Generate Single Hybrid (multi={is_multigpu})...")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    yolo_pt = os.path.join(get_output_dir("yolo11m"), "weights", "best.pt")
    if not os.path.exists(yolo_pt) or not os.path.exists(SAM2_PT):
        print(f"  ❌ File weights hybrid tidak lengkap!")
        return
        
    try:
        from ultralytics import YOLO, SAM
        yolo = YOLO(yolo_pt)
        sam  = SAM(SAM2_PT)
    except Exception as e:
        print(f"  ❌ Gagal load model Hybrid: {e}")
        return

    color = MODEL_COLORS.get("hybrid", (0, 165, 255))
    map_det = _read_map("hybrid", "det")
    map_seg = _read_map("hybrid", "seg")
    suffix = "_multigpu" if is_multigpu else ""
    
    samples = _get_sample_images()
    for img_path in samples:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_det = os.path.join(VISUALS_DIR, f"{fname}_hybrid_det{suffix}.jpg")
        out_seg = os.path.join(VISUALS_DIR, f"{fname}_hybrid_seg{suffix}.jpg")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        # 1. Deteksi
        det_img = img_bgr.copy()
        seg_img = img_bgr.copy()
        H, W = img_bgr.shape[:2]
        
        det = yolo.predict(img_bgr, conf=0.5, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
        names = det.names
        
        if det.boxes is not None and len(det.boxes) > 0:
            boxes = det.boxes.xyxy.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()
            clss  = det.boxes.cls.cpu().numpy().astype(int)
            
            # Predict SAM2 untuk seg
            sam_r = sam.predict(img_bgr, bboxes=boxes, verbose=False)
            if sam_r and sam_r[0].masks is not None:
                for m in sam_r[0].masks.data.cpu().numpy():
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    alpha = (m > 0.5).astype(np.uint8)
                    colored = np.zeros_like(seg_img)
                    colored[alpha == 1] = color
                    cv2.addWeighted(colored, 0.42, seg_img, 1.0, 0, seg_img)
            
            # Draw boxes di det & seg
            for out_canvas in [det_img, seg_img]:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cls_name = names.get(clss[i], f"cls{clss[i]}")
                    label_txt = f"{cls_name} {float(confs[i]):.2f}"
                    cv2.rectangle(out_canvas, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                    cv2.rectangle(out_canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(out_canvas, label_txt, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite(out_det, _add_header(det_img, "Hybrid (Det)", color, map_det))
        cv2.imwrite(out_seg, _add_header(seg_img, "Hybrid (Seg)", color, map_seg))

    del yolo, sam; gc.collect(); torch.cuda.empty_cache()
    print(f"  ✅ Selesai generate visual untuk Hybrid")


# ==============================================================================
# GRIDS (Di-run oleh Hybrid scripts di akhir pipeline)
# ==============================================================================
def _gap_col(h): return np.full((h, GAP, 3), BG_COLOR, dtype=np.uint8)
def _gap_row(w): return np.full((GAP, w, 3), BG_COLOR, dtype=np.uint8)

def _read_panel(filepath: str) -> np.ndarray:
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            return _resize(img)
    return _resize(_placeholder("Panel tidak tersedia"))

def generate_hybrid_grids(is_multigpu: bool):
    """Menyatukan gambar-gambar single yang sudah digenerate model-model sebelumnya ke dalam Grid"""
    print(f"\n[Visual] Generate Comparison Grids (multi={is_multigpu})...")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    suffix = "_multigpu" if is_multigpu else ""
    
    samples = _get_sample_images()
    for img_path in samples:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. Detection Grid (2x2)
        # Susunan: v8, v9, 11, hybrid
        d_v8  = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolov8m_det{suffix}.jpg"))
        d_v9  = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolov9m_det{suffix}.jpg"))
        d_11  = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolo11m_det{suffix}.jpg"))
        d_hyb = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_hybrid_det{suffix}.jpg"))
        
        row1 = np.hstack([d_v8, _gap_col(d_v8.shape[0]), d_v9])
        row2 = np.hstack([d_11, _gap_col(d_11.shape[0]), d_hyb])
        grid_det = np.vstack([row1, _gap_row(row1.shape[1]), row2])
        
        title_det = np.full((56, grid_det.shape[1], 3), (15, 15, 15), dtype=np.uint8)
        cv2.putText(title_det, "Detection Comparison — YOLOv8m | YOLOv9m | YOLO11m | Hybrid",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2, cv2.LINE_AA)
        grid_det = np.vstack([title_det, grid_det])
        cv2.imwrite(os.path.join(VISUALS_DIR, f"{fname}_visual_detection_comparison{suffix}.jpg"), grid_det)

        # 2. Segmentation Grid (3x2)
        # Susunan: v8, v9, 11, maskrcnn, hybrid, blank
        s_v8   = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolov8m_seg{suffix}.jpg"))
        s_v9   = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolov9c_seg{suffix}.jpg"))
        s_11   = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_yolo11m_seg{suffix}.jpg"))
        s_mask = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_maskrcnn_seg{suffix}.jpg"))
        s_hyb  = _read_panel(os.path.join(VISUALS_DIR, f"{fname}_hybrid_seg{suffix}.jpg"))
        blank  = _resize(np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8))
        
        r1 = np.hstack([s_v8, _gap_col(s_v8.shape[0]), s_v9, _gap_col(s_v9.shape[0]), s_11])
        r2 = np.hstack([s_mask, _gap_col(s_mask.shape[0]), s_hyb, _gap_col(s_hyb.shape[0]), blank])
        grid_seg = np.vstack([r1, _gap_row(r1.shape[1]), r2])
        
        title_seg = np.full((56, grid_seg.shape[1], 3), (15, 15, 15), dtype=np.uint8)
        cv2.putText(title_seg, "Segmentation Comparison — YOLOv8 | YOLOv9c | YOLO11 | Mask R-CNN | Hybrid",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2, cv2.LINE_AA)
        grid_seg = np.vstack([title_seg, grid_seg])
        cv2.imwrite(os.path.join(VISUALS_DIR, f"{fname}_visual_segmentation_comparison{suffix}.jpg"), grid_seg)

    print(f"  ✅ Selesai generate grids")
