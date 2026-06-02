"""
tmux kill-session -t generate_paper_visuals; 
tmux new-session -d -s generate_paper_visuals "cd /root/Trainning-Models/MyFineTunning-RunPOD && python3 -u utils/generals/generate_paper_visuals.py 2>&1 | tee utils/generals/generate_paper_visuals.log"

"""
# -*- coding: utf-8 -*-
import os
import sys
import json
import tarfile
import time
import shutil
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# PATH SETUP — resolve ROOT dari lokasi file ini (utils/generals/generate_paper_visuals.py)
# ==============================================================================
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_UTILS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, _UTILS_DIR)

from ultralytics import YOLO, SAM
from config_shared import DATASETS_DIR, get_output_dir, NUM_CLASSES, IMAGE_SIZE, WORKSPACE_DIR
from eval_unu_helpers import flush_gpu

# ====================================================================
# Konfigurasi Direktori
# ====================================================================
P2_DET_DIR = os.path.join(DATASETS_DIR, "standard_datasets_det", "valid")
P2_SEG_DIR = os.path.join(DATASETS_DIR, "standard_datasets_seg", "valid")

RUN_DIR = WORKSPACE_DIR
BASE_VIS_DIR = os.path.join(RUN_DIR, "visuals", "new_methods", "paper")

IMG_SAMPLE_DIR = os.path.join(BASE_VIS_DIR, "images_sample")
DET_OUT_DIR = os.path.join(BASE_VIS_DIR, "paper_eval_detection")
SEG_OUT_DIR = os.path.join(BASE_VIS_DIR, "paper_eval_segmentation")
COMP_DET_DIR = os.path.join(BASE_VIS_DIR, "comparison", "paper_eval_detection")
COMP_SEG_DIR = os.path.join(BASE_VIS_DIR, "comparison", "paper_eval_segmentation")

for d in [IMG_SAMPLE_DIR, DET_OUT_DIR, SEG_OUT_DIR, COMP_DET_DIR, COMP_SEG_DIR]:
    os.makedirs(d, exist_ok=True)

# ====================================================================
# Util Functions
# ====================================================================
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

CLASS_NAMES = []
COLORS = []

def init_classes_from_coco(coco_dir):
    global CLASS_NAMES, COLORS
    json_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        CLASS_NAMES = ["Class"] * NUM_CLASSES
        COLORS = [(0, 255, 0)] * NUM_CLASSES
        return
    
    with open(json_path, 'r') as f:
        coco_gt = json.load(f)
    
    # Abaikan supercategory (id: 0), ambil id > 0
    cats = [c for c in coco_gt['categories'] if c['id'] > 0]
    cats = sorted(cats, key=lambda x: x['id'])
    CLASS_NAMES = [c['name'] for c in cats]
    
    # Palet warna spesifik (BGR)
    color_palette = [
        (0, 255, 0),     # Hijau
        (255, 0, 0),     # Biru
        (0, 0, 255),     # Merah
        (0, 255, 255),   # Kuning
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Oranye
        (128, 0, 128),   # Ungu
        (0, 128, 128),   # Olive
    ]
    COLORS = [color_palette[i % len(color_palette)] for i in range(len(CLASS_NAMES))]

def get_text_color(bgr_color):
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def draw_custom(image_path, boxes, masks, confs, clss, color=None):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Draw masks
    if masks is not None:
        overlay = img.copy()
        for i in range(len(masks)):
            m = masks[i]
            c_color = color if color is not None else COLORS[int(clss[i]) % len(COLORS)]
            c_color_arr = np.array(c_color, dtype=np.uint8)
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[m > 0.5] = c_color_arr
            
            # Add mask to overlay
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        img = overlay
        
    # Draw bounding boxes and text
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = confs[i] if confs is not None else 1.0
        c = int(clss[i])
        c_color = color if color is not None else COLORS[c % len(COLORS)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), c_color, 2)
        
        label = f"{CLASS_NAMES[c % len(CLASS_NAMES)]} {conf:.2f}" if confs is not None else CLASS_NAMES[c % len(CLASS_NAMES)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), c_color, -1)
        
        text_color = get_text_color(c_color)
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img

def load_gt_annotations(coco_dir):
    json_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        coco_gt = json.load(f)
    
    from pycocotools.coco import COCO
    coco_api = COCO(json_path)
    return coco_api

def get_gt_data(coco_api, img_filename):
    img_ids = coco_api.getImgIds()
    target_id = None
    for i in img_ids:
        info = coco_api.loadImgs(i)[0]
        if info['file_name'] == img_filename:
            target_id = i
            break
            
    if target_id is None:
        return [], [], []
        
    ann_ids = coco_api.getAnnIds(imgIds=target_id)
    anns = coco_api.loadAnns(ann_ids)
    
    boxes = []
    masks = []
    clss = []
    for ann in anns:
        # COCO bbox is [x, y, w, h]
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x+w, y+h])
        clss.append(ann['category_id'] - 1)
        if 'segmentation' in ann and ann['segmentation']:
            mask = coco_api.annToMask(ann)
            masks.append(mask)
    
    return np.array(boxes) if boxes else [], np.array(masks) if masks else None, np.array(clss) if clss else []

def plot_grid(images, titles, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for i in range(6):
        if i < len(images) and images[i] is not None:
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(titles[i], fontsize=18, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ====================================================================
# Main Execution
# ====================================================================
def main():
    print("Mempersiapkan visualisasi untuk 164 gambar Golden Set...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Inisialisasi Class Names secara dinamis
    init_classes_from_coco(P2_SEG_DIR)
    print(f"✅ Kelas terdeteksi ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    
    # 1. Load Models (Detection)
    print("Loading Detection Models...")
    yolo8_det = YOLO(os.path.join(get_output_dir("yolov8m"), "weights", "best.pt"))
    yolo9_det = YOLO(os.path.join(get_output_dir("yolov9m"), "weights", "best.pt"))
    yolo11_det = YOLO(os.path.join(get_output_dir("yolo11l"), "weights", "best.pt"))
    mrcnn_det = load_maskrcnn(device) # We will extract just boxes
    
    # 2. Load Models (Segmentation)
    print("Loading Segmentation Models...")
    yolo8_seg = YOLO(os.path.join(get_output_dir("yolov8m_seg"), "weights", "best.pt"))
    yolo9_seg = YOLO(os.path.join(get_output_dir("yolov9c_seg"), "weights", "best.pt"))
    yolo11_seg = YOLO(os.path.join(get_output_dir("yolo11l_seg"), "weights", "best.pt"))
    sam2_model = SAM(os.path.join(ROOT, "models", "sam2.1_t.pt"))
    
    # 3. Load GT
    coco_det = load_gt_annotations(P2_DET_DIR)
    coco_seg = load_gt_annotations(P2_SEG_DIR)
    
    # Ambil daftar gambar (Kita asumsikan isi P2_DET dan P2_SEG sama persis file namanya)
    img_names = [f for f in os.listdir(P2_SEG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Total gambar ditemukan: {len(img_names)}")
    
    for idx, img_name in enumerate(img_names):
        print(f"[{idx+1}/{len(img_names)}] Memproses {img_name}...")
        img_path = os.path.join(P2_SEG_DIR, img_name)
        
        # Simpan ke images_sample
        shutil.copy(img_path, os.path.join(IMG_SAMPLE_DIR, img_name))
        
        # --- TAHAP 1: DETECTION ---
        # 1. YOLOv8m
        res8_det = yolo8_det.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b8 = res8_det.boxes.xyxy.cpu().numpy() if res8_det.boxes else []
        c8 = res8_det.boxes.conf.cpu().numpy() if res8_det.boxes else []
        cls8 = res8_det.boxes.cls.cpu().numpy() if res8_det.boxes else []
        img8_det = draw_custom(img_path, b8, None, c8, cls8, color=(0, 255, 0)) # Green
        if img8_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolov8m_{img_name}"), img8_det)
        
        # 2. YOLOv9m
        res9_det = yolo9_det.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b9 = res9_det.boxes.xyxy.cpu().numpy() if res9_det.boxes else []
        c9 = res9_det.boxes.conf.cpu().numpy() if res9_det.boxes else []
        cls9 = res9_det.boxes.cls.cpu().numpy() if res9_det.boxes else []
        img9_det = draw_custom(img_path, b9, None, c9, cls9, color=(255, 0, 0)) # Blue
        if img9_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolov9m_{img_name}"), img9_det)
        
        # 3. YOLO11m
        res11_det = yolo11_det.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b11 = res11_det.boxes.xyxy.cpu().numpy() if res11_det.boxes else []
        c11 = res11_det.boxes.conf.cpu().numpy() if res11_det.boxes else []
        cls11 = res11_det.boxes.cls.cpu().numpy() if res11_det.boxes else []
        img11_det = draw_custom(img_path, b11, None, c11, cls11, color=(0, 0, 255)) # Red
        if img11_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"yolo11l_{img_name}"), img11_det)
        
        # 4. Mask R-CNN
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
        pil = PILImage.open(img_path).convert("RGB")
        t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = mrcnn_det(t_img)[0]
        scores_mrcnn = preds["scores"].cpu().numpy()
        keep = scores_mrcnn >= 0.001
        bmrcnn = preds["boxes"].cpu().numpy()[keep]
        cmrcnn = scores_mrcnn[keep]
        clsmrcnn = preds["labels"].cpu().numpy()[keep] - 1
        mmrcnn = preds["masks"].cpu().numpy()[keep, 0] if "masks" in preds else None
        
        img_mrcnn_det = draw_custom(img_path, bmrcnn, None, cmrcnn, clsmrcnn, color=(0, 255, 255)) # Yellow
        if img_mrcnn_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"maskrcnn_{img_name}"), img_mrcnn_det)
        
        # 5. Hybrid Detection (Sama dengan YOLO11m)
        img_hybrid_det = draw_custom(img_path, b11, None, c11, cls11, color=(255, 0, 255)) # Magenta
        if img_hybrid_det is not None: cv2.imwrite(os.path.join(DET_OUT_DIR, f"hybrid_{img_name}"), img_hybrid_det)
        
        # 6. GT Detection
        gt_b_det, _, gt_c_det = get_gt_data(coco_det, img_name)
        img_gt_det = draw_custom(img_path, gt_b_det, None, None, gt_c_det, color=(255, 255, 0)) # Cyan
        
        # Plot Grid Detection
        plot_grid(
            [img8_det, img9_det, img11_det, img_mrcnn_det, img_hybrid_det, img_gt_det],
            ["YOLOv8m", "YOLOv9m", "YOLO11m", "Mask R-CNN", "Hybrid (YOLO11m)", "Ground Truth"],
            os.path.join(COMP_DET_DIR, f"grid_det_{img_name}")
        )
        
        # --- TAHAP 2: SEGMENTATION ---
        # 1. YOLOv8m-Seg
        res8_seg = yolo8_seg.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b8s = res8_seg.boxes.xyxy.cpu().numpy() if res8_seg.boxes else []
        c8s = res8_seg.boxes.conf.cpu().numpy() if res8_seg.boxes else []
        cls8s = res8_seg.boxes.cls.cpu().numpy() if res8_seg.boxes else []
        m8s = res8_seg.masks.data.cpu().numpy() if res8_seg.masks else None
        if m8s is not None:
            h, w = res8_seg.orig_img.shape[:2]
            m8s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m8s])
        else: m8s_resized = None
        img8_seg = draw_custom(img_path, b8s, m8s_resized, c8s, cls8s, color=(0, 255, 0)) # Green
        if img8_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolov8m_seg_{img_name}"), img8_seg)
        
        # 2. YOLOv9c-Seg
        res9_seg = yolo9_seg.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b9s = res9_seg.boxes.xyxy.cpu().numpy() if res9_seg.boxes else []
        c9s = res9_seg.boxes.conf.cpu().numpy() if res9_seg.boxes else []
        cls9s = res9_seg.boxes.cls.cpu().numpy() if res9_seg.boxes else []
        m9s = res9_seg.masks.data.cpu().numpy() if res9_seg.masks else None
        if m9s is not None:
            h, w = res9_seg.orig_img.shape[:2]
            m9s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m9s])
        else: m9s_resized = None
        img9_seg = draw_custom(img_path, b9s, m9s_resized, c9s, cls9s, color=(255, 0, 0)) # Blue
        if img9_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolov9c_seg_{img_name}"), img9_seg)
 
        # 3. YOLO11m-Seg
        res11_seg = yolo11_seg.predict(img_path, conf=0.001, iou=0.6, imgsz=IMAGE_SIZE, device="cuda:0", verbose=False)[0]
        b11s = res11_seg.boxes.xyxy.cpu().numpy() if res11_seg.boxes else []
        c11s = res11_seg.boxes.conf.cpu().numpy() if res11_seg.boxes else []
        cls11s = res11_seg.boxes.cls.cpu().numpy() if res11_seg.boxes else []
        m11s = res11_seg.masks.data.cpu().numpy() if res11_seg.masks else None
        if m11s is not None:
            h, w = res11_seg.orig_img.shape[:2]
            m11s_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in m11s])
        else: m11s_resized = None
        img11_seg = draw_custom(img_path, b11s, m11s_resized, c11s, cls11s, color=(0, 0, 255)) # Red
        if img11_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"yolo11l_seg_{img_name}"), img11_seg)
        
        # 4. Mask R-CNN Seg
        if mmrcnn is not None:
            h, w = pil.size[1], pil.size[0]
            mmrcnn_resized = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in mmrcnn])
        else: mmrcnn_resized = None
        img_mrcnn_seg = draw_custom(img_path, bmrcnn, mmrcnn_resized, cmrcnn, clsmrcnn, color=(0, 255, 255)) # Yellow
        if img_mrcnn_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"maskrcnn_{img_name}"), img_mrcnn_seg)
        
        # 5. Hybrid Seg
        sam_m = None
        if len(b11) > 0: # Gunakan box deteksi YOLO11m
            sam_res = sam2_model.predict(img_path, bboxes=b11, verbose=False)[0]
            if sam_res.masks is not None:
                sam_m = sam_res.masks.data.cpu().numpy()
                h, w = sam_res.orig_img.shape[:2]
                sam_m = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m])
        img_hybrid_seg = draw_custom(img_path, b11, sam_m, c11, cls11, color=(255, 0, 255)) # Magenta
        if img_hybrid_seg is not None: cv2.imwrite(os.path.join(SEG_OUT_DIR, f"hybrid_{img_name}"), img_hybrid_seg)
        
        # 6. GT Seg
        gt_b_seg, gt_m_seg, gt_c_seg = get_gt_data(coco_seg, img_name)
        img_gt_seg = draw_custom(img_path, gt_b_seg, gt_m_seg, None, gt_c_seg, color=(255, 255, 0)) # Cyan
        
        # Plot Grid Seg
        plot_grid(
            [img8_seg, img9_seg, img11_seg, img_mrcnn_seg, img_hybrid_seg, img_gt_seg],
            ["YOLOv8m-Seg", "YOLOv9c-Seg", "YOLO11m-Seg", "Mask R-CNN", "Hybrid (SAM2)", "Ground Truth"],
            os.path.join(COMP_SEG_DIR, f"grid_seg_{img_name}")
        )
        
        flush_gpu("Per Image")

    print("\nVisualisasi selesai. Mengompres direktori...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_name = f"new_methods_visuals_{timestamp}.tar.gz"
    tar_path = os.path.join(RUN_DIR, "visuals", tar_name)
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(BASE_VIS_DIR, arcname=os.path.basename(BASE_VIS_DIR))
        
    print(f"✅ Arsip visual berhasil dibuat di: {tar_path}")

if __name__ == "__main__":
    main()
