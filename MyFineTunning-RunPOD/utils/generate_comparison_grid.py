# -*- coding: utf-8 -*-
"""
utils/generate_comparison_grid.py
=================================
Generate gambar perbandingan 5 model secara LIVE (bukan dari cache).
Setiap gambar menjalankan inferensi langsung per model.

Hybrid pipeline: YOLO11l detect → boxes prompt → SAM2 segment.

Cara menjalankan:
    cd /home/my/Trainning-Models/MyFineTunning-RunPOD
    python3 utils/generate_comparison_grid.py --gpus 0

    tmux new-session -d -s comparison_grid "source Trainning-Models/MyFineTunning-RunPOD/.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-RunPOD/utils && \\
      python3 -u generate_comparison_grid.py --gpus 0 2>&1 | tee comparison_grid.log"

Output:
    visuals/new_methods/comparison/  — 10 gambar comparison grid
"""

import os, sys, gc, argparse
import numpy as np
import cv2

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
from config_shared import (
    DATASETS_DIR, IMAGE_SIZE, NUM_CLASSES,
    get_output_dir, VISUALS_DIR, MODEL_COLORS, PAPER1_VIS_DIR,
    REPORTS_DIR,
)
from telegram_utils import send_telegram_msg

# ==============================================================================
# CONSTANTS - TRAINING DATASET (Berdasarkan dataset_setup.py L153-L235)
# ==============================================================================
def _get_training_dataset_valid_dir():
    import os
    for d in os.listdir(DATASETS_DIR):
        if "seg" in d.lower() and "isempty" in d.lower() and "h61lr" not in d.lower() and "unu" not in d.lower():
            valid_path = os.path.join(DATASETS_DIR, d, "valid")
            if os.path.isdir(valid_path):
                return valid_path
    return os.path.join(DATASETS_DIR, "train_seg", "valid")

TRAIN_VALID_DIR  = _get_training_dataset_valid_dir()
PIPELINE_VISUALS_DIR = os.path.join(REPORTS_DIR, "visuals")
COMPARISON_DIR    = os.path.join(PIPELINE_VISUALS_DIR, "comparison")
_HYBRID_DIR     = os.path.join(ROOT, "hybrid")

TITLE_BAR_H = 30
N_SAMPLES   = 10

# Model configs: (label, color_BGR, type)
MODELS = [
    {"label": "YOLOv8m-Seg",         "key": "yolov8m_seg", "type": "yolo_seg",
     "color": (0, 255, 0)}, # Green
    {"label": "YOLOv9c-Seg",         "key": "yolov9c_seg", "type": "yolo_seg",
     "color": (255, 0, 0)}, # Blue
    {"label": "YOLO11l-Seg",         "key": "yolo11l_seg", "type": "yolo_seg",
     "color": (0, 0, 255)}, # Red
    {"label": "Mask R-CNN ResNet-50", "key": "maskrcnn",    "type": "maskrcnn",
     "color": (0, 255, 255)}, # Yellow
    {"label": "Hybrid (YOLO11l+SAM2)","key": "hybrid",      "type": "hybrid",
     "color": (255, 0, 255)}, # Magenta
]


# ==============================================================================
# HELPERS
# ==============================================================================

def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CLASS_NAMES = []

def get_text_color(bgr_color):
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def draw_custom(image_path, boxes, masks, confs, clss, color):
    img = cv2.imread(image_path)
    if img is None: return None
    
    c_color_arr = np.array(color, dtype=np.uint8)
    
    # Draw masks
    if masks is not None:
        overlay = img.copy()
        for i in range(len(masks)):
            m = masks[i]
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[m > 0.5] = c_color_arr
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        img = overlay
        
    # Draw bounding boxes and text
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = confs[i] if confs is not None else 1.0
        c = int(clss[i])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        c_name = CLASS_NAMES[c % len(CLASS_NAMES)] if len(CLASS_NAMES) > 0 else f"cls{c}"
        label = f"{c_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        
        text_color = get_text_color(color)
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img

def _render_yolo_seg(model, img_path, device_str, color):
    results = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, device=device_str, verbose=False)
    if not results: return cv2.imread(img_path)
    res = results[0]
    b = res.boxes.xyxy.cpu().numpy() if res.boxes else []
    c = res.boxes.conf.cpu().numpy() if res.boxes else []
    cls = res.boxes.cls.cpu().numpy() if res.boxes else []
    m = res.masks.data.cpu().numpy() if res.masks else None
    if m is not None:
        h, w = res.orig_img.shape[:2]
        m = np.array([cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for mask in m])
    
    img = draw_custom(img_path, b, m, c, cls, color)
    return img if img is not None else cv2.imread(img_path)

def _render_maskrcnn(model, img_path, device, color):
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    pil = PILImage.open(img_path).convert("RGB")
    t_img = TF.to_tensor(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(t_img)[0]

    scores = preds["scores"].cpu().numpy()
    keep = scores >= 0.5
    b = preds["boxes"].cpu().numpy()[keep]
    c = scores[keep]
    cls = preds["labels"].cpu().numpy()[keep] - 1
    m = preds["masks"].cpu().numpy()[keep, 0] if "masks" in preds else None
    if m is not None:
        h, w = pil.size[1], pil.size[0]
        m = np.array([cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for mask in m])
        
    img = draw_custom(img_path, b, m, c, cls, color)
    return img if img is not None else cv2.imread(img_path)

def _render_hybrid(yolo_model, sam_model, img_path, device_str, color):
    yolo_results = yolo_model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE, device=device_str, verbose=False)
    if not (yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0):
        return cv2.imread(img_path)

    b = yolo_results[0].boxes.xyxy.cpu().numpy()
    c = yolo_results[0].boxes.conf.cpu().numpy()
    cls = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    
    sam_m = None
    try:
        sam_results = sam_model.predict(yolo_results[0].orig_img, bboxes=yolo_results[0].boxes.xyxy, verbose=False)
        if sam_results and sam_results[0].masks is not None:
            sam_m = sam_results[0].masks.data.cpu().numpy()
            h, w = sam_results[0].orig_img.shape[:2]
            sam_m = np.array([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for m in sam_m])
    except Exception as e:
        print(f"    ⚠️ SAM2 error: {e}")

    img = draw_custom(img_path, b, sam_m, c, cls, color)
    return img if img is not None else cv2.imread(img_path)


# ==============================================================================
# MASK R-CNN MODEL BUILDER
# ==============================================================================

def _build_maskrcnn(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)

    pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
    if not os.path.exists(pt):
        print(f"  ❌ Mask R-CNN best.pt tidak ditemukan: {pt}")
        return None
    model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


# ==============================================================================
# MAIN
# ==============================================================================

def generate(device_str: str = "cuda:0"):
    from ultralytics import YOLO, SAM

    # Pastikan folder pipeline/visuals ada
    os.makedirs(PIPELINE_VISUALS_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    global CLASS_NAMES
    import yaml as _yaml, glob
    # Ambil class names dari data.yaml (bukan _annotations.coco.json)
    _data_yaml_path = os.path.join(os.path.dirname(TRAIN_VALID_DIR), "data.yaml")
    with open(_data_yaml_path, "r") as f:
        _data_cfg = _yaml.safe_load(f)
    _names = _data_cfg.get("names", [])
    if isinstance(_names, dict):
        CLASS_NAMES = [_names[k] for k in sorted(_names.keys())]
    else:
        CLASS_NAMES = list(_names)
    
    # Gunakan image sample dari TRAINING dataset validasi
    IMG_SAMPLE_DIR = os.path.join(TRAIN_VALID_DIR, "images")
    if not os.path.isdir(IMG_SAMPLE_DIR):
        IMG_SAMPLE_DIR = TRAIN_VALID_DIR  # fallback jika struktur flat
    img_files = sorted(glob.glob(os.path.join(IMG_SAMPLE_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_SAMPLE_DIR, "*.png")))
    samples = [{"file_name": os.path.basename(f)} for f in img_files][:N_SAMPLES]
    print(f"[CompGrid] {len(samples)} gambar dari {IMG_SAMPLE_DIR}.")

    # ── Load semua model sekali ──
    print("[CompGrid] Loading models ...")
    loaded = {}

    for mcfg in MODELS:
        mkey = mcfg["key"]
        mtype = mcfg["type"]

        if mtype == "yolo_seg":
            pt = os.path.join(get_output_dir(mkey), "weights", "best.pt")
            if os.path.exists(pt):
                loaded[mkey] = {"type": mtype, "model": YOLO(pt)}
                print(f"  ✅ {mcfg['label']}: {pt}")
            else:
                print(f"  ❌ {mcfg['label']}: {pt} NOT FOUND")
                loaded[mkey] = None

        elif mtype == "maskrcnn":
            m = _build_maskrcnn(device)
            if m:
                loaded[mkey] = {"type": mtype, "model": m}
                print(f"  ✅ {mcfg['label']}")
            else:
                loaded[mkey] = None

        elif mtype == "hybrid":
            yolo_pt = os.path.join(get_output_dir("yolo11l"), "weights", "best.pt")
            sam_pt  = os.path.join(ROOT, "models", "sam2.1_t.pt")

            if not os.path.exists(sam_pt):
                print(f"  ⏳ Mengunduh sam2.1_t.pt ...")
                try:
                    from ultralytics.utils.downloads import download
                    download("https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt",
                             dir=os.path.dirname(sam_pt))
                except Exception as e:
                    print(f"  ❌ Gagal unduh SAM2: {e}")

            if os.path.exists(yolo_pt) and os.path.exists(sam_pt):
                loaded[mkey] = {
                    "type": mtype,
                    "yolo": YOLO(yolo_pt),
                    "sam":  SAM(sam_pt),
                }
                print(f"  ✅ {mcfg['label']}: YOLO={yolo_pt} SAM={sam_pt}")
            else:
                print(f"  ❌ {mcfg['label']}: weights missing")
                loaded[mkey] = None

    # ── Generate comparison grids ──
    print(f"\n[CompGrid] Generating {len(samples)} comparison grids ...")

    for idx, img_info in enumerate(samples, 1):
        img_path = os.path.join(IMG_SAMPLE_DIR, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        panels = []
        for mcfg in MODELS:
            mkey = mcfg["key"]
            entry = loaded.get(mkey)

            if entry is None:
                # Model tidak tersedia — panel hitam
                panel = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(panel, "MODEL NOT FOUND", (W//4, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif entry["type"] == "yolo_seg":
                panel = _render_yolo_seg(entry["model"], img_path, device_str, mcfg["color"])
            elif entry["type"] == "maskrcnn":
                panel = _render_maskrcnn(entry["model"], img_path, device, mcfg["color"])
            elif entry["type"] == "hybrid":
                panel = _render_hybrid(entry["yolo"], entry["sam"],
                                       img_path, device_str, mcfg["color"])
            else:
                panel = img_bgr.copy()

            panels.append(panel)

        base = os.path.splitext(img_info["file_name"])[0]
        out = os.path.join(COMPARISON_DIR, f"comparison_{idx:02d}_{base}.jpg")
        
        # Plot grid menggunakan matplotlib seperti generate_paper_visuals.py
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        for i in range(6):
            if i < len(panels) and panels[i] is not None:
                img_rgb = cv2.cvtColor(panels[i], cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(MODELS[i]["label"], fontsize=18, fontweight='bold')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [{idx}/{len(samples)}] → {out}")

    # Cleanup
    del loaded
    _flush()

    print(f"\n✅ {len(samples)} comparison grids saved → {COMPARISON_DIR}")
    
    # Kompresi direktori new_methods
    print("\n[CompGrid] Melakukan kompresi arsip...")
    import tarfile
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_name = f"new_methods_comparison_{timestamp}.tar.gz"
    tar_path = os.path.join(VISUALS_DIR, tar_name)
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(PIPELINE_VISUALS_DIR, arcname="visuals")
        
    print(f"✅ Arsip visual berhasil dibuat di: {tar_path}")

    send_telegram_msg(
        f"✅ <b>Comparison Grid UNU</b>\n"
        f"{len(samples)} grids → <code>{COMPARISON_DIR}</code>\n"
        f"Archive: <code>{tar_name}</code>"
    )


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 5-model comparison grids — Live inference")
    parser.add_argument("--gpus", type=str, default="0",
        help="GPU ID. Default: '0'.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  CUDA tidak tersedia.")
        DEV = "cpu"
    else:
        gid = int(args.gpus.split(",")[0].strip())
        n = torch.cuda.device_count()
        if gid >= n:
            print(f"❌ GPU {gid} tidak tersedia ({n} GPU)."); sys.exit(1)
        DEV = f"cuda:{gid}"
        print(f"[Setup] GPU: {torch.cuda.get_device_name(gid)} ({DEV})")

    generate(device_str=DEV)
