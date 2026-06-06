# -*- coding: utf-8 -*-
"""
utils/generals/eval_unu_helpers.py
==================================
Shared helpers untuk eval_unu_dual.py — Boundary IoU bootstrap,
inference functions, evaluation runners, dan visual generators.
"""

import os, sys, gc, subprocess, copy, time
import numpy as np
import cv2

# ==============================================================================
# PATH SETUP — resolve ROOT dari lokasi file ini (utils/generals/eval_unu_helpers.py)
# ==============================================================================
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_UTILS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, _UTILS_DIR)

import torch
from config_shared import IMAGE_SIZE, NUM_CLASSES, get_output_dir, MODEL_COLORS


# ==============================================================================
# BOUNDARY-IOU BOOTSTRAP
# ==============================================================================

def ensure_boundary_iou_installed() -> bool:
    """Auto-install boundary-iou-api jika belum tersedia."""
    def _try_import():
        try:
            from boundary_iou.coco_instance_api.coco import COCO  # noqa
            from boundary_iou.coco_instance_api.cocoeval import COCOeval  # noqa
            return True
        except (ImportError, AttributeError):
            return False

    if _try_import():
        print("[BoundaryIoU] ✅ boundary-iou-api sudah terinstall.")
        return True

    print("[BoundaryIoU] ⏳ Menginstall boundary-iou-api via git clone...")
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="boundary_iou_")
    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/bowenc0221/boundary-iou-api.git",
             tmp_dir, "--depth=1", "--quiet"], timeout=120)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", tmp_dir, "--quiet"], timeout=120)
    except Exception as e:
        print(f"[BoundaryIoU] ❌ Instalasi gagal: {e}")
        return False

    _patch_boundary_iou_numpy(tmp_dir)

    import importlib, site
    importlib.invalidate_caches()
    for sp in site.getsitepackages():
        if sp not in sys.path:
            sys.path.insert(0, sp)

    if _try_import():
        print("[BoundaryIoU] ✅ Berhasil diinstall.")
        return True
    print("[BoundaryIoU] ❌ Import masih gagal setelah instalasi.")
    return False


def _patch_boundary_iou_numpy(clone_dir: str) -> None:
    """Patch deprecated np.float/np.bool aliases + loadRes bbox comparison."""
    import re
    targets = [
        os.path.join(clone_dir, "boundary_iou", "coco_instance_api", "cocoeval.py"),
        os.path.join(clone_dir, "boundary_iou", "lvis_instance_api", "eval.py"),
        os.path.join(clone_dir, "boundary_iou", "cityscapes_instance_api",
                     "evalInstanceLevelSemanticLabeling.py"),
    ]
    replacements = [
        (r"\bnp\.float\b", "np.float64"), (r"\bnp\.int\b", "np.int64"),
        (r"\bnp\.bool\b", "np.bool_"), (r"\bnp\.complex\b", "np.complex128"),
        (r"\bnp\.object\b", "np.object_"), (r"\bnp\.str\b", "np.str_"),
    ]
    for fpath in targets:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        patched = src
        for old, new in replacements:
            patched = re.sub(old, new, patched)
        if patched != src:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"  [Patch] ✅ {os.path.basename(fpath)}")

    coco_path = os.path.join(clone_dir, "boundary_iou", "coco_instance_api", "coco.py")
    if os.path.exists(coco_path):
        with open(coco_path, "r", encoding="utf-8") as f:
            src = f.read()
        old_line = "elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:"
        new_line = ("elif 'bbox' in anns[0] and "
                    "not (list(anns[0]['bbox']) "
                    "if hasattr(anns[0]['bbox'], '__iter__') "
                    "else [anns[0]['bbox']]) == []:")
        if old_line in src:
            with open(coco_path, "w", encoding="utf-8") as f:
                f.write(src.replace(old_line, new_line))
            print("  [Patch] ✅ loadRes bbox numpy-safe: coco.py")


# ==============================================================================
# GPU HELPERS
# ==============================================================================

def flush_gpu(label: str = ""):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        print(f"  [MemFlush] {label} — VRAM: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)


def gpu_str() -> str:
    from collections import Counter
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())


def mask_to_rle(bin_mask: np.ndarray) -> dict:
    from pycocotools import mask as maskUtils
    rle = maskUtils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ==============================================================================
# EVALUATION RUNNERS
# ==============================================================================

def run_boundary_eval(coco_gt, dt_segm: list, label: str) -> dict:
    """Jalankan COCOeval dengan iouType='boundary'."""
    from boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryCOCOeval

    if not dt_segm:
        return {k: "N/A" for k in ["BoundAP", "BoundAP50", "BoundAP75",
                                    "BoundAP_S", "BoundAP_M", "BoundAP_L"]}
    b_gt = BoundaryCOCO()
    b_gt.dataset = coco_gt.dataset
    b_gt.createIndex()
    b_dt = b_gt.loadRes(dt_segm)
    ev = BoundaryCOCOeval(b_gt, b_dt, iouType="boundary")
    ev.evaluate(); ev.accumulate()
    print(f"\n  [BoundaryIoU] ── {label} ──")
    ev.summarize()
    s = ev.stats
    return {"BoundAP": round(float(s[0]), 4), "BoundAP50": round(float(s[1]), 4),
            "BoundAP75": round(float(s[2]), 4), "BoundAP_S": round(float(s[3]), 4),
            "BoundAP_M": round(float(s[4]), 4), "BoundAP_L": round(float(s[5]), 4)}


def run_standard_mask_eval(coco_gt, dt_segm: list, label: str) -> dict:
    """Jalankan COCOeval STANDAR dengan iouType='segm'."""
    from pycocotools.coco import COCO as StdCOCO
    from pycocotools.cocoeval import COCOeval as StdCOCOeval

    if not dt_segm:
        return {k: "N/A" for k in ["MaskAP", "MaskAP50", "MaskAP75",
                                    "MaskAP_S", "MaskAP_M", "MaskAP_L"]}
    s_gt = StdCOCO()
    s_gt.dataset = coco_gt.dataset
    s_gt.createIndex()
    s_dt = s_gt.loadRes(dt_segm)
    ev = StdCOCOeval(s_gt, s_dt, iouType="segm")
    ev.evaluate(); ev.accumulate()
    print(f"\n  [MaskAP-Std] ── {label} ──")
    ev.summarize()
    s = ev.stats
    return {"MaskAP": round(float(s[0]), 4), "MaskAP50": round(float(s[1]), 4),
            "MaskAP75": round(float(s[2]), 4), "MaskAP_S": round(float(s[3]), 4),
            "MaskAP_M": round(float(s[4]), 4), "MaskAP_L": round(float(s[5]), 4)}


# ==============================================================================
# INFERENCE: YOLO-SEG
# ==============================================================================

def infer_yolo_seg(model_key: str, pt_path: str, image_ids: dict,
                   device_str: str) -> list:
    from ultralytics import YOLO

    if not os.path.exists(pt_path):
        print(f"  ❌ {model_key}: best.pt tidak ditemukan: {pt_path}")
        return []

    print(f"\n  [Infer] Loading {model_key} dari {pt_path} ...")
    model = YOLO(pt_path)
    dt_segm = []

    for img_path, img_id in image_ids.items():
        results = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                device=device_str, verbose=False)
        if not results or results[0].masks is None:
            continue
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]
        masks_data = results[0].masks.data.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        for i, mask in enumerate(masks_data):
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            bin_mask = (mask > 0.5).astype(np.uint8)
            rle = mask_to_rle(bin_mask)
            # YOLO 0-indexed → +1 untuk mencocokkan UNU category_id 1-7
            dt_segm.append({
                "image_id":     img_id,
                "category_id":  int(clss[i]) + 1,
                "segmentation": rle,
                "score":        float(confs[i]),
            })
    del model
    flush_gpu(model_key)
    print(f"  [Infer] {model_key}: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# INFERENCE: MASK R-CNN
# ==============================================================================

def _build_maskrcnn_model(device: torch.device) -> torch.nn.Module:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
    return model.to(device)


def infer_maskrcnn(pt_path: str, image_ids: dict, device: torch.device) -> list:
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    if not os.path.exists(pt_path):
        print(f"  ❌ Mask R-CNN: best.pt tidak ditemukan: {pt_path}")
        return []

    print(f"\n  [Infer] Loading Mask R-CNN dari {pt_path} ...")
    model = _build_maskrcnn_model(device)
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    model.eval()
    dt_segm = []

    with torch.no_grad():
        for img_path, img_id in image_ids.items():
            pil = PILImage.open(img_path).convert("RGB")
            W, H = pil.size
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
            preds = model(t_img)[0]
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            masks = preds["masks"].cpu().numpy()
            for i in range(len(scores)):
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                if bin_mask.shape != (H, W):
                    bin_mask = cv2.resize(bin_mask, (W, H),
                                          interpolation=cv2.INTER_NEAREST)
                rle = mask_to_rle(bin_mask)
                # Mask R-CNN sudah 1-indexed — cocok dengan UNU category_id 1-7
                dt_segm.append({
                    "image_id":     img_id,
                    "category_id":  int(labels[i]),
                    "segmentation": rle,
                    "score":        float(scores[i]),
                })
    del model
    flush_gpu("maskrcnn")
    print(f"  [Infer] Mask R-CNN: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# INFERENCE: HYBRID (YOLO11l + SAM2)
# ==============================================================================

def infer_hybrid(yolo_pt: str, sam_pt: str, image_ids: dict,
                 device_str: str) -> list:
    from ultralytics import YOLO, SAM

    if not os.path.exists(yolo_pt):
        print(f"  ❌ Hybrid YOLO11l: best.pt tidak ditemukan: {yolo_pt}")
        return []
    if not os.path.exists(sam_pt):
        print(f"  ⏳ Mengunduh sam2.1_t.pt ...")
        try:
            from ultralytics.utils.downloads import download
            download("https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt",
                     dir=os.path.dirname(sam_pt))
        except Exception as e:
            print(f"  ❌ Gagal unduh SAM2: {e}")
            return []

    print(f"\n  [Infer] Loading Hybrid YOLO11l + SAM2 ...")
    yolo_model = YOLO(yolo_pt)
    sam_model = SAM(sam_pt)
    dt_segm = []

    for img_path, img_id in image_ids.items():
        det = yolo_model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                 device=device_str, verbose=False)
        if not (det and det[0].boxes is not None and len(det[0].boxes) > 0):
            continue
        pred_boxes = det[0].boxes.xyxy.cpu().numpy()
        pred_confs = det[0].boxes.conf.cpu().numpy()
        pred_clss = det[0].boxes.cls.cpu().numpy().astype(int)

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]

        try:
            sam_result = sam_model.predict(det[0].orig_img,
                                           bboxes=pred_boxes, verbose=False)
        except Exception:
            sam_result = None

        if sam_result and sam_result[0].masks is not None:
            sam_masks = sam_result[0].masks.data.cpu().numpy()
        else:
            sam_masks = [None] * len(pred_boxes)

        for i in range(len(pred_boxes)):
            mask = sam_masks[i] if i < len(sam_masks) else None
            if mask is None:
                continue
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            bin_mask = (mask > 0.5).astype(np.uint8)
            rle = mask_to_rle(bin_mask)
            dt_segm.append({
                "image_id":     img_id,
                "category_id":  int(pred_clss[i]) + 1,
                "segmentation": rle,
                "score":        float(pred_confs[i]),
            })
    del yolo_model, sam_model
    flush_gpu("hybrid")
    print(f"  [Infer] Hybrid: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# VISUAL GENERATION
# ==============================================================================

def generate_visuals(model_key: str, dt_segm: list, coco_gt,
                     img_dir: str, boundary_dir: str, normal_dir: str):
    """Generate 164 boundary contour + 164 full mask overlay images."""
    from pycocotools import mask as maskUtils

    os.makedirs(boundary_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    theme_color = MODEL_COLORS.get(model_key, (255, 255, 0))
    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    # Group predictions by image_id
    preds_by_img = {}
    for pred in dt_segm:
        preds_by_img.setdefault(pred["image_id"], []).append(pred)

    count = 0
    for img_info in sorted(coco_gt.dataset["images"], key=lambda x: x["id"]):
        img_id = img_info["id"]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        preds = preds_by_img.get(img_id, [])
        count += 1
        base = os.path.splitext(img_info["file_name"])[0]

        # ── Boundary IoU Visual (kontur tepian) ──
        bnd_img = img_bgr.copy()
        for pred in preds:
            mask = maskUtils.decode(pred["segmentation"])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bnd_img, contours, -1, theme_color, 2)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                cat_name = cat_names.get(pred["category_id"], f"cls{pred['category_id']}")
                lbl = f"{cat_name} {pred['score']:.2f}"
                cv2.rectangle(bnd_img, (x1, y1), (x2, y2), theme_color, 1)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(bnd_img, (x1, y1 - th - 4), (x1 + tw + 2, y1), theme_color, -1)
                cv2.putText(bnd_img, lbl, (x1 + 1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(boundary_dir, f"{model_key}_{count:03d}_{base}.png"), bnd_img)

        # ── Normal COCO Visual (mask penuh semi-transparan) ──
        overlay = img_bgr.copy()
        for pred in preds:
            mask = maskUtils.decode(pred["segmentation"])
            colored = np.zeros_like(overlay)
            colored[mask == 1] = theme_color
            cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                cat_name = cat_names.get(pred["category_id"], f"cls{pred['category_id']}")
                lbl = f"{cat_name} {pred['score']:.2f}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_color, 2)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
                cv2.putText(overlay, lbl, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(normal_dir, f"{model_key}_{count:03d}_{base}.png"), overlay)

    print(f"  [Visual] {model_key}: {count} gambar → boundary_iou/ + normal_coco/")


# ==============================================================================
# COMPARISON GRID (5 Models × 10 Images)
# ==============================================================================

# Label → model_key mapping dan warna title bar (BGR)
_GRID_MODELS = [
    {"label": "YOLOv8m-Seg",       "key": "yolov8m_seg", "bar_color": (255,   0,   0)},   # Biru
    {"label": "YOLOv9c-Seg",       "key": "yolov9c_seg", "bar_color": (  0,   0, 255)},   # Merah
    {"label": "YOLO11l-Seg",       "key": "yolo11l_seg", "bar_color": (  0, 255,   0)},   # Hijau
    {"label": "Mask R-CNN ResNet-50","key": "maskrcnn",   "bar_color": (255,   0, 255)},   # Magenta
    {"label": "Hybrid (Seg)",       "key": "hybrid",      "bar_color": (  0, 165, 255)},   # Orange
]

_TITLE_BAR_H = 30   # Tinggi bar judul per panel


def _render_single_panel(img_bgr, preds, theme_color, cat_names):
    """Render overlay mask + bbox + label pada salinan gambar."""
    from pycocotools import mask as maskUtils

    overlay = img_bgr.copy()
    for pred in preds:
        mask = maskUtils.decode(pred["segmentation"])
        colored = np.zeros_like(overlay)
        colored[mask == 1] = theme_color
        cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            cat_name = cat_names.get(pred["category_id"], f"cls{pred['category_id']}")
            lbl = f"{cat_name} {pred['score']:.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_color, 2)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 2, y1), theme_color, -1)
            cv2.putText(overlay, lbl, (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _add_title_bar(panel, label, bar_color, bar_h=_TITLE_BAR_H):
    """Tambahkan bar judul berwarna di atas panel."""
    W = panel.shape[1]
    bar = np.zeros((bar_h, W, 3), dtype=np.uint8)
    bar[:] = bar_color
    cv2.putText(bar, label, (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def generate_comparison_grids(all_dt_segm: dict, coco_gt, img_dir: str,
                              out_dir: str, n_samples: int = 10):
    """
    Generate comparison grid images (5 model per gambar).
    Layout: 3 panel atas + 2 panel bawah (+ 1 slot kosong hitam).

    Args:
        all_dt_segm: dict {model_key: list_of_dt_segm}
        coco_gt: pycocotools COCO GT object
        img_dir: path ke folder gambar
        out_dir: folder output untuk menyimpan comparison grids
        n_samples: jumlah gambar sampel (default 10)
    """
    os.makedirs(out_dir, exist_ok=True)
    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    # Pilih 10 gambar sampel (merata dari dataset)
    all_imgs = sorted(coco_gt.dataset["images"], key=lambda x: x["id"])
    step = max(1, len(all_imgs) // n_samples)
    sample_imgs = all_imgs[::step][:n_samples]

    # Pre-group prediksi per model per image_id
    grouped = {}  # {model_key: {image_id: [preds]}}
    for mcfg in _GRID_MODELS:
        mkey = mcfg["key"]
        grouped[mkey] = {}
        for pred in all_dt_segm.get(mkey, []):
            grouped[mkey].setdefault(pred["image_id"], []).append(pred)

    print(f"\n  [Comparison] Generating {len(sample_imgs)} grid images ...")

    for idx, img_info in enumerate(sample_imgs, 1):
        img_id = img_info["id"]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        panels = []
        for mcfg in _GRID_MODELS:
            mkey = mcfg["key"]
            preds = grouped[mkey].get(img_id, [])
            theme = MODEL_COLORS.get(mkey, mcfg["bar_color"])
            rendered = _render_single_panel(img_bgr, preds, theme, cat_names)
            with_bar = _add_title_bar(rendered, mcfg["label"], mcfg["bar_color"])
            panels.append(with_bar)

        # Layout: baris atas 3 panel, baris bawah 2 panel + 1 slot hitam
        panel_h = panels[0].shape[0]  # H + bar
        panel_w = panels[0].shape[1]  # W

        row_top = np.hstack(panels[:3])

        # Slot kosong hitam untuk menyeimbangkan baris bawah
        black_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        row_bot = np.hstack(panels[3:] + [black_panel])

        # Pastikan lebar baris atas dan bawah sama
        if row_top.shape[1] != row_bot.shape[1]:
            target_w = row_top.shape[1]
            row_bot = cv2.resize(row_bot, (target_w, panel_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Header utama
        header_h = 35
        header = np.zeros((header_h, row_top.shape[1], 3), dtype=np.uint8)
        title = "Segmentation Comparison UNU | YOLOv8 | YOLOv9 | YOLO11 | Mask R-CNN | Hybrid"
        cv2.putText(header, title, (10, header_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

        grid = np.vstack([header, row_top, row_bot])

        base = os.path.splitext(img_info["file_name"])[0]
        out_path = os.path.join(out_dir, f"comparison_{idx:02d}_{base}.jpg")
        cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"    [{idx}/{len(sample_imgs)}] → {out_path}")

    print(f"  [Comparison] ✅ {len(sample_imgs)} comparison grids saved.")
