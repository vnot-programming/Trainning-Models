# -*- coding: utf-8 -*-
"""
hybrid/generate_visuals.py
===========================
Hasilkan 3 gambar visualisasi perbandingan pipeline CV:

  1. visual_hybrid_single.png          — YOLO bbox + SAM2 mask pada 1 gambar
  2. visual_detection_comparison.png   — Grid 2x2: YOLOv8|YOLOv9 / YOLO11|Hybrid
  3. visual_segmentation_comparison.png — Grid 3x2: v8seg|v9seg|11seg / MaskRCNN|Hybrid|blank

Dipanggil dari hybrid/main.py dan hybrid/eval_multigpu.py,
atau standalone: python3 generate_visuals.py [--img /path/to/image.jpg]
"""

import os, sys, gc, csv, argparse
import cv2
import numpy as np

_HYBRID_DIR      = os.path.abspath(os.path.dirname(__file__))
_FINETUNING_ROOT = os.path.abspath(os.path.join(_HYBRID_DIR, ".."))
_MASKRCNN_DIR    = os.path.join(_FINETUNING_ROOT, "mask-r-cnn")
sys.path.insert(0, _FINETUNING_ROOT)
sys.path.insert(0, _MASKRCNN_DIR)

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from config_shared import (
    IMAGE_SAMPLES_DIR, VISUALS_DIR, REPORTS_DIR, MODEL_COLORS,
    get_output_dir, DET_YAML, NUM_CLASSES, IMAGE_SIZE,
)

SAM2_PT = os.path.join(_HYBRID_DIR, "sam2.1_b.pt")
DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"

# Panel layout
PANEL_W  = 640
PANEL_H  = 480
HEADER_H = 52
GAP      = 6
BG_COLOR = (15, 15, 15)

CLASS_NAMES = ["dishwasher", "milk", "mineral", "non_mineral",
               "not_empty", "soda", "yogurt"]


# ── mAP reader ────────────────────────────────────────────────────────────────
def _read_map(model_key: str, task: str = "det") -> str:
    """Baca mAP50-95 dari CSV report di REPORTS_DIR."""
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
                        for key in ["mAP50-95(Box)", "mAP50-95", "mAP50_95"]:
                            v = row.get(key, "")
                            if v and v not in ("N/A", "ERR", ""):
                                return str(round(float(v), 4))
            except Exception:
                pass
    return "N/A"


# ── Panel helpers ──────────────────────────────────────────────────────────────
def _placeholder(msg: str = "Model not found") -> np.ndarray:
    img = np.full((PANEL_H, PANEL_W, 3), 30, dtype=np.uint8)
    cv2.putText(img, msg, (20, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (120, 120, 120), 2, cv2.LINE_AA)
    return img


def _add_header(panel: np.ndarray, label: str, color_bgr: tuple, map_str: str) -> np.ndarray:
    """Tambah bar header berwarna di atas panel."""
    bar = np.full((HEADER_H, PANEL_W, 3), color_bgr, dtype=np.uint8)
    cv2.putText(bar, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if map_str != "N/A":
        cv2.putText(bar, f"mAP50-95: {map_str}", (PANEL_W - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 180), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def _draw_yolo_result(img_bgr: np.ndarray, result, color: tuple,
                       draw_mask: bool = False) -> np.ndarray:
    """Gambar box + label kelas (nama, bukan index) + confidence dari result YOLO."""
    out   = img_bgr.copy()
    names = result.names   # dict {id: name}

    if result.masks is not None and draw_mask:
        H, W = out.shape[:2]
        for i, m in enumerate(result.masks.data.cpu().numpy()):
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
            conf_str = f"{float(confs[i]):.2f}"
            label_txt = f"{cls_name} {conf_str}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label_txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Panel renderers ────────────────────────────────────────────────────────────
def render_yolo_panel(img_bgr: np.ndarray, pt_path: str, color: tuple,
                       seg: bool = False) -> np.ndarray:
    if not os.path.exists(pt_path):
        return _resize(_placeholder(f"Not found:\n{os.path.basename(pt_path)}"))
    try:
        from ultralytics import YOLO
        model  = YOLO(pt_path)
        result = model.predict(img_bgr, conf=0.5, imgsz=IMAGE_SIZE,
                               device=DEVICE, verbose=False)[0]
        out = _draw_yolo_result(img_bgr.copy(), result, color, draw_mask=seg)
        del model; gc.collect(); torch.cuda.empty_cache()
        return _resize(out)
    except Exception as e:
        print(f"  [Visual] YOLO panel error: {e}")
        return _resize(_placeholder(str(e)[:60]))


def render_hybrid_panel(img_bgr: np.ndarray) -> np.ndarray:
    """YOLO bbox (orange) + SAM2 mask overlay."""
    yolo_pt = os.path.join(get_output_dir("yolo11m"), "weights", "best.pt")
    if not os.path.exists(yolo_pt) or not os.path.exists(SAM2_PT):
        return _resize(_placeholder("Hybrid model not found"))
    try:
        from ultralytics import YOLO, SAM
        color  = MODEL_COLORS.get("hybrid", (0, 165, 255))
        out    = img_bgr.copy()
        H, W   = out.shape[:2]

        yolo   = YOLO(yolo_pt)
        det    = yolo.predict(img_bgr, conf=0.5, imgsz=IMAGE_SIZE,
                              device=DEVICE, verbose=False)[0]
        names  = det.names

        if det.boxes is not None and len(det.boxes) > 0:
            boxes = det.boxes.xyxy.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()
            clss  = det.boxes.cls.cpu().numpy().astype(int)

            sam   = SAM(SAM2_PT)
            sam_r = sam.predict(img_bgr, bboxes=boxes, verbose=False)

            if sam_r and sam_r[0].masks is not None:
                for m in sam_r[0].masks.data.cpu().numpy():
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype(np.float32), (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                    alpha = (m > 0.5).astype(np.uint8)
                    colored = np.zeros_like(out)
                    colored[alpha == 1] = color
                    cv2.addWeighted(colored, 0.42, out, 1.0, 0, out)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cls_name = names.get(clss[i], f"cls{clss[i]}")
                conf_str = f"{float(confs[i]):.2f}"
                label_txt = f"{cls_name} {conf_str}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label_txt, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            del sam
        del yolo; gc.collect(); torch.cuda.empty_cache()
        return _resize(out)
    except Exception as e:
        print(f"  [Visual] Hybrid panel error: {e}")
        return _resize(_placeholder(str(e)[:60]))


def render_maskrcnn_panel(img_bgr: np.ndarray) -> np.ndarray:
    pt_path = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        return _resize(_placeholder("Mask R-CNN not found"))
    try:
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
        from maskrcnn_builder import build_model

        color  = MODEL_COLORS.get("maskrcnn", (255, 0, 255))
        device = torch.device(DEVICE)
        model  = build_model(device=device)
        ckpt   = torch.load(pt_path, map_location=device)
        state  = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()

        img_pil    = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_tensor = TF.to_tensor(img_pil).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        out = img_bgr.copy()
        H, W = out.shape[:2]
        boxes  = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks  = outputs["masks"].cpu().numpy()  # (N, 1, H, W)

        for i in range(len(boxes)):
            if scores[i] < 0.5:
                continue
            m = masks[i, 0]
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            alpha = (m > 0.5).astype(np.uint8)
            colored = np.zeros_like(out)
            colored[alpha == 1] = color
            cv2.addWeighted(colored, 0.40, out, 1.0, 0, out)

            x1, y1, x2, y2 = map(int, boxes[i])
            cls_idx  = int(labels[i]) - 1   # 0-indexed, 0=background
            cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"cls{cls_idx}"
            label_txt = f"{cls_name} {scores[i]:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label_txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        del model; gc.collect(); torch.cuda.empty_cache()
        return _resize(out)
    except Exception as e:
        print(f"  [Visual] MaskRCNN panel error: {e}")
        return _resize(_placeholder(str(e)[:60]))


# ── Grid assemblers ────────────────────────────────────────────────────────────
def _gap_col(h): return np.full((h, GAP, 3), BG_COLOR, dtype=np.uint8)
def _gap_row(w): return np.full((GAP, w, 3), BG_COLOR, dtype=np.uint8)


def _make_panel_full(img: np.ndarray, label: str, color: tuple, map_str: str) -> np.ndarray:
    return _add_header(_resize(img), label, color, map_str)


def assemble_2x2(panels_with_meta: list) -> np.ndarray:
    """panels_with_meta: list of (panel_img, label, color, map_str), 4 items."""
    row_imgs = []
    for r in range(2):
        cols = []
        for c in range(2):
            idx = r * 2 + c
            p, lbl, col, mp = panels_with_meta[idx]
            full = _make_panel_full(p, lbl, col, mp)
            cols.append(full)
            if c < 1:
                cols.append(_gap_col(full.shape[0]))
        row_imgs.append(np.hstack(cols))
        if r < 1:
            row_imgs.append(_gap_row(row_imgs[-1].shape[1]))
    canvas = np.vstack(row_imgs)
    return canvas


def assemble_3x2(panels_with_meta: list) -> np.ndarray:
    """panels_with_meta: list of (panel_img, label, color, map_str), 5–6 items.
    Last slot may be blank."""
    while len(panels_with_meta) < 6:
        blank = np.full((PANEL_H, PANEL_W, 3), BG_COLOR, dtype=np.uint8)
        panels_with_meta.append((blank, "", BG_COLOR, ""))

    row_imgs = []
    for r in range(2):
        cols = []
        for c in range(3):
            idx = r * 3 + c
            p, lbl, col, mp = panels_with_meta[idx]
            full = _make_panel_full(p, lbl, col, mp)
            cols.append(full)
            if c < 2:
                cols.append(_gap_col(full.shape[0]))
        row_imgs.append(np.hstack(cols))
        if r < 1:
            row_imgs.append(_gap_row(row_imgs[-1].shape[1]))
    return np.vstack(row_imgs)


# ── Gambar 1: Hybrid single ────────────────────────────────────────────────────
def generate_visual_1(img_bgr: np.ndarray, out_dir: str) -> str:
    print("\n  [Visual-1] Hybrid Single Pipeline...")
    panel = render_hybrid_panel(img_bgr)
    color = MODEL_COLORS.get("hybrid", (0, 165, 255))
    map_v = _read_map("hybrid", "det")

    # Title bar
    title_h = 56
    title   = np.full((title_h, panel.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(title, "Hybrid Pipeline: YOLO11m Detection → SAM2 Segmentation",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2, cv2.LINE_AA)

    header = np.full((HEADER_H, panel.shape[1], 3), color, dtype=np.uint8)
    cv2.putText(header, "Hybrid (YOLO11m + SAM2)", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if map_v != "N/A":
        cv2.putText(header, f"mAP50-95: {map_v}", (PANEL_W - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 180), 1, cv2.LINE_AA)

    result = np.vstack([title, header, panel])
    out_path = os.path.join(out_dir, "visual_hybrid_single.png")
    cv2.imwrite(out_path, result)
    print(f"  ✅ {out_path}")
    return out_path


# ── Gambar 2: Detection 2×2 ────────────────────────────────────────────────────
def generate_visual_2(img_bgr: np.ndarray, out_dir: str) -> str:
    print("\n  [Visual-2] Detection Comparison 2×2...")
    models = [
        ("yolov8m",  "YOLOv8m",  MODEL_COLORS.get("yolov8m",  (255, 0, 0)),   "det", False),
        ("yolov9m",  "YOLOv9m",  MODEL_COLORS.get("yolov9m",  (0, 0, 255)),   "det", False),
        ("yolo11m",  "YOLO11m",  MODEL_COLORS.get("yolo11m",  (0, 255, 0)),   "det", False),
    ]
    panels = []
    for key, label, color, task, seg in models:
        pt = os.path.join(get_output_dir(key), "weights", "best.pt")
        print(f"    {label}...")
        panel = render_yolo_panel(img_bgr, pt, color, seg=seg)
        mp    = _read_map(key, task)
        panels.append((panel, label, color, mp))

    # Hybrid sebagai panel ke-4
    print("    Hybrid...")
    hybrid_panel = render_hybrid_panel(img_bgr)
    hybrid_color = MODEL_COLORS.get("hybrid", (0, 165, 255))
    hybrid_map   = _read_map("hybrid", "det")
    panels.append((hybrid_panel, "Hybrid (YOLO11m+SAM2)", hybrid_color, hybrid_map))

    grid = assemble_2x2(panels)

    # Title bar
    title = np.full((56, grid.shape[1], 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(title, "Detection Comparison — YOLOv8m | YOLOv9m | YOLO11m | Hybrid",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2, cv2.LINE_AA)
    grid = np.vstack([title, grid])

    out_path = os.path.join(out_dir, "visual_detection_comparison.png")
    cv2.imwrite(out_path, grid)
    print(f"  ✅ {out_path}")
    return out_path


# ── Gambar 3: Segmentation 3×2 ────────────────────────────────────────────────
def generate_visual_3(img_bgr: np.ndarray, out_dir: str) -> str:
    print("\n  [Visual-3] Segmentation Comparison 3×2...")
    seg_models = [
        ("yolov8m_seg",  "YOLOv8m-Seg",       MODEL_COLORS.get("yolov8m_seg",  (255, 0, 0))),
        ("yolov9c_seg",  "YOLOv9c-Seg",        MODEL_COLORS.get("yolov9c_seg",  (0, 0, 255))),
        ("yolo11m_seg",  "YOLO11m-Seg",        MODEL_COLORS.get("yolo11m_seg",  (0, 255, 0))),
    ]
    panels = []
    for key, label, color in seg_models:
        pt = os.path.join(get_output_dir(key), "weights", "best.pt")
        print(f"    {label}...")
        panel = render_yolo_panel(img_bgr, pt, color, seg=True)
        mp    = _read_map(key, "seg")
        panels.append((panel, label, color, mp))

    # MaskRCNN
    print("    Mask R-CNN...")
    maskrcnn_color = MODEL_COLORS.get("maskrcnn", (255, 0, 255))
    maskrcnn_panel = render_maskrcnn_panel(img_bgr)
    maskrcnn_map   = _read_map("maskrcnn", "seg")
    panels.append((maskrcnn_panel, "Mask R-CNN ResNet-50 FPN-v2", maskrcnn_color, maskrcnn_map))

    # Hybrid
    print("    Hybrid...")
    hybrid_color = MODEL_COLORS.get("hybrid", (0, 165, 255))
    hybrid_panel = render_hybrid_panel(img_bgr)
    hybrid_map   = _read_map("hybrid", "seg")
    panels.append((hybrid_panel, "Hybrid (YOLO11m+SAM2)", hybrid_color, hybrid_map))

    grid = assemble_3x2(panels)

    title = np.full((56, grid.shape[1], 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(title,
                "Segmentation Comparison — YOLOv8 | YOLOv9c | YOLO11 | Mask R-CNN | Hybrid",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2, cv2.LINE_AA)
    grid = np.vstack([title, grid])

    out_path = os.path.join(out_dir, "visual_segmentation_comparison.png")
    cv2.imwrite(out_path, grid)
    print(f"  ✅ {out_path}")
    return out_path


# ── Entrypoint utama ───────────────────────────────────────────────────────────
def run_all_visuals(img_path: str = None) -> dict:
    """
    Dipanggil dari hybrid/main.py dan hybrid/eval_multigpu.py.
    Returns dict dengan path output.
    """
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Pilih gambar sampel
    if img_path is None or not os.path.exists(img_path):
        if os.path.isdir(IMAGE_SAMPLES_DIR):
            imgs = sorted([
                os.path.join(IMAGE_SAMPLES_DIR, f)
                for f in os.listdir(IMAGE_SAMPLES_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            img_path = imgs[0] if imgs else None

    if img_path is None:
        print("[Visual] ⚠️  Tidak ada gambar sampel ditemukan. Visualisasi dilewati.")
        return {}

    print(f"\n{'='*65}")
    print(f"  Generate Visual Comparisons")
    print(f"  Gambar: {img_path}")
    print(f"  Output: {VISUALS_DIR}")
    print(f"{'='*65}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[Visual] ❌ Gagal membaca gambar: {img_path}")
        return {}

    results = {}
    try:
        results["hybrid_single"]  = generate_visual_1(img_bgr, VISUALS_DIR)
    except Exception as e:
        print(f"[Visual] ❌ visual_1 error: {e}")

    try:
        results["detection_grid"] = generate_visual_2(img_bgr, VISUALS_DIR)
    except Exception as e:
        print(f"[Visual] ❌ visual_2 error: {e}")

    try:
        results["segmentation_grid"] = generate_visual_3(img_bgr, VISUALS_DIR)
    except Exception as e:
        print(f"[Visual] ❌ visual_3 error: {e}")

    print(f"\n✅ Visual generation selesai → {VISUALS_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual comparisons")
    parser.add_argument("--img", type=str, default=None,
                        help="Path gambar input (default: ambil dari IMAGE_SAMPLES_DIR)")
    args = parser.parse_args()
    run_all_visuals(args.img)
