# -*- coding: utf-8 -*-
"""
hybrid/main.py
==============
Evaluasi Final: Perbandingan semua model + Hybrid (YOLO11m + SAM2).

Langkah:
  1. Load semua best.pt (yolo8m, yolo9m, yolo11m, yolo8m-seg, yolo9c-seg,
     yolo11m-seg, maskrcnn)
  2. Jalankan YOLO11m + SAM2 hybrid pada 5 gambar random → 5 visual hybrid
  3. Buat 5 gambar perbandingan grid 1×5 (yolo8m, yolo9m, yolo11m, maskrcnn, hybrid)
     Total visual di subfolder 'hybrid': 10 gambar
  4. CSV detection comparison (yolo8m, yolo9m, yolo11m, hybrid)
  5. CSV segmentation comparison (yolo8m-seg, yolo9c-seg, yolo11m-seg, maskrcnn, hybrid)
  6. CSV latency hybrid per gambar

Cara menjalankan (setelah semua yolo & mask-r-cnn selesai):
    python -u main.py 2>&1 | tee hybrid_eval.log
"""

import os, sys, csv, gc, random, time
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config_shared import (
    WORKSPACE_DIR, SEG_DATASET_LOCATION, DET_YAML, SEG_YAML,
    IMAGE_SIZE, get_output_dir, compress_visuals,
)
from telegram_utils import send_telegram_msg
import torch
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"[Device] {DEVICE}")

# ==============================================================================
# HELPER: _benchmark_yolo (identik reporter.py)
# ==============================================================================
def _flush(label: str):
    gc.collect(); torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[MemFlush] {label} — VRAM bebas: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")


def _benchmark_yolo(model_path: str, yaml_path: str, task: str = "detect") -> dict:
    """Identik reporter.py _benchmark_yolo()."""
    empty = {"model_size_mb": "N/A", "map50_95": "N/A", "map50": "N/A",
             "map50_95_mask": "N/A", "precision": "N/A", "recall": "N/A",
             "preprocess_ms": "N/A", "inference_ms": "N/A",
             "postprocess_ms": "N/A", "fps": "N/A", "latency_ms": "N/A"}
    if not model_path or not os.path.exists(model_path):
        print(f"  ⚠️  {model_path} tidak ditemukan — skip")
        return empty
    try:
        model   = YOLO(model_path)
        metrics = model.val(data=yaml_path, imgsz=IMAGE_SIZE,
                            device="cuda:0", verbose=False, plots=False)
        size_mb = os.path.getsize(model_path) / 1e6
        speed   = metrics.speed if hasattr(metrics, "speed") else {}
        pre  = round(speed.get("preprocess",  0), 2)
        inf  = round(speed.get("inference",   0), 2)
        post = round(speed.get("postprocess", 0), 2)
        total = pre + inf + post
        fps   = round(1000 / total, 2) if total > 0 else "N/A"
        del model; _flush(f"benchmark {task}")
        if task == "segment":
            try:
                bm = round(float(metrics.box.map),   4)
                b5 = round(float(metrics.box.map50), 4)
                mm = round(float(metrics.seg.map),   4)
            except Exception:
                bm = b5 = mm = "N/A"
            return {"model_size_mb": round(size_mb, 2),
                    "map50_95": bm, "map50": b5, "map50_95_mask": mm,
                    "precision": "N/A", "recall": "N/A",
                    "preprocess_ms": pre, "inference_ms": inf,
                    "postprocess_ms": post, "fps": fps,
                    "latency_ms": round(total, 2)}
        else:
            try:
                bm = round(float(metrics.box.map),   4)
                b5 = round(float(metrics.box.map50), 4)
                pr = round(float(metrics.box.mp),    4)
                re = round(float(metrics.box.mr),    4)
            except Exception:
                bm = b5 = pr = re = "N/A"
            return {"model_size_mb": round(size_mb, 2),
                    "map50_95": bm, "map50": b5, "map50_95_mask": "N/A",
                    "precision": pr, "recall": re,
                    "preprocess_ms": pre, "inference_ms": inf,
                    "postprocess_ms": post, "fps": fps,
                    "latency_ms": round(total, 2)}
    except Exception as e:
        print(f"  ⚠️  Benchmark gagal {model_path}: {e}")
        return empty


# ==============================================================================
# 1. PATH SEMUA MODEL
# ==============================================================================
def _best(run_name): return os.path.join(get_output_dir(run_name), "weights", "best.pt")

paths = {
    "yolo8m_det":  _best("yolov8m"),
    "yolo9m_det":  _best("yolov9m"),
    "yolo11m_det": _best("yolo11m"),
    "yolo8m_seg":  _best("yolov8m_seg"),
    "yolo9c_seg":  _best("yolov9c_seg"),
    "yolo11m_seg": _best("yolo11m_seg"),
    "maskrcnn":    _best("maskrcnn"),
}

# YOLO11m digunakan sebagai prompt detector di hybrid
yolo11m_best_file = os.path.join(get_output_dir("yolo11m"), "weights", "best_path.txt")
if os.path.exists(yolo11m_best_file):
    with open(yolo11m_best_file) as _f:
        paths["yolo11m_det"] = _f.read().strip()

print("\n[Paths] Model yang akan digunakan:")
for k, v in paths.items():
    ok = "✅" if os.path.exists(v) else "❌"
    print(f"  {ok} {k:15s}: {v}")

# ==============================================================================
# 2. LOAD SAM2
# ==============================================================================
try:
    sam_model = SAM("sam2.1_b.pt")
    print("\n[Hybrid] SAM2 berhasil dimuat.")
except Exception as e:
    print(f"❌ SAM2 gagal dimuat: {e}")
    sys.exit(1)

# ==============================================================================
# 3. PILIH 5 GAMBAR SAMPEL RANDOM
# ==============================================================================
img_dir = os.path.join(SEG_DATASET_LOCATION, "test", "images")
if not os.path.isdir(img_dir):
    img_dir = os.path.join(SEG_DATASET_LOCATION, "valid", "images")

all_imgs    = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_imgs = random.sample(all_imgs, min(5, len(all_imgs)))
print(f"\n[Sampel] {len(sample_imgs)} gambar dari {img_dir}")

# ==============================================================================
# 4. HYBRID PIPELINE: YOLO11m → SAM2 → 5 visual hybrid
# ==============================================================================
visual_dir = os.path.join(WORKSPACE_DIR, "runs", "visuals", "hybrid")
os.makedirs(visual_dir, exist_ok=True)

yolo_det   = YOLO(paths["yolo11m_det"]) if os.path.exists(paths["yolo11m_det"]) else None
latency_rows = []

print("\n" + "="*65)
print("  Hybrid Pipeline: YOLO11m + SAM2")
print("="*65)

for idx, img_path in enumerate(sample_imgs, 1):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n[Hybrid] [{idx}/{len(sample_imgs)}] {img_path}")

    t0         = time.perf_counter()
    det_result = yolo_det.predict(img_path, conf=0.5, verbose=False,
                                  imgsz=IMAGE_SIZE) if yolo_det else None
    t_yolo     = (time.perf_counter() - t0) * 1000

    boxes      = (det_result[0].boxes.xyxy.cpu().numpy()
                  if det_result and det_result[0].boxes is not None
                  else np.zeros((0, 4)))

    if len(boxes) == 0:
        print("  ⚠️  Tidak ada deteksi — skip SAM2.")
        t_sam = 0.0; sam_result = None
    else:
        t1         = time.perf_counter()
        sam_result = sam_model.predict(det_result[0].orig_img,
                                       bboxes=boxes, verbose=False)
        t_sam      = (time.perf_counter() - t1) * 1000

    latency_rows.append({
        "Image":      img_name,
        "YOLO_ms":    round(t_yolo, 2),
        "SAM2_ms":    round(t_sam,  2),
        "Total_ms":   round(t_yolo + t_sam, 2),
        "FPS":        round(1000 / (t_yolo + t_sam), 2) if (t_yolo + t_sam) > 0 else "N/A",
        "Detections": len(boxes),
    })

    # --- Visualisasi single hybrid panel ---
    img_raw  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_yolo = cv2.cvtColor(det_result[0].plot(), cv2.COLOR_BGR2RGB) if det_result else img_raw
    img_hyb  = (cv2.cvtColor(sam_result[0].plot(masks=True, boxes=True), cv2.COLOR_BGR2RGB)
                if sam_result else img_raw)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(img_yolo)
    axes[0].set_title("YOLO11m Detection", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(img_hyb)
    axes[1].set_title("Hybrid (YOLO11m + SAM2)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle(f"Hybrid Pipeline — {img_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(visual_dir, f"hybrid_{idx:02d}_{img_name}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out_png} | YOLO={t_yolo:.1f}ms SAM2={t_sam:.1f}ms")
    gc.collect(); torch.cuda.empty_cache()

# ==============================================================================
# 5. EVALUASI PERBANDINGAN (det + seg via _benchmark_yolo)
# ==============================================================================
print("\n" + "="*65)
print("  Benchmark Semua Model Detection")
print("="*65)
det_data = {
    "YOLOv8m (Fine-tuned)":  _benchmark_yolo(paths["yolo8m_det"],  DET_YAML, "detect"),
    "YOLOv9m (Fine-tuned)":  _benchmark_yolo(paths["yolo9m_det"],  DET_YAML, "detect"),
    "YOLO11m (Fine-tuned)":  _benchmark_yolo(paths["yolo11m_det"], DET_YAML, "detect"),
}

print("\n" + "="*65)
print("  Benchmark Semua Model Segmentation")
print("="*65)
seg_data = {
    "YOLOv8m-Seg (Fine-tuned)":             _benchmark_yolo(paths["yolo8m_seg"],  SEG_YAML, "segment"),
    "YOLOv9c-Seg (Fine-tuned)":             _benchmark_yolo(paths["yolo9c_seg"],  SEG_YAML, "segment"),
    "YOLO11m-Seg (Fine-tuned)":             _benchmark_yolo(paths["yolo11m_seg"], SEG_YAML, "segment"),
    "Mask R-CNN ResNet-50 FPN (Fine-tuned)":{"model_size_mb": "N/A", "map50_95": "N/A",
                                             "map50_95_mask": "N/A",
                                             "latency_ms": "N/A", "fps": "N/A"},
}

# Hybrid latency rata-rata dari 5 sampel
_avg_hyb = (sum(r["Total_ms"] for r in latency_rows) / len(latency_rows)
            if latency_rows else 0)
_hyb_fps = round(1000 / _avg_hyb, 2) if _avg_hyb > 0 else "N/A"

seg_data["Hybrid (YOLO11m + SAM2)"] = {
    "model_size_mb": "N/A", "map50_95": "N/A", "map50_95_mask": "N/A",
    "latency_ms": round(_avg_hyb, 2) if _avg_hyb > 0 else "N/A", "fps": _hyb_fps,
}
det_data["Hybrid (YOLO11m + SAM2)"] = {
    "model_size_mb": "N/A", "map50_95": "N/A", "map50": "N/A",
    "precision": "N/A", "recall": "N/A",
    "preprocess_ms": "N/A", "inference_ms": "N/A", "postprocess_ms": "N/A",
    "fps": _hyb_fps, "latency_ms": round(_avg_hyb, 2) if _avg_hyb > 0 else "N/A",
}

# Load maskrcnn latency dari CSV jika sudah ada
_mrcnn_csv = os.path.join(WORKSPACE_DIR, "runs", "reports", "report_maskrcnn_seg.csv")
if os.path.exists(_mrcnn_csv):
    import csv as _csv
    with open(_mrcnn_csv, newline="", encoding="utf-8") as _f:
        _row = list(_csv.DictReader(_f))
    if _row:
        seg_data["Mask R-CNN ResNet-50 FPN (Fine-tuned)"]["latency_ms"] = _row[0].get("Latency(ms)", "N/A")
        seg_data["Mask R-CNN ResNet-50 FPN (Fine-tuned)"]["fps"]        = _row[0].get("FPS", "N/A")
        seg_data["Mask R-CNN ResNet-50 FPN (Fine-tuned)"]["model_size_mb"] = _row[0].get("Model Size (MB)", "N/A")

# ==============================================================================
# 6. CSV REPORTS
# ==============================================================================
report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)

# -- Detection comparison CSV
det_fields = ["Model", "Model Size (MB)", "mAP50-95", "mAP50",
              "Precision", "Recall",
              "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]
det_rows = []
for label, m in det_data.items():
    det_rows.append({
        "Model":            label,
        "Model Size (MB)":  m["model_size_mb"],
        "mAP50-95":         m["map50_95"],
        "mAP50":            m.get("map50", "N/A"),
        "Precision":        m.get("precision", "N/A"),
        "Recall":           m.get("recall", "N/A"),
        "Preprocess (ms)":  m.get("preprocess_ms", "N/A"),
        "Inference (ms)":   m.get("inference_ms", "N/A"),
        "Postprocess (ms)": m.get("postprocess_ms", "N/A"),
    })
det_csv = os.path.join(report_dir, "report_det_comparison.csv")
with open(det_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=det_fields); w.writeheader(); w.writerows(det_rows)
print(f"\n✅ Det Comparison : {det_csv}")

# -- Segmentation comparison CSV
seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS"]
seg_rows = []
for label, m in seg_data.items():
    seg_rows.append({
        "Model":           label,
        "Model Size (MB)": m["model_size_mb"],
        "mAP50-95(Box)":   m.get("map50_95", "N/A"),
        "mAP50-95(Mask)":  m.get("map50_95_mask", "N/A"),
        "Latency(ms)":     m.get("latency_ms", "N/A"),
        "FPS":             m.get("fps", "N/A"),
    })
seg_csv = os.path.join(report_dir, "report_seg_comparison.csv")
with open(seg_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=seg_fields); w.writeheader(); w.writerows(seg_rows)
print(f"✅ Seg Comparison : {seg_csv}")

# -- Hybrid latency CSV
hyb_csv = os.path.join(report_dir, "report_hybrid_latency.csv")
with open(hyb_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Image", "YOLO_ms", "SAM2_ms",
                                       "Total_ms", "FPS", "Detections"])
    w.writeheader(); w.writerows(latency_rows)
print(f"✅ Hybrid Latency : {hyb_csv}")
if _avg_hyb > 0:
    print(f"   Rata-rata: {_avg_hyb:.2f} ms ({_hyb_fps} FPS)")

# ==============================================================================
# 7. GRID COMPARISON 1×5: 5 gambar masing-masing 5 panel
# ==============================================================================
print("\n" + "="*65)
print("  Membuat Comparison Grid 1×5")
print("="*65)

# Load model untuk rendering comparison (hanya yang ada)
_CLASS_COLORS = [
    (255,  56,  56), (255, 157,  51), ( 50, 205,  50),
    ( 30, 144, 255), (238, 130, 238), (255, 215,   0),
    (  0, 206, 209), (255,  99,  71),
]

def _render_yolo(model_path, img_path, fallback):
    if not model_path or not os.path.exists(model_path):
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
    try:
        m = YOLO(model_path)
        r = m.predict(img_path, conf=0.5, verbose=False, imgsz=IMAGE_SIZE)
        del m; gc.collect(); torch.cuda.empty_cache()
        return cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  ⚠️  render YOLO gagal: {e}")
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)


def _render_maskrcnn(best_pt, img_path, fallback):
    if not best_pt or not os.path.exists(best_pt):
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
    try:
        from config_shared import NUM_CLASSES
        from models.maskrcnn_builder import build_mask_rcnn as _build
        from PIL import Image as _PILImg
        import torchvision.transforms.functional as _TF
        _mm = _build(num_classes=NUM_CLASSES+1, use_parallel=False, device=DEVICE)
        _mm.load_state_dict(torch.load(best_pt, map_location=DEVICE))
        _mm.eval()
        _pil  = _PILImg.open(img_path).convert("RGB")
        _t    = _TF.to_tensor(_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = _mm([_t[0]])[0]
        canvas = cv2.cvtColor(np.array(_pil), cv2.COLOR_RGB2BGR)
        scores = preds["scores"].cpu().numpy()
        masks  = preds["masks"].cpu().numpy()
        boxes  = preds["boxes"].cpu().numpy().astype(int)
        overlay = canvas.copy()
        for i, (sc, mk, bx) in enumerate(zip(scores, masks, boxes)):
            if sc < 0.5: continue
            color = _CLASS_COLORS[i % len(_CLASS_COLORS)]
            m = (mk[0] > 0.5).astype(np.uint8)
            col_m = np.zeros_like(overlay); col_m[m==1] = color
            cv2.addWeighted(col_m, 0.45, overlay, 1.0, 0, overlay)
            cv2.rectangle(overlay, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
        del _mm; gc.collect(); torch.cuda.empty_cache()
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  ⚠️  render MaskRCNN gagal: {e}")
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)


def _render_hybrid_panel(yolo_path, sam_mdl, img_path, fallback):
    if not yolo_path or not os.path.exists(yolo_path) or sam_mdl is None:
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
    try:
        _yd = YOLO(yolo_path)
        _dr = _yd.predict(img_path, conf=0.5, verbose=False, imgsz=IMAGE_SIZE)
        _bx = (_dr[0].boxes.xyxy.cpu().numpy()
               if _dr[0].boxes is not None else np.zeros((0, 4)))
        if len(_bx) == 0:
            del _yd; return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
        _sr = sam_mdl.predict(_dr[0].orig_img, bboxes=_bx, verbose=False)
        canvas  = _dr[0].orig_img.copy()
        overlay = canvas.copy()
        _masks  = (_sr[0].masks.data.cpu().numpy()
                   if _sr and _sr[0].masks is not None else [])
        _cls    = _dr[0].boxes.cls.cpu().numpy().astype(int)
        _confs  = _dr[0].boxes.conf.cpu().numpy()
        H, W    = canvas.shape[:2]
        for i in range(len(_cls)):
            color = _CLASS_COLORS[_cls[i] % len(_CLASS_COLORS)]
            if i < len(_masks):
                mk = _masks[i]
                if mk.shape != (H, W):
                    mk = cv2.resize(mk.astype(np.float32), (W, H),
                                    interpolation=cv2.INTER_NEAREST)
                bn = (mk > 0.5).astype(np.uint8)
                cm = np.zeros_like(overlay); cm[bn==1] = color
                cv2.addWeighted(cm, 0.45, overlay, 1.0, 0, overlay)
            x1,y1,x2,y2 = map(int, _bx[i])
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        del _yd; gc.collect(); torch.cuda.empty_cache()
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  ⚠️  render Hybrid gagal: {e}")
        return cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)


# Panel titles
PANELS = [
    ("YOLOv8m\n(Detection)",             "yolo8m_det"),
    ("YOLOv9m\n(Detection)",             "yolo9m_det"),
    ("YOLO11m\n(Detection)",             "yolo11m_det"),
    ("Mask R-CNN\n(Segmentation)",       "maskrcnn"),
    ("Hybrid\n(YOLO11m + SAM2)",         "hybrid"),
]

for idx, img_path in enumerate(sample_imgs, 1):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    fallback = cv2.imread(img_path)
    print(f"\n[Grid] [{idx}/{len(sample_imgs)}] {img_name}")

    panels_img = [
        _render_yolo(paths["yolo8m_det"],  img_path, fallback),
        _render_yolo(paths["yolo9m_det"],  img_path, fallback),
        _render_yolo(paths["yolo11m_det"], img_path, fallback),
        _render_maskrcnn(paths["maskrcnn"],img_path, fallback),
        _render_hybrid_panel(paths["yolo11m_det"], sam_model, img_path, fallback),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 7))
    fig.suptitle(f"Perbandingan 5 Model — {img_name}",
                 fontsize=14, fontweight="bold")
    for ax, (img, (title, _)) in zip(axes, zip(panels_img, PANELS)):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
    plt.tight_layout()
    out_png = os.path.join(visual_dir, f"comparison_{idx:02d}_{img_name}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out_png}")
    gc.collect(); torch.cuda.empty_cache()

del yolo_det, sam_model
gc.collect(); torch.cuda.empty_cache()

# ==============================================================================
# 8. KOMPRES & RINGKASAN
# ==============================================================================
compress_visuals()

print("\n✅ Hybrid evaluation selesai.")
print("\n" + "="*65)
print("  SEMUA PIPELINE SELESAI")
print(f"  Workspace  : {WORKSPACE_DIR}")
print(f"  Det CSV    : {det_csv}")
print(f"  Seg CSV    : {seg_csv}")
print(f"  Hybrid CSV : {hyb_csv}")
print(f"  Visuals    : {visual_dir}/ ({len(sample_imgs)*2} gambar)")
print(f"               {len(sample_imgs)} hybrid + {len(sample_imgs)} comparison grid")
print(f"  Download   : {WORKSPACE_DIR}/runs/visuals.tar.gz")
print("="*65)
send_telegram_msg(f"🏁 <b>ALL PIPELINES FINISHED!</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>\nAll models trained and evaluated.")
