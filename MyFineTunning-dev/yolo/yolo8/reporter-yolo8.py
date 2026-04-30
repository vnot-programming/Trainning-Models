# -*- coding: utf-8 -*-
"""
yolo/yolo8/reporter-yolo8.py
=============================
Standalone evaluasi & laporan untuk YOLOv8m (Detection + Segmentation).
Format CSV identik dengan MyTrainEngine/evaluation/reporter.py.

Output:
  - runs/reports/report_yolov8m_det.csv   → format det_performance
  - runs/reports/report_yolov8m_seg.csv   → format seg_performance

Workspace : /workspace/MyFineTunning-20260423_030331
best.pt   : runs/yolov8m/weights/best.pt
            runs/yolov8m_seg/weights/best.pt

Cara menjalankan:
    python -u reporter-yolo8.py 2>&1 | tee yolov8_reporter.log
"""

import os, sys, csv, gc

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Paksa workspace ke sesi yang benar ───────────────────────────────────────
TARGET_WORKSPACE = "/workspace/MyFineTunning-20260423_030331"
os.environ["WORKSPACE_BASE"] = "/workspace"

_FINETUNING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WS_ID_FILE = os.path.join(_FINETUNING_ROOT, ".workspace_id")
with open(_WS_ID_FILE, "w") as _f:
    _f.write("20260423_030331")

ROOT = _FINETUNING_ROOT
sys.path.insert(0, ROOT)

from config_shared import (WORKSPACE_DIR, DET_YAML, SEG_YAML, IMAGE_SIZE,
                           get_output_dir, measure_yolo_complexity)
import argparse, torch
from ultralytics import YOLO

assert WORKSPACE_DIR == TARGET_WORKSPACE, (
    f"WORKSPACE_DIR mismatch!\n  expected: {TARGET_WORKSPACE}\n  got: {WORKSPACE_DIR}"
)
print(f"[Workspace] ✅ {WORKSPACE_DIR}")

parser = argparse.ArgumentParser(description="YOLOv8m Reporter")
parser.add_argument("--device", type=str, default="0",
    help="GPU device untuk val(). Default: '0'")
args   = parser.parse_args()
DEVICE = args.device


# ── Helper identik reporter.py ────────────────────────────────────────────────
def _flush(label: str):
    gc.collect(); torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[MemFlush] {label} — VRAM bebas: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")


def _benchmark_yolo(model_path: str, yaml_path: str, task: str = "detect") -> dict:
    """Identik dengan reporter.py _benchmark_yolo() — menangkap AttributeError."""
    empty = {
        "model_size_mb":  "N/A", "map50_95":      "N/A",
        "map50":          "N/A", "map50_95_mask": "N/A",
        "precision":      "N/A", "recall":        "N/A",
        "preprocess_ms":  "N/A", "inference_ms":  "N/A",
        "postprocess_ms": "N/A", "fps":           "N/A",
        "latency_ms":     "N/A",
    }
    try:
        model   = YOLO(model_path)
        metrics = model.val(
            data=yaml_path, imgsz=IMAGE_SIZE,
            device=DEVICE, verbose=False, plots=False,
        )
        size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else "N/A"
        speed   = metrics.speed if hasattr(metrics, "speed") else {}
        pre     = round(speed.get("preprocess",  0), 2)
        inf     = round(speed.get("inference",   0), 2)
        post    = round(speed.get("postprocess", 0), 2)
        total   = pre + inf + post
        fps     = round(1000 / total, 2) if total > 0 else "N/A"

        del model
        _flush(f"benchmark {task}")

        if task == "segment":
            try:
                box_map5095  = round(float(metrics.box.map),   4)
                box_map50    = round(float(metrics.box.map50), 4)
                mask_map5095 = round(float(metrics.seg.map),   4)
            except Exception:
                box_map5095 = box_map50 = mask_map5095 = "N/A"
            return {
                "model_size_mb":  round(size_mb, 2),
                "map50_95":       box_map5095,
                "map50":          box_map50,
                "map50_95_mask":  mask_map5095,
                "precision":      "N/A",
                "recall":         "N/A",
                "preprocess_ms":  pre,
                "inference_ms":   inf,
                "postprocess_ms": post,
                "fps":            fps,
                "latency_ms":     round(total, 2),
            }
        else:
            try:
                map5095   = round(float(metrics.box.map),   4)
                map50     = round(float(metrics.box.map50), 4)
                precision = round(float(metrics.box.mp),    4)
                recall    = round(float(metrics.box.mr),    4)
            except Exception:
                map5095 = map50 = precision = recall = "N/A"
            return {
                "model_size_mb":  round(size_mb, 2),
                "map50_95":       map5095,
                "map50":          map50,
                "map50_95_mask":  "N/A",
                "precision":      precision,
                "recall":         recall,
                "preprocess_ms":  pre,
                "inference_ms":   inf,
                "postprocess_ms": post,
                "fps":            fps,
                "latency_ms":     round(total, 2),
            }
    except Exception as e:
        print(f"  ⚠️  Benchmark gagal untuk {model_path}: {e}")
        return empty


# ── Lokasi best.pt ────────────────────────────────────────────────────────────
best_det = os.path.join(get_output_dir("yolov8m"),     "weights", "best.pt")
best_seg = os.path.join(get_output_dir("yolov8m_seg"), "weights", "best.pt")

print(f"\n[Model] Detection  : {best_det}")
print(f"[Model] Segmentation: {best_seg}")

for name, path in [("Detection", best_det), ("Segmentation", best_seg)]:
    if not os.path.exists(path):
        print(f"  ⚠️  {name} best.pt tidak ditemukan: {path}")

# ── Report dir ────────────────────────────────────────────────────────────────
report_dir = os.path.join(WORKSPACE_DIR, "runs", "reports")
os.makedirs(report_dir, exist_ok=True)

# ── Detection CSV (format identik report_det_performance.csv) ─────────────────
print("\n" + "="*60)
print("  [1/2] Benchmark Detection — YOLOv8m")
print("="*60)
if os.path.exists(best_det):
    m_det = _benchmark_yolo(best_det, DET_YAML, task="detect")
    det_row = {
        "Model":            "YOLOv8m (Fine-tuned)",
        "Model Size (MB)":  m_det["model_size_mb"],
        "mAP50-95":         m_det["map50_95"],
        "mAP50":            m_det["map50"],
        "Precision":        m_det["precision"],
        "Recall":           m_det["recall"],
        "Preprocess (ms)":  m_det["preprocess_ms"],
        "Inference (ms)":   m_det["inference_ms"],
        "Postprocess (ms)": m_det["postprocess_ms"],
    }
else:
    det_row = {"Model": "YOLOv8m (Fine-tuned)",
               **{k: "N/A" for k in ["Model Size (MB)", "mAP50-95", "mAP50",
                                      "Precision", "Recall",
                                      "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]}}

det_fields = ["Model", "Model Size (MB)", "mAP50-95", "mAP50",
              "Precision", "Recall",
              "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]
det_csv = os.path.join(report_dir, "report_yolov8m_det.csv")
with open(det_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=det_fields); w.writeheader(); w.writerow(det_row)
print(f"\n✅ Det CSV : {det_csv}")
print(f"   mAP50-95={det_row['mAP50-95']}  mAP50={det_row['mAP50']}  "
      f"Prec={det_row['Precision']}  Rec={det_row['Recall']}")

# ── Segmentation CSV (format identik report_seg_performance.csv) ──────────────
print("\n" + "="*60)
print("  [2/2] Benchmark Segmentation — YOLOv8m-Seg")
print("="*60)
if os.path.exists(best_seg):
    m_seg = _benchmark_yolo(best_seg, SEG_YAML, task="segment")
    seg_row = {
        "Model":           "YOLOv8m-Seg (Fine-tuned)",
        "Model Size (MB)": m_seg["model_size_mb"],
        "mAP50-95(Box)":   m_seg["map50_95"],
        "mAP50-95(Mask)":  m_seg["map50_95_mask"],
        "Latency(ms)":     m_seg["latency_ms"],
        "FPS":             m_seg["fps"],
    }
else:
    seg_row = {"Model": "YOLOv8m-Seg (Fine-tuned)",
               **{k: "N/A" for k in ["Model Size (MB)", "mAP50-95(Box)",
                                      "mAP50-95(Mask)", "Latency(ms)", "FPS"]}}

seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS"]
seg_csv = os.path.join(report_dir, "report_yolov8m_seg.csv")
with open(seg_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=seg_fields); w.writeheader(); w.writerow(seg_row)
print(f"\n✅ Seg CSV : {seg_csv}")
print(f"   mAP50-95(Box)={seg_row['mAP50-95(Box)']}  "
      f"mAP50-95(Mask)={seg_row['mAP50-95(Mask)']}  "
      f"Latency={seg_row['Latency(ms)']}ms  FPS={seg_row['FPS']}")

print(f"\n📊 Semua laporan tersimpan di: {report_dir}")

# ── Complexity CSV (Parameters, GFLOPs, Peak VRAM) ───────────────────────────
print("\n" + "="*60)
print("  [3/3] Complexity Measurement — YOLOv8m")
print("="*60)

cplx_rows = []
for model_label, pt_path, arch_type in [
    ("YOLOv8m (Fine-tuned)",     best_det, "Object Detection"),
    ("YOLOv8m-Seg (Fine-tuned)", best_seg, "End-to-End Segmentation"),
]:
    print(f"  ▶ {model_label}")
    c = measure_yolo_complexity(pt_path) if os.path.exists(pt_path) else \
        {"params_m": "N/A", "gflops": "N/A", "vram_gb": "N/A"}
    cplx_rows.append({
        "Model":            model_label,
        "Architecture Type":arch_type,
        "Parameters (M)":   c["params_m"],
        "GFLOPs":           c["gflops"],
        "Max VRAM (GB)":    c["vram_gb"],
        "Notes":            "",
    })
    print(f"     Params={c['params_m']} M  GFLOPs={c['gflops']}  VRAM={c['vram_gb']} GB")

cplx_fields = ["Model", "Architecture Type", "Parameters (M)",
               "GFLOPs", "Max VRAM (GB)", "Notes"]
cplx_csv = os.path.join(report_dir, "report_yolov8m_complexity.csv")
with open(cplx_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cplx_fields); w.writeheader(); w.writerows(cplx_rows)
print(f"\n✅ Complexity CSV : {cplx_csv}")
