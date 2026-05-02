# -*- coding: utf-8 -*-
"""
yolo/yolo9/det_eval_only.py
============================
Evaluasi + Visual Samples untuk YOLOv9m Detection ONLY.
Menggunakan best.pt yang sudah ada dari sesi training sebelumnya.

Workspace target : /workspace/MyFineTunning-20260423_104653
Model            : runs/yolov9m/weights/best.pt

Cara menjalankan:
    python -u det_eval_only.py 2>&1 | tee yolov9_det_eval.log
"""

import os, sys, csv, gc

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Paksa workspace ke direktori session sebelumnya ───────────────────────────
TARGET_WORKSPACE = "/workspace/MyFineTunning-20260423_104653"
os.environ["WORKSPACE_BASE"] = "/workspace"

_FINETUNING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WS_ID_FILE = os.path.join(_FINETUNING_ROOT, ".workspace_id")
with open(_WS_ID_FILE, "w") as _f:
    _f.write("20260423_104653")

# ── Import setelah .workspace_id di-patch ────────────────────────────────────
ROOT = _FINETUNING_ROOT
sys.path.insert(0, ROOT)

from config_shared import (
    WORKSPACE_DIR, DET_YAML,
    IMAGE_SIZE, get_output_dir,
    save_yolo_visual_samples, parse_device,
)
import argparse
import torch
from ultralytics import YOLO

# Sanity check
assert WORKSPACE_DIR == TARGET_WORKSPACE, (
    f"WORKSPACE_DIR mismatch!\n  expected: {TARGET_WORKSPACE}\n  got: {WORKSPACE_DIR}"
)
print(f"[Workspace] ✅ {WORKSPACE_DIR}")

# ── Device ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="YOLOv9m Detection — Eval + Visual")
parser.add_argument("--device", type=str, default=None,
    help="GPU: '0', '1,2', 'cpu'. Default: GPU 0 untuk eval.")
args = parser.parse_args()

if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    DEVICE = 0 if torch.cuda.is_available() else "cpu"

print(f"[Device] YOLOv9m Eval → {DEVICE}")


# ── Helper ────────────────────────────────────────────────────────────────────
def _flush(label: str):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"[MemFlush] {label} — VRAM bebas: {free_gb:.2f} GB")


def _eval(label: str, pt: str, yaml: str, task: str) -> dict:
    """Jalankan validasi YOLO dan kembalikan dict metrik."""
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=DEVICE,
                    verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre, inf, post = (round(sp.get(k, 0), 2)
                          for k in ("preprocess", "inference", "postprocess"))
        tot = pre + inf + post
        box = met.box
        del m
        _flush(f"eval {label}")
        return {
            "Model"    : label,
            "Task"     : task,
            "Size(MB)" : round(os.path.getsize(pt) / 1e6, 2),
            "mAP50-95" : round(float(box.map),  4),
            "mAP50"    : round(float(box.map50), 4),
            "Precision": round(float(box.mp), 4) if task == "detect" else "N/A",
            "Recall"   : round(float(box.mr), 4) if task == "detect" else "N/A",
            "Pre(ms)"  : pre, "Inf(ms)": inf, "Post(ms)": post,
            "FPS"      : round(1000 / tot, 2) if tot > 0 else "N/A",
        }
    except Exception as e:
        print(f"  ⚠️  {label}: {e}")
        return {"Model": label, "Task": task,
                **{k: "ERR" for k in ["Size(MB)", "mAP50-95", "mAP50",
                                      "Precision", "Recall",
                                      "Pre(ms)", "Inf(ms)", "Post(ms)", "FPS"]}}


# ── Cari best.pt detection ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  YOLOv9m Detection — Evaluasi & Visual Samples")
print("="*65)

best_det = os.path.join(get_output_dir("yolov9m"), "weights", "best.pt")
if not os.path.exists(best_det):
    raise FileNotFoundError(
        f"best.pt tidak ditemukan: {best_det}\n"
        "Pastikan training detection sudah selesai terlebih dahulu."
    )
print(f"[Model] ✅ {best_det}  ({round(os.path.getsize(best_det)/1e6, 1)} MB)")

# ── Evaluasi ──────────────────────────────────────────────────────────────────
print("\n[Eval] Menjalankan validasi YOLOv9m Detection ...")
rows = [_eval("YOLOv9m (Fine-tuned)", best_det, DET_YAML, "detect")]

report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)
csv_path = os.path.join(report_dir, "report_yolov9m_det.csv")
fields = ["Model", "Task", "Size(MB)", "mAP50-95", "mAP50",
          "Precision", "Recall", "Pre(ms)", "Inf(ms)", "Post(ms)", "FPS"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\n✅ Report: {csv_path}")
for r in rows:
    print(f"   mAP50={r['mAP50']}  mAP50-95={r['mAP50-95']}  "
          f"Prec={r['Precision']}  Rec={r['Recall']}  FPS={r['FPS']}")

# ── Visual Samples ────────────────────────────────────────────────────────────
print("\n[Visual] Membuat visual samples ...")
_img_dir = os.path.join(os.path.dirname(DET_YAML), "test", "images")
if not os.path.isdir(_img_dir):
    _img_dir = os.path.join(os.path.dirname(DET_YAML), "valid", "images")

save_yolo_visual_samples(best_det, "yolov9m", _img_dir)

print("\n✅ YOLOv9m Detection — Eval & Visual selesai.")
print(f"   Hasil di: {WORKSPACE_DIR}/runs/")
