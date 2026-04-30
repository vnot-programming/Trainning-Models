# -*- coding: utf-8 -*-
"""
yolo/yolo9/seg_only.py
======================
Fine-tuning YOLOv9c-Seg (Segmentation ONLY) — melanjutkan workspace
yang sudah ada dari sesi sebelumnya.

Workspace target : /workspace/MyFineTunning-20260423_104653
Model            : yolov9c-seg.pt  (Ultralytics assets v8.4.0)

Cara menjalankan:
    python -u seg_only.py 2>&1 | tee yolov9_seg_only.log
    python -u seg_only.py --device 0 2>&1 | tee yolov9_seg_only.log
"""

import os, sys, csv, gc

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Paksa workspace ke direktori session sebelumnya ───────────────────────────
TARGET_WORKSPACE = "/workspace/MyFineTunning-20260423_104653"
os.environ["WORKSPACE_BASE"] = "/workspace"

# Tulis ulang .workspace_id agar config_shared membaca timestamp yang benar
_FINETUNING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WS_ID_FILE = os.path.join(_FINETUNING_ROOT, ".workspace_id")
with open(_WS_ID_FILE, "w") as _f:
    _f.write("20260423_104653")

# ── Import setelah .workspace_id di-patch ────────────────────────────────────
ROOT = _FINETUNING_ROOT
sys.path.insert(0, ROOT)

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML,
    EPOCHS, IMAGE_SIZE, YOLO_BATCH_SIZE,
    get_output_dir, compress_run,
    save_yolo_visual_samples, parse_device,
)
import argparse
import torch
from ultralytics import YOLO

# Sanity check — pastikan workspace benar
assert WORKSPACE_DIR == TARGET_WORKSPACE, (
    f"WORKSPACE_DIR mismatch!\n  expected: {TARGET_WORKSPACE}\n  got: {WORKSPACE_DIR}"
)
print(f"[Workspace] ✅ {WORKSPACE_DIR}")

# ── Device ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="YOLOv9c-Seg Segmentation Only")
parser.add_argument("--device", type=str, default=None,
    help="GPU: '0', '1,2', '0,1,2', 'cpu'. Default: semua GPU.")
args = parser.parse_args()

if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    n = torch.cuda.device_count()
    DEVICE = list(range(n)) if n > 1 else (0 if n == 1 else "cpu")

print(f"[Device] YOLOv9c-Seg → {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _flush(label):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[MemFlush] {label} — VRAM bebas: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")


def _train(model_pt, yaml_path, run_name, label):
    out_dir  = get_output_dir(run_name)
    best_pt  = os.path.join(out_dir, "weights", "best.pt")
    last_pt  = os.path.join(out_dir, "weights", "last.pt")

    if os.path.exists(best_pt):
        print(f"\n[SKIP] {label}: training sudah selesai.\n  best.pt: {best_pt}")
        return best_pt

    if os.path.exists(last_pt):
        print(f"\n[RESUME] {label}: melanjutkan dari last.pt\n  {last_pt}")
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        model = YOLO(model_pt)
        model.train(
            data     = yaml_path,
            epochs   = EPOCHS,
            imgsz    = IMAGE_SIZE,
            batch    = YOLO_BATCH_SIZE,
            project  = os.path.dirname(out_dir),
            name     = os.path.basename(out_dir),
            exist_ok = True,
            device   = DEVICE,
        )

    result = str(model.trainer.best)
    del model
    _flush(label)
    return result


def _eval(label, pt, yaml, task):
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=0, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre, inf, post = (round(sp.get(k, 0), 2) for k in ("preprocess", "inference", "postprocess"))
        tot = pre + inf + post
        box = met.box
        del m
        _flush(f"eval {label}")
        return {
            "Model"     : label,
            "Task"      : task,
            "Size(MB)"  : round(os.path.getsize(pt) / 1e6, 2),
            "mAP50-95"  : round(float(box.map),  4),
            "mAP50"     : round(float(box.map50), 4),
            "Precision" : round(float(box.mp), 4) if task == "detect"  else "N/A",
            "Recall"    : round(float(box.mr), 4) if task == "detect"  else "N/A",
            "Pre(ms)"   : pre, "Inf(ms)": inf, "Post(ms)": post,
            "FPS"       : round(1000 / tot, 2) if tot > 0 else "N/A",
        }
    except Exception as e:
        print(f"  ⚠️  {label}: {e}")
        return {"Model": label, "Task": task,
                **{k: "ERR" for k in ["Size(MB)", "mAP50-95", "mAP50",
                                      "Precision", "Recall",
                                      "Pre(ms)", "Inf(ms)", "Post(ms)", "FPS"]}}


# ── Main ──────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  YOLOv9c-Seg — Segmentation Fine-tuning")
print("="*65)

# Path model weight — cari di direktori script ini dulu
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_PT = os.path.join(_SCRIPT_DIR, "yolov9c-seg.pt")
if not os.path.exists(SEG_PT):
    raise FileNotFoundError(
        f"yolov9c-seg.pt tidak ditemukan di {_SCRIPT_DIR}\n"
        "Download: wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c-seg.pt"
    )

best_seg = _train(SEG_PT, SEG_YAML, "yolov9c_seg", "YOLOv9c-Seg Segmentation")

# ── Evaluasi ──────────────────────────────────────────────────────────────────
rows = [_eval("YOLOv9c-Seg (Fine-tuned)", best_seg, SEG_YAML, "segment")]

report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)
csv_path = os.path.join(report_dir, "report_yolov9c_seg.csv")
fields = ["Model", "Task", "Size(MB)", "mAP50-95", "mAP50",
          "Precision", "Recall", "Pre(ms)", "Inf(ms)", "Post(ms)", "FPS"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\n✅ Report: {csv_path}")

# ── Visual Samples ────────────────────────────────────────────────────────────
_img_dir = os.path.join(os.path.dirname(SEG_YAML), "test", "images")
if not os.path.isdir(_img_dir):
    _img_dir = os.path.join(os.path.dirname(SEG_YAML), "valid", "images")

save_yolo_visual_samples(best_seg, "yolov9c_seg", _img_dir)

# ── Compress ──────────────────────────────────────────────────────────────────
compress_run("yolov9c_seg")

print("\n✅ YOLOv9c-Seg selesai.")
