# -*- coding: utf-8 -*-
"""
yolo/yolo11/main.py
===================
Fine-tuning YOLO11 Medium pada dataset botol plastik (RVM).

Model:
  - yolo11m.pt      → Detection  (Primary — digunakan sebagai Prompt di Hybrid)
  - yolo11m-seg.pt  → Instance Segmentation

Cara menjalankan:
    python -u main.py 2>&1 | tee yolo11_train.log
"""

import os, sys, csv, gc
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# GPU Fan Manager
try:
    from gpu_fan_manager import start_fan_manager
    start_fan_manager()
except ImportError:
    print("[Warning] gpu_fan_manager.py not found in ROOT.")

from config_shared import (
    WORKSPACE_DIR, DET_YAML, SEG_YAML, MODELS_DIR,
    EPOCHS, IMAGE_SIZE, YOLO_BATCH_SIZE, get_output_dir, compress_run,
    save_yolo_visual_samples, parse_device, download_and_move_model, REPORTS_DIR
)
from telegram_utils import get_yolo_callbacks, send_telegram_msg
import argparse
import torch
from ultralytics import YOLO, settings

# Arahkan semua download model Ultralytics ke MODELS_DIR
settings.update({'weights_dir': MODELS_DIR})

parser = argparse.ArgumentParser(description="YOLO11m Fine-tuning")
parser.add_argument("--device", type=str, default=None,
    help="GPU: '0', '1,2', '0,1,2', 'cpu'. Default: semua GPU.")
args = parser.parse_args()

if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    n = torch.cuda.device_count()
    DEVICE = list(range(n)) if n > 1 else (0 if n == 1 else "cpu")

print(f"[Device] YOLO11m → {DEVICE}")


def get_gpu_report_str(device):
    if device == "cpu":
        return "1x CPU"
    from collections import Counter
    ids = [device] if isinstance(device, int) else device
    gpu_names = [torch.cuda.get_device_name(i) for i in ids]
    counts = Counter(gpu_names)
    return ", ".join([f"{count}x {name}" for name, count in counts.items()])


def _flush(label):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"[MemFlush] {label} — VRAM bebas: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")


def _train(model_pt, yaml_path, run_name, label):
    out_dir = get_output_dir(run_name)
    best_pt = os.path.join(out_dir, "weights", "best.pt")
    last_pt = os.path.join(out_dir, "weights", "last.pt")

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
        # Tambahkan Telegram Callbacks
        for k, v in get_yolo_callbacks(label).items():
            model.add_callback(k, v)

        model.train(data=yaml_path, epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=YOLO_BATCH_SIZE,
                    project=os.path.dirname(out_dir), name=os.path.basename(out_dir),
                    exist_ok=True, device=DEVICE)

    result = str(model.trainer.best)
    del model; _flush(label)
    return result


def _eval_det(label, pt, yaml):
    """Detection — format reporter.py det_performance."""
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=0, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        box  = met.box
        del m; _flush(f"eval {label}")
        return {
            "Model":            label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95":        round(float(box.map),   4),
            "mAP50":           round(float(box.map50), 4),
            "Precision":       round(float(box.mp),    4),
            "Recall":          round(float(box.mr),    4),
            "Preprocess (ms)": pre,
            "Inference (ms)":  inf,
            "Postprocess (ms)":post,
        }
    except Exception as e:
        print(f"  ⚠️ {label}: {e}")
        return {"Model": label, **{k: "ERR" for k in [
            "Model Size (MB)", "mAP50-95", "mAP50", "Precision", "Recall",
            "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]}}


def _eval_seg(label, pt, yaml):
    """Segmentation — format reporter.py seg_performance."""
    try:
        m   = YOLO(pt)
        met = m.val(data=yaml, imgsz=IMAGE_SIZE, device=0, verbose=False, plots=False)
        sp  = getattr(met, "speed", {})
        pre  = round(sp.get("preprocess",  0), 2)
        inf  = round(sp.get("inference",   0), 2)
        post = round(sp.get("postprocess", 0), 2)
        total_ms = round(pre + inf + post, 2)
        fps = round(1000 / total_ms, 2) if total_ms > 0 else "N/A"
        try:
            box_map  = round(float(met.box.map), 4)
            mask_map = round(float(met.seg.map), 4)
        except Exception:
            box_map = mask_map = "N/A"
        del m; _flush(f"eval {label}")
        return {
            "Model":           label,
            "Model Size (MB)": round(os.path.getsize(pt)/1e6, 2),
            "mAP50-95(Box)":   box_map,
            "mAP50-95(Mask)":  mask_map,
            "Latency(ms)":     total_ms,
            "FPS":             fps,
            "GPUs":            get_gpu_report_str(DEVICE),
        }
    except Exception as e:
        print(f"  ⚠️ {label}: {e}")
        return {"Model": label, **{k: "ERR" for k in [
            "Model Size (MB)", "mAP50-95(Box)", "mAP50-95(Mask)",
            "Latency(ms)", "FPS"]}}


print("\n" + "="*65 + "\n  YOLO11m Fine-tuning\n" + "="*65)

# Download & Pindahkan model dasar terlebih dahulu
det_model_path = download_and_move_model("yolo11m.pt")
seg_model_path = download_and_move_model("yolo11m-seg.pt")

# Detection — best.pt ini yang akan digunakan oleh hybrid/main.py sebagai prompt
best_det = _train(det_model_path,     DET_YAML, "yolo11m",    "YOLO11m Detection (Primary)")
best_seg = _train(seg_model_path, SEG_YAML, "yolo11m_seg","YOLO11m-Seg Segmentation")

# Simpan path best_det ke file agar hybrid/main.py bisa membacanya
det_path_file = os.path.join(get_output_dir("yolo11m"), "weights", "best_path.txt")
with open(det_path_file, "w") as f:
    f.write(best_det)
print(f"[Info] Path YOLO11m best.pt disimpan ke: {det_path_file}")

report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)

# Detection CSV
det_row    = _eval_det("YOLO11m (Fine-tuned)", best_det, DET_YAML)
det_fields = ["Model", "Model Size (MB)", "mAP50-95", "mAP50",
              "Precision", "Recall",
              "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"]
det_csv = os.path.join(report_dir, "report_yolo11m_det.csv")
with open(det_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=det_fields); w.writeheader(); w.writerow(det_row)
print(f"\n✅ Det Report : {det_csv}")

# Segmentation CSV
seg_row    = _eval_seg("YOLO11m-Seg (Fine-tuned)", best_seg, SEG_YAML)
seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS", "GPUs"]
seg_csv = os.path.join(report_dir, "report_yolo11m_seg.csv")
with open(seg_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=seg_fields); w.writeheader(); w.writerow(seg_row)
print(f"✅ Seg Report : {seg_csv}")

_img_dir = os.path.join(os.path.dirname(DET_YAML), "test", "images")
if not os.path.isdir(_img_dir):
    _img_dir = os.path.join(os.path.dirname(DET_YAML), "valid", "images")
_seg_img_dir = _img_dir.replace(os.path.dirname(DET_YAML), os.path.dirname(SEG_YAML))
if not os.path.isdir(_seg_img_dir):
    _seg_img_dir = _img_dir  # fallback ke det dir

save_yolo_visual_samples(best_det, "yolo11m",     _img_dir)
save_yolo_visual_samples(best_seg, "yolo11m_seg", _seg_img_dir)

compress_run("yolo11m")
compress_run("yolo11m_seg")

print("\n✅ YOLO11m selesai.")
send_telegram_msg(f"✅ <b>YOLO11m Pipeline Finished</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>")
print(f"\n[Next] Lanjutkan ke: cd ../../mask-r-cnn && python -u main.py")
