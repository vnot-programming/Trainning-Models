# -*- coding: utf-8 -*-
"""
yolo/yolov10/main.py
====================
Fine-tuning YOLOv10 pada dataset botol plastik (RVM).

Model:
  - yolov10m.pt      → Detection (Medium)
  - yolov10x.pt      → Detection (Extra Large)

Catatan:
  - YOLOv10 tidak memiliki varian segmentasi resmi di Ultralytics, 
    sehingga tugas segmentasi dilewati secara penuh.

Cara menjalankan:
    python -u main.py 2>&1 | tee yolov10_train.log
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
    WORKSPACE_DIR, DET_YAML, MODELS_DIR,
    EPOCHS, IMAGE_SIZE, YOLO_BATCH_SIZE, get_output_dir, compress_run,
    save_yolo_visual_samples, parse_device, download_and_move_model, REPORTS_DIR,
    EARLY_STOPPING_PATIENCE, NUM_WORKERS
)
from telegram_utils import get_yolo_callbacks, send_telegram_msg
import argparse
import torch
from ultralytics import YOLO, settings

# Arahkan semua download model Ultralytics ke MODELS_DIR
settings.update({'weights_dir': MODELS_DIR})

# Import COCO eval utils
sys.path.insert(0, os.path.join(ROOT, ".."))  # Agar bisa import coco_eval_utils
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, 
    save_detection_report_csv, check_pycocotools
)

parser = argparse.ArgumentParser(description="YOLOv10 Fine-tuning")
parser.add_argument("--device", type=str, default=None,
    help="GPU: '0', '1,2', '0,1,2', 'cpu'. Default: semua GPU.")
parser.add_argument("--skip-eval", action="store_true",
    help="Skip automatic evaluation after training")
args = parser.parse_args()

if args.device is not None:
    DEVICE = parse_device(args.device)
else:
    n = torch.cuda.device_count()
    DEVICE = list(range(n)) if n > 1 else (0 if n == 1 else "cpu")

print(f"[Device] YOLOv10 → {DEVICE}")


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
    done_flag = os.path.join(out_dir, "training_done.flag")

    if os.path.exists(done_flag) and os.path.exists(best_pt):
        print(f"\n[SKIP] {label}: training sudah selesai.\n  best.pt: {best_pt}")
        return best_pt

    # Auto-scale workers per GPU to avoid multiprocessing BrokenPipe / EOFError
    n_gpus = len(DEVICE) if isinstance(DEVICE, list) else 1
    train_workers = max(1, NUM_WORKERS // n_gpus)

    if os.path.exists(last_pt):
        print(f"\n[RESUME] {label}: melanjutkan dari last.pt\n  {last_pt}")
        model = YOLO(last_pt)
        # Tambahkan Telegram Callbacks (meskipun resume)
        for k, v in get_yolo_callbacks(label).items():
            model.add_callback(k, v)
        model.train(resume=True, patience=EARLY_STOPPING_PATIENCE, workers=train_workers)
    else:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        model = YOLO(model_pt)
        # Tambahkan Telegram Callbacks
        for k, v in get_yolo_callbacks(label).items():
            model.add_callback(k, v)

        model.train(data=yaml_path, epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=YOLO_BATCH_SIZE,
                    project=os.path.dirname(out_dir), name=os.path.basename(out_dir),
                    exist_ok=True, device=DEVICE, patience=EARLY_STOPPING_PATIENCE,
                    workers=train_workers)

    # Tandai training selesai
    with open(done_flag, "w") as f:
        f.write("done")

    result = str(model.trainer.best) if hasattr(model, 'trainer') and model.trainer else best_pt
    del model; _flush(label)
    return result


print("\n" + "="*65 + "\n  YOLOv10 Multi-Model Fine-tuning (Medium & Extra Large)\n" + "="*65)

models_to_train = [
    {"pt": "yolov10m.pt", "yaml": DET_YAML, "run_name": "yolov10m", "label": "YOLOv10m Detection"},
    {"pt": "yolov10x.pt", "yaml": DET_YAML, "run_name": "yolov10x", "label": "YOLOv10x Detection"},
]

for spec in models_to_train:
    model_path = download_and_move_model(spec["pt"])
    best_weights = _train(model_path, spec["yaml"], spec["run_name"], spec["label"])
    
    # Simpan visualisasi sampel
    try:
        save_yolo_visual_samples(best_weights, spec["run_name"], "", n=10, conf=0.5)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan visualisasi sampel untuk {spec['run_name']}: {e}")

report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)

# Evaluasi Multi-GPU menggunakan eval_multigpu.py
if not args.skip_eval:
    print("\n" + "="*65 + "\n  Menjalankan Evaluasi Multi-GPU untuk YOLOv10 (m, x)\n" + "="*65)
    import subprocess
    import sys
    eval_gpu = args.device if args.device is not None else "all"
    try:
        subprocess.run([sys.executable, "-u", "eval_multigpu.py", "--gpus", str(eval_gpu)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Evaluasi Multi-GPU gagal: {e}")
else:
    print("\n[Skip] Evaluasi Multi-GPU dilewati karena argumen --skip-eval aktif.")

# ------ Kompres folder hasil training ------
for spec in models_to_train:
    try:
        compress_run(spec["run_name"])
    except Exception as e:
        print(f"⚠️ Gagal kompres {spec['run_name']}: {e}")

print("\n✅ Seluruh YOLOv10 Pipeline selesai.")
send_telegram_msg(f"✅ <b>YOLOv10 Pipeline Finished (Medium & Extra Large)</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>")
