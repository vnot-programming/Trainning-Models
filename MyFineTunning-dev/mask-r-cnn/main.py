# -*- coding: utf-8 -*-
"""
mask-r-cnn/main.py
==================
Fine-tuning Mask R-CNN — proses TERPISAH, clean GPU.

Cara menjalankan:
    python -u main.py 2>&1 | tee maskrcnn_train.log
"""

import os, sys, csv, gc
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_ENGINE = os.environ.get("TRAIN_ENGINE_PATH", "/root/MyTrainEngine")
sys.path.insert(0, ROOT)
sys.path.insert(0, TRAIN_ENGINE)

from config_shared import (
    WORKSPACE_DIR, SEG_DATASET_LOCATION,
    EPOCHS, MASKRCNN_BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, get_output_dir, compress_run,
    parse_device,
)
import argparse
import torch, torch._dynamo
torch._dynamo.disable()

parser = argparse.ArgumentParser(description="Mask R-CNN Fine-tuning")
parser.add_argument(
    "--device", type=str, default="0",
    help="GPU single yang digunakan (Mask R-CNN tidak DDP). "
         "Contoh: '0', '1', '2'. Default: '0'."
)
args = parser.parse_args()

_dev = parse_device(args.device)
# Mask R-CNN HARUS single GPU — ambil GPU pertama jika list
if isinstance(_dev, list):
    _dev = _dev[0]
    print(f"[Device] ⚠️  Mask R-CNN tidak support multi-GPU. Digunakan GPU {_dev}.")

DEVICE = torch.device(f"cuda:{_dev}") if torch.cuda.is_available() and _dev != "cpu" else torch.device("cpu")
print(f"[Device] Mask R-CNN → {DEVICE}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"[MemCheck] {free/1e9:.2f} GB bebas / {total/1e9:.2f} GB total")

from models.maskrcnn_builder import build_mask_rcnn
from models.maskrcnn_trainer import train_mask_rcnn

out_dir = get_output_dir("maskrcnn")
best_pt = os.path.join(out_dir, "weights", "best.pt")

if os.path.exists(best_pt):
    print(f"\n[SKIP] Mask R-CNN sudah ada: {best_pt}")
else:
    print(f"\n{'='*60}\n  Mask R-CNN Fine-tuning\n{'='*60}")
    model = build_mask_rcnn(num_classes=NUM_CLASSES+1, use_parallel=False, device=DEVICE)
    best_pt = train_mask_rcnn(
        model=model, dataset_location=SEG_DATASET_LOCATION,
        epochs=EPOCHS, batch_size=MASKRCNN_BATCH_SIZE,
        device=DEVICE, num_workers=NUM_WORKERS,
        output_dir=os.path.join(out_dir, "weights"),
    )
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"✅ Selesai: {best_pt}")

# ------ Evaluasi latency manual (Mask R-CNN tidak punya .val()) ------
import time, random as _random
_size_mb = round(os.path.getsize(best_pt)/1e6, 2) if os.path.exists(best_pt) else "N/A"

_lat_ms  = "N/A"
_fps_val = "N/A"
if os.path.exists(best_pt):
    _img_dir_eval = os.path.join(SEG_DATASET_LOCATION, "test", "images")
    if not os.path.isdir(_img_dir_eval):
        _img_dir_eval = os.path.join(SEG_DATASET_LOCATION, "valid", "images")
    _eval_imgs = [os.path.join(_img_dir_eval, f)
                  for f in os.listdir(_img_dir_eval)
                  if f.lower().endswith((".jpg",".jpeg",".png"))]
    _eval_samples = _random.sample(_eval_imgs, min(5, len(_eval_imgs)))

    # Load model untuk timing
    from models.maskrcnn_builder import build_mask_rcnn as _build_eval
    from torchvision.transforms import functional as _TF
    _eval_model = _build_eval(num_classes=NUM_CLASSES+1, use_parallel=False, device=DEVICE)
    _eval_model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
    _eval_model.eval()

    _times = []
    print("\n[Eval] Mengukur latency Mask R-CNN pada sampel...")
    for _ip in _eval_samples:
        from PIL import Image as _Img
        import torchvision.transforms.functional as _TF2
        _pil = _Img.open(_ip).convert("RGB")
        _t   = _TF2.to_tensor(_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _t0 = time.perf_counter()
            _eval_model(_t)
            _times.append((time.perf_counter() - _t0) * 1000)
    del _eval_model; gc.collect(); torch.cuda.empty_cache()

    if _times:
        _lat_ms  = round(sum(_times) / len(_times), 2)
        _fps_val = round(1000 / _lat_ms, 2)
    print(f"[Eval] Rata-rata latency: {_lat_ms} ms ({_fps_val} FPS)")

# CSV format seg_performance (identik reporter.py)
report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)
csv_path   = os.path.join(report_dir, "report_maskrcnn_seg.csv")
seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=seg_fields)
    w.writeheader()
    w.writerow({
        "Model":           "Mask R-CNN ResNet-50 FPN (Fine-tuned)",
        "Model Size (MB)": _size_mb,
        "mAP50-95(Box)":   "N/A",   # Mask R-CNN tidak punya YOLO val()
        "mAP50-95(Mask)":  "N/A",   # diisi manual dari evaluasi custom jika ada
        "Latency(ms)":     _lat_ms,
        "FPS":             _fps_val,
    })

print(f"\n✅ Report: {csv_path}")

# ------ Visualisasi 5 sampel Mask R-CNN ------
import random, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.transforms import functional as TF

visual_dir = os.path.join(WORKSPACE_DIR, "runs", "visuals")
os.makedirs(visual_dir, exist_ok=True)

if os.path.exists(best_pt):
    img_dir = os.path.join(SEG_DATASET_LOCATION, "test", "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(SEG_DATASET_LOCATION, "valid", "images")

    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
    samples = random.sample(all_imgs, min(5, len(all_imgs)))
    print(f"\n[Visual] Mask R-CNN — {len(samples)} sampel")

    # Reload model untuk inferensi
    from models.maskrcnn_builder import build_mask_rcnn as _build
    vis_model = _build(num_classes=NUM_CLASSES+1, use_parallel=False, device=DEVICE)
    vis_model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
    vis_model.eval()

    COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

    for idx, img_path in enumerate(samples, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = vis_model(img_tensor)[0]

        # Overlay masks dan boxes
        overlay = img_rgb.copy()
        masks   = preds.get("masks", torch.zeros(0,1,1,1))
        boxes   = preds.get("boxes", torch.zeros(0,4))
        scores  = preds.get("scores", torch.zeros(0))
        labels  = preds.get("labels", torch.zeros(0, dtype=torch.int64))

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            if score.item() < 0.5:
                continue
            color = (COLORS[i % 20, :3] * 255).astype(np.uint8)
            m = mask[0].cpu().numpy() > 0.5
            overlay[m] = (overlay[m] * 0.5 + color * 0.5).astype(np.uint8)
            x1,y1,x2,y2 = box.int().tolist()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color.tolist(), 2)
            cv2.putText(overlay, f"{score:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)

        out_path = os.path.join(visual_dir, f"maskrcnn_{idx:02d}_{base}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  [{idx}/{len(samples)}] → {out_path}")

    del vis_model; gc.collect(); torch.cuda.empty_cache()
else:
    print("[Visual] ⚠️  Mask R-CNN best.pt tidak ada, skip visualisasi.")

compress_run("maskrcnn")

print("\n✅ Mask R-CNN selesai.")
print("[Next] Lanjutkan ke: cd ../hybrid && python -u main.py")
