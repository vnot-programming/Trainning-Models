# -*- coding: utf-8 -*-
"""
mask-r-cnn/maskrcnn_eval.py
============================
Evaluasi Mask R-CNN menggunakan COCO JSON dari Roboflow.
Menghasilkan mAP50-95(Box) dan mAP50-95(Mask) yang akurat.

Output:
  - runs/reports/report_maskrcnn_seg.csv   (format seg_performance)
  - runs/reports/report_maskrcnn_complexity.csv

Cara menjalankan (setelah maskrcnn/main.py training selesai):
    python -u maskrcnn_eval.py 2>&1 | tee maskrcnn_eval.log
"""

import os, sys, csv, gc, json, time
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_ENGINE = os.environ.get("TRAIN_ENGINE_PATH", "/root/MyTrainEngine")
sys.path.insert(0, ROOT)
sys.path.insert(0, TRAIN_ENGINE)

from config_shared import (
    WORKSPACE_DIR, SEG_DATASET_LOCATION, NUM_CLASSES,
    get_output_dir, measure_vram_peak,
)
import torch, torch._dynamo
torch._dynamo.disable()

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"[Device] {DEVICE}")

best_pt = os.path.join(get_output_dir("maskrcnn"), "weights", "best.pt")
if not os.path.exists(best_pt):
    print(f"❌ best.pt tidak ditemukan: {best_pt}")
    print("   Pastikan mask-r-cnn/main.py sudah dijalankan.")
    sys.exit(1)

print(f"[Model] ✅ {best_pt}  ({round(os.path.getsize(best_pt)/1e6, 1)} MB)")

# ==============================================================================
# 1. DOWNLOAD COCO JSON (SEGMENTATION) dari Roboflow jika belum ada
# ==============================================================================
COCO_SEG_DIR = os.path.join(os.path.dirname(SEG_DATASET_LOCATION), "coco_seg_eval")
COCO_ANN_FILE = None

# Cari annotations yang sudah ada
for candidate in [
    os.path.join(SEG_DATASET_LOCATION, "test", "_annotations.coco.json"),
    os.path.join(SEG_DATASET_LOCATION, "valid", "_annotations.coco.json"),
    os.path.join(COCO_SEG_DIR, "test", "_annotations.coco.json"),
    os.path.join(COCO_SEG_DIR, "valid", "_annotations.coco.json"),
]:
    if os.path.exists(candidate):
        COCO_ANN_FILE = candidate
        print(f"[COCO] Annotations ditemukan: {COCO_ANN_FILE}")
        break

if COCO_ANN_FILE is None:
    print("[COCO] Annotations tidak ditemukan — download dari Roboflow...")
    try:
        from roboflow import Roboflow
        rf      = Roboflow(api_key="F0VtV8b5YBdJHZbasy0w")
        project = rf.workspace("wbc-laboratory").project("segpoligon-me-bottle-isempty3")
        version = project.version(5)
        dataset = version.download("coco-segmentation", location=COCO_SEG_DIR)
        # Cari file annotations
        for root_d, dirs, files in os.walk(COCO_SEG_DIR):
            for fname in files:
                if fname.endswith(".json"):
                    COCO_ANN_FILE = os.path.join(root_d, fname)
                    break
            if COCO_ANN_FILE:
                break
        if COCO_ANN_FILE:
            print(f"[COCO] ✅ Downloaded: {COCO_ANN_FILE}")
        else:
            print("[COCO] ❌ JSON tidak ditemukan setelah download")
    except Exception as e:
        print(f"[COCO] ❌ Download gagal: {e}")

# Tentukan img_dir dari lokasi annotations
if COCO_ANN_FILE:
    COCO_IMG_DIR = os.path.dirname(COCO_ANN_FILE)
else:
    COCO_IMG_DIR = os.path.join(SEG_DATASET_LOCATION, "test", "images")
    if not os.path.isdir(COCO_IMG_DIR):
        COCO_IMG_DIR = os.path.join(SEG_DATASET_LOCATION, "valid", "images")

print(f"[COCO] Image dir: {COCO_IMG_DIR}")

# ==============================================================================
# 2. LOAD MODEL
# ==============================================================================
from models.maskrcnn_builder import build_mask_rcnn
from torchvision.transforms import functional as TF
from PIL import Image

model = build_mask_rcnn(num_classes=NUM_CLASSES+1, use_parallel=False, device=DEVICE)
model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
model.eval()
print("[Model] Mask R-CNN dimuat untuk evaluasi.")

# ==============================================================================
# 3. LATENCY & PEAK VRAM (timing manual pada 10 sampel)
# ==============================================================================
import random, glob
all_imgs = glob.glob(os.path.join(COCO_IMG_DIR, "*.jpg")) + \
           glob.glob(os.path.join(COCO_IMG_DIR, "*.png"))
latency_samples = random.sample(all_imgs, min(10, len(all_imgs)))

_times = []
print(f"\n[Eval] Mengukur latency pada {len(latency_samples)} sampel ...")
for img_path in latency_samples:
    pil = Image.open(img_path).convert("RGB")
    t   = TF.to_tensor(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        t0 = time.perf_counter()
        model([t[0]])
        _times.append((time.perf_counter() - t0) * 1000)

lat_ms  = round(sum(_times) / len(_times), 2) if _times else "N/A"
fps_val = round(1000 / lat_ms, 2) if isinstance(lat_ms, float) else "N/A"
print(f"[Eval] Rata-rata latency: {lat_ms} ms ({fps_val} FPS)")

# Peak VRAM
vram_gb = "N/A"
if torch.cuda.is_available() and all_imgs:
    _pil_v = Image.open(all_imgs[0]).convert("RGB")
    _t_v   = TF.to_tensor(_pil_v).unsqueeze(0).to(DEVICE)
    vram_gb = measure_vram_peak(lambda: model([_t_v[0]]), device_id=0)
    print(f"[Eval] Peak VRAM: {vram_gb} GB")

# ==============================================================================
# 4. EVALUASI mAP dengan pycocotools (jika COCO JSON tersedia)
# ==============================================================================
box_map5095  = "N/A"
mask_map5095 = "N/A"
footnote     = ""

if COCO_ANN_FILE and os.path.exists(COCO_ANN_FILE):
    print(f"\n[mAP] Menjalankan evaluasi COCO pada: {COCO_ANN_FILE}")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import numpy as np

        coco_gt  = COCO(COCO_ANN_FILE)
        img_ids  = sorted(coco_gt.getImgIds())

        det_results  = []  # untuk box mAP
        seg_results  = []  # untuk mask mAP

        print(f"[mAP] Inferensi pada {len(img_ids)} gambar ...")
        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            pil = Image.open(img_path).convert("RGB")
            t   = TF.to_tensor(pil).to(DEVICE)
            with torch.no_grad():
                preds = model([t])[0]

            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            boxes  = preds["boxes"].cpu().numpy()
            masks  = preds.get("masks", None)

            for i in range(len(scores)):
                if scores[i] < 0.05:
                    continue
                x1, y1, x2, y2 = boxes[i]
                w, h = float(x2 - x1), float(y2 - y1)
                # COCO label: Mask R-CNN label 1-indexed (0=background)
                cat_id = int(labels[i])

                det_results.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [float(x1), float(y1), w, h],
                    "score":       float(scores[i]),
                })

                if masks is not None and i < len(masks):
                    from pycocotools import mask as mask_util
                    m_bin = (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                    rle   = mask_util.encode(np.asfortranarray(m_bin))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    seg_results.append({
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "segmentation":rle,
                        "score":       float(scores[i]),
                    })

        def _run_coco_eval(results, iou_type):
            if not results:
                return "N/A"
            coco_dt  = coco_gt.loadRes(results)
            evaluator = COCOeval(coco_gt, coco_dt, iou_type)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            return round(float(evaluator.stats[0]), 4)  # mAP50-95

        print("\n[mAP] Evaluasi Box ...")
        box_map5095  = _run_coco_eval(det_results,  "bbox")
        print(f"[mAP] Box mAP50-95 = {box_map5095}")

        print("\n[mAP] Evaluasi Mask ...")
        mask_map5095 = _run_coco_eval(seg_results, "segm")
        print(f"[mAP] Mask mAP50-95 = {mask_map5095}")

        footnote = "Diukur pada dataset test fine-tune (COCO JSON, pycocotools)"
    except Exception as e:
        print(f"[mAP] ❌ Evaluasi COCO gagal: {e}")
        box_map5095  = "N/A"
        mask_map5095 = "N/A"
        footnote     = "†Gagal evaluasi COCO — lihat log"

else:
    # Fallback: nilai dari literatur
    box_map5095  = "38.2†"
    mask_map5095 = "34.6†"
    footnote     = "†COCO val2017 baseline (He et al., 2017); bukan hasil fine-tune dataset ini"
    print(f"[mAP] COCO JSON tidak tersedia → pakai nilai literatur: box={box_map5095}, mask={mask_map5095}")

del model; gc.collect(); torch.cuda.empty_cache()

# ==============================================================================
# 5. CSV REPORTS
# ==============================================================================
report_dir = REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)

# --- Segmentation Performance CSV ---
seg_fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS", "Notes"]
seg_csv = os.path.join(report_dir, "report_maskrcnn_seg.csv")
with open(seg_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=seg_fields)
    w.writeheader()
    w.writerow({
        "Model":           "Mask R-CNN ResNet-50 FPN (Fine-tuned)",
        "Model Size (MB)": round(os.path.getsize(best_pt)/1e6, 2),
        "mAP50-95(Box)":   box_map5095,
        "mAP50-95(Mask)":  mask_map5095,
        "Latency(ms)":     lat_ms,
        "FPS":             fps_val,
        "Notes":           footnote,
    })
print(f"\n✅ Seg CSV  : {seg_csv}")

# --- Complexity CSV ---
complexity_fields = ["Model", "Architecture Type", "Parameters (M)",
                     "GFLOPs", "Max VRAM (GB)", "Notes"]
cplx_csv = os.path.join(report_dir, "report_maskrcnn_complexity.csv")
with open(cplx_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=complexity_fields)
    w.writeheader()
    w.writerow({
        "Model":             "Mask R-CNN ResNet-50 FPN (Fine-tuned)",
        "Architecture Type": "End-to-End Segmentation",
        "Parameters (M)":   "45.7†",
        "GFLOPs":            "~134.4†",
        "Max VRAM (GB)":     vram_gb,
        "Notes":             "†He et al. (2017); GFLOPs estimasi Torchvision",
    })
print(f"✅ Complexity CSV: {cplx_csv}")

print(f"\n📊 Hasil Evaluasi Mask R-CNN:")
print(f"   mAP50-95(Box)  = {box_map5095}")
print(f"   mAP50-95(Mask) = {mask_map5095}")
print(f"   Latency        = {lat_ms} ms")
print(f"   FPS            = {fps_val}")
print(f"   Peak VRAM      = {vram_gb} GB")
