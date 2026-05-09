# -*- coding: utf-8 -*-
"""
hybrid/eval_multigpu.py
========================
Distributed Multi-GPU Evaluation untuk Hybrid Pipeline (YOLO11m + SAM2).

Arsitektur Hybrid:
  - YOLO11m → Fast detection (bounding boxes + class predictions)
  - SAM2     → High-quality instance segmentation dari bounding box prompts
  
Strategi Evaluasi Terdistribusi:
  - Setiap GPU memuat YOLO11m + SAM2 secara terpisah (independent replicas)
  - Setiap GPU memproses subset gambar yang berbeda (data parallelism)
  - Hasil prediksi (bbox + mask RLE) dikumpulkan dari file pickle per-rank
  - COCOeval dijalankan di proses utama untuk konsistensi (single-threaded)
  - Throughput = total_gambar / wall_clock_time (ukuran throughput nyata)

Catatan:
  - SAM2 (sam2.1_b.pt) harus tersedia di direktori hybrid/
  - YOLO11m best.pt diambil dari get_output_dir("yolo11m")/weights/best.pt

Cara menjalankan:
    python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log
    python -u eval_multigpu.py --gpus 0,1 2>&1 | tee eval_multigpu.log

    # tmux background:
    tmux new-session -d -s hybrideval "source ../.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-dev/hybrid && \\
      python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log"

Output:
    <REPORTS_DIR>/report_hybrid_det_multigpu.csv
    <REPORTS_DIR>/report_hybrid_seg_multigpu.csv
    <REPORTS_DIR>/hybrid_detailed_predictions_multigpu.csv
"""

import os, sys, gc, csv, time, argparse, pickle, tempfile
import numpy as np
import cv2

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HYBRID_DIR      = os.path.abspath(os.path.dirname(__file__))
_FINETUNING_ROOT = os.path.abspath(os.path.join(_HYBRID_DIR, ".."))
sys.path.insert(0, _FINETUNING_ROOT)

import torch
import torch.multiprocessing as mp

from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE,
    get_output_dir, REPORTS_DIR, DATA_FILES_DIR
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools
)

MODEL_LABEL = "Hybrid (YOLO11m+SAM2, MultiGPU)"
SAM_MODEL_PATH = os.path.join(_HYBRID_DIR, "sam2.1_b.pt")

# ==============================================================================
# HELPERS
# ==============================================================================

def _flush_gpu(gpu_id: int, label: str):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu_id)
        free, total = torch.cuda.mem_get_info(gpu_id)
        print(f"  [GPU:{gpu_id}][MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB",
              flush=True)


def _gpu_report_str(gpu_ids: list) -> str:
    from collections import Counter
    names  = [torch.cuda.get_device_name(i) for i in gpu_ids]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())


def _partition_images(img_dir: str, rank: int, world_size: int) -> list:
    all_imgs = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return all_imgs[rank::world_size]


def _resolve_img_dir(yaml_path: str) -> str:
    base = os.path.dirname(yaml_path)
    for split in ("valid", "test"):
        d = os.path.join(base, split, "images")
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Tidak ditemukan valid/ atau test/images di {base}")


def _get_model_size_str() -> str:
    try:
        yolo_pt  = os.path.join(get_output_dir("yolo11m"), "weights", "best.pt")
        yolo_mb  = os.path.getsize(yolo_pt) / 1e6 if os.path.exists(yolo_pt) else 0
        sam_mb   = os.path.getsize(SAM_MODEL_PATH) / 1e6 if os.path.exists(SAM_MODEL_PATH) else 0
        return f"{yolo_mb + sam_mb:.2f}"
    except Exception:
        return "N/A"


# ==============================================================================
# WORKER: HYBRID INFERENCE PER GPU
# ==============================================================================

def _infer_worker_hybrid(rank: int, gpu_ids: list,
                         yolo_pt: str, sam_pt: str,
                         img_dir: str, image_ids: dict,
                         tmp_dir: str):
    """
    Worker hybrid: setiap GPU jalankan YOLO11m → SAM2 per gambar.
    Simpan prediksi (bbox + mask RLE) ke pickle.
    """
    from ultralytics import YOLO, SAM
    from pycocotools import mask as maskUtils

    gpu    = gpu_ids[rank]
    device_str = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)

    print(f"  [GPU:{gpu}] Rank {rank} loading YOLO11m + SAM2...", flush=True)
    try:
        yolo_model = YOLO(yolo_pt)
        sam_model  = SAM(sam_pt)
        print(f"  [GPU:{gpu}] Models loaded.", flush=True)
    except Exception as e:
        print(f"  [GPU:{gpu}] ❌ Gagal load model: {e}", flush=True)
        with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
            pickle.dump({"dt_bbox": [], "dt_segm": [],
                         "times": [], "n_imgs": 0,
                         "yolo_times": [], "sam_times": []}, f)
        return

    subset  = _partition_images(img_dir, rank, len(gpu_ids))
    dt_bbox = []
    dt_segm = []
    yolo_times = []
    sam_times  = []

    for img_path in subset:
        if img_path not in image_ids:
            continue
        img_id = image_ids[img_path]

        # YOLO detection
        t_yolo = time.perf_counter()
        det_result = yolo_model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                        device=device_str, verbose=False)
        yolo_times.append((time.perf_counter() - t_yolo) * 1000)

        if not (det_result and det_result[0].boxes is not None
                and len(det_result[0].boxes) > 0):
            continue

        pred_boxes = det_result[0].boxes.xyxy.cpu().numpy()
        pred_confs = det_result[0].boxes.conf.cpu().numpy()
        pred_clss  = det_result[0].boxes.cls.cpu().numpy().astype(int)

        # SAM2 segmentation
        try:
            t_sam = time.perf_counter()
            sam_result = sam_model.predict(
                det_result[0].orig_img, bboxes=pred_boxes, verbose=False
            )
            sam_times.append((time.perf_counter() - t_sam) * 1000)
        except Exception as e:
            print(f"  [GPU:{gpu}] ⚠️ SAM2 error pada {os.path.basename(img_path)}: {e}",
                  flush=True)
            sam_result = None

        # Load image dimensions
        img_cv = cv2.imread(img_path)
        if img_cv is None: continue
        H, W = img_cv.shape[:2]

        masks_np = []
        if sam_result and sam_result[0].masks is not None:
            for m in sam_result[0].masks.data.cpu().numpy():
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                masks_np.append(m > 0.5)
        else:
            masks_np = [None] * len(pred_boxes)

        for i in range(len(pred_boxes)):
            score  = float(pred_confs[i])
            cat_id = int(pred_clss[i]) + 1   # 0-indexed → 1-indexed COCO
            x1, y1, x2, y2 = pred_boxes[i].tolist()
            bw, bh = x2-x1, y2-y1

            dt_bbox.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, bw, bh],
                "score":       score,
            })

            if i < len(masks_np) and masks_np[i] is not None:
                bin_mask = masks_np[i].astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(bin_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                dt_segm.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":       score,
                })

    print(f"  [GPU:{gpu}] {len(dt_bbox)} prediksi bbox, {len(dt_segm)} prediksi mask "
          f"dari {len(subset)} gambar", flush=True)

    with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
        pickle.dump({
            "dt_bbox":    dt_bbox,
            "dt_segm":    dt_segm,
            "yolo_times": yolo_times,
            "sam_times":  sam_times,
            "n_imgs":     len(subset),
        }, f)

    del yolo_model, sam_model
    _flush_gpu(gpu, f"hybrid rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


# ==============================================================================
# EVALUASI UTAMA
# ==============================================================================

def eval_hybrid_distributed(gpu_ids: list, tmp_dir: str) -> tuple:
    """
    Evaluasi Hybrid YOLO11m + SAM2 secara terdistribusi.
    Returns: (det_row, seg_row) — dua dict hasil evaluasi
    """
    print("\n" + "="*65)
    print("  Distributed Eval: Hybrid Pipeline (YOLO11m + SAM2)")
    print("="*65)

    yolo_pt = os.path.join(get_output_dir("yolo11m"), "weights", "best.pt")
    sam_pt  = SAM_MODEL_PATH

    if not os.path.exists(yolo_pt):
        print(f"  ❌ YOLO11m best.pt tidak ditemukan: {yolo_pt}")
        print("  💡 Jalankan yolo/yolo11/main.py terlebih dahulu.")
        send_telegram_msg(
            f"❌ <b>Hybrid MultiGPU Eval</b>\n"
            f"YOLO11m best.pt tidak ditemukan:\n<code>{yolo_pt}</code>"
        )
        return None, None

    if not os.path.exists(sam_pt):
        print(f"  ❌ SAM2 model tidak ditemukan: {sam_pt}")
        print("  💡 Pastikan sam2.1_b.pt ada di direktori hybrid/")
        send_telegram_msg(
            f"❌ <b>Hybrid MultiGPU Eval</b>\n"
            f"SAM2 model tidak ditemukan:\n<code>{sam_pt}</code>"
        )
        return None, None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall."); return None, None

    # Build COCO GT (menggunakan SEG_YAML untuk konsistensi dengan Mask R-CNN)
    print("  [GT] Membangun COCO ground truth dari SEG_YAML...")
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT"); return None, None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
    world_size = len(gpu_ids)

    yolo_mb = round(os.path.getsize(yolo_pt) / 1e6, 2)
    sam_mb  = round(os.path.getsize(sam_pt) / 1e6, 2)
    total_mb_str = f"{yolo_mb + sam_mb:.2f}"
    print(f"  [Model] YOLO11m: {yolo_mb}MB | SAM2: {sam_mb}MB | Total: {total_mb_str}MB")

    print(f"  [Spawn] Menjalankan {world_size} GPU worker Hybrid secara paralel...")
    t_wall_start = time.perf_counter()
    mp.spawn(
        _infer_worker_hybrid,
        args=(gpu_ids, yolo_pt, sam_pt, img_dir, image_ids, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall_start

    # Kumpulkan hasil
    all_dt_bbox  = []
    all_dt_segm  = []
    all_yolo_t   = []
    all_sam_t    = []
    total_imgs   = 0

    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            all_dt_bbox.extend(data["dt_bbox"])
            all_dt_segm.extend(data["dt_segm"])
            all_yolo_t.extend(data["yolo_times"])
            all_sam_t.extend(data["sam_times"])
            total_imgs += data["n_imgs"]

    print(f"\n  [Collect] {len(all_dt_bbox)} bbox prediksi | {len(all_dt_segm)} mask prediksi")
    print(f"  [Throughput] Wall clock: {total_wall:.2f}s | {total_imgs} gambar diproses")

    if not all_dt_bbox:
        print("  ❌ Tidak ada prediksi. Pipeline Hybrid tidak menghasilkan output.")
        return None, None

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_ms         = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"

    avg_yolo_ms = round(sum(all_yolo_t) / len(all_yolo_t), 2) if all_yolo_t else "N/A"
    avg_sam_ms  = round(sum(all_sam_t) / len(all_sam_t), 2) if all_sam_t else "N/A"
    print(f"  [Latency] Avg YOLO: {avg_yolo_ms}ms | Avg SAM2: {avg_sam_ms}ms | Total: {lat_ms}ms")

    # COCOeval — Detection (BBox)
    print("\n  [COCOeval] ── Detection (BBox) ──")
    mAP50_box, mAP50_95_box = evaluate_coco_predictions(
        coco_gt_dict, image_ids,
        # Convert back to compatible format
        [{"image": None, "pred_box": None, "pred_cls": None, "pred_conf": None}],  # dummy
        iou_type="bbox"
    )
    # Gunakan langsung dari COCOeval dengan format dt yang benar
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()

        # Detection evaluation
        if all_dt_bbox:
            coco_dt_box = coco_gt.loadRes(all_dt_bbox)
            eval_box    = COCOeval(coco_gt, coco_dt_box, iouType="bbox")
            eval_box.evaluate(); eval_box.accumulate(); eval_box.summarize()
            mAP50_box    = round(float(eval_box.stats[1]), 4)  # stats[1] = mAP@.50
            mAP50_95_box = round(float(eval_box.stats[0]), 4)  # stats[0] = mAP@.50:.95
        else:
            mAP50_box = mAP50_95_box = "N/A"

        # Segmentation evaluation
        if all_dt_segm:
            print("\n  [COCOeval] ── Segmentation (Mask) ──")
            coco_dt_seg  = coco_gt.loadRes(all_dt_segm)
            eval_seg     = COCOeval(coco_gt, coco_dt_seg, iouType="segm")
            eval_seg.evaluate(); eval_seg.accumulate(); eval_seg.summarize()
            mAP50_mask    = round(float(eval_seg.stats[1]), 4)
            mAP50_95_mask = round(float(eval_seg.stats[0]), 4)
        else:
            mAP50_mask = mAP50_95_mask = "N/A"

    except Exception as e:
        print(f"  ⚠️ COCOeval error: {e}")
        mAP50_box = mAP50_95_box = mAP50_mask = mAP50_95_mask = "ERR"

    print(f"\n  ✅ Detection  — mAP50={mAP50_box}  mAP50-95={mAP50_95_box}")
    print(f"  ✅ Segmentation — mAP50={mAP50_mask}  mAP50-95={mAP50_95_mask}")
    print(f"  ✅ FPS (Throughput): {throughput_fps}")

    gpu_str = _gpu_report_str(gpu_ids)

    det_row = {
        "Model":             MODEL_LABEL,
        "Model Size (MB)":   total_mb_str,
        "mAP50-95":          mAP50_95_box,
        "mAP50":             mAP50_box,
        "Precision":         "N/A (MultiGPU)",
        "Recall":            "N/A (MultiGPU)",
        "Preprocess (ms)":   avg_yolo_ms,
        "Inference (ms)":    avg_sam_ms,
        "Postprocess (ms)":  "N/A (Distributed)",
        "Latency (ms)":      lat_ms,
        "FPS":               throughput_fps,
        "GPUs":              gpu_str,
        "Evaluator":         "COCOeval (MultiGPU)",
    }

    seg_row = {
        "Model":            MODEL_LABEL,
        "Model Size (MB)":  total_mb_str,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "Latency (ms)":     lat_ms,
        "FPS":              throughput_fps,
        "GPUs":             gpu_str,
        "Evaluator":        "COCOeval (MultiGPU)",
    }

    print(f"\n  📊 Hasil Hybrid Pipeline MultiGPU:")
    for k, v in det_row.items():
        print(f"     {k}: {v}")

    return det_row, seg_row


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid (YOLO11m+SAM2) Distributed Multi-GPU Evaluation"
    )
    parser.add_argument("--gpus", type=str, default="all",
        help="GPU yang digunakan. Contoh: '0,1' atau 'all'. Default: semua GPU.")
    args = parser.parse_args()

    GPU_IDS = list(range(torch.cuda.device_count())) if args.gpus.strip().lower() == "all" \
              else [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    if not torch.cuda.is_available() or not GPU_IDS:
        print("❌ CUDA tidak tersedia atau tidak ada GPU."); sys.exit(1)

    n_avail = torch.cuda.device_count()
    for g in GPU_IDS:
        if g >= n_avail:
            print(f"❌ GPU {g} tidak tersedia (sistem punya {n_avail} GPU)."); sys.exit(1)

    print("=" * 65)
    print("  Hybrid Pipeline — Distributed Multi-GPU Evaluation")
    print("=" * 65)
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {len(GPU_IDS)}")
    print(f"  YOLO11m pt         : {os.path.join(get_output_dir('yolo11m'), 'weights', 'best.pt')}")
    print(f"  SAM2 pt            : {SAM_MODEL_PATH}")
    print(f"  SEG_YAML           : {SEG_YAML}")
    print(f"  REPORTS_DIR        : {REPORTS_DIR}")
    print("=" * 65 + "\n")

    gc.collect(); torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(GPU_IDS[0])[0] / 1e9
    print(f"[MemClean] VRAM bebas GPU:{GPU_IDS[0]}: {free_gb:.2f} GB\n")

    send_telegram_msg(
        f"🚀 <b>Hybrid MultiGPU Eval Dimulai</b>\n"
        f"GPUs: <code>{GPU_IDS}</code>\n"
        f"World Size: {len(GPU_IDS)}\n"
        f"Pipeline: YOLO11m → SAM2 (Distributed Inference)"
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hybrideval_") as tmp_dir:
        t_total_start = time.perf_counter()

        det_row, seg_row = eval_hybrid_distributed(GPU_IDS, tmp_dir)

        # Simpan Detection Report
        if det_row:
            det_fields = [
                "Model", "Model Size (MB)", "mAP50-95", "mAP50",
                "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
                "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
            ]
            det_csv = os.path.join(REPORTS_DIR, "report_hybrid_det_multigpu.csv")
            with open(det_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=det_fields)
                w.writeheader(); w.writerow(det_row)
            print(f"\n✅ Det Report: {det_csv}")
            send_telegram_msg(
                f"✅ <b>Hybrid Det MultiGPU Selesai</b>\n"
                f"mAP50-95: <code>{det_row['mAP50-95']}</code>\n"
                f"mAP50: <code>{det_row['mAP50']}</code>\n"
                f"YOLO Latency: <code>{det_row['Preprocess (ms)']}ms</code>\n"
                f"SAM2 Latency: <code>{det_row['Inference (ms)']}ms</code>\n"
                f"FPS (Throughput): <code>{det_row['FPS']}</code>\n"
                f"GPUs: <code>{det_row['GPUs']}</code>"
            )

        # Simpan Segmentation Report
        if seg_row:
            seg_fields = [
                "Model", "Model Size (MB)", "mAP50-95(Box)",
                "mAP50-95(Mask)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
            ]
            seg_csv = os.path.join(REPORTS_DIR, "report_hybrid_seg_multigpu.csv")
            with open(seg_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=seg_fields)
                w.writeheader(); w.writerow(seg_row)
            print(f"✅ Seg Report: {seg_csv}")
            send_telegram_msg(
                f"✅ <b>Hybrid Seg MultiGPU Selesai</b>\n"
                f"mAP50-95(Box): <code>{seg_row['mAP50-95(Box)']}</code>\n"
                f"mAP50-95(Mask): <code>{seg_row['mAP50-95(Mask)']}</code>\n"
                f"FPS (Throughput): <code>{seg_row['FPS']}</code>\n"
                f"GPUs: <code>{seg_row['GPUs']}</code>"
            )

        if not det_row and not seg_row:
            print("\n❌ Evaluasi Hybrid gagal total — tidak ada report yang disimpan.")
            send_telegram_msg("❌ <b>Hybrid MultiGPU Eval GAGAL</b>\nCek log untuk detail error.")

        total_elapsed = round(time.perf_counter() - t_total_start, 1)
        print(f"\n✅ Total waktu evaluasi Hybrid: {total_elapsed}s")
        send_telegram_msg(
            f"🏁 <b>Hybrid MultiGPU Eval Selesai</b>\n"
            f"Total waktu: <code>{total_elapsed}s</code>\n"
            f"Report: <code>{REPORTS_DIR}</code>"
        )
