# -*- coding: utf-8 -*-
"""
yolo/yolo8/eval_multigpu.py
============================
Distributed Multi-GPU Evaluation untuk YOLOv8m (Detection + Segmentation).

Strategi:
  - Setiap GPU memproses subset gambar secara paralel (mp.spawn)
  - Hasil prediksi per-rank dikumpulkan via pickle file → rank 0 menjalankan COCOeval
  - Throughput diukur sebagai total_gambar / total_waktu_wall_clock (bukan rata-rata latency)

Cara menjalankan:
    # Semua GPU otomatis:
    python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log

    # GPU spesifik:
    python -u eval_multigpu.py --gpus 0,1 2>&1 | tee eval_multigpu.log

    # tmux background:
    tmux new-session -d -s yolo8eval "source ../../.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-dev/yolo/yolo8 && \\
      python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log"

Output:
    <REPORTS_DIR>/report_yolov8m_det_multigpu.csv
    <REPORTS_DIR>/report_yolov8m_seg_multigpu.csv
"""

import os, sys, gc, csv, time, json, argparse, pickle, tempfile
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.multiprocessing as mp

from config_shared import (
    WORKSPACE_DIR, DET_YAML, SEG_YAML, IMAGE_SIZE,
    get_output_dir, REPORTS_DIR
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import (
    build_coco_ground_truth, evaluate_coco_predictions, check_pycocotools
)

MODEL_LABEL_DET = "YOLOv8m (Fine-tuned, MultiGPU)"
MODEL_LABEL_SEG = "YOLOv8m-Seg (Fine-tuned, MultiGPU)"

# ==============================================================================
# HELPERS
# ==============================================================================

def _flush_gpu(rank: int, label: str):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(rank)
    free, total = torch.cuda.mem_get_info(rank)
    print(f"  [GPU:{rank}][MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)


def _gpu_report_str(gpu_ids: list) -> str:
    from collections import Counter
    names  = [torch.cuda.get_device_name(i) for i in gpu_ids]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())


def _partition_images(img_dir: str, rank: int, world_size: int) -> list:
    """Bagi daftar gambar secara merata ke setiap GPU-rank."""
    all_imgs = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return all_imgs[rank::world_size]


def _resolve_img_dir(yaml_path: str) -> str:
    """Cari folder valid/ atau test/ dari YAML."""
    base = os.path.dirname(yaml_path)
    for split in ("valid", "test"):
        d = os.path.join(base, split, "images")
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Tidak ditemukan valid/ atau test/images di {base}")


# ==============================================================================
# WORKER: INFERENCE PER GPU
# ==============================================================================

def _infer_worker_det(rank: int, gpu_ids: list, pt_path: str,
                      img_dir: str, image_ids: dict,
                      tmp_dir: str, barrier_file: str):
    """
    Worker deteksi: setiap GPU proses subset gambar, simpan prediksi ke pickle.
    Juga mengumpulkan timing internal YOLO (preprocess/inference/postprocess) per gambar.
    """
    from ultralytics import YOLO

    gpu     = gpu_ids[rank]
    device  = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens deteksi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []

    # Timing internal YOLO per gambar (ms)
    speed_pre  = []   # preprocess
    speed_inf  = []   # inference
    speed_post = []   # postprocess

    t_start = time.perf_counter()
    for img_path in subset:
        if img_path not in image_ids:
            continue
        result = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                               device=gpu, verbose=False)
        if result:
            # Kumpulkan timing internal dari Ultralytics
            spd = result[0].speed  # {"preprocess": ms, "inference": ms, "postprocess": ms}
            speed_pre.append(spd.get("preprocess", 0.0))
            speed_inf.append(spd.get("inference",  0.0))
            speed_post.append(spd.get("postprocess", 0.0))

            if result[0].boxes is not None:
                boxes = result[0].boxes.xyxy.cpu().numpy()
                confs = result[0].boxes.conf.cpu().numpy()
                clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(boxes, confs, clss):
                    preds.append({
                        "image":     img_path,
                        "pred_box":  box.tolist(),
                        "pred_cls":  int(cls),
                        "pred_conf": float(conf),
                    })
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi dari {len(subset)} gambar "
          f"dalam {elapsed:.1f}s", flush=True)

    out_pkl = os.path.join(tmp_dir, f"det_rank{rank}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed,
                     "n_imgs": len(subset),
                     "speed_pre":  speed_pre,
                     "speed_inf":  speed_inf,
                     "speed_post": speed_post}, f)

    del model
    _flush_gpu(gpu, f"det rank{rank}")

    done_flag = os.path.join(tmp_dir, f"done_det_rank{rank}.flag")
    open(done_flag, "w").close()
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _infer_worker_seg(rank: int, gpu_ids: list, pt_path: str,
                      img_dir: str, image_ids: dict, tmp_dir: str):
    """Worker segmentasi: simpan prediksi mask + box + speed ke pickle."""
    from ultralytics import YOLO

    gpu    = gpu_ids[rank]
    torch.cuda.set_device(gpu)

    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens segmentasi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []

    speed_pre  = []
    speed_inf  = []
    speed_post = []

    t_start = time.perf_counter()
    for img_path in subset:
        if img_path not in image_ids:
            continue
        result = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                               device=gpu, verbose=False)
        if result:
            spd = result[0].speed
            speed_pre.append(spd.get("preprocess", 0.0))
            speed_inf.append(spd.get("inference",  0.0))
            speed_post.append(spd.get("postprocess", 0.0))

            if result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy()
                boxes = result[0].boxes.xyxy.cpu().numpy()
                confs = result[0].boxes.conf.cpu().numpy()
                clss  = result[0].boxes.cls.cpu().numpy().astype(int)
                for mask, box, conf, cls in zip(masks, boxes, confs, clss):
                    preds.append({
                        "image":     img_path,
                        "pred_mask": mask,
                        "pred_box":  box.tolist(),
                        "pred_cls":  int(cls),
                        "pred_conf": float(conf),
                    })
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print(f"  [GPU:{gpu}] {len(preds)} prediksi-mask dari {len(subset)} gambar "
          f"dalam {elapsed:.1f}s", flush=True)

    out_pkl = os.path.join(tmp_dir, f"seg_rank{rank}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed,
                     "n_imgs": len(subset),
                     "speed_pre":  speed_pre,
                     "speed_inf":  speed_inf,
                     "speed_post": speed_post}, f)

    del model
    _flush_gpu(gpu, f"seg rank{rank}")
    open(os.path.join(tmp_dir, f"done_seg_rank{rank}.flag"), "w").close()
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


# ==============================================================================
# EVALUASI DETECTION (Distributed)
# ==============================================================================

def eval_detection_distributed(gpu_ids: list, tmp_dir: str) -> dict | None:
    """Evaluasi detection dengan distribusi beban ke semua GPU."""
    print("\n" + "="*65)
    print("  Distributed Eval: YOLOv8m Detection")
    print("="*65)

    pt_path = os.path.join(get_output_dir("yolov8m"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ best.pt tidak ditemukan: {pt_path}")
        send_telegram_msg(f"❌ <b>YOLOv8m MultiGPU Eval</b>\nDet best.pt tidak ditemukan:\n<code>{pt_path}</code>")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return None

    # Build COCO ground truth (sekali, di proses utama)
    print("  [GT] Membangun COCO ground truth...")
    coco_gt_dict, image_ids = build_coco_ground_truth(DET_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir = _resolve_img_dir(DET_YAML)
    world_size = len(gpu_ids)

    # Spawn workers
    print(f"  [Spawn] Menjalankan {world_size} GPU worker secara paralel...")
    t_wall_start = time.perf_counter()
    mp.spawn(
        _infer_worker_det,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir, ""),
        nprocs=world_size,
        join=True,
    )
    t_wall_end = time.perf_counter()
    total_wall = t_wall_end - t_wall_start

    # Kumpulkan prediksi dan timing dari semua rank
    all_preds = []
    all_pre_t = []; all_inf_t = []; all_post_t = []
    total_imgs_processed = 0
    for r in range(world_size):
        pkl_path = os.path.join(tmp_dir, f"det_rank{r}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            all_preds.extend(data["preds"])
            total_imgs_processed += data["n_imgs"]
            all_pre_t.extend(data.get("speed_pre",  []))
            all_inf_t.extend(data.get("speed_inf",  []))
            all_post_t.extend(data.get("speed_post", []))

    print(f"  [Collect] Total: {len(all_preds)} prediksi dari {total_imgs_processed} gambar")
    print(f"  [Throughput] Wall clock: {total_wall:.2f}s")

    def _avg_ms(lst): return round(sum(lst)/len(lst), 2) if lst else "N/A"
    avg_pre_ms  = _avg_ms(all_pre_t)
    avg_inf_ms  = _avg_ms(all_inf_t)
    avg_post_ms = _avg_ms(all_post_t)
    print(f"  [Speed] Pre={avg_pre_ms}ms | Inf={avg_inf_ms}ms | Post={avg_post_ms}ms (avg/gambar)")

    # Throughput Distributed
    throughput_fps = round(total_imgs_processed / total_wall, 2) if total_wall > 0 else "N/A"
    lat_per_img_ms = round(total_wall * 1000 / total_imgs_processed, 2) if total_imgs_processed > 0 else "N/A"

    # COCOeval (sekali, di rank 0)
    print("  [COCOeval] Menghitung mAP50 & mAP50-95...")
    mAP50, mAP50_95 = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="bbox"
    )
    print(f"  ✅ mAP50={mAP50}  mAP50-95={mAP50_95}")

    # Hitung Precision & Recall manual (IoU@0.5)
    # Build GT lookup
    gt_det = {}
    for img_path, img_id in image_ids.items():
        gt_det[img_path] = {"boxes": [], "clss": []}
        for ann in coco_gt_dict["annotations"]:
            if ann["image_id"] == img_id:
                b = ann["bbox"]
                gt_det[img_path]["boxes"].append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
                gt_det[img_path]["clss"].append(ann["category_id"] - 1)

    gt_per_class = {}
    for g in gt_det.values():
        for c in g["clss"]:
            gt_per_class[c] = gt_per_class.get(c, 0) + 1

    total_tp, total_fp, total_gt = 0, 0, 0
    num_classes = max(gt_per_class.keys()) + 1 if gt_per_class else 1
    for cls_id in range(num_classes):
        cls_preds = [p for p in all_preds if p["pred_cls"] == cls_id]
        cls_gt_count = gt_per_class.get(cls_id, 0)
        total_gt += cls_gt_count
        if not cls_preds:
            continue
        gt_matched = set()
        gt_boxes_cls = [
            (ip, idx, box)
            for ip, g in gt_det.items()
            for idx, (box, cls) in enumerate(zip(g["boxes"], g["clss"]))
            if cls == cls_id
        ]
        for pred in sorted(cls_preds, key=lambda x: x["pred_conf"], reverse=True):
            best_iou, best_key = 0.0, None
            pb = pred["pred_box"]
            for gi_path, gi_idx, gb in gt_boxes_cls:
                if pred["image"] != gi_path:
                    continue
                x1 = max(pb[0], gb[0]); y1 = max(pb[1], gb[1])
                x2 = min(pb[2], gb[2]); y2 = min(pb[3], gb[3])
                iw = max(0, x2-x1); ih = max(0, y2-y1)
                ia = iw * ih
                pa = (pb[2]-pb[0])*(pb[3]-pb[1])
                ga = (gb[2]-gb[0])*(gb[3]-gb[1])
                ua = pa + ga - ia
                iou = ia / ua if ua > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou; best_key = (gi_path, gi_idx)
            if best_iou >= 0.5 and best_key not in gt_matched:
                total_tp += 1; gt_matched.add(best_key)
            else:
                total_fp += 1

    prec = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    rec  = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0
    print(f"  ✅ Precision={prec}  Recall={rec}")

    size_mb = round(os.path.getsize(pt_path) / 1e6, 2)
    gpu_str = _gpu_report_str(gpu_ids)

    row = {
        "Model":             MODEL_LABEL_DET,
        "Model Size (MB)":   size_mb,
        "mAP50-95":          mAP50_95,
        "mAP50":             mAP50,
        "Precision":         prec,
        "Recall":            rec,
        "Preprocess (ms)":   avg_pre_ms,
        "Inference (ms)":    avg_inf_ms,
        "Postprocess (ms)":  avg_post_ms,
        "Latency (ms)":      lat_per_img_ms,
        "FPS":               throughput_fps,
        "GPUs":              gpu_str,
        "Evaluator":         "COCOeval (MultiGPU)",
    }
    print(f"\n  📊 Hasil Detection MultiGPU:")
    for k, v in row.items():
        print(f"     {k}: {v}")
    return row


# ==============================================================================
# EVALUASI SEGMENTATION (Distributed)
# ==============================================================================

def eval_segmentation_distributed(gpu_ids: list, tmp_dir: str) -> dict | None:
    """Evaluasi segmentasi dengan distribusi beban ke semua GPU."""
    print("\n" + "="*65)
    print("  Distributed Eval: YOLOv8m-Seg Segmentation")
    print("="*65)

    pt_path = os.path.join(get_output_dir("yolov8m_seg"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ best.pt tidak ditemukan: {pt_path}")
        send_telegram_msg(f"❌ <b>YOLOv8m MultiGPU Eval</b>\nSeg best.pt tidak ditemukan:\n<code>{pt_path}</code>")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall.")
        return None

    print("  [GT] Membangun COCO ground truth segmentasi...")
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT")
        return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi secara paralel...")
    t_wall_start = time.perf_counter()
    mp.spawn(
        _infer_worker_seg,
        args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    t_wall_end = time.perf_counter()
    total_wall = t_wall_end - t_wall_start

    all_preds = []
    all_pre_t = []; all_inf_t = []; all_post_t = []
    total_imgs_processed = 0
    for r in range(world_size):
        pkl_path = os.path.join(tmp_dir, f"seg_rank{r}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            all_preds.extend(data["preds"])
            total_imgs_processed += data["n_imgs"]
            all_pre_t.extend(data.get("speed_pre",  []))
            all_inf_t.extend(data.get("speed_inf",  []))
            all_post_t.extend(data.get("speed_post", []))

    print(f"  [Collect] Total: {len(all_preds)} prediksi dari {total_imgs_processed} gambar")
    print(f"  [Throughput] Wall clock: {total_wall:.2f}s")

    def _avg_ms(lst): return round(sum(lst)/len(lst), 2) if lst else "N/A"
    avg_pre_ms  = _avg_ms(all_pre_t)
    avg_inf_ms  = _avg_ms(all_inf_t)
    avg_post_ms = _avg_ms(all_post_t)
    print(f"  [Speed] Pre={avg_pre_ms}ms | Inf={avg_inf_ms}ms | Post={avg_post_ms}ms (avg/gambar)")

    throughput_fps = round(total_imgs_processed / total_wall, 2) if total_wall > 0 else "N/A"
    lat_per_img_ms = round(total_wall * 1000 / total_imgs_processed, 2) if total_imgs_processed > 0 else "N/A"

    print("  [COCOeval] Menghitung mAP Mask & Box...")
    mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="segm"
    )
    _, mAP50_95_box = evaluate_coco_predictions(
        coco_gt_dict, image_ids, all_preds, iou_type="bbox"
    )
    print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")

    size_mb = round(os.path.getsize(pt_path) / 1e6, 2)
    gpu_str = _gpu_report_str(gpu_ids)

    row = {
        "Model":            MODEL_LABEL_SEG,
        "Model Size (MB)":  size_mb,
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "Latency (ms)":     lat_per_img_ms,
        "FPS":              throughput_fps,
        "GPUs":             gpu_str,
        "Evaluator":        "COCOeval (MultiGPU)",
    }
    print(f"\n  📊 Hasil Segmentation MultiGPU:")
    for k, v in row.items():
        print(f"     {k}: {v}")
    return row


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8m Distributed Multi-GPU Evaluation")
    parser.add_argument("--gpus", type=str, default="all",
        help="GPU yang digunakan. Contoh: '0,1' atau 'all'. Default: semua GPU.")
    parser.add_argument("--skip-det", action="store_true",
        help="Lewati evaluasi detection.")
    parser.add_argument("--skip-seg", action="store_true",
        help="Lewati evaluasi segmentation.")
    args = parser.parse_args()

    if args.gpus.strip().lower() == "all":
        n_gpus  = torch.cuda.device_count()
        GPU_IDS = list(range(n_gpus))
    else:
        GPU_IDS = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    if not torch.cuda.is_available() or not GPU_IDS:
        print("❌ CUDA tidak tersedia atau tidak ada GPU terpilih.")
        sys.exit(1)

    n_avail = torch.cuda.device_count()
    for g in GPU_IDS:
        if g >= n_avail:
            print(f"❌ GPU {g} tidak tersedia (sistem punya {n_avail} GPU).")
            sys.exit(1)

    print("=" * 65)
    print("  YOLOv8m — Distributed Multi-GPU Evaluation")
    print("=" * 65)
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {len(GPU_IDS)}")
    print(f"  DET_YAML           : {DET_YAML}")
    print(f"  SEG_YAML           : {SEG_YAML}")
    print(f"  REPORTS_DIR        : {REPORTS_DIR}")
    print("=" * 65 + "\n")

    # Bersihkan VRAM sebelum mulai
    gc.collect()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(GPU_IDS[0])[0] / 1e9
    print(f"[MemClean] VRAM bebas GPU:{GPU_IDS[0]}: {free_gb:.2f} GB\n")

    send_telegram_msg(
        f"🚀 <b>YOLOv8m MultiGPU Eval Dimulai</b>\n"
        f"GPUs: <code>{GPU_IDS}</code>\n"
        f"World Size: {len(GPU_IDS)}"
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Gunakan tmpdir bersama untuk pickle antar worker
    with tempfile.TemporaryDirectory(prefix="yolo8eval_") as tmp_dir:
        t_total_start = time.perf_counter()

        # ── Detection ─────────────────────────────────────────────────────────
        det_row = None
        if not args.skip_det:
            send_telegram_msg("🔍 <b>YOLOv8m Det Eval</b>\nMemulai distributed inference deteksi...", force=False)
            det_row = eval_detection_distributed(GPU_IDS, tmp_dir)
            if det_row:
                det_fields = [
                    "Model", "Model Size (MB)", "mAP50-95", "mAP50",
                    "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
                    "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
                ]
                det_csv = os.path.join(REPORTS_DIR, "report_yolov8m_det_multigpu.csv")
                with open(det_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=det_fields)
                    w.writeheader(); w.writerow(det_row)
                print(f"\n✅ Det Report: {det_csv}")
                send_telegram_msg(
                    f"✅ <b>YOLOv8m Det MultiGPU Selesai</b>\n"
                    f"mAP50-95: <code>{det_row['mAP50-95']}</code>\n"
                    f"mAP50: <code>{det_row['mAP50']}</code>\n"
                    f"Precision: <code>{det_row['Precision']}</code>\n"
                    f"Recall: <code>{det_row['Recall']}</code>\n"
                    f"FPS (Throughput): <code>{det_row['FPS']}</code>\n"
                    f"GPUs: <code>{det_row['GPUs']}</code>"
                )

        # ── Segmentation ──────────────────────────────────────────────────────
        seg_row = None
        if not args.skip_seg:
            send_telegram_msg("🔍 <b>YOLOv8m Seg Eval</b>\nMemulai distributed inference segmentasi...", force=False)
            seg_row = eval_segmentation_distributed(GPU_IDS, tmp_dir)
            if seg_row:
                seg_fields = [
                    "Model", "Model Size (MB)", "mAP50-95(Box)",
                    "mAP50-95(Mask)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
                ]
                seg_csv = os.path.join(REPORTS_DIR, "report_yolov8m_seg_multigpu.csv")
                with open(seg_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=seg_fields)
                    w.writeheader(); w.writerow(seg_row)
                print(f"✅ Seg Report: {seg_csv}")
                send_telegram_msg(
                    f"✅ <b>YOLOv8m Seg MultiGPU Selesai</b>\n"
                    f"mAP50-95(Box): <code>{seg_row['mAP50-95(Box)']}</code>\n"
                    f"mAP50-95(Mask): <code>{seg_row['mAP50-95(Mask)']}</code>\n"
                    f"FPS (Throughput): <code>{seg_row['FPS']}</code>\n"
                    f"GPUs: <code>{seg_row['GPUs']}</code>"
                )

        t_total_end = time.perf_counter()
        total_elapsed = round(t_total_end - t_total_start, 1)
        print(f"\n✅ YOLOv8m MultiGPU Evaluation selesai dalam {total_elapsed}s")
        send_telegram_msg(
            f"🏁 <b>YOLOv8m MultiGPU Eval Selesai</b>\n"
            f"Total waktu: <code>{total_elapsed}s</code>\n"
            f"Report: <code>{REPORTS_DIR}</code>"
        )
