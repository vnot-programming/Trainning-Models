# -*- coding: utf-8 -*-
"""
yolo/yolo9/eval_multigpu.py
============================
Distributed Multi-GPU Evaluation untuk YOLOv9m (Detection) + YOLOv9c-Seg (Segmentation).

Strategi:
  - mp.spawn: setiap GPU memproses subset gambar secara paralel
  - Prediksi dikumpulkan dari file pickle, COCOeval dijalankan di proses utama
  - FPS diukur sebagai throughput terdistribusi (total_imgs / wall_clock)

Cara menjalankan:
    python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log
    python -u eval_multigpu.py --gpus 0,1 2>&1 | tee eval_multigpu.log

    # tmux background:
    tmux new-session -d -s yolo9eval "source ../../.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-RunPOD/yolo/yolo9 && \\
      python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log"

Output:
    <REPORTS_DIR>/report_yolov9m_det_multigpu.csv
    <REPORTS_DIR>/report_yolov9c_seg_multigpu.csv
"""

import os, sys, gc, csv, time, argparse, pickle, tempfile, subprocess
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

MODEL_LABEL_DET = "YOLOv9m"
MODEL_LABEL_SEG = "YOLOv9c-Seg"

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


# ==============================================================================
# WORKERS
# ==============================================================================

def _infer_worker_det(rank: int, gpu_ids: list, pt_path: str,
                      img_dir: str, image_ids: dict, tmp_dir: str):
    from ultralytics import YOLO

    gpu = gpu_ids[rank]
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens deteksi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []
    speed_pre = []; speed_inf = []; speed_post = []

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
    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    with open(os.path.join(tmp_dir, f"det_rank{rank}.pkl"), "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf,
                     "speed_post": speed_post}, f)

    del model; _flush_gpu(gpu, f"det rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


def _infer_worker_seg(rank: int, gpu_ids: list, pt_path: str,
                      img_dir: str, image_ids: dict, tmp_dir: str):
    from ultralytics import YOLO

    gpu = gpu_ids[rank]
    torch.cuda.set_device(gpu)
    print(f"  [GPU:{gpu}] Rank {rank} mulai inferens segmentasi...", flush=True)

    model  = YOLO(pt_path)
    subset = _partition_images(img_dir, rank, len(gpu_ids))
    preds  = []
    speed_pre = []; speed_inf = []; speed_post = []

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
    elapsed = time.perf_counter() - t_start
    print(f"  [GPU:{gpu}] {len(preds)} prediksi-mask dari {len(subset)} gambar dalam {elapsed:.1f}s", flush=True)

    with open(os.path.join(tmp_dir, f"seg_rank{rank}.pkl"), "wb") as f:
        pickle.dump({"preds": preds, "elapsed": elapsed, "n_imgs": len(subset),
                     "speed_pre": speed_pre, "speed_inf": speed_inf,
                     "speed_post": speed_post}, f)

    del model; _flush_gpu(gpu, f"seg rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


# ==============================================================================
# EVALUASI
# ==============================================================================

def _collect_results(tmp_dir: str, prefix: str, world_size: int):
    """Kumpulkan prediksi + timing speed dari semua rank pickle."""
    all_preds, total_imgs = [], 0
    all_pre = []; all_inf = []; all_post = []
    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"{prefix}_rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            all_preds.extend(data["preds"])
            total_imgs += data["n_imgs"]
            all_pre.extend(data.get("speed_pre",  []))
            all_inf.extend(data.get("speed_inf",  []))
            all_post.extend(data.get("speed_post", []))

    def _avg(lst): return round(sum(lst)/len(lst), 2) if lst else "N/A"
    return all_preds, total_imgs, _avg(all_pre), _avg(all_inf), _avg(all_post)


def _calc_prec_recall(all_preds, coco_gt_dict, image_ids):
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
        cls_preds   = [p for p in all_preds if p["pred_cls"] == cls_id]
        cls_gt_cnt  = gt_per_class.get(cls_id, 0)
        total_gt   += cls_gt_cnt
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
                if pred["image"] != gi_path: continue
                x1=max(pb[0],gb[0]);y1=max(pb[1],gb[1]);x2=min(pb[2],gb[2]);y2=min(pb[3],gb[3])
                iw=max(0,x2-x1);ih=max(0,y2-y1);ia=iw*ih
                pa=(pb[2]-pb[0])*(pb[3]-pb[1]);ga=(gb[2]-gb[0])*(gb[3]-gb[1]);ua=pa+ga-ia
                iou=ia/ua if ua>0 else 0.0
                if iou > best_iou: best_iou=iou; best_key=(gi_path,gi_idx)
            if best_iou >= 0.5 and best_key not in gt_matched:
                total_tp+=1; gt_matched.add(best_key)
            else:
                total_fp+=1

    prec = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    rec  = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0
    return prec, rec


def eval_detection_distributed(gpu_ids: list, tmp_dir: str) -> dict | None:
    print("\n" + "="*65)
    print("  Distributed Eval: YOLOv9m Detection")
    print("="*65)

    pt_path = os.path.join(get_output_dir("yolov9m"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ best.pt tidak ditemukan: {pt_path}")
        send_telegram_msg(f"❌ <b>YOLOv9m MultiGPU Eval</b>\nDet best.pt tidak ditemukan:\n<code>{pt_path}</code>")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall."); return None

    print("  [GT] Membangun COCO ground truth...")
    coco_gt_dict, image_ids = build_coco_ground_truth(DET_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT"); return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(DET_YAML)
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker deteksi secara paralel...")
    t_wall_start = time.perf_counter()
    mp.spawn(_infer_worker_det,
             args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir),
             nprocs=world_size, join=True)
    total_wall = time.perf_counter() - t_wall_start

    all_preds, total_imgs, avg_pre, avg_inf, avg_post = _collect_results(tmp_dir, "det", world_size)
    print(f"  [Collect] {len(all_preds)} prediksi dari {total_imgs} gambar | Wall: {total_wall:.2f}s")
    print(f"  [Speed] Pre={avg_pre}ms | Inf={avg_inf}ms | Post={avg_post}ms (avg/gambar)")

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_ms         = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"

    print("  [COCOeval] Menghitung mAP...")
    mAP50, mAP50_95 = evaluate_coco_predictions(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    print(f"  ✅ mAP50={mAP50}  mAP50-95={mAP50_95}")

    prec, rec = _calc_prec_recall(all_preds, coco_gt_dict, image_ids)
    print(f"  ✅ Precision={prec}  Recall={rec}")

    row = {
        "Model":             MODEL_LABEL_DET,
        "Model Size (MB)":   round(os.path.getsize(pt_path)/1e6, 2),
        "mAP50-95":          mAP50_95,
        "mAP50":             mAP50,
        "Precision":         prec,
        "Recall":            rec,
        "Preprocess (ms)":   avg_pre,
        "Inference (ms)":    avg_inf,
        "Postprocess (ms)":  avg_post,
        "Latency (ms)":      lat_ms,
        "FPS":               throughput_fps,
        "GPUs":              _gpu_report_str(gpu_ids),
        "Evaluator":         "COCOeval (MultiGPU)",
    }
    print(f"\n  📊 Hasil Detection MultiGPU:")
    for k, v in row.items():
        print(f"     {k}: {v}")
    return row


def eval_segmentation_distributed(gpu_ids: list, tmp_dir: str) -> dict | None:
    print("\n" + "="*65)
    print("  Distributed Eval: YOLOv9c-Seg Segmentation")
    print("="*65)

    pt_path = os.path.join(get_output_dir("yolov9c_seg"), "weights", "best.pt")
    if not os.path.exists(pt_path):
        print(f"  ❌ best.pt tidak ditemukan: {pt_path}")
        send_telegram_msg(f"❌ <b>YOLOv9c-Seg MultiGPU Eval</b>\nSeg best.pt tidak ditemukan:\n<code>{pt_path}</code>")
        return None

    if not check_pycocotools():
        print("  ❌ pycocotools tidak terinstall."); return None

    print("  [GT] Membangun COCO ground truth segmentasi...")
    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split="valid")
    if coco_gt_dict is None:
        print("  ❌ Gagal membangun COCO GT"); return None
    print(f"  [GT] {len(image_ids)} gambar ditemukan.")

    img_dir    = _resolve_img_dir(SEG_YAML)
    world_size = len(gpu_ids)

    print(f"  [Spawn] Menjalankan {world_size} GPU worker segmentasi secara paralel...")
    t_wall_start = time.perf_counter()
    mp.spawn(_infer_worker_seg,
             args=(gpu_ids, pt_path, img_dir, image_ids, tmp_dir),
             nprocs=world_size, join=True)
    total_wall = time.perf_counter() - t_wall_start

    all_preds, total_imgs, avg_pre, avg_inf, avg_post = _collect_results(tmp_dir, "seg", world_size)
    print(f"  [Collect] {len(all_preds)} prediksi dari {total_imgs} gambar | Wall: {total_wall:.2f}s")
    print(f"  [Speed] Pre={avg_pre}ms | Inf={avg_inf}ms | Post={avg_post}ms (avg/gambar)")

    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    lat_ms         = round(total_wall * 1000 / total_imgs, 2) if total_imgs > 0 else "N/A"

    print("  [COCOeval] Menghitung mAP Mask & Box...")
    mAP50_mask, mAP50_95_mask = evaluate_coco_predictions(coco_gt_dict, image_ids, all_preds, iou_type="segm")
    _, mAP50_95_box           = evaluate_coco_predictions(coco_gt_dict, image_ids, all_preds, iou_type="bbox")
    print(f"  ✅ mAP50-95(Box)={mAP50_95_box}  mAP50-95(Mask)={mAP50_95_mask}")

    row = {
        "Model":            MODEL_LABEL_SEG,
        "Model Size (MB)":  round(os.path.getsize(pt_path)/1e6, 2),
        "mAP50-95(Box)":    mAP50_95_box,
        "mAP50-95(Mask)":   mAP50_95_mask,
        "Preprocess (ms)":  avg_pre,
        "Inference (ms)":   avg_inf,
        "Postprocess (ms)": avg_post,
        "Latency (ms)":     lat_ms,
        "FPS":              throughput_fps,
        "GPUs":             _gpu_report_str(gpu_ids),
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
    parser = argparse.ArgumentParser(description="YOLOv9 Distributed Multi-GPU Evaluation")
    parser.add_argument("--gpus", type=str, default="all",
        help="GPU yang digunakan. Contoh: '0,1' atau 'all'. Default: semua GPU.")
    parser.add_argument("--skip-det", action="store_true", help="Lewati evaluasi detection.")
    parser.add_argument("--skip-seg", action="store_true", help="Lewati evaluasi segmentation.")
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
    print("  YOLOv9 — Distributed Multi-GPU Evaluation")
    print("=" * 65)
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {len(GPU_IDS)}")
    print(f"  DET_YAML           : {DET_YAML}")
    print(f"  SEG_YAML           : {SEG_YAML}")
    print(f"  REPORTS_DIR        : {REPORTS_DIR}")
    print("=" * 65 + "\n")

    gc.collect(); torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(GPU_IDS[0])[0] / 1e9
    print(f"[MemClean] VRAM bebas GPU:{GPU_IDS[0]}: {free_gb:.2f} GB\n")

    send_telegram_msg(
        f"🚀 <b>YOLOv9 MultiGPU Eval Dimulai</b>\n"
        f"GPUs: <code>{GPU_IDS}</code>\n"
        f"World Size: {len(GPU_IDS)}"
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yolo9eval_") as tmp_dir:
        t_total_start = time.perf_counter()

        # ── Detection ─────────────────────────────────────────────────────────
        det_row = None
        if not args.skip_det:
            send_telegram_msg("🔍 <b>YOLOv9m Det Eval</b>\nMemulai distributed inference deteksi...", force=False)
            det_row = eval_detection_distributed(GPU_IDS, tmp_dir)
            if det_row:
                det_fields = [
                    "Model", "Model Size (MB)", "mAP50-95", "mAP50",
                    "Precision", "Recall", "Preprocess (ms)", "Inference (ms)",
                    "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
                ]
                det_csv = os.path.join(REPORTS_DIR, "report_yolov9m_det_multigpu.csv")
                with open(det_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=det_fields)
                    w.writeheader(); w.writerow(det_row)
                print(f"\n✅ Det Report: {det_csv}")
                send_telegram_msg(
                    f"✅ <b>YOLOv9m Det MultiGPU Selesai</b>\n"
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
            send_telegram_msg("🔍 <b>YOLOv9c-Seg Eval</b>\nMemulai distributed inference segmentasi...", force=False)
            seg_row = eval_segmentation_distributed(GPU_IDS, tmp_dir)
            if seg_row:
                seg_fields = [
                    "Model", "Model Size (MB)", "mAP50-95(Box)",
                    "mAP50-95(Mask)", "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)", "Latency (ms)", "FPS", "GPUs", "Evaluator"
                ]
                seg_csv = os.path.join(REPORTS_DIR, "report_yolov9c_seg_multigpu.csv")
                with open(seg_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=seg_fields)
                    w.writeheader(); w.writerow(seg_row)
                print(f"✅ Seg Report: {seg_csv}")
                send_telegram_msg(
                    f"✅ <b>YOLOv9c-Seg MultiGPU Selesai</b>\n"
                    f"mAP50-95(Box): <code>{seg_row['mAP50-95(Box)']}</code>\n"
                    f"mAP50-95(Mask): <code>{seg_row['mAP50-95(Mask)']}</code>\n"
                    f"FPS (Throughput): <code>{seg_row['FPS']}</code>\n"
                    f"GPUs: <code>{seg_row['GPUs']}</code>"
                )

        # ------ Generate Comparison Grid ------
        print("\n" + "="*65 + "\n  Generating Comparison Grid\n" + "="*65)
        try:
            subprocess.run([sys.executable, "-u", os.path.join(ROOT, "utils", "generate_comparison_grid.py")], check=False)
        except Exception as e:
            print(f"⚠️ Gagal memanggil generate_comparison_grid: {e}")

        total_elapsed = round(time.perf_counter() - t_total_start, 1)
        print(f"\n✅ YOLOv9 MultiGPU Evaluation selesai dalam {total_elapsed}s")
        send_telegram_msg(
            f"🏁 <b>YOLOv9 MultiGPU Eval Selesai</b>\n"
            f"Total waktu: <code>{total_elapsed}s</code>\n"
            f"Report: <code>{REPORTS_DIR}</code>"
        )
