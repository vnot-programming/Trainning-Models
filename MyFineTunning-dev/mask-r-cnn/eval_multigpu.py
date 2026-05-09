# -*- coding: utf-8 -*-
"""
mask-r-cnn/eval_multigpu.py
============================
Distributed Multi-GPU Evaluation untuk Mask R-CNN ResNet-50-FPN-v2.

Strategi evaluasi terdistribusi:
  - mp.spawn: setiap GPU memproses subset gambar secara paralel
  - Inferens: load model yang SAMA di setiap GPU (model parallelism = False,
    data parallelism = True) — ini adalah cara yang benar untuk eval
  - Prediksi (bbox + mask RLE) dikumpulkan dari file pickle per-rank
  - COCOeval dijalankan di proses utama setelah semua rank selesai
  - Latency diukur per-rank lalu dirata-ratakan; FPS = throughput terdistribusi

PENTING: Beda dengan train_multigpu.py yang menggunakan DDP (gradient sync),
  script ini TIDAK perlu dist.init_process_group karena tidak ada backward pass.

Cara menjalankan:
    python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log
    python -u eval_multigpu.py --gpus 0,1 2>&1 | tee eval_multigpu.log

    # tmux background:
    tmux new-session -d -s maskeval "source ../.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-dev/mask-r-cnn && \\
      python -u eval_multigpu.py 2>&1 | tee eval_multigpu.log"

Output:
    <REPORTS_DIR>/report_maskrcnn_ddp_seg_multigpu.csv
"""

import os, sys, gc, csv, time, argparse, pickle, tempfile
import numpy as np

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_SCRIPT_DIR      = os.path.abspath(os.path.dirname(__file__))
_FINETUNING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _FINETUNING_ROOT)

import torch
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

from config_shared import (
    WORKSPACE_DIR, SEG_DATASET_LOCATION, REPORTS_DIR,
    NUM_CLASSES, MODEL_COLORS
)
from telegram_utils import send_telegram_msg
from maskrcnn_builder import build_model

MODEL_LABEL = "Mask R-CNN ResNet-50 FPN-v2"
OUTPUT_KEY  = "maskrcnn"

# ==============================================================================
# HELPERS
# ==============================================================================

def _flush_gpu(gpu_id: int, label: str):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    free, total = torch.cuda.mem_get_info(gpu_id)
    print(f"  [GPU:{gpu_id}][MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)


def _gpu_report_str(gpu_ids: list) -> str:
    from collections import Counter
    names  = [torch.cuda.get_device_name(i) for i in gpu_ids]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())


def _resolve_eval_dirs() -> tuple:
    """Cari direktori evaluasi (test/ atau valid/)."""
    for split in ("test", "valid"):
        img_dir = os.path.join(SEG_DATASET_LOCATION, split, "images")
        lbl_dir = os.path.join(SEG_DATASET_LOCATION, split, "labels")
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            return img_dir, lbl_dir, split
    raise FileNotFoundError(f"Tidak ditemukan test/ atau valid/ di {SEG_DATASET_LOCATION}")


def _get_best_pt() -> str:
    """Temukan best.pt Mask R-CNN di workspace aktif."""
    best_pt = os.path.join(WORKSPACE_DIR, "runs", OUTPUT_KEY, "weights", "best.pt")
    return best_pt


def _partition_files(file_list: list, rank: int, world_size: int) -> list:
    return file_list[rank::world_size]


# ==============================================================================
# COCO GROUND TRUTH BUILDER (dari YOLO polygon labels)
# ==============================================================================

CLASS_NAMES = ["dishwasher", "milk", "mineral", "non_mineral",
               "not_empty", "soda", "yogurt"]

def _build_coco_gt(img_dir: str, lbl_dir: str, img_files: list) -> tuple:
    """Bangun struktur COCO ground truth dari YOLO polygon labels."""
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as maskUtils
    except ImportError:
        print("  ❌ pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return None, None

    coco_gt_dict = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": i+1, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    ann_id    = 1
    img_id_map = {}

    for img_id, fname in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, fname)
        pil      = PILImage.open(img_path).convert("RGB")
        W, H     = pil.size
        base     = os.path.splitext(fname)[0]

        coco_gt_dict["images"].append({
            "id": img_id, "file_name": fname, "width": W, "height": H
        })
        img_id_map[img_path] = img_id

        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0]) + 1   # 0-indexed → 1-indexed COCO
                coords = list(map(float, parts[1:]))
                poly   = [c * W if i % 2 == 0 else c * H
                          for i, c in enumerate(coords)]
                if len(poly) < 6: continue

                xs = poly[0::2]; ys = poly[1::2]
                x1c, y1c = min(xs), min(ys)
                bw, bh   = max(xs)-x1c, max(ys)-y1c

                rle = maskUtils.frPyObjects([poly], H, W)
                rle = maskUtils.merge(rle)
                area = float(maskUtils.area(rle))

                coco_gt_dict["annotations"].append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cls_id,
                    "segmentation": rle,
                    "bbox":        [x1c, y1c, bw, bh],
                    "area":        area,
                    "iscrowd":     0,
                })
                ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    print(f"  [GT] {len(coco_gt_dict['images'])} gambar | {len(coco_gt_dict['annotations'])} anotasi")
    return coco_gt, img_id_map


# ==============================================================================
# WORKER: INFERENCE PER GPU
# ==============================================================================

def _infer_worker(rank: int, gpu_ids: list, best_pt: str,
                  img_dir: str, lbl_dir: str, img_files: list,
                  img_id_map: dict, tmp_dir: str):
    """
    Worker evaluasi: setiap GPU inferens subset gambar, simpan prediksi ke pickle.
    Prediksi format: {image_id, category_id, bbox [xywh], score, segmentation RLE}
    """
    from pycocotools import mask as maskUtils
    import cv2

    gpu    = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    print(f"  [GPU:{gpu}] Rank {rank} loading Mask R-CNN model...", flush=True)
    model = build_model(device)
    model.load_state_dict(torch.load(best_pt, map_location=device))
    model.eval()
    print(f"  [GPU:{gpu}] Rank {rank} model loaded. Mulai inferens...", flush=True)

    subset    = _partition_files(img_files, rank, len(gpu_ids))
    dt_bbox   = []   # COCO detection results
    dt_segm   = []   # COCO segmentation results
    times     = []   # Latency per image (ms)

    with torch.no_grad():
        for fname in subset:
            img_path = os.path.join(img_dir, fname)
            img_id   = img_id_map.get(img_path)
            if img_id is None:
                continue

            pil   = PILImage.open(img_path).convert("RGB")
            W, H  = pil.size
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device)

            t0    = time.perf_counter()
            preds = model(t_img)[0]
            times.append((time.perf_counter() - t0) * 1000)

            boxes  = preds["boxes"].cpu().numpy()    # (N, 4) xyxy
            scores = preds["scores"].cpu().numpy()   # (N,)
            labels = preds["labels"].cpu().numpy()   # (N,) 1-indexed
            masks  = preds["masks"].cpu().numpy()    # (N, 1, H, W)

            for i in range(len(scores)):
                score  = float(scores[i])
                cat_id = int(labels[i])   # Sudah 1-indexed
                x1, y1, x2, y2 = boxes[i].tolist()
                bw, bh = x2-x1, y2-y1

                dt_bbox.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, bw, bh],
                    "score":       score,
                })

                # Konversi mask float → binary → RLE
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                if bin_mask.shape != (H, W):
                    import cv2 as _cv2
                    bin_mask = _cv2.resize(bin_mask, (W, H),
                                           interpolation=_cv2.INTER_NEAREST)
                rle = maskUtils.encode(np.asfortranarray(bin_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                dt_segm.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":       score,
                })

    avg_lat = round(sum(times) / len(times), 2) if times else "N/A"
    print(f"  [GPU:{gpu}] {len(dt_bbox)} prediksi dari {len(subset)} gambar | "
          f"Avg latency: {avg_lat}ms", flush=True)

    with open(os.path.join(tmp_dir, f"rank{rank}.pkl"), "wb") as f:
        pickle.dump({
            "dt_bbox": dt_bbox,
            "dt_segm": dt_segm,
            "times":   times,
            "n_imgs":  len(subset),
        }, f)

    del model; _flush_gpu(gpu, f"infer rank{rank}")
    print(f"  [GPU:{gpu}] Rank {rank} selesai.", flush=True)


# ==============================================================================
# EVALUASI UTAMA
# ==============================================================================

def eval_maskrcnn_distributed(gpu_ids: list, tmp_dir: str) -> dict | None:
    print("\n" + "="*65)
    print("  Distributed Eval: Mask R-CNN ResNet-50 FPN-v2")
    print("="*65)

    best_pt = _get_best_pt()
    if not os.path.exists(best_pt):
        print(f"  ❌ best.pt tidak ditemukan: {best_pt}")
        print("  💡 Jalankan mask-r-cnn/train_multigpu.py terlebih dahulu.")
        send_telegram_msg(
            f"❌ <b>Mask R-CNN MultiGPU Eval</b>\n"
            f"best.pt tidak ditemukan:\n<code>{best_pt}</code>"
        )
        return None

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  ❌ pycocotools tidak terinstall."); return None

    # Resolve dataset dirs
    try:
        img_dir, lbl_dir, split_name = _resolve_eval_dirs()
    except FileNotFoundError as e:
        print(f"  ❌ {e}"); return None

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not img_files:
        print("  ❌ Tidak ada gambar di direktori evaluasi."); return None

    print(f"  [Dataset] Split: {split_name} | Jumlah gambar: {len(img_files)}")

    # Build COCO GT di proses utama
    print("  [GT] Membangun COCO ground truth...")
    coco_gt, img_id_map = _build_coco_gt(img_dir, lbl_dir, img_files)
    if coco_gt is None:
        return None

    world_size = len(gpu_ids)
    print(f"  [Spawn] Menjalankan {world_size} GPU worker secara paralel...")

    t_wall_start = time.perf_counter()
    mp.spawn(
        _infer_worker,
        args=(gpu_ids, best_pt, img_dir, lbl_dir, img_files, img_id_map, tmp_dir),
        nprocs=world_size,
        join=True,
    )
    total_wall = time.perf_counter() - t_wall_start

    # Kumpulkan hasil
    all_dt_bbox  = []
    all_dt_segm  = []
    all_times    = []
    total_imgs   = 0
    for r in range(world_size):
        pkl = os.path.join(tmp_dir, f"rank{r}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            all_dt_bbox.extend(data["dt_bbox"])
            all_dt_segm.extend(data["dt_segm"])
            all_times.extend(data["times"])
            total_imgs += data["n_imgs"]

    print(f"\n  [Collect] {len(all_dt_bbox)} prediksi dari {total_imgs} gambar")
    print(f"  [Throughput] Wall clock: {total_wall:.2f}s")

    if not all_dt_bbox:
        print("  ❌ Tidak ada prediksi. Cek threshold confidence atau model.")
        return None

    # Throughput & latency
    throughput_fps = round(total_imgs / total_wall, 2) if total_wall > 0 else "N/A"
    avg_lat_ms     = round(sum(all_times) / len(all_times), 2) if all_times else "N/A"

    # COCOeval — BBox
    print("\n  [COCOeval] ── BBox ──")
    coco_dt_box = coco_gt.loadRes(all_dt_bbox)
    eval_box    = COCOeval(coco_gt, coco_dt_box, iouType="bbox")
    eval_box.evaluate(); eval_box.accumulate(); eval_box.summarize()
    map_box = round(float(eval_box.stats[0]), 4)

    # COCOeval — Segm
    print("\n  [COCOeval] ── Segm ──")
    coco_dt_seg = coco_gt.loadRes(all_dt_segm)
    eval_seg    = COCOeval(coco_gt, coco_dt_seg, iouType="segm")
    eval_seg.evaluate(); eval_seg.accumulate(); eval_seg.summarize()
    map_mask = round(float(eval_seg.stats[0]), 4)

    print(f"\n  ✅ mAP50-95(Box)={map_box}  mAP50-95(Mask)={map_mask}")
    print(f"  ✅ Avg Latency per gambar: {avg_lat_ms}ms | FPS (Throughput): {throughput_fps}")

    size_mb = round(os.path.getsize(best_pt) / 1e6, 2)
    gpu_str = _gpu_report_str(gpu_ids)

    row = {
        "Model":            MODEL_LABEL,
        "Model Size (MB)":  size_mb,
        "mAP50-95(Box)":    map_box,
        "mAP50-95(Mask)":   map_mask,
        "Latency (ms)":     avg_lat_ms,
        "FPS":              throughput_fps,
        "GPUs":             gpu_str,
        "Evaluator":        "COCOeval (MultiGPU)",
    }
    print(f"\n  📊 Hasil Mask R-CNN MultiGPU:")
    for k, v in row.items():
        print(f"     {k}: {v}")
    return row


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN Distributed Multi-GPU Evaluation"
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
    print("  Mask R-CNN — Distributed Multi-GPU Evaluation")
    print("=" * 65)
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {len(GPU_IDS)}")
    print(f"  SEG_DATASET        : {SEG_DATASET_LOCATION}")
    print(f"  REPORTS_DIR        : {REPORTS_DIR}")
    print(f"  WORKSPACE_DIR      : {WORKSPACE_DIR}")
    print("=" * 65 + "\n")

    best_pt_info = _get_best_pt()
    print(f"  Best model target  : {best_pt_info}\n")

    gc.collect(); torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(GPU_IDS[0])[0] / 1e9
    print(f"[MemClean] VRAM bebas GPU:{GPU_IDS[0]}: {free_gb:.2f} GB\n")

    send_telegram_msg(
        f"🚀 <b>Mask R-CNN MultiGPU Eval Dimulai</b>\n"
        f"GPUs: <code>{GPU_IDS}</code>\n"
        f"World Size: {len(GPU_IDS)}\n"
        f"Model: <code>{best_pt_info}</code>"
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="maskrcnn_eval_") as tmp_dir:
        t_total_start = time.perf_counter()

        row = eval_maskrcnn_distributed(GPU_IDS, tmp_dir)

        if row:
            fields   = ["Model", "Model Size (MB)", "mAP50-95(Box)",
                        "mAP50-95(Mask)", "Latency (ms)", "FPS", "GPUs", "Evaluator"]
            csv_path = os.path.join(REPORTS_DIR, "report_maskrcnn_ddp_seg_multigpu.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerow(row)
            print(f"\n✅ Report: {csv_path}")
            send_telegram_msg(
                f"✅ <b>Mask R-CNN MultiGPU Eval Selesai</b>\n"
                f"mAP50-95(Box):  <code>{row['mAP50-95(Box)']}</code>\n"
                f"mAP50-95(Mask): <code>{row['mAP50-95(Mask)']}</code>\n"
                f"Avg Latency:    <code>{row['Latency (ms)']}ms</code>\n"
                f"FPS (Throughput): <code>{row['FPS']}</code>\n"
                f"GPUs: <code>{row['GPUs']}</code>\n"
                f"Report: <code>{csv_path}</code>"
            )
        else:
            print("\n❌ Evaluasi gagal — tidak ada report yang disimpan.")
            send_telegram_msg("❌ <b>Mask R-CNN MultiGPU Eval GAGAL</b>\nCek log untuk detail error.")

        total_elapsed = round(time.perf_counter() - t_total_start, 1)
        print(f"\n✅ Total waktu evaluasi: {total_elapsed}s")
        send_telegram_msg(
            f"🏁 <b>Mask R-CNN MultiGPU Eval Selesai</b>\n"
            f"Total waktu: <code>{total_elapsed}s</code>"
        )
