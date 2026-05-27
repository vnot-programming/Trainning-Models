#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask-r-cnn/train_multigpu.py
============================
Fine-tuning Mask R-CNN dengan multi-GPU menggunakan DistributedDataParallel (DDP).

Kenapa DDP dan bukan DataParallel?
  - DataParallel gather() scalar-dict losses semua ke GPU-0 → OOM
  - DDP: setiap proses hitung loss sendiri, sync gradient via all-reduce →
    aman untuk Mask R-CNN (tidak ada gather bottleneck)

GPU target: cuda:1 dan cuda:2  (ubah --gpus jika perlu)

Cara menjalankan:
    cd /root/MyFineTunning/mask-r-cnn
    python train_multigpu.py --gpus 1,2 2>&1 | tee train_multigpu.log

    # Atau GPU lain (single GPU):
    python train_multigpu.py --gpus 1

Output:
    /workspace/MyFineTunning-20260423_104653/runs/maskrcnn_multigpu/weights/best.pt
"""

import os, sys, gc, csv, time, random, argparse

# ─── CRITICAL: matikan dynamo SEBELUM torch di-import ────────────────────────
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── Batasi thread per proses agar tidak crash saat DDP multi-GPU ─────────────
# Tanpa ini: 5 GPU × 32 workers × 96 OpenBLAS threads = ~15.000+ threads → crash
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS",   "1")
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch._dynamo
torch._dynamo.disable()

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR

# Path setup — agar bisa import dari MyTrainEngine dan MyFineTunning root
_SCRIPT_DIR    = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_TRAIN_ENGINE  = os.environ.get("TRAIN_ENGINE_PATH", "/root/MyTrainEngine")
sys.path.insert(0, ROOT)
sys.path.insert(0, _TRAIN_ENGINE)

# ==============================================================================
# KONFIGURASI — Ubah sesuai kebutuhan
# ==============================================================================
# Import from config_shared
from config_shared import (
    WORKSPACE_DIR, SEG_DATASET_LOCATION, REPORTS_DIR, VISUALS_DIR,
    IMAGE_SAMPLES_DIR, MODEL_COLORS, NUM_CLASSES, EPOCHS,
    MASKRCNN_BATCH_SIZE, NUM_WORKERS, compress_run, EARLY_STOPPING_PATIENCE
)
from telegram_utils import send_telegram_msg

BATCH_SIZE_PER_GPU  = MASKRCNN_BATCH_SIZE
LR                  = 0.005
LR_STEP             = 5
LR_GAMMA            = 0.5
CONF_THRESH_VISUAL  = 0.5
N_VISUAL_SAMPLES    = 5
LOG_EVERY_N_BATCH   = 5

OUTPUT_KEY     = "maskrcnn"
OUTPUT_DIR     = os.path.join(WORKSPACE_DIR, "runs", OUTPUT_KEY, "weights")
REPORT_DIR     = REPORTS_DIR
VISUAL_DIR     = VISUALS_DIR

# Nama kelas (urutan sesuai data.yaml)
CLASS_NAMES = ["dishwasher", "milk", "mineral", "non_mineral",
               "not_empty", "soda", "yogurt"]

# ==============================================================================
# ARGPARSE
# ==============================================================================


# ==============================================================================
# DATASET
# ==============================================================================
from maskrcnn_dataset import RoboflowSegToMaskRCNN, collate_fn


def build_dataloaders(rank: int, world_size: int):
    """Buat DataLoader dengan DistributedSampler untuk proses DDP rank `rank`."""
    # Auto-scale workers: total workers = world_size × workers_per_gpu
    # Batasi agar total tidak melebihi CPU cores dan tidak meledakkan thread
    cpu_count = os.cpu_count() or 4
    max_workers_per_gpu = max(1, min(NUM_WORKERS, cpu_count // world_size, 8))
    if rank == 0 and max_workers_per_gpu != NUM_WORKERS:
        print(f"[DataLoader] NUM_WORKERS auto-scaled: {NUM_WORKERS} → {max_workers_per_gpu}/GPU "
              f"(total: {max_workers_per_gpu * world_size}, CPUs: {cpu_count})")

    train_ds = RoboflowSegToMaskRCNN(
        images_dir=os.path.join(SEG_DATASET_LOCATION, "train", "images"),
        labels_dir=os.path.join(SEG_DATASET_LOCATION, "train", "labels"),
    )
    val_ds = RoboflowSegToMaskRCNN(
        images_dir=os.path.join(SEG_DATASET_LOCATION, "valid", "images"),
        labels_dir=os.path.join(SEG_DATASET_LOCATION, "valid", "labels"),
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=max_workers_per_gpu,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        num_workers=max_workers_per_gpu,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if rank == 0:
        print(f"[DataLoader] Train: {len(train_ds)} gambar | Val: {len(val_ds)} gambar")
    return train_loader, val_loader, train_sampler


# ==============================================================================
# MODEL
# ==============================================================================
def build_model(device: torch.device) -> torch.nn.Module:
    """Bangun Mask R-CNN ResNet-50-FPN-v2 dengan head kustom."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )

    # Ganti box head
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)

    # Ganti mask head
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)

    model = model.to(device)
    return model


# ==============================================================================
# TRAIN SATU EPOCH
# ==============================================================================
def run_train_epoch(model, loader, optimizer, device, epoch, total_epochs, rank):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if rank == 0 and (batch_idx + 1) % LOG_EVERY_N_BATCH == 0:
            print(
                f"  Epoch [{epoch}/{total_epochs}]  "
                f"Batch [{batch_idx + 1}/{len(loader)}]  "
                f"Loss: {losses.item():.4f}",
                flush=True
            )
            # Heartbeat notification (10-min interval via force=False)
            msg = (f"📈 <b>Progress Update</b> (Mask R-CNN)\n"
                   f"Epoch: {epoch}/{total_epochs}\n"
                   f"Batch: {batch_idx + 1}/{len(loader)}\n"
                   f"Loss: {losses.item():.4f}")
            send_telegram_msg(msg, force=False)

    return total_loss / max(len(loader), 1)


# ==============================================================================
# VALIDASI SATU EPOCH
# ==============================================================================
def run_val_epoch(model, loader, device):
    model.train()   # mode train agar loss dict tersedia
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss  += sum(loss for loss in loss_dict.values()).item()

    return val_loss / max(len(loader), 1)


# ==============================================================================
# SAVE BEST MODEL (hanya dari rank 0)
# ==============================================================================
def save_best_model(model, path: str, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ambil state_dict dari dalam DDP wrapper
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(state, path)
    print(f"  ✅ Best model disimpan (val_loss={val_loss:.4f}) → {path}")


def save_checkpoint(epoch: int, model, optimizer, scheduler,
                    val_loss: float, path: str) -> None:
    """
    Simpan checkpoint training (last.pt) agar bisa dilanjutkan jika terputus.
    Hanya dipanggil dari rank 0.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        "epoch":           epoch,
        "model_state":     state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":        val_loss,
    }, path)
    print(f"  💾 Checkpoint disimpan: epoch {epoch} → {path}")


# ==============================================================================
# TRAINING WORKER (1 proses per GPU)
# ==============================================================================
def train_worker(local_rank: int, gpu_ids: list, best_pt_path: str, world_size: int):
    """
    Dijalankan oleh setiap proses DDP.

    Parameters
    ----------
    local_rank : int
        Index dalam gpu_ids (0 = GPU pertama dalam list, dst.)
    gpu_ids : list[int]
        Daftar GPU index aktual (misal [1, 2])
    best_pt_path : str
        Path untuk menyimpan best.pt
    """
    global_gpu = gpu_ids[local_rank]
    device     = torch.device(f"cuda:{global_gpu}")
    last_pt    = os.path.join(os.path.dirname(best_pt_path), "last.pt")
    checkpoint_pt = os.path.join(os.path.dirname(best_pt_path), "last_checkpoint.pt")

    # ── Init proses group DDP ────────────────────────────────────────────────
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(global_gpu)

    if local_rank == 0:
        free, total = torch.cuda.mem_get_info(global_gpu)
        print(f"[GPU {global_gpu}] VRAM: {free/1e9:.2f} GB bebas / {total/1e9:.2f} GB total")

    # ── Build model & wrap DDP ───────────────────────────────────────────────
    model = build_model(device)
    model = DDP(model, device_ids=[global_gpu], output_device=global_gpu)

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Mask R-CNN] Parameter trainable: {n_params:,}")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_dataloaders(local_rank, world_size)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch   = 1

    # ── Resume dari checkpoint jika ada ─────────────────────────────────────
    if os.path.exists(checkpoint_pt):
        if local_rank == 0:
            print(f"\n[RESUME] Melanjutkan dari checkpoint: {checkpoint_pt}")
        ckpt = torch.load(checkpoint_pt, map_location=device)
        # Load model state ke DDP module
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        if local_rank == 0:
            print(f"  ✅ Melanjutkan dari epoch {start_epoch}/{EPOCHS}  "
                  f"(best_val_loss={best_val_loss:.4f})")

    # ── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        train_sampler.set_epoch(epoch)   # Penting untuk shuffling DDP

        avg_train = run_train_epoch(
            model, train_loader, optimizer, device, epoch, EPOCHS, local_rank
        )
        avg_val = run_val_epoch(model, val_loader, device)
        scheduler.step()

        # Sync val loss antar GPU (ambil rata-rata)
        val_tensor = torch.tensor(avg_val, device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
        avg_val_sync = val_tensor.item()

        if local_rank == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}]  "
                f"Train Loss: {avg_train:.4f} | "
                f"Val Loss: {avg_val_sync:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            # Simpan checkpoint setiap epoch (last.pt full checkpoint)
            save_checkpoint(epoch, model, optimizer, scheduler,
                            avg_val_sync, checkpoint_pt)
            # Simpan last.pt (weights only, untuk eval cepat)
            save_best_model(model, last_pt, avg_val_sync)

            if avg_val_sync < best_val_loss:
                best_val_loss = avg_val_sync
                patience_counter = 0
                save_best_model(model, best_pt_path, best_val_loss)
            else:
                patience_counter += 1
                print(f"  [EarlyStopping] Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"  [EarlyStopping] Patience threshold reached on rank 0!")

        # Sync stop flag across all processes in DDP group
        stop_tensor = torch.tensor(1 if patience_counter >= EARLY_STOPPING_PATIENCE else 0, device=device)
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        if stop_tensor.item() == 1:
            if local_rank == 0:
                print(f"  [EarlyStopping] Synchronized early stopping triggered across all GPUs at epoch {epoch}.")
            break

    if local_rank == 0:
        print(f"\n✅ Training selesai. Best model: {best_pt_path}")

    dist.destroy_process_group()


# ==============================================================================
# EVALUASI LATENCY (single GPU, setelah training)
# ==============================================================================
def measure_latency(best_pt: str, device: torch.device) -> tuple:
    """Ukur rata-rata latency dari 5 sampel gambar test."""
    img_dir = os.path.join(SEG_DATASET_LOCATION, "test", "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(SEG_DATASET_LOCATION, "valid", "images")

    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    samples  = random.sample(all_imgs, min(N_VISUAL_SAMPLES, len(all_imgs)))

    # Load model untuk eval
    eval_model = build_model(device)
    eval_model.load_state_dict(torch.load(best_pt, map_location=device))
    eval_model.eval()

    from PIL import Image as _Img
    import torchvision.transforms.functional as _TF

    times = []
    print("\n[Eval] Mengukur latency Mask R-CNN ...")
    for img_path in samples:
        pil   = _Img.open(img_path).convert("RGB")
        t_img = _TF.to_tensor(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            t0 = time.perf_counter()
            eval_model(t_img)
            times.append((time.perf_counter() - t0) * 1000)

    del eval_model; gc.collect(); torch.cuda.empty_cache()

    if times:
        lat_ms  = round(sum(times) / len(times), 2)
        fps_val = round(1000 / lat_ms, 2)
        print(f"[Eval] Latency: {lat_ms} ms ({fps_val} FPS)")
        return lat_ms, fps_val
    return "N/A", "N/A"


# ==============================================================================
# EVALUASI mAP — COCO-style (pycocotools)
# ==============================================================================
def evaluate_map(best_pt: str, device: torch.device) -> tuple:
    """
    Hitung mAP50-95 (Box) dan mAP50-95 (Mask) menggunakan COCO API.

    Alur:
      1. Baca ground-truth dari label YOLO-polygon (txt) → konversi ke COCO format
      2. Jalankan model pada seluruh gambar test/valid → kumpulkan prediksi
      3. Jalankan COCOeval untuk task 'bbox' dan 'segm'

    Returns
    -------
    (map_box, map_mask) : tuple[float | str, float | str]
        mAP50-95 untuk bbox dan segm. Nilai "N/A" jika pycocotools tidak ada.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools import mask as maskUtils
    except ImportError:
        print("[mAP] ⚠️  pycocotools tidak terinstall. "
              "Jalankan: pip install pycocotools")
        return "N/A", "N/A"

    import numpy as np
    from PIL import Image as _Img
    import torchvision.transforms.functional as _TF

    # ── Tentukan split evaluasi ───────────────────────────────────────────────
    eval_split = "test"
    img_dir    = os.path.join(SEG_DATASET_LOCATION, eval_split, "images")
    lbl_dir    = os.path.join(SEG_DATASET_LOCATION, eval_split, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        eval_split = "valid"
        img_dir    = os.path.join(SEG_DATASET_LOCATION, eval_split, "images")
        lbl_dir    = os.path.join(SEG_DATASET_LOCATION, eval_split, "labels")

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not img_files:
        print("[mAP] ⚠️  Tidak ada gambar untuk evaluasi.")
        return "N/A", "N/A"

    print(f"\n[mAP] Evaluasi {len(img_files)} gambar dari split '{eval_split}' ...")

    # ── Bangun struktur COCO ground-truth ────────────────────────────────────
    coco_gt_dict = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": i + 1, "name": n}
                        for i, n in enumerate(CLASS_NAMES)],
    }
    ann_id    = 1
    img_id_map = {}   # filename (no ext) → image_id

    for img_id, fname in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, fname)
        pil      = _Img.open(img_path).convert("RGB")
        W, H     = pil.size
        base     = os.path.splitext(fname)[0]

        coco_gt_dict["images"].append({
            "id": img_id, "file_name": fname, "width": W, "height": H
        })
        img_id_map[base] = img_id

        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0]) + 1   # YOLO 0-indexed → COCO 1-indexed
                coords = list(map(float, parts[1:]))
                # Denormalisasi koordinat polygon
                poly = []
                for k in range(0, len(coords), 2):
                    poly.extend([coords[k] * W, coords[k + 1] * H])
                if len(poly) < 6:
                    continue   # Polygon minimal 3 titik

                # Hitung bbox dari polygon
                xs = poly[0::2]; ys = poly[1::2]
                x1c, y1c = min(xs), min(ys)
                bw, bh   = max(xs) - x1c, max(ys) - y1c

                # Konversi polygon ke RLE untuk COCOeval segm
                rle = maskUtils.frPyObjects([poly], H, W)
                rle = maskUtils.merge(rle)
                area = float(maskUtils.area(rle))

                coco_gt_dict["annotations"].append({
                    "id":            ann_id,
                    "image_id":      img_id,
                    "category_id":   cls_id,
                    "segmentation":  rle,
                    "bbox":          [x1c, y1c, bw, bh],
                    "area":          area,
                    "iscrowd":       0,
                })
                ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # ── Load model ───────────────────────────────────────────────────────────
    eval_model = build_model(device)
    eval_model.load_state_dict(torch.load(best_pt, map_location=device))
    eval_model.eval()

    # ── Jalankan prediksi & kumpulkan hasil ──────────────────────────────────
    dt_bbox = []   # detection results bbox
    dt_segm = []   # detection results segm

    with torch.no_grad():
        for fname in img_files:
            base   = os.path.splitext(fname)[0]
            img_id = img_id_map[base]

            img_path = os.path.join(img_dir, fname)
            pil      = _Img.open(img_path).convert("RGB")
            W, H     = pil.size
            t_img    = _TF.to_tensor(pil).unsqueeze(0).to(device)

            preds   = eval_model(t_img)[0]
            boxes   = preds["boxes"].cpu().numpy()    # (N, 4) xyxy
            scores  = preds["scores"].cpu().numpy()   # (N,)
            labels  = preds["labels"].cpu().numpy()   # (N,) 1-indexed
            masks   = preds["masks"].cpu().numpy()    # (N, 1, H, W) float

            for i in range(len(scores)):
                score   = float(scores[i])
                cat_id  = int(labels[i])   # sudah 1-indexed (Mask R-CNN)
                x1, y1, x2, y2 = boxes[i].tolist()
                bw, bh  = x2 - x1, y2 - y1

                # BBox result
                dt_bbox.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, bw, bh],   # COCO: xywh
                    "score":       score,
                })

                # Mask result — konversi ke RLE
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                if bin_mask.shape != (H, W):
                    import cv2 as _cv2
                    bin_mask = _cv2.resize(
                        bin_mask, (W, H), interpolation=cv2.INTER_NEAREST
                    )
                rle = maskUtils.encode(np.asfortranarray(bin_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                dt_segm.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":        score,
                })

    del eval_model; gc.collect(); torch.cuda.empty_cache()

    if not dt_bbox:
        print("[mAP] ⚠️  Tidak ada prediksi — pastikan conf_thresh tidak terlalu tinggi.")
        return "N/A", "N/A"

    # ── COCOeval ─────────────────────────────────────────────────────────────
    def _run_coco_eval(dt_list: list, iou_type: str) -> float:
        coco_dt  = coco_gt.loadRes(dt_list)
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return round(float(evaluator.stats[0]), 4)   # stats[0] = mAP@.50:.95

    print("\n[mAP] ── BBox ──")
    map_box  = _run_coco_eval(dt_bbox, "bbox")
    print("\n[mAP] ── Segm ──")
    map_mask = _run_coco_eval(dt_segm, "segm")

    print(f"\n[mAP] mAP50-95(Box)  = {map_box}")
    print(f"[mAP] mAP50-95(Mask) = {map_mask}")
    return map_box, map_mask


# ==============================================================================
# VISUALISASI (single GPU, setelah training)
# ==============================================================================
def run_visualization(best_pt: str, device: torch.device) -> None:
    """Render 5 sampel gambar dengan mask + label kelas yang benar."""
    import cv2, numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as _Img
    import torchvision.transforms.functional as _TF

    os.makedirs(VISUAL_DIR, exist_ok=True)

    target_img_dir = IMAGE_SAMPLES_DIR
    if not os.path.isdir(target_img_dir):
        print(f"[Visual] ⚠️  IMAGE_SAMPLES_DIR tidak ditemukan: {target_img_dir}")
        return

    all_imgs = sorted([os.path.join(target_img_dir, f) for f in os.listdir(target_img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    samples  = all_imgs[:10]
    
    if not samples:
        print(f"[Visual] ⚠️  Tidak ada gambar di: {target_img_dir}")
        return

    vis_model = build_model(device)
    vis_model.load_state_dict(torch.load(best_pt, map_location=device))
    vis_model.eval()

    theme_color = MODEL_COLORS.get("maskrcnn", (255, 0, 255))

    print(f"\n[Visual] Mask R-CNN DDP — {len(samples)} sampel")
    for idx, img_path in enumerate(samples, 1):
        base    = os.path.splitext(os.path.basename(img_path))[0]
        bgr     = cv2.imread(img_path)
        pil     = _Img.open(img_path).convert("RGB")
        t_img   = _TF.to_tensor(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = vis_model(t_img)[0]

        overlay = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        H, W    = overlay.shape[:2]

        masks   = preds.get("masks",  torch.zeros(0, 1, H, W))
        boxes   = preds.get("boxes",  torch.zeros(0, 4))
        scores  = preds.get("scores", torch.zeros(0))
        labels  = preds.get("labels", torch.zeros(0, dtype=torch.long))

        for i, (mask, box, score, lbl) in enumerate(zip(masks, boxes, scores, labels)):
            if score.item() < CONF_THRESH_VISUAL:
                continue

            cls_id   = int(lbl.item())
            # Mask R-CNN: label 0 = background, kelas mulai dari 1
            cls_name = CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= NUM_CLASSES else f"cls{cls_id}"
            conf_str = f"{score.item():.2f}"

            # Overlay mask semi-transparan
            m = mask[0].cpu().numpy() > 0.5
            if m.shape != (H, W):
                import cv2 as _cv2
                m = _cv2.resize(m.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            colored = np.zeros_like(overlay)
            colored[m] = theme_color
            cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)

            # Bounding box
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_color, 2)

            # Label teks dengan nama kelas
            label_txt = f"{cls_name} {conf_str}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), theme_color, -1)
            cv2.putText(overlay, label_txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(VISUAL_DIR, f"maskrcnn_ddp_{idx:02d}_{base}.png")
        cv2.imwrite(out_path, overlay)
        print(f"  [{idx}/{len(samples)}] → {out_path}")

    del vis_model; gc.collect(); torch.cuda.empty_cache()


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    # ── Parse Argumen ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Mask R-CNN Multi-GPU DDP Training")
    parser.add_argument(
        "--gpus", type=str, default="all",
        help="GPU index yang digunakan, pisahkan koma. Default: 'all' (semua GPU). "
             "Contoh: '1,2' untuk GPU 1 dan 2, '0' untuk single GPU 0."
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation phase after training"
    )
    args = parser.parse_args()

    # Auto-detect semua GPU jika --gpus tidak dispesifikasi atau "all"
    if args.gpus.strip().lower() == "all":
        n_gpus = torch.cuda.device_count()
        GPU_IDS = list(range(n_gpus))
    else:
        GPU_IDS = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    WORLD_SIZE = len(GPU_IDS)

    # ── Header Print ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Mask R-CNN Multi-GPU DDP Training")
    print(f"{'='*60}")
    print(f"  GPU yang digunakan : {GPU_IDS}")
    print(f"  World size         : {WORLD_SIZE}")
    print(f"  Epochs             : {EPOCHS}")
    print(f"  Batch/GPU          : {BATCH_SIZE_PER_GPU}")
    print(f"  Total batch        : {BATCH_SIZE_PER_GPU * WORLD_SIZE}")
    print(f"  Dataset            : {SEG_DATASET_LOCATION}")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── Validasi GPU ─────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("❌ CUDA tidak tersedia. Script ini membutuhkan GPU.")
        sys.exit(1)

    n_gpu_avail = torch.cuda.device_count()
    for g in GPU_IDS:
        if g >= n_gpu_avail:
            print(f"❌ GPU {g} tidak tersedia. Sistem hanya punya {n_gpu_avail} GPU (0–{n_gpu_avail-1}).")
            sys.exit(1)

    # GPU Fan Manager (Main process only)
    try:
        from gpu_fan_manager import start_fan_manager
        start_fan_manager()
    except ImportError:
        print("[Warning] gpu_fan_manager.py not found in ROOT.")

    best_pt = os.path.join(OUTPUT_DIR, "best.pt")

    # ── Cek skip ────────────────────────────────────────────────────────────
    if os.path.exists(best_pt):
        print(f"\n[SKIP] Model sudah ada: {best_pt}")
        print("Hapus file tersebut jika ingin training ulang.\n")
    else:
        # ── Set env untuk DDP init_method env:// ────────────────────────────
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")   # Port berbeda dari YOLO DDP

        # ── Paksa bersihkan VRAM sebelum DDP spawn ──────────────────────────
        # Mask R-CNN butuh ruang penuh di setiap GPU untuk RPN + RoIAlign.
        # Jika dijalankan setelah YOLO, sisa tensor bisa sebabkan OOM.
        import gc as _gc
        import torch as _torch_clean
        if _torch_clean.cuda.is_available():
            _torch_clean.cuda.empty_cache()
            _torch_clean.cuda.synchronize()
            _gc.collect()
            free_gb  = _torch_clean.cuda.mem_get_info(0)[0] / 1e9
            total_gb = _torch_clean.cuda.mem_get_info(0)[1] / 1e9
            # print(f"[MemClean] VRAM bebas sebelum DDP spawn: {free_gb:.2f} GB / {total_gb:.2f} GB")
            print("[MemClean] ✅ ")

        # ── Spawn satu proses per GPU ────────────────────────────────────────
        mp.spawn(
            train_worker,
            args=(GPU_IDS, best_pt, WORLD_SIZE),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # ── Post-training: latency + report + visual (semua di GPU pertama) ──────
    eval_device = torch.device(f"cuda:{GPU_IDS[0]}")
    csv_path    = "N/A (Skipped - Akan dievaluasi terpisah oleh evaluator pipeline)"

    if not args.skip_eval:
        size_mb         = round(os.path.getsize(best_pt) / 1e6, 2)
        lat_ms, fps_val = measure_latency(best_pt, eval_device)
        map_box, map_mask = evaluate_map(best_pt, eval_device)

        # ── CSV Report ───────────────────────────────────────────────────────────
        os.makedirs(REPORT_DIR, exist_ok=True)
        csv_path = os.path.join(REPORT_DIR, "report_maskrcnn_ddp_seg.csv")
        fields   = ["Model", "Model Size (MB)", "mAP50-95(Box)",
                    "mAP50-95(Mask)", "Latency (ms)", "FPS", "GPUs", "Evaluator"]

        # Ambil nama merk GPU & hitung jumlahnya (e.g. "2x NVIDIA RTX 3060")
        from collections import Counter
        gpu_names = [torch.cuda.get_device_name(i) for i in GPU_IDS]
        counts = Counter(gpu_names)
        gpu_report_str = ", ".join([f"{count}x {name}" for name, count in counts.items()])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({
                "Model":           "Mask R-CNN ResNet-50 FPN-v2",
                "Model Size (MB)": size_mb,
                "mAP50-95(Box)":   map_box,
                "mAP50-95(Mask)":  map_mask,
                "Latency (ms)":     lat_ms,
                "FPS":             fps_val,
                "GPUs":            gpu_report_str,
                "Evaluator":       "COCOeval",
            })
        print(f"\n✅ Report: {csv_path}")

        # ── Evaluasi via generate_report_single_model.py ─────────────────────────────────────
        print("\n" + "="*65 + "\n  Menjalankan Evaluasi Mask R-CNN via generate_report_single_model.py\n" + "="*65)
        try:
            import subprocess
            eval_script = os.path.join(ROOT, "utils", "generate_report_single_model.py")
            if not os.path.exists(eval_script):
                print(f"⚠️ Skrip evaluasi tidak ditemukan: {eval_script}")
            else:
                subprocess.run(
                    [sys.executable, "-u", eval_script, "--family", "maskrcnn", "--gpus", args.gpus],
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Evaluasi Mask R-CNN gagal: {e}")
    else:
        print("\n[Skip] Evaluasi Mask R-CNN dilewati karena argumen --skip-eval aktif.")

    # ── Kompres folder hasil training ───────────────────────────────────────
    compress_run(OUTPUT_KEY)

    print(f"\n{'='*60}")
    print(f"✅ Mask R-CNN DDP selesai.")
    print(f"   Best model  : {best_pt}")
    print(f"   Report      : {csv_path}")
    print(f"   Visualisasi : {VISUAL_DIR}")
    print(f"{'='*60}")
    send_telegram_msg(f"✅ <b>Mask R-CNN Pipeline Finished</b>\nWorkspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>")
