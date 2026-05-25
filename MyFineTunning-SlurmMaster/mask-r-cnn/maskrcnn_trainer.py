# -*- coding: utf-8 -*-
"""
models/maskrcnn_trainer.py
==========================
Loop training dan validasi untuk Mask R-CNN.

Fungsi publik:
    train_mask_rcnn(model, dataset_location, epochs, batch_size,
                    device, num_workers) -> str
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# NONAKTIFKAN torch._dynamo untuk Mask R-CNN
# ─────────────────────────────────────────────────────────────────────────────
# Bug: _dynamo mencoba mengompilasi ulang _roi_align setiap kali batch_size
# berubah (batch terakhir sering tidak genap). Setelah mencapai batas
# recompilasi (8x), fallback-nya corrupt dan mencoba mengalokasikan memori
# tidak masuk akal (>100 GiB) → OOM palsu.
# Solusi: matikan JIT compiler, paksa eager mode untuk seluruh sesi Mask R-CNN.
# ─────────────────────────────────────────────────────────────────────────────
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.disable()

# Kurangi fragmentasi memori VRAM saat Mask R-CNN backward pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from torch.optim.lr_scheduler import StepLR

from models.maskrcnn_dataset import RoboflowSegToMaskRCNN, collate_fn


# ==============================================================================
# KONSTANTA INTERNAL
# ==============================================================================
_DEFAULT_OUTPUT_DIR = "exp5_training/mask_rcnn_custom/weights"
_BEST_MODEL_NAME    = "best.pt"
_LOG_EVERY_N_BATCH  = 20   # Cetak log setiap N batch


# ==============================================================================
# FUNGSI PUBLIK
# ==============================================================================
def train_mask_rcnn(
    model,
    dataset_location: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 2,
    output_dir: str  = _DEFAULT_OUTPUT_DIR,
) -> str:
    """
    Jalankan training loop Mask R-CNN dengan validasi per epoch.

    Dataset yang digunakan harus dalam format YOLO Segmentation (polygon),
    dengan struktur direktori:
        <dataset_location>/train/images/
        <dataset_location>/train/labels/
        <dataset_location>/valid/images/
        <dataset_location>/valid/labels/

    Model terbaik (berdasarkan val loss terendah) disimpan otomatis
    ke `output_dir/best.pt`.

    Parameters
    ----------
    model : nn.Module
        Model Mask R-CNN yang sudah di-build dan di-.to(device).
    dataset_location : str
        Root direktori dataset segmentasi.
    epochs : int
        Jumlah epoch training.
    batch_size : int
        Batch size untuk train dan validasi.
    device : torch.device
        Device target komputasi.
    num_workers : int
        Jumlah worker DataLoader.
    output_dir : str
        Direktori penyimpanan best.pt.

    Returns
    -------
    str
        Path absolut ke file best.pt.
    """
    print("\n--- [Tahap 2C] Training Mask R-CNN (Segmentation) ---")

    train_loader, val_loader = _build_dataloaders(
        dataset_location, batch_size, num_workers
    )
    optimizer, scheduler     = _build_optimizer(model)
    best_model_path          = _prepare_output_path(output_dir)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        avg_train_loss = _run_train_epoch(
            model, train_loader, optimizer, device, epoch, epochs
        )
        avg_val_loss = _run_val_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch [{epoch}/{epochs}]  "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            _save_best_model(model, best_model_path, best_val_loss)

    print(f"\n✅ Training selesai. Best model: {best_model_path}")
    return best_model_path


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================

def _build_dataloaders(
    dataset_location: str,
    batch_size: int,
    num_workers: int,
) -> tuple:
    """Buat DataLoader untuk split train dan valid."""
    train_ds = RoboflowSegToMaskRCNN(
        images_dir=os.path.join(dataset_location, "train", "images"),
        labels_dir=os.path.join(dataset_location, "train", "labels"),
    )
    val_ds = RoboflowSegToMaskRCNN(
        images_dir=os.path.join(dataset_location, "valid", "images"),
        labels_dir=os.path.join(dataset_location, "valid", "labels"),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_ds)} gambar | Val: {len(val_ds)} gambar")
    return train_loader, val_loader


def _build_optimizer(model):
    """Bangun SGD optimizer dan StepLR scheduler (sesuai paper Mask R-CNN)."""
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    return optimizer, scheduler


def _prepare_output_path(output_dir: str) -> str:
    """Buat direktori output dan kembalikan path best.pt."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, _BEST_MODEL_NAME)


def _run_train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Satu epoch training — kembalikan rata-rata loss."""
    model.train()
    total_loss = 0.0

    # Tentukan device dari model (selalu single GPU untuk Mask R-CNN)
    _device = next(model.parameters()).device

    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(_device) for img in images]
        targets = [{k: v.to(_device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # loss_dict adalah dict biasa (bukan DataParallel output)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Bebaskan cache VRAM setiap batch untuk cegah fragmentasi
        torch.cuda.empty_cache()

        if (batch_idx + 1) % _LOG_EVERY_N_BATCH == 0:
            print(
                f"  Epoch [{epoch}/{total_epochs}]  "
                f"Batch [{batch_idx + 1}/{len(loader)}]  "
                f"Loss: {losses.item():.4f}"
            )

    return total_loss / len(loader)


def _run_val_epoch(model, loader, device: torch.device) -> float:
    """
    Satu epoch validasi — kembalikan rata-rata loss.

    Mask R-CNN harus tetap dalam mode .train() agar loss dict tersedia,
    tetapi gradien dinonaktifkan via torch.no_grad().
    """
    model.train()   # Loss dict hanya tersedia dalam mode train
    val_loss  = 0.0
    _device   = next(model.parameters()).device

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(_device) for img in images]
            targets = [{k: v.to(_device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()
            torch.cuda.empty_cache()

    return val_loss / len(loader)


def _save_best_model(model, path: str, val_loss: float) -> None:
    """Simpan state_dict model terbaik (DataParallel-aware)."""
    state = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state, path)
    print(f"  ✅ Best model disimpan (val_loss={val_loss:.4f}) → {path}")
