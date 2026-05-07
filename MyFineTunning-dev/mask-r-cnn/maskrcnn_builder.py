# -*- coding: utf-8 -*-
"""
models/maskrcnn_builder.py
==========================
Fungsi untuk membangun arsitektur Mask R-CNN dengan kepala klasifikasi
yang disesuaikan (fine-tuned head) untuk jumlah kelas dataset kustom.

Fungsi publik:
    build_mask_rcnn(num_classes, use_parallel, device) -> nn.Module
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor


def build_mask_rcnn(
    num_classes: int,
    use_parallel: bool = False,   # Diabaikan — lihat catatan di bawah
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Bangun Mask R-CNN dengan backbone ResNet-50-v2 + FPN (pretrained COCO)
    dan ganti kepala klasifikasi + mask sesuai jumlah kelas kustom.

    Catatan: `num_classes` harus sudah INCLUDE background (index 0).
    Contoh: 7 kelas objek → num_classes = 8.

    ⚠️  DataParallel DINONAKTIFKAN untuk Mask R-CNN:
        Mask R-CNN mengembalikan dict-of-scalar-losses; DataParallel
        mencoba gather() scalar tersebut ke GPU-0 (master) sehingga
        seluruh gradient dari semua GPU berkumpul di satu GPU dan
        menyebabkan OOM. Gunakan single-GPU (cuda:0) untuk Mask R-CNN
        dan biarkan YOLO yang memanfaatkan multi-GPU via DDP.

    Parameters
    ----------
    num_classes : int
        Jumlah kelas total termasuk background.
    use_parallel : bool
        Parameter ini sengaja diabaikan untuk Mask R-CNN.
    device : torch.device | None
        Device target model. Jika None, default ke CPU.
        Disarankan: torch.device("cuda:0") agar selalu ke GPU pertama.

    Returns
    -------
    torch.nn.Module
        Model Mask R-CNN siap training, sudah di-.to(device).
    """
    if device is None:
        device = torch.device("cpu")

    # ----------------------------------------------------------------
    # 1. Load pretrained backbone (COCO weights), bukan head-nya
    # ----------------------------------------------------------------
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )

    # ----------------------------------------------------------------
    # 2. Ganti Box Head (classification + regression)
    # ----------------------------------------------------------------
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box, num_classes
    )

    # ----------------------------------------------------------------
    # 3. Ganti Mask Head
    # ----------------------------------------------------------------
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # ----------------------------------------------------------------
    # 4. Single-GPU — DataParallel SENGAJA dinonaktifkan untuk Mask R-CNN
    #    (DataParallel + dict-of-scalar-losses → gather error + GPU-0 OOM)
    # ----------------------------------------------------------------
    single_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device is None:
        device = single_device
    print(f"[Mask R-CNN] Berjalan di single device: {device} (DataParallel dinonaktifkan)")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Mask R-CNN] Siap. Parameter trainable: {n_params:,}")

    return model
