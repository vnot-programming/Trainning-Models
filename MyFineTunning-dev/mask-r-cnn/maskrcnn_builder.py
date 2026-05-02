import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_mask_rcnn(num_classes: int, use_parallel: bool = False, device: torch.device = None) -> torch.nn.Module:
    """
    Membangun arsitektur Mask R-CNN menggunakan backbone ResNet-50-FPN-v2.
    Head (box dan mask) akan diganti ukurannya agar sesuai dengan num_classes.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Kustomisasi Box Predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
    # Kustomisasi Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer_dim = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer_dim, num_classes)
    
    # Transfer ke device
    if device is not None:
        model = model.to(device)
    
    # Jika multigpu diizinkan dan tidak menggunakan DDP (misal DataParallel standard)
    if use_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    return model
