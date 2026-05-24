import os, sys, gc
sys.path.insert(0, "/home/my/Trainning-Models/MyFineTunning-dev")
from ultralytics import YOLO, SAM
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch

NUM_CLASSES = 3

def get_sz(m):
    params = sum(p.nelement() for p in m.parameters())
    sz = sum(p.nelement() * p.element_size() for p in m.parameters()) + sum(b.nelement() * b.element_size() for b in m.buffers())
    return params, sz

results = {}

try:
    m = YOLO("/home/my/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260505_034341/runs/yolov8m/weights/best.pt").model
    p, s = get_sz(m)
    results["YOLOv8m"] = (p, s)
    del m; gc.collect()
except Exception as e: print(e)

try:
    m = YOLO("/home/my/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260505_034341/runs/yolov9c_seg/weights/best.pt").model
    p, s = get_sz(m)
    results["YOLOv9c-Seg"] = (p, s)
    del m; gc.collect()
except Exception as e: print(e)

try:
    m = YOLO("/home/my/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260505_034341/runs/yolo11l/weights/best.pt").model
    p, s = get_sz(m)
    results["YOLO11l"] = (p, s)
    del m; gc.collect()
except Exception as e: print(e)

try:
    m = YOLO("/home/my/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260505_034341/runs/yolo11l_seg/weights/best.pt").model
    p, s = get_sz(m)
    results["YOLO11l-Seg"] = (p, s)
    del m; gc.collect()
except Exception as e: print(e)

try:
    m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
    p, s = get_sz(m)
    results["Mask R-CNN"] = (p, s)
    del m; gc.collect()
except Exception as e: print(e)

try:
    sam = SAM("/home/my/Trainning-Models/MyFineTunning-dev/models/sam2.1_t.pt").model
    p, s = get_sz(sam)
    results["SAM2.1_t"] = (p, s)
    del sam; gc.collect()
except Exception as e: print(e)

for k, v in results.items():
    print(f"{k}: Params: {v[0]/1e6:.2f} M | Size: {v[1]/1024**2:.2f} MB")

