import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class RoboflowSegToMaskRCNN(Dataset):
    """
    Konversi Dataset YOLO Segmentasi dari Roboflow 
    menjadi format Tensor yang dibutuhkan oleh torchvision Mask R-CNN.
    """
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.imgs = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        lbl_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels, masks = [], [], []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.read().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    
                    # YOLO label index dimulai dari 0.
                    # Mask R-CNN butuh index mulai dari 1 (0 adalah background).
                    cls_id = int(parts[0]) + 1 
                    
                    coords = list(map(float, parts[1:]))
                    pts = np.array(
                        [[coords[i] * w, coords[i+1] * h] for i in range(0, len(coords)-1, 2)], 
                        dtype=np.float32
                    )
                    
                    if len(pts) < 3: 
                        continue

                    # Buat binary mask 2D dari poligon
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                    
                    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
                    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

                    # Abaikan bounding box yang luasnya kurang dari 1 pixel
                    if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                        continue
                        
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cls_id)
                    masks.append(mask)

        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area_t = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            # Handle empty image
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            area_t = torch.zeros((0,), dtype=torch.float32)
            
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "area": area_t,
            "iscrowd": torch.zeros_like(labels_t, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        # Convert image ke PyTorch Tensor dengan normalisasi [0, 1]
        img_t = TF.to_tensor(img)
        return img_t, target

def collate_fn(batch):
    """
    Collate function khusus untuk Object Detection/Instance Segmentation,
    karena batch-nya terdiri dari sekumpulan image dengan dimensi target berbeda-beda.
    """
    return tuple(zip(*batch))