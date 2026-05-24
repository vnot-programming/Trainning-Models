import torch
from ultralytics import SAM
import numpy as np
sam_model = SAM("models/sam2.1_t.pt")
img = np.zeros((640, 640, 3), dtype=np.uint8)
bboxes = [[10, 10, 50, 50], [60, 60, 100, 100]]
res = sam_model(img, bboxes=bboxes, verbose=False)
print("Type of res:", type(res))
print("Len of res:", len(res))
if len(res) > 0:
    print("Shape of first mask:", res[0].masks.data.shape if res[0].masks else "None")
