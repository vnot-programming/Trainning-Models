import os
import sys
import json
_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "utils"))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)
from config_shared import SEG_DATASET_LOCATION, NUM_CLASSES
CLASS_NAMES = []
def init_classes_from_coco(coco_dir):
    global CLASS_NAMES
    json_path = os.path.join(coco_dir, "valid", "_annotations.coco.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        CLASS_NAMES = ["Class"] * NUM_CLASSES
        print("Fallback!")
        return
    with open(json_path, 'r') as f:
        coco_gt = json.load(f)
    cats = [c for c in coco_gt['categories'] if c['id'] > 0]
    cats = sorted(cats, key=lambda x: x['id'])
    CLASS_NAMES = [c['name'] for c in cats]
    print(f"✅ Kelas terdeteksi ({len(CLASS_NAMES)}): {CLASS_NAMES}")

init_classes_from_coco(SEG_DATASET_LOCATION)
