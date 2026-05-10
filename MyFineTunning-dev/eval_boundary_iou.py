# -*- coding: utf-8 -*-
"""
eval_boundary_iou.py  (ROOT — MyFineTunning-dev)
=================================================
Evaluasi Boundary Mask AP untuk semua model segmentasi menggunakan
Boundary IoU (Cheng et al., CVPR 2021).

Metrik: BoundAP, BoundAP50, BoundAP75, BoundAP_S, BoundAP_M, BoundAP_L

Model yang dievaluasi:
  1. YOLOv8m-Seg
  2. YOLOv9c-Seg
  3. YOLO11m-Seg
  4. Mask R-CNN (ResNet-50 FPN-v2, torchvision state_dict)
  5. Hybrid (YOLO11m + SAM2)

Cara menjalankan:
    python eval_boundary_iou.py
    python eval_boundary_iou.py --gpus 0
    python eval_boundary_iou.py --gpus 0,1

Atau via tmux (recommended):
tmux new-session -d -s eval_boundary_iou "source /home/my/Trainning-Models/MyFineTunning-dev/.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-dev && \\
      python -u eval_boundary_iou.py 2>&1 | tee eval_boundary_iou.log"

tmux new-session -d -s eval_boundary_iou "cd /home/my/Trainning-Models/MyFineTunning-dev && python3 eval_boundary_iou.py 2>&1 | tee eval_boundary_iou.log"

Output:
    REPORTS_DIR/report_boundary_iou_comparison.csv
    REPORTS_DIR/narrative_reports/Laporan_Boundary_IoU_Comparison.md
"""

import os, sys, gc, csv, time, argparse, subprocess
import numpy as np
import cv2

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import torch
from config_shared import (
    WORKSPACE_DIR, SEG_YAML, DET_YAML, IMAGE_SIZE,
    get_output_dir, REPORTS_DIR, NUM_CLASSES,
    SEG_DATASET_LOCATION
)
from telegram_utils import send_telegram_msg
from coco_eval_utils import build_coco_ground_truth

# ==============================================================================
# DEPENDENCY BOOTSTRAP
# ==============================================================================

def _ensure_boundary_iou_installed() -> bool:
    """Auto-install boundary-iou-api jika belum tersedia.

    Catatan: `pip install git+...` untuk repo ini menghasilkan package KOSONG
    (hanya __init__.py tanpa submodule). Solusi: clone manual lalu `pip install -e`.
    """
    def _try_import():
        try:
            from boundary_iou.coco_instance_api.coco import COCO  # noqa
            from boundary_iou.coco_instance_api.cocoeval import COCOeval  # noqa
            return True
        except (ImportError, AttributeError):
            return False

    if _try_import():
        print("[BoundaryIoU] ✅ boundary-iou-api sudah terinstall.")
        return True

    print("[BoundaryIoU] ⏳ boundary-iou-api belum ada — menginstall via git clone...")
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix="boundary_iou_")
    try:
        # Clone repo (lebih andal dari pip install git+)
        subprocess.check_call(
            ["git", "clone", "https://github.com/bowenc0221/boundary-iou-api.git",
             tmp_dir, "--depth=1", "--quiet"],
            timeout=120
        )
        # Install editable agar subpackage tersedia
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", tmp_dir, "--quiet"],
            timeout=120
        )
    except Exception as e:
        print(f"[BoundaryIoU] ❌ Instalasi gagal: {e}")
        return False

    # Invalidate import cache Python
    import importlib, site
    importlib.invalidate_caches()
    for sp in site.getsitepackages():
        if sp not in sys.path:
            sys.path.insert(0, sp)

    if _try_import():
        print("[BoundaryIoU] ✅ boundary-iou-api berhasil diinstall dan di-load.")
        return True

    print("[BoundaryIoU] ❌ Import masih gagal setelah instalasi.")
    print(f"   Jalankan manual di dalam .venv:")
    print(f"   git clone https://github.com/bowenc0221/boundary-iou-api.git /tmp/biou")
    print(f"   pip install -e /tmp/biou")
    return False




# ==============================================================================
# HELPERS
# ==============================================================================

def _flush_gpu(label: str = ""):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        print(f"  [MemFlush] {label} — VRAM bebas: {free/1e9:.2f}/{total/1e9:.2f} GB",
              flush=True)


def _gpu_str() -> str:
    from collections import Counter
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    counts = Counter(names)
    return ", ".join(f"{c}x {n}" for n, c in counts.items())


def _mask_to_rle(bin_mask: np.ndarray) -> dict:
    from pycocotools import mask as maskUtils
    rle = maskUtils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _run_boundary_eval(coco_gt, dt_segm: list, label: str) -> dict:
    """Jalankan COCOeval dengan iouType='boundary'."""
    from boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryCOCOeval

    if not dt_segm:
        print(f"  [BoundaryIoU] ⚠️  {label}: tidak ada prediksi mask.")
        return {k: "N/A" for k in ["BoundAP", "BoundAP50", "BoundAP75",
                                    "BoundAP_S", "BoundAP_M", "BoundAP_L"]}

    b_gt = BoundaryCOCO()
    b_gt.dataset = coco_gt.dataset
    b_gt.createIndex()

    b_dt = b_gt.loadRes(dt_segm)
    evaluator = BoundaryCOCOeval(b_gt, b_dt, iouType="boundary")
    evaluator.evaluate()
    evaluator.accumulate()
    print(f"\n  [BoundaryIoU] ── {label} ──")
    evaluator.summarize()

    s = evaluator.stats
    return {
        "BoundAP":   round(float(s[0]), 4),
        "BoundAP50": round(float(s[1]), 4),
        "BoundAP75": round(float(s[2]), 4),
        "BoundAP_S": round(float(s[3]), 4),
        "BoundAP_M": round(float(s[4]), 4),
        "BoundAP_L": round(float(s[5]), 4),
    }


# ==============================================================================
# GROUND TRUTH
# ==============================================================================

def _build_gt(split: str = "valid"):
    """Build pycocotools COCO GT object dari SEG_YAML."""
    from pycocotools.coco import COCO

    coco_gt_dict, image_ids = build_coco_ground_truth(SEG_YAML, split=split)
    if coco_gt_dict is None:
        raise RuntimeError(f"Gagal membangun COCO GT dari {SEG_YAML}")

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    print(f"[GT] {len(image_ids)} gambar, {len(coco_gt_dict['annotations'])} anotasi.")
    return coco_gt, image_ids, coco_gt_dict


# ==============================================================================
# INFERENCE: YOLO-SEG (yolov8m, yolov9c, yolo11m)
# ==============================================================================

def _infer_yolo_seg(model_key: str, pt_path: str, image_ids: dict,
                    device_str: str) -> list:
    """Jalankan inference YOLO-Seg, kembalikan dt_segm COCO-format."""
    from ultralytics import YOLO
    from pycocotools import mask as maskUtils

    if not os.path.exists(pt_path):
        print(f"  ❌ {model_key}: best.pt tidak ditemukan: {pt_path}")
        return []

    print(f"\n  [Infer] Loading {model_key} dari {pt_path} ...")
    model = YOLO(pt_path)
    dt_segm = []

    for img_path, img_id in image_ids.items():
        results = model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                device=device_str, verbose=False)
        if not results or results[0].masks is None:
            continue

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]

        masks_data = results[0].masks.data.cpu().numpy()
        confs  = results[0].boxes.conf.cpu().numpy()
        clss   = results[0].boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks_data):
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            bin_mask = (mask > 0.5).astype(np.uint8)
            rle = _mask_to_rle(bin_mask)
            dt_segm.append({
                "image_id":     img_id,
                "category_id":  int(clss[i]) + 1,
                "segmentation": rle,
                "score":        float(confs[i]),
            })

    del model
    _flush_gpu(model_key)
    print(f"  [Infer] {model_key}: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# INFERENCE: MASK R-CNN (torchvision state_dict)
# ==============================================================================

def _build_maskrcnn_model(device: torch.device) -> torch.nn.Module:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, NUM_CLASSES + 1)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES + 1)
    return model.to(device)


def _infer_maskrcnn(pt_path: str, image_ids: dict, device: torch.device) -> list:
    """Load Mask R-CNN dari state_dict best.pt, jalankan inference."""
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    if not os.path.exists(pt_path):
        print(f"  ❌ Mask R-CNN: best.pt tidak ditemukan: {pt_path}")
        return []

    print(f"\n  [Infer] Loading Mask R-CNN dari {pt_path} ...")
    model = _build_maskrcnn_model(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    dt_segm = []
    with torch.no_grad():
        for img_path, img_id in image_ids.items():
            pil = PILImage.open(img_path).convert("RGB")
            W, H = pil.size
            t_img = TF.to_tensor(pil).unsqueeze(0).to(device)
            preds = model(t_img)[0]

            boxes  = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            masks  = preds["masks"].cpu().numpy()  # (N,1,H,W)

            for i in range(len(scores)):
                bin_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                if bin_mask.shape != (H, W):
                    bin_mask = cv2.resize(bin_mask, (W, H),
                                          interpolation=cv2.INTER_NEAREST)
                rle = _mask_to_rle(bin_mask)
                x1, y1, x2, y2 = boxes[i].tolist()
                dt_segm.append({
                    "image_id":     img_id,
                    "category_id":  int(labels[i]),  # Mask R-CNN sudah 1-indexed
                    "segmentation": rle,
                    "score":        float(scores[i]),
                })

    del model
    _flush_gpu("maskrcnn")
    print(f"  [Infer] Mask R-CNN: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# INFERENCE: HYBRID (YOLO11m + SAM2)
# ==============================================================================

def _infer_hybrid(yolo_pt: str, sam_pt: str, image_ids: dict,
                  device_str: str) -> list:
    """YOLO11m deteksi → SAM2 segmentasi per bbox."""
    from ultralytics import YOLO, SAM

    if not os.path.exists(yolo_pt):
        print(f"  ❌ Hybrid YOLO11m: best.pt tidak ditemukan: {yolo_pt}")
        return []

    # Auto-download SAM2 jika belum ada
    if not os.path.exists(sam_pt):
        print(f"  ⏳ Mengunduh sam2.1_b.pt ...")
        try:
            from ultralytics.utils.downloads import download
            download("https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt",
                     dir=os.path.dirname(sam_pt))
        except Exception as e:
            print(f"  ❌ Gagal mengunduh SAM2: {e}")
            return []

    print(f"\n  [Infer] Loading Hybrid YOLO11m + SAM2 ...")
    yolo_model = YOLO(yolo_pt)
    sam_model  = SAM(sam_pt)
    dt_segm    = []

    for img_path, img_id in image_ids.items():
        det_result = yolo_model.predict(img_path, conf=0.5, imgsz=IMAGE_SIZE,
                                        device=device_str, verbose=False)
        if not (det_result and det_result[0].boxes is not None
                and len(det_result[0].boxes) > 0):
            continue

        pred_boxes = det_result[0].boxes.xyxy.cpu().numpy()
        pred_confs = det_result[0].boxes.conf.cpu().numpy()
        pred_clss  = det_result[0].boxes.cls.cpu().numpy().astype(int)

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]

        try:
            sam_result = sam_model.predict(det_result[0].orig_img,
                                           bboxes=pred_boxes, verbose=False)
        except Exception as e:
            print(f"  ⚠️ SAM2 error: {e}")
            sam_result = None

        if sam_result and sam_result[0].masks is not None:
            sam_masks = sam_result[0].masks.data.cpu().numpy()
        else:
            sam_masks = [None] * len(pred_boxes)

        for i in range(len(pred_boxes)):
            mask = sam_masks[i] if i < len(sam_masks) else None
            if mask is None:
                continue
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            bin_mask = (mask > 0.5).astype(np.uint8)
            rle = _mask_to_rle(bin_mask)
            dt_segm.append({
                "image_id":     img_id,
                "category_id":  int(pred_clss[i]) + 1,
                "segmentation": rle,
                "score":        float(pred_confs[i]),
            })

    del yolo_model, sam_model
    _flush_gpu("hybrid")
    print(f"  [Infer] Hybrid: {len(dt_segm)} prediksi mask.")
    return dt_segm


# ==============================================================================
# SAVE REPORTS
# ==============================================================================

def _save_csv(rows: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "report_boundary_iou_comparison.csv")
    fields = ["Model", "BoundAP", "BoundAP50", "BoundAP75",
              "BoundAP_S", "BoundAP_M", "BoundAP_L", "GPUs", "Evaluator"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ CSV: {csv_path}")
    return csv_path


def _save_markdown(rows: list, out_dir: str):
    narr_dir = os.path.join(out_dir, "narrative_reports")
    os.makedirs(narr_dir, exist_ok=True)
    md_path = os.path.join(narr_dir, "Laporan_Boundary_IoU_Comparison.md")

    lines = [
        "# Laporan Evaluasi Boundary IoU — Perbandingan Semua Model",
        "",
        "**Metrik**: Boundary Mask AP (Cheng et al., CVPR 2021)",
        "**Evaluator**: `boundary-iou-api` (iouType='boundary')",
        "**Dataset Split**: valid",
        "",
        "---",
        "",
        "## 📊 Tabel Perbandingan Boundary AP",
        "",
        "| Model | BoundAP | BoundAP50 | BoundAP75 | BoundAP_S | BoundAP_M | BoundAP_L | GPUs | Evaluator |",
        "|-------|---------|-----------|-----------|-----------|-----------|-----------|------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['Model']}** | {r['BoundAP']} | {r['BoundAP50']} | "
            f"{r['BoundAP75']} | {r['BoundAP_S']} | {r['BoundAP_M']} | "
            f"{r['BoundAP_L']} | {r['GPUs']} | {r['Evaluator']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 🔍 Interpretasi",
        "",
        "Boundary IoU mengukur ketajaman tepian mask dengan cara mengevaluasi",
        "prediksi hanya di zona batas (*boundary zone*), bukan seluruh area mask.",
        "Model yang menghasilkan mask halus dengan tepian presisi (seperti SAM2)",
        "akan mendapat skor **BoundAP** jauh lebih tinggi dibanding model yang",
        "menghasilkan mask blok kasar meskipun memiliki COCO Mask AP yang serupa.",
        "",
        "**Hipotesis**: Hybrid (YOLO11m + SAM2) diharapkan unggul secara signifikan",
        "khususnya pada **BoundAP75** (threshold IoU ketat) karena SAM2 dirancang",
        "untuk menghasilkan tepian objek sub-piksel yang akurat.",
        "",
        f"*Dihasilkan otomatis oleh eval_boundary_iou.py*",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown: {md_path}")
    return md_path


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_evaluation(device_str: str = "cuda:0"):
    """Evaluasi Boundary AP untuk semua model secara sequential."""
    print("\n" + "="*65)
    print("  Boundary IoU Evaluation — Semua Model Segmentasi")
    print("="*65)

    # 1. Bootstrap dependency
    if not _ensure_boundary_iou_installed():
        print("❌ boundary-iou-api tidak tersedia. Evaluasi dibatalkan.")
        sys.exit(1)

    # 2. Build GT
    print("\n[GT] Membangun COCO Ground Truth ...")
    coco_gt, image_ids, _ = _build_gt(split="valid")

    gpu_report = _gpu_str()
    rows       = []

    # ── Definisi model ────────────────────────────────────────────────────────
    models_cfg = [
        {
            "label":     "YOLOv8m-Seg",
            "type":      "yolo_seg",
            "model_key": "yolov8m_seg",
        },
        {
            "label":     "YOLOv9c-Seg",
            "type":      "yolo_seg",
            "model_key": "yolov9c_seg",
        },
        {
            "label":     "YOLO11m-Seg",
            "type":      "yolo_seg",
            "model_key": "yolo11m_seg",
        },
        {
            "label":     "Mask R-CNN ResNet-50 FPN-v2",
            "type":      "maskrcnn",
            "model_key": "maskrcnn",
        },
        {
            "label":     "Hybrid (YOLO11m+SAM2)",
            "type":      "hybrid",
            "yolo_key":  "yolo11m",
            "sam_path":  os.path.join(_SCRIPT_DIR, "hybrid", "sam2.1_b.pt"),
        },
    ]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    send_telegram_msg(
        f"🚀 <b>Boundary IoU Eval Dimulai</b>\n"
        f"GPU: <code>{gpu_report}</code>\n"
        f"Models: {len(models_cfg)}"
    )

    for cfg in models_cfg:
        label = cfg["label"]
        print(f"\n{'='*55}")
        print(f"  Evaluasi: {label}")
        print(f"{'='*55}")

        t0 = time.perf_counter()

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            if cfg["type"] == "yolo_seg":
                pt = os.path.join(get_output_dir(cfg["model_key"]), "weights", "best.pt")
                dt_segm = _infer_yolo_seg(cfg["model_key"], pt, image_ids, device_str)

            elif cfg["type"] == "maskrcnn":
                pt = os.path.join(get_output_dir(cfg["model_key"]), "weights", "best.pt")
                dt_segm = _infer_maskrcnn(pt, image_ids, device)

            elif cfg["type"] == "hybrid":
                yolo_pt = os.path.join(get_output_dir(cfg["yolo_key"]), "weights", "best.pt")
                dt_segm = _infer_hybrid(yolo_pt, cfg["sam_path"], image_ids, device_str)

            else:
                dt_segm = []

        except Exception as e:
            print(f"  ❌ Inference gagal untuk {label}: {e}")
            dt_segm = []

        # ── Boundary AP Evaluation ─────────────────────────────────────────────
        metrics = _run_boundary_eval(coco_gt, dt_segm, label)
        elapsed = round(time.perf_counter() - t0, 1)

        row = {
            "Model":     label,
            "GPUs":      gpu_report,
            "Evaluator": "BoundaryIoU (Cheng et al., CVPR 2021)",
            **metrics,
        }
        rows.append(row)

        print(f"\n  ✅ {label} selesai dalam {elapsed}s")
        print(f"     BoundAP={metrics['BoundAP']} | BoundAP50={metrics['BoundAP50']} | BoundAP75={metrics['BoundAP75']}")

        send_telegram_msg(
            f"✅ <b>{label}</b>\n"
            f"BoundAP: <code>{metrics['BoundAP']}</code>\n"
            f"BoundAP50: <code>{metrics['BoundAP50']}</code>\n"
            f"BoundAP75: <code>{metrics['BoundAP75']}</code>"
        )

    # ── Simpan laporan ────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = _save_csv(rows, REPORTS_DIR)
    _save_markdown(rows, REPORTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BOUNDARY IoU EVALUATION SELESAI")
    print("="*65)
    print(f"{'Model':<35} {'BoundAP':>8} {'BoundAP50':>10} {'BoundAP75':>10}")
    print("-"*65)
    for r in rows:
        print(f"{r['Model']:<35} {str(r['BoundAP']):>8} "
              f"{str(r['BoundAP50']):>10} {str(r['BoundAP75']):>10}")

    # Telegram final summary
    summary_lines = "\n".join(
        f"• {r['Model']}: BoundAP={r['BoundAP']} | AP50={r['BoundAP50']}"
        for r in rows
    )
    send_telegram_msg(
        f"🏁 <b>Boundary IoU Eval Selesai</b>\n\n{summary_lines}\n\n"
        f"Report: <code>{csv_path}</code>"
    )

    return rows


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Boundary IoU Evaluation — Semua Model Segmentasi"
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="GPU untuk inference (single GPU). Contoh: '0' atau '1'. Default: '0'."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  CUDA tidak tersedia. Menggunakan CPU (sangat lambat).")
        DEVICE_STR = "cpu"
    else:
        gpu_id = int(args.gpus.split(",")[0].strip())
        n_avail = torch.cuda.device_count()
        if gpu_id >= n_avail:
            print(f"❌ GPU {gpu_id} tidak tersedia (sistem punya {n_avail} GPU).")
            sys.exit(1)
        DEVICE_STR = f"cuda:{gpu_id}"
        print(f"[Setup] GPU: {torch.cuda.get_device_name(gpu_id)} ({DEVICE_STR})")

    run_evaluation(device_str=DEVICE_STR)
