# -*- coding: utf-8 -*-
"""
utils/eval_unu_dual.py
======================
Evaluasi Dual (Boundary IoU + Standard COCO Mask mAP) untuk semua model
segmentasi menggunakan dataset UNU COCO-segmentation.

Dataset: me-bottle-isempty-unu3-sem-seg-1-coco (format COCO native)
GT: valid/_annotations.coco.json (164 gambar, 179 anotasi, 8 kategori)

Model yang dievaluasi:
  1. YOLOv8m-Seg      4. Mask R-CNN (ResNet-50 FPN-v2)
  2. YOLOv9c-Seg      5. Hybrid (YOLO11l + SAM2)
  3. YOLO11l-Seg

Cara menjalankan:
    python3 utils/eval_unu_dual.py --gpus 0

    tmux new-session -d -s eval_unu " source Trainning-Models/MyFineTunning-dev/.venv/bin/activate && \\
      cd /home/my/Trainning-Models/MyFineTunning-dev/utils && \\
      python3 -u eval_unu_dual.py --gpus 0 2>&1 | tee eval_unu_dual.log"
      

Output:
    visuals/new_methods/boundary_iou/  — 164×5 = 820 gambar (kontur tepian)
    visuals/new_methods/normal_coco/   — 164×5 = 820 gambar (mask penuh)
    reports/boundary_iou/              — CSV + Markdown
    reports/normal_coco/               — CSV + Markdown
"""

import os, sys, csv, time, argparse, copy

os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, _UTILS_DIR)

import torch
from config_shared import (
    DATASETS_DIR, get_output_dir, REPORTS_DIR, VISUALS_DIR
)
from telegram_utils import send_telegram_msg
from eval_unu_helpers import (
    ensure_boundary_iou_installed, flush_gpu, gpu_str,
    run_boundary_eval, run_standard_mask_eval,
    infer_yolo_seg, infer_maskrcnn, infer_hybrid,
    generate_visuals, generate_comparison_grids,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================
UNU_DATASET_DIR  = os.path.join(DATASETS_DIR, "me-bottle-isempty-unu3-sem-seg-1-coco")
UNU_VALID_DIR    = os.path.join(UNU_DATASET_DIR, "valid")
UNU_ANNOTATIONS  = os.path.join(UNU_VALID_DIR, "_annotations.coco.json")

BOUNDARY_VIS_DIR = os.path.join(VISUALS_DIR, "new_methods", "boundary_iou")
NORMAL_VIS_DIR   = os.path.join(VISUALS_DIR, "new_methods", "normal_coco")
BOUNDARY_RPT_DIR = os.path.join(REPORTS_DIR, "boundary_iou")
NORMAL_RPT_DIR   = os.path.join(REPORTS_DIR, "normal_coco")
COMPARISON_DIR   = os.path.join(VISUALS_DIR, "new_methods", "comparison")

_HYBRID_DIR = os.path.join(ROOT, "hybrid")


# ==============================================================================
# GT LOADING — COCO JSON NATIVE
# ==============================================================================
def _load_unu_gt():
    """Load ground truth langsung dari _annotations.coco.json (COCO native)."""
    from pycocotools.coco import COCO

    if not os.path.exists(UNU_ANNOTATIONS):
        raise FileNotFoundError(f"COCO JSON tidak ditemukan: {UNU_ANNOTATIONS}")

    print(f"[GT] Loading COCO JSON: {UNU_ANNOTATIONS}")
    coco_gt = COCO(UNU_ANNOTATIONS)

    # Build image_ids mapping: {absolute_img_path: coco_image_id}
    image_ids = {}
    for img_info in coco_gt.dataset["images"]:
        img_path = os.path.join(UNU_VALID_DIR, img_info["file_name"])
        image_ids[img_path] = img_info["id"]

    n_imgs = len(image_ids)
    n_anns = len(coco_gt.dataset["annotations"])
    cats = [c["name"] for c in coco_gt.dataset["categories"]]
    print(f"[GT] {n_imgs} gambar, {n_anns} anotasi, {len(cats)} kategori: {cats}")
    return coco_gt, image_ids


# ==============================================================================
# CSV REPORTS
# ==============================================================================
def _save_boundary_csv(rows: list, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_boundary_iou_unu.csv")
    fields = ["Model", "BoundAP", "BoundAP50", "BoundAP75",
              "BoundAP_S", "BoundAP_M", "BoundAP_L", "GPUs", "Evaluator"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ Boundary CSV: {path}")
    return path


def _save_normal_csv(rows: list, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_normal_coco_unu.csv")
    fields = ["Model", "MaskAP", "MaskAP50", "MaskAP75",
              "MaskAP_S", "MaskAP_M", "MaskAP_L", "GPUs", "Evaluator"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ Normal COCO CSV: {path}")
    return path


# ==============================================================================
# MARKDOWN REPORTS
# ==============================================================================
def _save_boundary_md(rows: list, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Laporan_Boundary_IoU_UNU.md")
    lines = [
        "# Laporan Evaluasi Boundary AP — Dataset UNU COCO-Segmentation",
        "",
        "## Dataset: me-bottle-isempty-unu3-sem-seg (164 gambar, 179 anotasi)",
        "",
        "Boundary IoU (Cheng et al., CVPR 2021) hanya mengevaluasi prediksi di zona",
        "tepian objek, **mengabaikan area interior** yang terpengaruh bias anotasi.",
        "",
        "## 📊 Tabel Perbandingan",
        "",
        "| Model | BoundAP | BoundAP50 | BoundAP75 | BoundAP_S | BoundAP_M | BoundAP_L |",
        "|-------|---------|-----------|-----------|-----------|-----------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['Model']}** | {r['BoundAP']} | {r['BoundAP50']} | {r['BoundAP75']} "
            f"| {r['BoundAP_S']} | {r['BoundAP_M']} | {r['BoundAP_L']} |"
        )
    lines += ["", f"*Dihasilkan otomatis oleh eval_unu_dual.py*"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Boundary Markdown: {path}")
    return path


def _save_normal_md(rows: list, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Laporan_Normal_COCO_UNU.md")
    lines = [
        "# Laporan Evaluasi Standard COCO Mask mAP — Dataset UNU COCO-Segmentation",
        "",
        "## Dataset: me-bottle-isempty-unu3-sem-seg (164 gambar, 179 anotasi)",
        "",
        "Standard COCO Mask mAP menggunakan **pixel-level IoU** antara prediksi dan GT.",
        "",
        "## 📊 Tabel Perbandingan",
        "",
        "| Model | MaskAP | MaskAP50 | MaskAP75 | MaskAP_S | MaskAP_M | MaskAP_L |",
        "|-------|--------|----------|----------|----------|----------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['Model']}** | {r['MaskAP']} | {r['MaskAP50']} | {r['MaskAP75']} "
            f"| {r['MaskAP_S']} | {r['MaskAP_M']} | {r['MaskAP_L']} |"
        )
    lines += ["", f"*Dihasilkan otomatis oleh eval_unu_dual.py*"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Normal COCO Markdown: {path}")
    return path


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_evaluation(device_str: str = "cuda:0"):
    """Evaluasi Dual untuk semua 5 model terhadap dataset UNU."""

    print("\n" + "=" * 65)
    print("  Dual Evaluation (Boundary IoU + Normal COCO) — Dataset UNU")
    print("=" * 65)

    # 1. Bootstrap boundary-iou-api
    if not ensure_boundary_iou_installed():
        print("❌ boundary-iou-api tidak tersedia. Evaluasi dibatalkan.")
        sys.exit(1)

    # 2. Load GT
    print("\n[GT] Membangun COCO Ground Truth dari UNU dataset ...")
    coco_gt, image_ids = _load_unu_gt()

    gpu_report   = gpu_str()
    all_dt_segm  = {}  # {model_key: dt_segm} untuk comparison grids
    boundary_rows = []
    normal_rows   = []

    # 3. Definisi model
    models_cfg = [
        {"label": "YOLOv8m-Seg",  "type": "yolo_seg", "model_key": "yolov8m_seg"},
        {"label": "YOLOv9c-Seg",  "type": "yolo_seg", "model_key": "yolov9c_seg"},
        {"label": "YOLO11l-Seg",  "type": "yolo_seg", "model_key": "yolo11l_seg"},
        {"label": "Mask R-CNN ResNet-50 FPN-v2", "type": "maskrcnn", "model_key": "maskrcnn"},
        {"label": "Hybrid (YOLO11l+SAM2)", "type": "hybrid",
         "model_key": "hybrid", "yolo_key": "yolo11l",
         "sam_path": os.path.join(ROOT, "models", "sam2.1_t.pt")},
    ]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    send_telegram_msg(
        f"🚀 <b>Dual Eval UNU Dimulai</b>\n"
        f"GPU: <code>{gpu_report}</code>\n"
        f"Models: {len(models_cfg)} | Dataset: UNU (164 gambar)"
    )

    # 4. Evaluasi per model
    for cfg in models_cfg:
        label = cfg["label"]
        model_key = cfg.get("model_key", cfg.get("yolo_key", "unknown"))
        print(f"\n{'=' * 55}")
        print(f"  Evaluasi: {label}")
        print(f"{'=' * 55}")

        t0 = time.perf_counter()

        # ── Inference ──
        try:
            if cfg["type"] == "yolo_seg":
                pt = os.path.join(get_output_dir(cfg["model_key"]), "weights", "best.pt")
                dt_segm = infer_yolo_seg(cfg["model_key"], pt, image_ids, device_str)
            elif cfg["type"] == "maskrcnn":
                pt = os.path.join(get_output_dir(cfg["model_key"]), "weights", "best.pt")
                dt_segm = infer_maskrcnn(pt, image_ids, device)
            elif cfg["type"] == "hybrid":
                yolo_pt = os.path.join(get_output_dir(cfg["yolo_key"]), "weights", "best.pt")
                dt_segm = infer_hybrid(yolo_pt, cfg["sam_path"], image_ids, device_str)
            else:
                dt_segm = []
        except Exception as e:
            print(f"  ❌ Inference gagal untuk {label}: {e}")
            dt_segm = []

        # Simpan prediksi untuk comparison grids
        all_dt_segm[model_key] = dt_segm

        # ── Generate Visuals ──
        print(f"  [Visual] Generating {label} visuals ...")
        generate_visuals(model_key, dt_segm, coco_gt, UNU_VALID_DIR,
                         BOUNDARY_VIS_DIR, NORMAL_VIS_DIR)

        # ── Dual Evaluation ──
        # deepcopy wajib — pycocotools.loadRes() modifikasi dt_segm in-place
        std_metrics   = run_standard_mask_eval(coco_gt, copy.deepcopy(dt_segm), label)
        bound_metrics = run_boundary_eval(coco_gt, copy.deepcopy(dt_segm), label)
        elapsed = round(time.perf_counter() - t0, 1)

        boundary_rows.append({
            "Model": label, "GPUs": gpu_report,
            "Evaluator": "BoundaryIoU (Cheng et al., CVPR 2021)",
            **bound_metrics,
        })
        normal_rows.append({
            "Model": label, "GPUs": gpu_report,
            "Evaluator": "pycocotools COCOeval (segm)",
            **std_metrics,
        })

        print(f"\n  ✅ {label} selesai dalam {elapsed}s")
        print(f"     Standard  → MaskAP={std_metrics['MaskAP']} | MaskAP50={std_metrics['MaskAP50']}")
        print(f"     Boundary  → BoundAP={bound_metrics['BoundAP']} | BoundAP50={bound_metrics['BoundAP50']}")

        send_telegram_msg(
            f"✅ <b>{label}</b> (UNU)\n"
            f"MaskAP: <code>{std_metrics['MaskAP']}</code> | "
            f"BoundAP: <code>{bound_metrics['BoundAP']}</code>"
        )

    # 5. Save reports
    _save_boundary_csv(boundary_rows, BOUNDARY_RPT_DIR)
    _save_normal_csv(normal_rows, NORMAL_RPT_DIR)
    _save_boundary_md(boundary_rows, BOUNDARY_RPT_DIR)
    _save_normal_md(normal_rows, NORMAL_RPT_DIR)

    # 6. Generate Comparison Grids (10 gambar × 5 model)
    print("\n  [Comparison] Generating 5-model comparison grids (10 samples) ...")
    generate_comparison_grids(all_dt_segm, coco_gt, UNU_VALID_DIR,
                              COMPARISON_DIR, n_samples=10)

    # 7. Summary
    print("\n" + "=" * 85)
    print("  EVALUASI SELESAI — Standard Mask mAP vs Boundary AP (Dataset UNU)")
    print("=" * 85)
    print(f"{'Model':<32} {'MaskAP':>7} {'MaskAP50':>9}  {'BoundAP':>8} {'BndAP50':>8}")
    print("-" * 85)
    for br, nr in zip(boundary_rows, normal_rows):
        print(f"{br['Model']:<32} {str(nr['MaskAP']):>7} {str(nr['MaskAP50']):>9}  "
              f"{str(br['BoundAP']):>8} {str(br['BoundAP50']):>8}")

    summary = "\n".join(
        f"• {nr['Model']}: MaskAP={nr['MaskAP']} | BoundAP={br['BoundAP']}"
        for br, nr in zip(boundary_rows, normal_rows)
    )
    send_telegram_msg(
        f"🏁 <b>Dual Eval UNU Selesai</b>\n\n{summary}\n\n"
        f"Visuals: <code>{BOUNDARY_VIS_DIR}</code>\n"
        f"Reports: <code>{BOUNDARY_RPT_DIR}</code>"
    )

    return boundary_rows, normal_rows


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual Evaluation (Boundary IoU + Normal COCO) — Dataset UNU"
    )
    parser.add_argument("--gpus", type=str, default="0",
        help="GPU untuk inference (single). Contoh: '0' atau '1'. Default: '0'.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  CUDA tidak tersedia. Menggunakan CPU.")
        DEVICE_STR = "cpu"
    else:
        gpu_id = int(args.gpus.split(",")[0].strip())
        n_avail = torch.cuda.device_count()
        if gpu_id >= n_avail:
            print(f"❌ GPU {gpu_id} tidak tersedia ({n_avail} GPU)."); sys.exit(1)
        DEVICE_STR = f"cuda:{gpu_id}"
        print(f"[Setup] GPU: {torch.cuda.get_device_name(gpu_id)} ({DEVICE_STR})")

    run_evaluation(device_str=DEVICE_STR)
