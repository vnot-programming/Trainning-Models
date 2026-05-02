import os, sys, torch, csv

# Path setup
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_FINETUNING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _FINETUNING_ROOT)

from config_shared import WORKSPACE_DIR, REPORTS_DIR
from train_multigpu import evaluate_map, measure_latency

def main():
    # Setup paths
    OUTPUT_KEY = "maskrcnn"
    OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "runs", OUTPUT_KEY, "weights")
    best_pt = os.path.join(OUTPUT_DIR, "best.pt")
    csv_path = os.path.join(REPORTS_DIR, "report_maskrcnn_ddp_seg.csv")

    if not os.path.exists(best_pt):
        print(f"❌ Error: {best_pt} tidak ditemukan.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai evaluasi ulang untuk Mask R-CNN menggunakan: {device}")
    print(f"   Model: {best_pt}")

    # Hitung metrik
    size_mb = round(os.path.getsize(best_pt) / 1e6, 2)
    lat_ms, fps_val = measure_latency(best_pt, device)
    map_box, map_mask = evaluate_map(best_pt, device)

    # Ambil nama merk GPU
    gpu_name = f"1x {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "1x CPU"

    # Update CSV
    fields = ["Model", "Model Size (MB)", "mAP50-95(Box)",
              "mAP50-95(Mask)", "Latency(ms)", "FPS", "GPUs"]

    print(f"📝 Memperbarui report: {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "Model":           "Mask R-CNN ResNet-50 FPN-v2 (DDP Fine-tuned)",
            "Model Size (MB)": size_mb,
            "mAP50-95(Box)":   map_box,
            "mAP50-95(Mask)":  map_mask,
            "Latency(ms)":     lat_ms,
            "FPS":             fps_val,
            "GPUs":            gpu_name,
        })

    print("✅ Selesai! Nilai mAP telah diperbarui.")

if __name__ == "__main__":
    main()
