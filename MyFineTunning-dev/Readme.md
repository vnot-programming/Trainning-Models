```
# ==============================================================================
# README — MyFineTunning-dev Training Pipeline
# ==============================================================================
# Dokumentasi untuk pipeline training model YOLO dan Mask R-CNN
# ==============================================================================

## Struktur Output Training

Berdasarkan workspace ID: `20260502_105159` (dari file `.env`)

```

# Workspace directory (dibuat otomatis oleh main.py):

MyFineTunning-20260502_105159/
├── models/ ← model pra-terlatih (yolov8m.pt, yolov9m.pt, dll)
├── reports/ ← CSV reports (tidak dikompres)
│ ├── report_yolov8m_det.csv
│ ├── report_yolov8m_seg.csv
│ ├── report_yolov9m_det.csv
│ ├── report_yolov9c_seg.csv
│ ├── report_yolo11m_det.csv
│ ├── report_yolo11m_seg.csv
│ ├── report_hybrid_map.csv
│ └── hybrid_detailed_predictions.csv
├── visuals/ ← PNG hasil visualisasi sampel
├── image_samples/ ← 10 gambar sampel untuk visualisasi
└── runs/
├── yolov8m/ → YOLOv8m Detection
│ └── weights/
│ ├── best.pt
│ └── last.pt
├── yolov8m_seg/ → YOLOv8m Segmentation
│ └── weights/
│ ├── best.pt
│ └── last.pt
├── yolov9m/ → YOLOv9m Detection
│ └── weights/
│ ├── best.pt
│ └── last.pt
├── yolov9c_seg/ → YOLOv9c Segmentation
│ └── weights/
│ ├── best.pt
│ └── last.pt
├── yolo11m/ → YOLOv11m Detection (prompt untuk Hybrid)
│ └── weights/
│ ├── best.pt
│ └── last.pt
├── yolo11m_seg/ → YOLOv11m Segmentation
│ └── weights/
│ ├── best.pt
│ └── last.pt
└── maskrcnn/ → Mask R-CNN
└── weights/
├── best.pt
├── last.pt
└── last_checkpoint.pt

```

**Catatan kompresi (dijalankan otomatis setelah training):**
- Folder dalam `runs/` dikompres: `yolov8m.tar.gz`, `yolov8m_seg.tar.gz`, dll
- File `best.pt` dan `last.pt` tidak dikompres, langsung di-backup ke Google Drive
- Folder `reports/`, `visuals/`, dan `image_samples/` tidak dikompres (ukuran kecil)
```

```bash
# Gunakan GPU 1 dan 2 (GPU 0 idle untuk Mask R-CNN nanti)
cd /root/MyFineTunning/yolo/yolo8
python -u main.py --device 1,2 2>&1 | tee yolov8_train.log

cd /root/MyFineTunning/yolo/yolo9
python -u main.py --device 1,2 2>&1 | tee yolov9_train.log

cd /root/MyFineTunning/yolo/yolo11
python -u main.py --device 1,2 2>&1 | tee yolo11_train.log

# Mask R-CNN — single GPU, pilih GPU 0 yang bebas
cd /root/MyFineTunning/mask-r-cnn
python -u main.py --device 0 2>&1 | tee maskrcnn_train.log
```

**Ringkasan aturan:**
| Script | `--device` | Behaviour |
|---|---|---|
| `yolo8`, `yolo9`, `yolo11` | `1,2` | DDP ke GPU 1+2 (Ultralytics native) |
| `yolo8`, `yolo9`, `yolo11` | `0` | Single GPU 0 |
| `mask-r-cnn` | `0` / `1` / `2` | Single GPU saja (DDP tidak support) |
| Semua | tidak diisi | Auto-detect semua GPU |

> Catatan: Mask R-CNN akan auto-ambil GPU pertama dari list jika salah ketik `--device 1,2` (ada warning di output).

**Xorg untuk pengaturan FAN**: Kontrol ini akan terus aktif selama `cv-host` tidak di-reboot lagi. (Jika di-reboot, Anda cukup jalankan `sudo nohup Xorg :0 &` sekali lagi).

```bash
tmux kill-session -t yolo8training 2>/dev/null; echo "Killed old Yolo9 session" && tmux new-session -d -s yolo8training "source /home/my/Computer-Vision/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Computer-Vision/MyFineTunning-dev/yolo/yolo8 && python -u main.py 2>&1 | tee yolo8training.log" && echo "New Yolo8 session started"

tmux kill-session -t yolo9training 2>/dev/null; echo "Killed old Yolo9 session" && tmux new-session -d -s yolo9training "source /home/my/Computer-Vision/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Computer-Vision/MyFineTunning-dev/yolo/yolo9 && python -u main.py 2>&1 | tee yolo9training.log" && echo "New Yolo9 session started"

tmux kill-session -t yolo11training 2>/dev/null; echo "Killed old Yolo11 session" && tmux new-session -d -s yolo11training "source /home/my/Computer-Vision/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Computer-Vision/MyFineTunning-dev/yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log" && echo "New Yolo11 session started"

tmux kill-session -t maskrcnntraining 2>/dev/null; echo "Killed old Mask R-CNN session" && tmux new-session -d -s maskrcnntraining "source /home/my/Computer-Vision/MyFineTuning-dev/.venv/bin/activate && cd /home/my/Computer-Vision/MyFineTuning-dev/mask-r-cnn && python -u train_multigpu.py 2>&1 | tee train_multigpu.log" && echo "New Mask R-CNN session started"
```

## Konfigurasi Environment

Pastikan file konfigurasi berikut sudah disiapkan:

- `.env` — Konfigurasi credential (jangan commit ke Git)
- `.env.example` — Template untuk konfigurasi
- `rclone.conf` — Konfigurasi rclone untuk backup (jangan commit ke Git)
- `rclone.conf.example` — Template konfigurasi rclone

## Menjalankan Pipeline

1. **Setup Environment**:

   ```bash
   source .venv/bin/activate
   ```

2. **Jalankan Training**:
   - Gunakan perintah tmux di atas untuk setiap model
   - Monitor dengan: `tmux attach -t yolo8training`

3. **Backup ke Google Drive**:
   ```bash
   ./rclone_sync.sh
   ```

## Catatan Penting

- Pastikan GPU tersedia sebelum menjalankan training
- Gunakan `nvidia-smi` untuk memantau penggunaan GPU
- Backup otomatis akan berjalan setelah training selesai
- Notifikasi Telegram akan dikirim untuk status training dan backup
