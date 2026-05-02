/workspace/MyFineTunning-20260423_023800/
├── runs/
│   ├── yolov8m/           → dikompres → yolov8m.tar.gz ✅ setelah selesai
│   ├── yolov8m_seg/       → dikompres → yolov8m_seg.tar.gz ✅
│   ├── reports/           ← CSV dari semua model (tidak dikompres, kecil)
│   └── visuals/           ← PNG dari hybrid
└── .workspace_id          ← file penanda timestamp (dibaca semua sub-script)

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

tmux kill-session -t maskrcnntraining 2>/dev/null; echo "Killed old Mask R-CNN session" && tmux new-session -d -s maskrcnntraining "source /home/my/Computer-Vision/MyFineTunning-dev/.venv/bin/activate && cd /home/my/Computer-Vision/MyFineTunning-dev/mask-r-cnn && python -u train_multigpu.py 2>&1 | tee train_multigpu.log" && echo "New Mask R-CNN session started"
```