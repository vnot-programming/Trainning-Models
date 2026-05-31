#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Setup environment untuk MyFineTunning (Slurm / RunPOD / GPU Server)
# Menggunakan Conda (yolo_env) secara eksklusif.
# ==============================================================================

: <<'CARA_PAKAI'

==============================================================================
PANDUAN EKSEKUSI SETUP
==============================================================================

LANGKAH 1: Masuk ke Node GPU (Slurm)
Pastikan Anda mengeksekusi ini di dalam Node Komputasi GPU, BUKAN di Login Node.
Gunakan utilitas yang sudah disediakan:
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
    ./attach_gpu.sh

LANGKAH 2: Eksekusi Setup
Anda memiliki dua opsi eksekusi setelah berada di dalam Node GPU:

Opsi A. Eksekusi Background via TMUX (DIREKOMENDASIKAN AGAR TIDAK TERPUTUS):
    tmux new-session -d -s setup "source /data/programs/anaconda3/bin/activate yolo_env && cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && bash setup.sh 2>&1 | tee setup.log"

Opsi B. Eksekusi Langsung (Interaktif):
    source /data/programs/anaconda3/bin/activate yolo_env
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster
    bash setup.sh

Opsi C. Eksekusi Background didalam TMUX (DIREKOMENDASIKAN AGAR TIDAK TERPUTUS):
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && bash setup.sh 2>&1 | tee setup.log

LANGKAH 3: Jalankan Pipeline Orchestrator (Setelah Setup Selesai)
    python run_pipeline_parallel.py
==============================================================================
CARA_PAKAI

set -e  # Exit langsung jika ada error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  MyFineTunning — Setup Environment (yolo_env)"
echo "  Dir: $SCRIPT_DIR"
echo "============================================================"

# ------------------------------------------------------------------------------
# Mengaktifkan yolo_env
# ------------------------------------------------------------------------------
echo "[Setup] Mengaktifkan conda environment yolo_env..."
source /data/programs/anaconda3/bin/activate yolo_env
echo "[Setup] Python aktif: $(which python3) — $(python3 --version)"

# Upgrade pip
pip install --upgrade pip --quiet

# ------------------------------------------------------------------------------
# Install requirements
# ------------------------------------------------------------------------------
echo ""
echo "[Setup] Menginstall requirements.txt ke dalam yolo_env ..."
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo "[Setup] ✅ Semua package terinstall."

# ------------------------------------------------------------------------------
# Verifikasi package & CUDA
# ------------------------------------------------------------------------------
echo ""
echo "[Setup] Verifikasi package..."
python3 -c "
import torch, torchvision, ultralytics, cv2, matplotlib, numpy
print(f'  torch       : {torch.__version__} | CUDA: {torch.cuda.is_available()}')
print(f'  torchvision : {torchvision.__version__}')
print(f'  ultralytics : {ultralytics.__version__}')
print(f'  opencv      : {cv2.__version__}')
print(f'  numpy       : {numpy.__version__}')
n_gpu = torch.cuda.device_count()
print(f'  GPU tersedia: {n_gpu}')
for i in range(n_gpu):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)')
"

echo ""
echo "============================================================"
echo "  ✅ Setup selesai!"
echo ""
echo "  Aktifkan env  : source /data/programs/anaconda3/bin/activate yolo_env"
echo "  Jalankan setup: python main.py"
echo "  Konfigurasi  : config_shared.py"
echo ""
echo "  Urutan training (Default menggunakan GPU 0 | Jika Multi maka Paralel Aktif):"
echo "  1. YOLO8   : cd yolo/yolo8   && python -u main.py 2>&1 | tee yolo8training.log"
echo "  2. YOLO9   : cd yolo/yolo9   && python -u main.py 2>&1 | tee yolo9training.log"
echo "  3. YOLO10  : cd yolo/yolov10 && python -u main.py 2>&1 | tee yolo10training.log"
echo "  4. YOLO11  : cd yolo/yolo11  && python -u main.py 2>&1 | tee yolo11training.log"
echo "  5. MaskRCNN: cd mask-r-cnn   && python -u train_multigpu.py 2>&1 | tee maskrcnntraining.log"
echo "  6. Hybrid  : cd hybrid       && python -u main.py 2>&1 | tee hybridtraining.log"
echo ""
echo "  💡 Tips GPU: Tambahkan '--device 1,2' jika ingin menggunakan GPU nomor 1 dan 2 saja."
echo ""
echo "  🛠️ Background Execution (tmux):"
echo "  - YOLO8 :"
echo "    tmux new-session -d -s yolo8training \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/yolo/yolo8 && python -u main.py 2>&1 | tee yolo8training.log\""
echo ""
echo "  - YOLO9 :"
echo "    tmux new-session -d -s yolo9training \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/yolo/yolo9 && python -u main.py 2>&1 | tee yolo9training.log\""
echo ""
echo "  - YOLO10:"
echo "    tmux new-session -d -s yolo10training \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/yolo/yolov10 && python -u main.py 2>&1 | tee yolo10training.log\""
echo ""
echo "  - YOLO11:"
echo "    tmux new-session -d -s yolo11training \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log\""
echo ""
echo "  - MaskRCNN:"
echo "    tmux new-session -d -s maskrcnntraining \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/mask-r-cnn && python -u train_multigpu.py 2>&1 | tee maskrcnntraining.log\""
echo ""
echo "  - Hybrid:"
echo "    tmux new-session -d -s hybridtraining \"source /data/programs/anaconda3/bin/activate yolo_env && cd $SCRIPT_DIR/hybrid && python -u main.py 2>&1 | tee hybridtraining.log\""
echo ""

if ! command -v tmux &> /dev/null; then
    echo "  ⚠️  Peringatan: 'tmux' belum terinstall. Menginstall menggunakan conda..."
    conda install -y -c conda-forge tmux ncurses
fi

if command -v tmux &> /dev/null; then
    echo "  ✅ tmux terinstal: $(tmux -V)"
else
    echo "  ⚠️ tmux masih belum terinstal."
fi
echo "============================================================"

# ==============================================================================
# AUTO-RUN main.py (Setup workspace + verifikasi/download dataset)
# ==============================================================================
echo ""
echo "============================================================"
echo "  🚀 Menjalankan main.py (setup workspace & dataset)..."
echo "============================================================"
python "$SCRIPT_DIR/main.py"
