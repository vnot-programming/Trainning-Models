#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Setup environment untuk MyFineTunning (RunPOD)
# ==============================================================================
#
# Cara pakai:
#
# 1. Background Execution (TMUX):
#    tmux new-session -d -s setup_session "cd /root/Trainning-Models/MyFineTunning-dev && bash setup.sh 2>&1 | tee SetupReport.log"
#    atau
#    tmux new-session -d -s setup_session "cd C:\Users\Server\Documents\~S3\Tunning\MyFineTunning-20260502_134510-RunPOD && bash setup.sh 2>&1 | tee SetupReport.log"
#
# 2. Direct Execution:
#    cd /Trainning-Models/MyFineTunning-dev && bash setup.sh
#    atau
#    cd /mnt/c/Users/Server/Documents/~S3/Tunning/MyFineTunning-20260502_134510-RunPOD && bash setup.sh
# 3. Mode Options:
#    bash setup.sh            ← Standard (New .venv)
#    bash setup.sh --reuse    ← Reuse MyTrainEngine .venv
#
# Setelah setup:
#   source .venv/bin/activate
#   python main.py           ← Jalankan root setup
# ==============================================================================

set -e  # Exit langsung jika ada error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REUSE_PATH="/root/MyTrainEngine/.venv"

echo "============================================================"
echo "  MyFineTunning — Setup Environment"
echo "  Dir: $SCRIPT_DIR"
echo "============================================================"

# ------------------------------------------------------------------------------
# Opsi --reuse: symlink ke .venv MyTrainEngine yang sudah lengkap
# ------------------------------------------------------------------------------
if [[ "$1" == "--reuse" ]]; then
    if [[ -d "$REUSE_PATH" ]]; then
        echo "[Setup] Reuse .venv dari: $REUSE_PATH"
        if [[ ! -L "$VENV_DIR" ]]; then
            ln -sf "$REUSE_PATH" "$VENV_DIR"
            echo "[Setup] Symlink dibuat: $VENV_DIR → $REUSE_PATH"
        else
            echo "[Setup] Symlink sudah ada."
        fi
        echo ""
        echo "✅ Siap. Aktifkan dengan:"
        echo "   source $VENV_DIR/bin/activate"
        exit 0
    else
        echo "[Setup] ⚠️  $REUSE_PATH tidak ditemukan. Buat .venv baru."
    fi
fi

# ------------------------------------------------------------------------------
# Buat .venv baru jika belum ada
# ------------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[Setup] Membuat virtual environment di: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[Setup] .venv sudah ada: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[Setup] Python: $(which python3) — $(python3 --version)"

# ------------------------------------------------------------------------------
# Upgrade pip
# ------------------------------------------------------------------------------
pip install --upgrade pip --quiet

# Cek apakah torch sudah terinstall dan kompatibel
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[Setup] torch dengan dukungan CUDA sudah tersedia."
else
    echo "[Setup] torch belum ada atau CUDA tidak terdeteksi — install versi kompatibel (cu121)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --quiet
fi

# ------------------------------------------------------------------------------
# Install requirements
# ------------------------------------------------------------------------------
echo ""
echo "[Setup] Menginstall requirements.txt ..."
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo "[Setup] ✅ Semua package terinstall."

# ------------------------------------------------------------------------------
# Verifikasi kunci
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
echo "  Aktifkan env  : source $VENV_DIR/bin/activate"
echo "  Jalankan setup: python main.py"
echo "  Konfigurasi  : config_shared.py"
echo ""
echo "  Urutan training (Default menggunakan GPU 0 | Jika Multi maka Paralel Aktif):"
echo "  1. YOLO8   : cd yolo/yolo8  && python -u main.py 2>&1 | tee yolo8training.log"
echo "  2. YOLO9   : cd yolo/yolo9  && python -u main.py 2>&1 | tee yolo9training.log"
echo "  3. YOLO11  : cd yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log"
echo "  4. MaskRCNN: cd mask-r-cnn  && python -u train_multigpu.py 2>&1 | tee maskrcnntraining.log"
echo "  5. Hybrid  : cd hybrid      && python -u main.py 2>&1 | tee hybridtraining.log"
echo ""
echo "  💡 Tips GPU: Tambahkan '--device 1,2' jika ingin menggunakan GPU nomor 1 dan 2 saja."
echo ""
echo "  🛠️ Background Execution (tmux):"
echo "  - YOLO8 :"
echo "    tmux new-session -d -s yolo8training \"source $VENV_DIR/bin/activate && cd $SCRIPT_DIR/yolo/yolo8 && python -u main.py 2>&1 | tee yolo8training.log\""
echo ""
echo "  - YOLO9 :"
echo "    tmux new-session -d -s yolo9training \"source $VENV_DIR/bin/activate && cd $SCRIPT_DIR/yolo/yolo9 && python -u main.py 2>&1 | tee yolo9training.log\""
echo ""
echo "  - YOLO11:"
echo "    tmux new-session -d -s yolo11training \"source $VENV_DIR/bin/activate && cd $SCRIPT_DIR/yolo/yolo11 && python -u main.py 2>&1 | tee yolo11training.log\""
echo ""
echo "  - MaskRCNN:"
echo "    tmux new-session -d -s maskrcnntraining \"source $VENV_DIR/bin/activate && cd $SCRIPT_DIR/mask-r-cnn && python -u train_multigpu.py 2>&1 | tee maskrcnntraining.log\""
echo ""
echo "  - Hybrid:"
echo "    tmux new-session -d -s hybridtraining \"source $VENV_DIR/bin/activate && cd $SCRIPT_DIR/hybrid && python -u main.py 2>&1 | tee hybridtraining.log\""
echo ""
if ! command -v tmux &> /dev/null; then
    echo "  ⚠️  Peringatan: 'tmux' belum terinstall. Jalankan: sudo apt update && sudo apt install tmux -y"
fi
echo "============================================================"

# ==============================================================================
# AUTO-RUN main.py (Setup workspace + verifikasi/download dataset)
# ==============================================================================
echo ""
echo "============================================================"
echo "  🚀 Menjalankan main.py (setup workspace & dataset)..."
echo "============================================================"
python3 "$SCRIPT_DIR/main.py"
