#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Setup environment untuk MyFineTunning (RunPOD / GPU Server)
# ==============================================================================
#
# Cara pakai:
#
# 1. Background Execution via TMUX (Sangat Direkomendasikan untuk Cloud):
#    tmux new-session -d -s setup_session "cd /home/my/Trainning-Models/MyFineTunning-RunPOD && bash setup.sh 2>&1 | tee SetupReport.log"
#
# 2. Direct Execution:
#    cd /home/my/Trainning-Models/MyFineTunning-RunPOD && bash setup.sh
#
# 3. Mode Options:
#    bash setup.sh            ← Standar (Membuat/memperbarui .venv lokal terisolasi)
#    bash setup.sh --reuse    ← Menggunakan ulang .venv dari MyFineTunning-dev
#
# Setelah setup sukses:
#   source /data/programs/anaconda3/bin/activate && conda activate yolo_env
#   python run_pipeline_parallel.py              ← Menjalankan Multi-GPU Parallel Pipeline Scheduler (Cerdas)
# ==============================================================================

set -e  # Exit langsung jika ada error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REUSE_PATH="/home/my/Trainning-Models/MyFineTunning-dev/.venv"

echo "============================================================"
echo "  MyFineTunning — Setup Environment"
echo "  Dir: $SCRIPT_DIR"
echo "============================================================"

# ------------------------------------------------------------------------------
# Opsi --reuse: symlink ke .venv MyFineTunning-dev yang sudah lengkap
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
# Cek Host CUDA Version terlebih dahulu
# ------------------------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -i "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/')
    USE_CU130=$(python3 -c "print(1 if float('${CUDA_VER:-0.0}') >= 13.0 else 0)" 2>/dev/null || echo 0)
    USE_CU128=$(python3 -c "print(1 if float('${CUDA_VER:-0.0}') >= 12.8 else 0)" 2>/dev/null || echo 0)
    USE_CU126=$(python3 -c "print(1 if float('${CUDA_VER:-0.0}') >= 12.6 else 0)" 2>/dev/null || echo 0)
else
    CUDA_VER="0.0"
    USE_CU130=0; USE_CU128=0; USE_CU126=0
fi

# Fungsi Cek Kompatibilitas Torch CUDA
check_torch_cuda() {
    python3 -c "
import torch
import sys
if not torch.cuda.is_available():
    print(0)
    sys.exit(0)
# Jika torch.cuda.is_available() True, berarti versi cu berapapun yang terinstal sudah berfungsi dengan baik.
print(1)
" 2>/dev/null || echo 0
}

# ------------------------------------------------------------------------------
# Menjamin Tersedianya Virtual Environment (.venv) - Terisolasi & Dinamis
# ------------------------------------------------------------------------------
echo "[Setup] Mencari virtual environment lokal (.venv) di: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[Setup] .venv belum ada. Membuat virtual environment baru di: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[Setup] .venv lokal sudah ada: $VENV_DIR"
fi

echo "[Setup] Mengaktifkan virtual environment lokal..."
source "$VENV_DIR/bin/activate"
echo "[Setup] Python aktif: $(which python3) — $(python3 --version)"

# Upgrade pip
pip install --upgrade pip --quiet

# Fungsi Cek Kompatibilitas Torch CUDA (dalam venv)
check_torch_cuda() {
    python3 -c "
import torch
import sys
if not torch.cuda.is_available():
    print(0)
    sys.exit(0)
print(1)
" 2>/dev/null || echo 0
}

# Cek cu versi terinstal setelah venv aktif
echo "[Setup] Mengecek versi PyTorch CUDA di dalam .venv..."
TORCH_OK=$(check_torch_cuda)

if [ "$TORCH_OK" -eq 1 ]; then
    echo "[Setup] ✅ PyTorch dengan dukungan CUDA yang sesuai (Host: $CUDA_VER) sudah aktif di dalam .venv."
else
    # Install / Download jika belum sesuai di dalam .venv
    echo "[Setup] ⚠️ PyTorch CUDA belum sesuai di dalam .venv. Memulai instalasi..."
    if [ "$USE_CU130" -eq 1 ] || [ "$USE_CU128" -eq 1 ]; then
        echo "[Setup] CUDA >= 12.8 terdeteksi (Host: $CUDA_VER). Arsitektur Blackwell membutuhkan cu128..."
        pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --quiet
    elif [ "$USE_CU126" -eq 1 ]; then
        echo "[Setup] CUDA >= 12.6 terdeteksi (Host: $CUDA_VER) — install versi kompatibel (cu126)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --quiet
    else
        echo "[Setup] CUDA < 12.6 terdeteksi (Host: $CUDA_VER) — install versi kompatibel (cu121)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --quiet
    fi
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
"$VENV_DIR/bin/python" "$SCRIPT_DIR/main.py"
