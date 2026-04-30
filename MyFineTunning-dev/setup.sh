#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Setup environment untuk MyFineTunning (RunPOD)
# ==============================================================================
#
# Cara pakai:
#   bash setup.sh            ← Setup standar (gunakan .venv baru)
#   bash setup.sh --reuse    ← Reuse .venv dari MyTrainEngine jika ada
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

# ------------------------------------------------------------------------------
# Cek apakah torch sudah terinstall (RunPOD biasanya sudah ada)
# ------------------------------------------------------------------------------
if python3 -c "import torch; print(f'[Setup] torch {torch.__version__} sudah tersedia (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
    echo "[Setup] Melewati instalasi torch (sudah ada)."
else
    echo "[Setup] torch belum ada — install dari PyPI..."
    pip install torch torchvision --quiet
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
echo ""
echo "  Urutan training (satu per satu):"
echo "  1. cd yolo/yolo8  && python -u main.py --device 1,2 2>&1 | tee yolov8_train.log"
echo "  2. cd yolo/yolo9  && python -u main.py --device 1,2 2>&1 | tee yolov9_train.log"
echo "  3. cd yolo/yolo11 && python -u main.py --device 1,2 2>&1 | tee yolo11_train.log"
echo "  4. cd mask-r-cnn  && python -u main.py --device 0   2>&1 | tee maskrcnn_train.log"
echo "  5. cd hybrid      && python -u main.py               2>&1 | tee hybrid_eval.log"
echo "============================================================"
