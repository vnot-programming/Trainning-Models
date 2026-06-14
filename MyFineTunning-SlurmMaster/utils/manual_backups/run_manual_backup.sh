#!/bin/bash
# ==============================================================================
# # Otomatis buat tmux dan jalankan:
#   /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/manual_backups/run_manual_backup.sh
# Wrapper: Membuka sesi tmux 'manual_backup' dan menjalankan script di dalamnya.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_NAME="manual_backup"
PYTHON_SCRIPT="${SCRIPT_DIR}/manual_backup.py"

# Coba aktifkan Anaconda Conda bawaan jika conda tidak terdeteksi
if ! command -v conda &> /dev/null; then
    if [ -f "/data/programs/anaconda3/bin/activate" ]; then
        echo "conda terdeteksi di /data/programs/anaconda3. Mengaktifkan base..."
        source /data/programs/anaconda3/bin/activate base
    fi
fi

# Periksa apakah python3 terinstal
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: 'python3' tidak terinstal atau tidak ditemukan di PATH."
    echo "   Mencoba mengaktifkan environment 'yolo_env'..."
    if [ -f "/data/programs/anaconda3/bin/activate" ]; then
        source /data/programs/anaconda3/bin/activate yolo_env
    fi
    # Cek kembali setelah aktivasi conda
    if ! command -v python3 &> /dev/null; then
        echo "❌ Gagal mengaktifkan Python3 secara otomatis."
        exit 1
    fi
fi

# Periksa apakah tmux terinstal
if ! command -v tmux &> /dev/null; then
    echo "⚠️  'tmux' tidak ditemukan."
    echo "🚀 Mencoba menginstal 'tmux' secara rootless menggunakan conda..."
    if command -v conda &> /dev/null; then
        # Instal tmux + ncurses dari conda-forge agar kompatibel dan bebas warning ncurses
        conda install -y -c conda-forge tmux ncurses
        if ! command -v tmux &> /dev/null; then
            echo "❌ Instalasi tmux via Conda gagal."
            exit 1
        else
            echo "✅ 'tmux' berhasil terinstal via Conda!"
        fi
    else
        echo "❌ Conda tidak terdeteksi. Tidak dapat menginstal tmux secara rootless."
        exit 1
    fi
fi

# Periksa apakah sesi tmux sudah ada
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Sesi tmux '${SESSION_NAME}' sudah berjalan."
    echo "   Gunakan: tmux attach -t ${SESSION_NAME}"
    exit 1
fi

echo "🚀 Membuat sesi tmux '${SESSION_NAME}'..."
echo "   Script: ${PYTHON_SCRIPT}"
echo ""
echo "💡 Setelah masuk tmux:"
echo "   - Ikuti menu interaktif untuk memilih file/folder"
echo "   - Setelah selesai memilih, DETACH tmux (Ctrl+B, lalu D) untuk membiarkan proses berjalan di background"
echo "   - Untuk kembali: tmux attach -t ${SESSION_NAME}"
echo ""

# Buat sesi tmux baru dan langsung jalankan script
tmux new-session -s "$SESSION_NAME" "cd ${SCRIPT_DIR} && python3 ${PYTHON_SCRIPT} 2>&1 | tee ${SCRIPT_DIR}/manual_backup.log; echo ''; echo 'Tekan Enter untuk menutup sesi tmux...'; read"
