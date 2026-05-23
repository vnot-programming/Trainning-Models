#!/usr/bin/env bash
# ==============================================================================
# force_reupload_archive.sh
# Script untuk memaksa kompresi ulang dan re-upload arsip ZIP ke Google Drive
# ==============================================================================
# You can also run it in the background using tmux to prevent disconnection issues:
# ```bash
# tmux new-session -d -s force_reupload_archive "/root/Trainning-Models/MyFineTunning-RunPOD/force_reupload_archive.sh 2>&1 | tee force_reupload_archive.log"
# ```
set -euo pipefail

# Konfigurasi Path
WORKSPACE_ID=$(cat .workspace_id 2>/dev/null || echo "20260523_022725")
WORKSPACE_DIR="MyFineTunning-${WORKSPACE_ID}"
HOSTNAME=$(hostname)
REMOTE_ARCHIVE_PATH="gdrive:gdrive-backup/${HOSTNAME}/${WORKSPACE_DIR}/fine_models_archive"

echo "================================================================="
echo "  🔄 FORCE REPACK & REUPLOAD ARCHIVE"
echo "================================================================="
echo "Target Workspace : ${WORKSPACE_DIR}"
echo "Remote Path      : ${REMOTE_ARCHIVE_PATH}"
echo ""

echo "🗑️  Langkah 1: Menghapus arsip lama (.tar.gz / .tar.zst) dari Google Drive..."
# Menghapus file kompresi lama agar rclone_sync.sh tidak men-skip proses repack
if rclone delete --include "*.tar.*" "${REMOTE_ARCHIVE_PATH}"; then
    echo "✅ Berhasil. Arsip lama di GDrive telah dihapus (atau memang tidak ada)."
else
    echo "⚠️  Peringatan: Gagal menghapus atau folder belum terbentuk di GDrive."
fi
echo ""

echo "📦 Langkah 2: Menjalankan rclone_sync.sh upload..."
# Mengeksekusi script rclone bawaan Anda
# Karena file di GDrive sudah hilang, script ini otomatis akan melakukan kompresi 
# dari awal dan mengupload versi terbarunya!
# Pindah ke direktori utama agar rclone_sync.sh ditemukan
cd /root/Trainning-Models/MyFineTunning-RunPOD
bash rclone_sync.sh upload

echo ""
echo "🏁 Selesai! Seluruh data terbaru Anda telah diamankan ke cloud."
