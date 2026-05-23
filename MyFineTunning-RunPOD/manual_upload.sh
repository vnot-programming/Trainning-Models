#!/usr/bin/env bash
# ==============================================================================
# manual_upload.sh — Manual Upload Workspace to Google Drive via Rclone
# ==============================================================================
#
# Deskripsi:
#   Script shell interaktif & aman untuk mengunggah berkas hasil training
#   (weights, reports, visuals, logs) dari workspace aktif ke Google Drive.
#
# Penggunaan:
#   bash manual_upload.sh                      # Mode interaktif
#   bash manual_upload.sh --remote gdrive      # Tentukan remote name
# 
# Viewed main.py:475-489
# Searched for "rclone"
# Viewed rclone_sync.sh:1-157
# Viewed custom_upload.py:1-346
# Edited manual_upload.sh

# I have successfully created an elegant, high-quality, and robust shell script named `manual_upload.sh` in your workspace root!

# ### 🌟 Key Features of `manual_upload.sh`:
# 1. **Dynamic Workspace Discovery**: Automatically reads the active workspace ID from `.workspace_id`. If not found, it intelligently scans the `data-files/` folder to target the latest active run (`MyFineTunning-*`).
# 2. **Interactive Remote Selection**: Automatically lists all configured Rclone remotes and prompts you to select one, or specify a custom remote manually.
# 3. **Smart Archiving / Tar Compression**: Scans your `runs` directory and checks if model weights (YOLO, Mask R-CNN) are archived (`.tar.gz`). If a folder is not zipped, it will automatically compress it on the fly before uploading to save bandwidth and storage.
# 4. **Clean GDrive Structure**: Organizes your upload into:
#    - `gdrive-backup/runpod/MyFineTunning-{timestamp}/runs/*.tar.gz` (archives)
#    - `gdrive-backup/runpod/MyFineTunning-{timestamp}/reports/`
#    - `gdrive-backup/runpod/MyFineTunning-{timestamp}/visuals/`
#    - `gdrive-backup/runpod/MyFineTunning-{timestamp}/image_samples/`
# 5. **Real-time Progress Indicator**: Uses `rclone copy --progress` to display a beautiful live progress bar with exact ETA, speed, and percentage directly in your terminal.

# ### 🚀 How to Run:
# Ensure the script is executable and execute it:
# ```bash
# cd /root/Trainning-Models/MyFineTunning-RunPOD
# chmod +x manual_upload.sh
# ./manual_upload.sh
# ```

# You can also run it in the background using tmux to prevent disconnection issues:
# ```bash
# tmux new-session -d -s manual_upload "cd /root/Trainning-Models/MyFineTunning-RunPOD && manual_upload.sh --remote gdrive 2>&1 | tee manual_upload.log"

# tmux new-session -d -s manual_upload "cd /root/Trainning-Models/MyFineTunning-RunPOD && ./manual_upload.sh --remote gdrive 2>&1 | tee manual_upload.log"

# ```

# ==============================================================================

set -euo pipefail

# ── WARNA TERMINAL ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

log_info() {
    echo -e "${CYAN}[Info]${RESET} $1"
}
log_success() {
    echo -e "${GREEN}[Success]${RESET} ${BOLD}$1${RESET}"
}
log_warn() {
    echo -e "${YELLOW}[Warning]${RESET} $1"
}
log_err() {
    echo -e "${RED}[Error]${RESET} ${BOLD}$1${RESET}"
}

# ── VALIDASI ALAT ──────────────────────────────────────────────────────────────
if ! command -v rclone &> /dev/null; then
    log_err "rclone tidak terdeteksi! Silakan install rclone terlebih dahulu:"
    echo "   curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

# ── TEMUKAN WORKSPACE ROOT ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── TELEGRAM NOTIFICATION HELPER ───────────────────────────────────────────────
send_telegram() {
    local msg="$1"
    python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
try:
    from telegram_utils import send_telegram_msg
    send_telegram_msg('''$msg''')
except Exception as e:
    pass
" &>/dev/null || true
}

SUCCESS=false

failure_notification() {
    if [[ "$SUCCESS" == "false" ]]; then
        log_err "Upload manual dibatalkan atau terjadi kesalahan!"
        send_telegram "⚠️ <b>Manual Upload Failed / Aborted</b>
Workspace: <code>MyFineTunning-${WORKSPACE_ID:-unknown}</code>"
    fi
}
trap failure_notification EXIT ERR INT

# Baca .workspace_id
WORKSPACE_ID=""
if [[ -f ".workspace_id" ]]; then
    WORKSPACE_ID=$(cat .workspace_id | tr -d '\r\n[:space:]')
fi

# Temukan folder di data-files
DATA_FILES_DIR="${SCRIPT_DIR}/data-files"
WORKSPACE_DIR=""

if [[ -n "$WORKSPACE_ID" && -d "${DATA_FILES_DIR}/MyFineTunning-${WORKSPACE_ID}" ]]; then
    WORKSPACE_DIR="${DATA_FILES_DIR}/MyFineTunning-${WORKSPACE_ID}"
else
    # Fallback: Cari folder MyFineTunning-* paling baru
    if [[ -d "$DATA_FILES_DIR" ]]; then
        LATEST_DIR=$(ls -td "${DATA_FILES_DIR}"/MyFineTunning-* 2>/dev/null | head -1 || true)
        if [[ -n "$LATEST_DIR" ]]; then
            WORKSPACE_DIR="$LATEST_DIR"
            WORKSPACE_ID=$(basename "$LATEST_DIR" | sed 's/MyFineTunning-//')
        fi
    fi
fi

# Cetak header
echo -e "${CYAN}${BOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 RClone Manual Upload Utility"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${RESET}"

if [[ -z "$WORKSPACE_DIR" || ! -d "$WORKSPACE_DIR" ]]; then
    log_err "Tidak ditemukan workspace aktif di data-files/!"
    exit 1
fi

log_info "Workspace Aktif: ${BOLD}MyFineTunning-${WORKSPACE_ID}${RESET}"
log_info "Path Workspace : ${WORKSPACE_DIR}"

# ── MEMILIH REMOTE ──────────────────────────────────────────────────────────────
RCLONE_REMOTE="gdrive"
if [[ "${1:-}" == "--remote" && -n "${2:-}" ]]; then
    RCLONE_REMOTE="$2"
else
    # Cek daftar remote yang tersedia
    REMOTES=$(rclone listremotes | tr -d ':' || true)
    if [[ -n "$REMOTES" ]]; then
        echo -e "\n${BOLD}Daftar Remote Rclone:${RESET}"
        select opt in $REMOTES "Tulis manual..."; do
            if [[ "$opt" == "Tulis manual..." ]]; then
                read -rp "Masukkan nama remote rclone: " custom_remote
                RCLONE_REMOTE="$custom_remote"
                break
            elif [[ -n "$opt" ]]; then
                RCLONE_REMOTE="$opt"
                break
            fi
        done
    else
        log_warn "Tidak ada remote terkonfigurasi. Silakan jalankan 'rclone config' terlebih dahulu."
        read -rp "Atau ketik nama remote jika sudah yakin: " custom_remote
        RCLONE_REMOTE="$custom_remote"
    fi
fi

# Validasi remote terpilih
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:" 2>/dev/null; then
    log_warn "Remote '${RCLONE_REMOTE}' mungkin belum terkonfigurasi dengan benar."
fi

# Tentukan path di Google Drive
GDRIVE_DEST="gdrive-backup/runpod/MyFineTunning-${WORKSPACE_ID}"

echo -e "\n${BOLD}Konfigurasi Upload:${RESET}"
echo -e "  Remote GDrive  : ${GREEN}${RCLONE_REMOTE}:${GDRIVE_DEST}${RESET}"
echo -e "  Sumber Lokal   : ${YELLOW}${WORKSPACE_DIR}${RESET}"

echo -e "\n${BOLD}Ingin melanjutkan upload? [y/N]${RESET}"
read -rp "> " choice
if [[ ! "$choice" =~ ^[Yy]$ ]]; then
    log_info "Upload dibatalkan oleh pengguna."
    exit 0
fi

# ── PREPARATION: KOMPRESI FOLDER HASIL TRAINING ──────────────────────────────────
RUNS_DIR="${WORKSPACE_DIR}/runs"
if [[ -d "$RUNS_DIR" ]]; then
    log_info "Memeriksa arsip kompresi .tar.gz untuk setiap model di folder runs..."
    for model_dir in "$RUNS_DIR"/*/; do
        if [[ -d "$model_dir" ]]; then
            model_name=$(basename "$model_dir")
            # Lewati jika extension adalah tar.gz
            if [[ "$model_name" =~ \.tar\.gz$ ]]; then
                continue
            fi
            
            tar_path="${RUNS_DIR}/${model_name}.tar.gz"
            if [[ ! -f "$tar_path" ]]; then
                log_info "Mengompres folder training: ${model_name}..."
                tar -czf "$tar_path" -C "$RUNS_DIR" "$model_name"
                log_success "Arsip berhasil dibuat: $(basename "$tar_path") ($(du -sh "$tar_path" | cut -f1))"
            else
                log_info "Arsip ${model_name}.tar.gz sudah ada. Lewati kompresi."
            fi
        fi
    done
fi

# ── MEMULAI UPLOAD ─────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
log_success "Memulai upload ke Google Drive..."

send_telegram "☁️ <b>Manual Upload Started</b>
Workspace: <code>MyFineTunning-${WORKSPACE_ID}</code>
Remote: <code>${RCLONE_REMOTE}:${GDRIVE_DEST}</code>"

# List item untuk diupload
# 1. Upload semua file .tar.gz di runs
if [[ -d "$RUNS_DIR" ]]; then
    log_info "Mengunggah file kompresi weights (.tar.gz)..."
    rclone copy "$RUNS_DIR" "${RCLONE_REMOTE}:${GDRIVE_DEST}/runs" \
        --include "*.tar.gz" \
        --progress \
        --transfers 4 \
        --log-level NOTICE
fi

# 2. Upload folder-folder pendukung
SUPPORT_FOLDERS=("reports" "visuals" "image_samples")
for folder in "${SUPPORT_FOLDERS[@]}"; do
    local_folder="${WORKSPACE_DIR}/${folder}"
    if [[ -d "$local_folder" ]]; then
        log_info "Mengunggah folder: ${folder}/..."
        rclone copy "$local_folder" "${RCLONE_REMOTE}:${GDRIVE_DEST}/${folder}" \
            --progress \
            --transfers 4 \
            --log-level NOTICE
    fi
done

# ── SELESAI ────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

log_success "Semua berkas berhasil diunggah!"
echo -e "  ⏱️  Total Waktu : ${GREEN}${MINUTES}m ${SECONDS}s${RESET}"
echo -e "  📂 Lokasi GDrive: ${GREEN}${RCLONE_REMOTE}:${GDRIVE_DEST}/${RESET}"

SUCCESS=true
trap - EXIT ERR INT

send_telegram "🏁 <b>Manual Upload Finished</b>
Workspace: <code>MyFineTunning-${WORKSPACE_ID}</code>
Duration: <code>${MINUTES}m ${SECONDS}s</code>
Destination: <code>${RCLONE_REMOTE}:${GDRIVE_DEST}</code>"

# Tampilkan hasil upload di remote
echo -e "\n${CYAN}Isi direktori tujuan di Google Drive:${RESET}"
rclone lsf "${RCLONE_REMOTE}:${GDRIVE_DEST}/" --max-depth 2 || true
