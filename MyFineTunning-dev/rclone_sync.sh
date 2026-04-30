#!/usr/bin/env bash
# ==============================================================================
# rclone_sync.sh — Sinkronisasi hasil training ke Google Drive
# ==============================================================================
#
# Cara pakai:
#   bash rclone_sync.sh upload    → Push semua hasil training ke GDrive
#   bash rclone_sync.sh download  → Pull weights dari GDrive
#   bash rclone_sync.sh status    → Tampilkan info & isi GDrive
#
# Struktur folder di GDrive:
#   gdrive-backup/
#   └── {hostname}/
#       └── MyFineTunning-{timestamp}/
#           ├── data-files/reports/
#           ├── data-files/visuals/
#           └── runs/  (berisi best.pt per model)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Baca .env ──────────────────────────────────────────────────────────────────
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Konfigurasi (default jika tidak ada di .env) ────────────────────────────────
RCLONE_BIN="${RCLONE_BIN:-$HOME/.local/bin/rclone}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
RCLONE_DEST="${RCLONE_DEST:-gdrive-backup}"
HOSTNAME_VAL="$(hostname)"

# ── Baca workspace timestamp ────────────────────────────────────────────────────
WS_ID_FILE="$SCRIPT_DIR/.workspace_id"
if [[ -f "$WS_ID_FILE" ]]; then
    TIMESTAMP="$(cat "$WS_ID_FILE")"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi

# ── Path remote ─────────────────────────────────────────────────────────────────
REMOTE_FOLDER="${RCLONE_REMOTE}:${RCLONE_DEST}/${HOSTNAME_VAL}/MyFineTunning-${TIMESTAMP}"

# ── Warna terminal ───────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Cek rclone tersedia ──────────────────────────────────────────────────────────
if ! command -v "$RCLONE_BIN" &>/dev/null && ! command -v rclone &>/dev/null; then
    echo -e "${RED}❌ rclone tidak ditemukan.${RESET}"
    echo "   Install: curl https://rclone.org/install.sh | bash"
    echo "   Atau manual ke ~/.local/bin/"
    exit 1
fi

# Gunakan rclone dari ~/.local/bin jika ada, fallback ke PATH
RCLONE_CMD="$RCLONE_BIN"
if ! command -v "$RCLONE_BIN" &>/dev/null; then
    RCLONE_CMD="rclone"
fi

# ==============================================================================
# FUNGSI
# ==============================================================================

show_info() {
    echo -e "${CYAN}${BOLD}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RClone Sync — MyFineTunning-dev"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${RESET}"
    echo "  Hostname     : $HOSTNAME_VAL"
    echo "  Timestamp    : $TIMESTAMP"
    echo "  Remote Folder: $REMOTE_FOLDER"
    echo "  Local Root   : $SCRIPT_DIR"
    echo ""
}

do_upload() {
    show_info
    echo -e "${CYAN}${BOLD}📤 UPLOAD ke Google Drive...${RESET}"
    echo ""

    # 1. Upload data-files (reports + visuals)
    if [[ -d "$SCRIPT_DIR/data-files" ]]; then
        echo -e "${YELLOW}→ Upload data-files/ (reports & visuals)${RESET}"
        "$RCLONE_CMD" copy "$SCRIPT_DIR/data-files" \
            "$REMOTE_FOLDER/data-files" \
            --progress --create-empty-src-dirs
        echo ""
    fi

    # 2. Upload runs (weights) dari semua workspace folder
    for ws_dir in "$SCRIPT_DIR"/MyFineTunning-*/; do
        if [[ -d "$ws_dir/runs" ]]; then
            ws_name="$(basename "$ws_dir")"
            echo -e "${YELLOW}→ Upload runs/ dari $ws_name${RESET}"
            "$RCLONE_CMD" copy "$ws_dir/runs" \
                "$REMOTE_FOLDER/runs" \
                --include "*.pt" \
                --include "*.csv" \
                --include "*.png" \
                --progress
            echo ""
        fi
    done

    echo -e "${GREEN}${BOLD}✅ Upload selesai!${RESET}"
    echo -e "   📁 GDrive: ${REMOTE_FOLDER}"
}

do_download() {
    show_info
    echo -e "${CYAN}${BOLD}📥 DOWNLOAD dari Google Drive...${RESET}"
    echo ""

    LOCAL_DOWNLOADS="$SCRIPT_DIR/downloads"
    mkdir -p "$LOCAL_DOWNLOADS/weights"

    echo -e "${YELLOW}→ Download weights (*.pt) → $LOCAL_DOWNLOADS/weights/${RESET}"
    "$RCLONE_CMD" copy "$REMOTE_FOLDER/runs" \
        "$LOCAL_DOWNLOADS/weights" \
        --include "*.pt" \
        --progress

    echo ""
    echo -e "${GREEN}${BOLD}✅ Download selesai!${RESET}"
    echo -e "   📁 Lokal: $LOCAL_DOWNLOADS/weights/"
}

do_status() {
    show_info
    echo -e "${CYAN}${BOLD}📋 Status Google Drive (gdrive-backup/${HOSTNAME_VAL}/)${RESET}"
    echo ""

    # List semua folder session di bawah hostname
    "$RCLONE_CMD" lsd "${RCLONE_REMOTE}:${RCLONE_DEST}/${HOSTNAME_VAL}/" 2>/dev/null || \
        echo "  (Belum ada data untuk host ini)"

    echo ""
    echo -e "${CYAN}${BOLD}📋 Folder aktif saat ini: $REMOTE_FOLDER${RESET}"
    "$RCLONE_CMD" ls "$REMOTE_FOLDER" 2>/dev/null | head -n 20 || \
        echo "  (Belum ada data di folder ini)"
}

# ==============================================================================
# MAIN
# ==============================================================================
ACTION="${1:-}"

case "$ACTION" in
    upload)
        do_upload
        ;;
    download)
        do_download
        ;;
    status)
        do_status
        ;;
    *)
        echo -e "${BOLD}Penggunaan:${RESET}"
        echo "  bash rclone_sync.sh upload    → Upload hasil training ke GDrive"
        echo "  bash rclone_sync.sh download  → Download weights dari GDrive"
        echo "  bash rclone_sync.sh status    → Tampilkan isi GDrive"
        exit 0
        ;;
esac
