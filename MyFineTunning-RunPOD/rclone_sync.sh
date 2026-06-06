#!/usr/bin/env bash
# ==============================================================================
# rclone_sync.sh — Sinkronisasi hasil training ke Google Drive
# ==============================================================================
# Cara pakai:
#   bash rclone_sync.sh upload    → Push semua hasil training ke GDrive
#   bash rclone_sync.sh download  → Pull weights dari GDrive
#   bash rclone_sync.sh status    → Tampilkan info & isi GDrive
#
#   Background Execution (TMUX):
#   1. Upload = tmux new-session -d -s rclone_session "cd /root/Trainning-Models/MyFineTunning-RunPOD && bash rclone_sync.sh upload 2>&1 | tee RCloneReport.log"
#
# Struktur folder di GDrive:
#   gdrive-backup/
#   └── {hostname}/
#       └── MyFineTunning-{timestamp}/
#           ├── reports/
#           ├── visuals/
#           ├── image_samples/
#           ├── fine_models/          ← {model}-best.pt
#           └── fine_models_archive/  ← arsip .tar.zst (backup penuh)
#
# Alur Upload:
#   1. Kompres seluruh folder MyFineTunning-xxxx → rclone_local/*.tar.zst
#   2. Upload arsip → GDrive: fine_models_archive/
#   3. Hapus arsip lokal setelah berhasil
#   4. Upload reports/, visuals/, image_samples/ satu per satu
#   5. Upload best.pt per model → fine_models/{model}-best.pt
#
# Alur Download:
#   1. Download arsip dari fine_models_archive/
#   2. Ekstrak otomatis ke data-files/
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

# Fallback ke python sistem jika venv tidak ada
if [[ ! -f "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# ── Warna terminal ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Validasi Python ────────────────────────────────────────────────────────────
if [[ -z "$PYTHON_BIN" ]]; then
    echo -e "${RED}❌ Python tidak ditemukan.${RESET}"
    exit 1
fi

# ── Info ───────────────────────────────────────────────────────────────────────
show_info() {
    echo -e "${CYAN}${BOLD}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RClone Sync — MyFineTunning-RunPOD"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${RESET}"
    echo "  Hostname  : $(hostname)"
    echo "  Python    : $PYTHON_BIN"
    echo "  Project   : $SCRIPT_DIR"
    echo ""
}

# ── Upload ─────────────────────────────────────────────────────────────────────
do_upload() {
    show_info
    echo -e "${CYAN}${BOLD}📤 UPLOAD ke Google Drive...${RESET}"
    echo ""

    cd "$SCRIPT_DIR"
    "$PYTHON_BIN" - <<'EOF'
from rclone_utils import upload_results
import sys
ok = upload_results()
sys.exit(0 if ok else 1)
EOF

    if [[ $? -eq 0 ]]; then
        echo -e "\n${GREEN}${BOLD}✅ Upload selesai!${RESET}"
    else
        echo -e "\n${RED}${BOLD}❌ Upload selesai dengan beberapa error.${RESET}"
        exit 1
    fi
}

# ── Download ───────────────────────────────────────────────────────────────────
do_download() {
    show_info
    echo -e "${CYAN}${BOLD}📥 DOWNLOAD dari Google Drive...${RESET}"
    echo ""

    cd "$SCRIPT_DIR"
    "$PYTHON_BIN" - <<'EOF'
from rclone_utils import download_weights
import sys
ok = download_weights(auto_extract=True)
sys.exit(0 if ok else 1)
EOF

    if [[ $? -eq 0 ]]; then
        echo -e "\n${GREEN}${BOLD}✅ Download & ekstraksi selesai!${RESET}"
    else
        echo -e "\n${RED}${BOLD}❌ Download selesai dengan beberapa error.${RESET}"
        exit 1
    fi
}

# ── Status ─────────────────────────────────────────────────────────────────────
do_status() {
    show_info
    echo -e "${CYAN}${BOLD}📋 Status Google Drive...${RESET}"
    echo ""

    cd "$SCRIPT_DIR"
    "$PYTHON_BIN" - <<'EOF'
from rclone_utils import get_status, list_remote
status = get_status()
print("  Konfigurasi RClone:")
for k, v in status.items():
    print(f"    {k:20s}: {v}")
print()
list_remote()
EOF
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
        echo "  bash rclone_sync.sh download  → Download & ekstrak dari GDrive"
        echo "  bash rclone_sync.sh status    → Tampilkan info & isi GDrive"
        exit 0
        ;;
esac
