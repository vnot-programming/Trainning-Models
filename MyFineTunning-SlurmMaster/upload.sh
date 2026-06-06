#!/usr/bin/env bash
# ==============================================================================
# upload.sh — Skrip Pengarsipan dan Kompresi Hasil Training Lokal
# ==============================================================================
# Cara pakai:
#   bash upload.sh pack    → Melakukan proses pemindahan model & kompresi folder lokal
#
# Alur Eksekusi:
#   1. Mengaktifkan lingkungan Conda yolo_env (jika tersedia).
#   2. Menjalankan modul Python utils/upload_utils.py.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

# ── Deteksi Otomatis Conda Environment yolo_env ──────────────────────────────────
CONDA_ACTIVATE="/data/programs/anaconda3/bin/activate"
ENV_NAME="yolo_env"

if [[ -f "$CONDA_ACTIVATE" ]]; then
    # Mengaktifkan conda environment yolo_env
    source "$CONDA_ACTIVATE" "$ENV_NAME"
    PYTHON_BIN="python3"
else
    # Fallback ke python3 bawaan sistem jika conda tidak ditemukan
    if [[ ! -f "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi

# ── Skema Warna Bio-Digital Minimalism (Calibrated HSL Terminal Colors) ──────────
GREEN='\033[38;2;46;204;113m'    # Hijau Zamrud menenangkan
YELLOW='\033[38;2;241;196;15m'   # Kuning hangat lembut
CYAN='\033[38;2;52;152;219m'     # Biru muda dingin/bersih
RED='\033[38;2;231;76;60m'       # Merah pastel aman (bukan pekat)
BOLD='\033[1m'
RESET='\033[0m'

# ── Validasi Keberadaan Python ──────────────────────────────────────────────────
if [[ -z "$PYTHON_BIN" ]]; then
    echo -e "${RED}❌ Kesalahan: Python3 tidak ditemukan di sistem Anda!${RESET}"
    echo -e "   Pastikan environment Conda atau instalasi python3 tersedia."
    exit 1
fi

# ── Tampilan Header ─────────────────────────────────────────────────────────────
show_header() {
    echo -e "${CYAN}${BOLD}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📦 Local Archiver & Compressing Suite — MyFineTunning"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${RESET}"
    echo -e "  • ${BOLD}Waktu Lokal${RESET}  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "  • ${BOLD}Server Node${RESET}  : $(hostname)"
    echo -e "  • ${BOLD}Python Exec${RESET}  : $PYTHON_BIN"
    echo -e "  • ${BOLD}Workspace  ${RESET}  : $SCRIPT_DIR"
    echo ""
}

# ── Eksekusi Pemaketan Lokal ────────────────────────────────────────────────────
do_pack() {
    show_header
    echo -e "${CYAN}${BOLD}⏳ Memproses pemindahan model dan kompresi folder...${RESET}"
    echo ""

    cd "$SCRIPT_DIR"
    
    # Jalankan utilitas python
    if "$PYTHON_BIN" "utils/upload_utils.py"; then
        echo -e "${GREEN}${BOLD}🎉 SELESAI: Seluruh file berhasil dipindahkan dan dikompresi!${RESET}"
        echo -e "   Silakan periksa folder hasil di: ${BOLD}$SCRIPT_DIR/datas/${RESET}"
    else
        echo -e "\n${RED}${BOLD}❌ ERROR: Terjadi kegagalan saat pengarsipan lokal.${RESET}"
        echo -e "   Silakan tinjau pesan detail di atas untuk mengetahui penyebabnya."
        exit 1
    fi
}

# ── Eksekusi Upload ke Google Drive ─────────────────────────────────────────────
do_upload() {
    # Jalankan pemaketan lokal terlebih dahulu
    do_pack

    echo -e "\n${CYAN}${BOLD}⏳ Memulai proses upload ke Google Drive dengan rclone...${RESET}"
    echo ""

    cd "$SCRIPT_DIR"
    if [[ ! -d "datas" ]]; then
        echo -e "${RED}${BOLD}❌ ERROR: Direktori datas/ tidak ditemukan!${RESET}"
        exit 1
    fi

    # Menentukan konfigurasi remote path
    HOSTNAME_VAL=$(hostname)
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    REMOTE_PATH="gdrive:gdrive-backup/${HOSTNAME_VAL}/MyFineTunning-${TIMESTAMP}/datas"

    echo -e "   ${BOLD}Target Upload:${RESET} ${REMOTE_PATH}"
    
    # Menambahkan --exclude agar file kompresi workspace utuh tidak terupload
    if rclone copy ./datas "$REMOTE_PATH" -P --exclude "MyFineTunning-*.tar.gz"; then
        echo -e "\n${GREEN}${BOLD}🎉 SELESAI: Upload ke Google Drive berhasil!${RESET}"
    else
        echo -e "\n${RED}${BOLD}❌ ERROR: Terjadi kegagalan saat proses upload menggunakan rclone.${RESET}"
        exit 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
ACTION="${1:-}"

case "$ACTION" in
    pack)
        do_pack
        ;;
    upload)
        do_upload
        ;;
    *)
        show_header
        echo -e "${BOLD}Panduan Penggunaan:${RESET}"
        echo -e "  bash upload.sh pack      → Menjalankan pemindahan model & kompresi folder"
        echo -e "  bash upload.sh upload    → Menjalankan pemaketan & upload ke Google Drive"
        echo ""
        exit 0
        ;;
esac

