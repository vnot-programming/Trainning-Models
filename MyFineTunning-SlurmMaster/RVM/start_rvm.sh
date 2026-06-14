#!/bin/bash
# ==============================================================================
# start_rvm.sh — Launcher Visual Evaluation Web App
# ==============================================================================
# Menjalankan Frontend (login node) dan Backend (compute node GPU) dengan rapi.
#
# Berdasarkan Workflow Standar:
#   1. Jalankan frontend:
#      bash RVM/start_rvm.sh frontend
#   2. Jalankan backend (Otomatis membuat tmux rvm_backend & attach GPU):
#      bash RVM/start_rvm.sh backend
# ==============================================================================

set -euo pipefail

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ACTIVATE="/data/programs/anaconda3/bin/activate"
CONDA_ENV="yolo_env"
BACKEND_PORT=8602
TMUX_BIN="/usr/bin/tmux"

print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  🔬 Visual Evaluation Web App — RVM Launcher${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
}

start_frontend() {
    print_header
    SESSION_FRONTEND="rvm_frontend"
    echo -e "  ${CYAN}📱 Memulai Frontend Server (Login Node)${NC}"

    if "$TMUX_BIN" has-session -t "$SESSION_FRONTEND" 2>/dev/null; then
        echo -e "  ${YELLOW}⚠️  Sesi '$SESSION_FRONTEND' sudah ada. Mematikan sesi lama...${NC}"
        "$TMUX_BIN" kill-session -t "$SESSION_FRONTEND"
    fi

    # Jalankan frontend di dalam tmux rvm_frontend (Login Node) agar aman saat terminal ditutup
    "$TMUX_BIN" new-session -d -s "$SESSION_FRONTEND"
    "$TMUX_BIN" send-keys -t "$SESSION_FRONTEND" "source $CONDA_ACTIVATE $CONDA_ENV && cd $ROOT_DIR && python RVM/serve_frontend.py 2>&1 | tee RVM/backend/logs/frontend.log" C-m

    echo -e "  ${YELLOW}➔ Menunggu Frontend berjalan di port 8601 (timeout 15s)...${NC}"
    local timeout=15
    local elapsed=0
    while ! ss -tln 2>/dev/null | grep -q ":8601 " && [ $elapsed -lt $timeout ]; do
        echo -ne "  ${CYAN}⏳ Loading... ($elapsed s)\r${NC}"
        sleep 1
        elapsed=$((elapsed+1))
    done
    echo ""

    if [ $elapsed -ge $timeout ]; then
        echo -e "  ${RED}⚠️  Timeout! Frontend (Port 8601) mungkin gagal berjalan.${NC}"
        echo -e "  ${YELLOW}➔ Cek log dengan: tmux attach -t $SESSION_FRONTEND${NC}"
    else
        echo -e "  ${GREEN}✅ Frontend berhasil berjalan dan listening di port 8601!${NC}"
        echo -e "  ${CYAN}➔ Gunakan perintah berikut untuk melihat log frontend:${NC}"
        echo -e "     tmux attach -t $SESSION_FRONTEND"
        echo -e "  ${YELLOW}➔ Tekan Ctrl+B lalu D untuk keluar (detach) dari sesi tersebut.${NC}\n"
    fi
}

update_class_mapping_json() {
    echo -e "  ${CYAN}🔄 Menyinkronkan class_mapping.json dengan data.yaml...${NC}"
    local yaml_file="${ROOT_DIR}/datasets/training_seg/data.yaml"
    local json_file="${ROOT_DIR}/RVM/backend/class_mapping.json"
    
    if [ ! -f "$yaml_file" ]; then
        echo -e "  ${RED}⚠️  File '$yaml_file' tidak ditemukan. Gagal memperbarui class_mapping.json.${NC}"
        return
    fi
    
    # Jalankan inline python dengan mengaktifkan conda environment yolo_env di subshell
    if ( source "$CONDA_ACTIVATE" "$CONDA_ENV" && python -c "
import yaml
import json
import os

yaml_path = '$yaml_file'
json_path = '$json_file'

try:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if names:
        mapping = {'0': 'background'}
        for idx, name in enumerate(names):
            mapping[str(idx + 1)] = name
        with open(json_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print('SUCCESS')
    else:
        print('EMPTY_NAMES')
except Exception as e:
    print(f'ERROR: {e}')
" ) 2>&1 | grep -q "SUCCESS"; then
        echo -e "  ${GREEN}✅ class_mapping.json berhasil disinkronkan dengan data.yaml.${NC}"
    else
        echo -e "  ${RED}⚠️  Gagal menyinkronkan class_mapping.json. Silakan cek format data.yaml.${NC}"
    fi
}

start_backend() {
    print_header
    echo -e "  ${CYAN}⚙️  Memulai Backend API (Compute Node + GPU)${NC}"

    # Sinkronisasi class_mapping.json sebelum meluncurkan backend
    update_class_mapping_json

    HOSTNAME=$(hostname)
    SESSION_BACKEND="rvm_backend"

    # Jika dipanggil dari login node, otomatisasi pembuatan tmux rvm_backend dan attach GPU
    if [[ "$HOSTNAME" != ai* ]]; then
        echo -e "  ${YELLOW}➔ Membuat sesi tmux '$SESSION_BACKEND' di Login Node...${NC}"
        
        if "$TMUX_BIN" has-session -t "$SESSION_BACKEND" 2>/dev/null; then
            echo -e "  ${YELLOW}⚠️  Sesi '$SESSION_BACKEND' sudah ada. Mematikan sesi lama...${NC}"
            "$TMUX_BIN" kill-session -t "$SESSION_BACKEND"
        fi
        
        # Buat sesi baru
        "$TMUX_BIN" new-session -d -s "$SESSION_BACKEND"
        
        # Jalankan RVM Backend Daemon di dalam Tmux
        "$TMUX_BIN" send-keys -t "$SESSION_BACKEND" "bash ${ROOT_DIR}/RVM/run_backend_daemon.sh" C-m
        
        echo -e "  ${YELLOW}➔ Daemon Auto-Resume berjalan di Tmux '$SESSION_BACKEND'...${NC}"
        echo -e "  ${YELLOW}➔ Daemon ini akan otomatis menunggu dan menautkan GPU untuk Backend Anda.${NC}"
        echo -e "  ${GREEN}✅ Tmux '$SESSION_BACKEND' berhasil dikonfigurasi!${NC}"
        echo -e "  ${CYAN}➔ Gunakan perintah berikut untuk masuk dan memantau log backend:${NC}"
        echo -e "     tmux attach -t $SESSION_BACKEND"
        echo -e "  ${YELLOW}➔ Tekan Ctrl+B lalu D untuk keluar (detach) dari sesi tersebut.${NC}\n"
    else
        # Jika dipanggil langsung di dalam compute node (jarang digunakan langsung, namun disediakan)
        echo -e "  ${GREEN}✅ Menjalankan Backend API langsung di Compute Node...${NC}"
        
        if pgrep -u $USER -f "ssh -N.*-R ${BACKEND_PORT}" > /dev/null 2>&1; then
            echo -e "  ${YELLOW}⚠️  SSH tunnel ke port ${BACKEND_PORT} sudah aktif.${NC}"
        else
            ssh -N -f -R ${BACKEND_PORT}:localhost:${BACKEND_PORT} slurmmaster
            echo -e "  ${GREEN}✅ SSH Reverse Tunnel diaktifkan.${NC}"
        fi
        
        source "$CONDA_ACTIVATE" "$CONDA_ENV"
        cd "$ROOT_DIR"
        python -u RVM/backend/visual_eval_api.py 2>&1 | tee RVM/backend/logs/backend.log
    fi
}

stop_services() {
    print_header
    echo -e "  ${CYAN}■ Menghentikan services...${NC}\n"

    # Hentikan tmux rvm_backend di login node jika ada
    if "$TMUX_BIN" has-session -t "rvm_backend" 2>/dev/null; then
        "$TMUX_BIN" kill-session -t "rvm_backend"
        echo -e "  ${GREEN}✅ Sesi tmux 'rvm_backend' dihentikan.${NC}"
    fi

    # Hentikan tmux rvm_frontend di login node jika ada
    if "$TMUX_BIN" has-session -t "rvm_frontend" 2>/dev/null; then
        "$TMUX_BIN" kill-session -t "rvm_frontend"
        echo -e "  ${GREEN}✅ Sesi tmux 'rvm_frontend' dihentikan.${NC}"
    fi

    # Matikan SSH tunnel sisa di login node jika ada
    pkill -f "ssh -N.*-R ${BACKEND_PORT}" || true
    echo -e "  ${GREEN}✅ Semua proses local dihentikan.${NC}\n"
}

stop_backend() {
    print_header
    echo -e "  ${CYAN}■ Menghentikan backend service...${NC}\n"
    if "$TMUX_BIN" has-session -t "rvm_backend" 2>/dev/null; then
        "$TMUX_BIN" kill-session -t "rvm_backend"
        echo -e "  ${GREEN}✅ Sesi tmux 'rvm_backend' dihentikan.${NC}"
    fi
    pkill -f "ssh -N.*-R ${BACKEND_PORT}" || true
    echo -e "  ${GREEN}✅ Layanan backend Flask & SSH tunnel dihentikan.${NC}\n"
}

stop_frontend() {
    print_header
    echo -e "  ${CYAN}■ Menghentikan frontend service...${NC}\n"
    if "$TMUX_BIN" has-session -t "rvm_frontend" 2>/dev/null; then
        "$TMUX_BIN" kill-session -t "rvm_frontend"
        echo -e "  ${GREEN}✅ Sesi tmux 'rvm_frontend' dihentikan.${NC}"
    fi
}

check_status() {
    set +e # Nonaktifkan sementara errexit agar skrip tidak crash jika service/port tidak aktif
    print_header
    echo -e "  ${CYAN}📊 Status Ports & Sesi Tmux:${NC}\n"

    # Cek sesi tmux
    for sess in rvm_frontend rvm_backend; do
        if "$TMUX_BIN" has-session -t "$sess" 2>/dev/null; then
            echo -e "  Tmux Sesi '$sess': ${GREEN}● ACTIVE${NC}"
        else
            echo -e "  Tmux Sesi '$sess': ${RED}○ INACTIVE${NC}"
        fi
    done

    # Cek Cloudflared
    if pgrep -u $USER cloudflared > /dev/null 2>&1; then
        echo -e "  Cloudflared      : ${GREEN}● RUNNING${NC}"
    else
        echo -e "  Cloudflared      : ${RED}○ STOPPED${NC}"
    fi

    # Cek Port listening di Login Node
    echo -e "\n  ${CYAN}🔌 Ports (Login Node):${NC}"
    for port in 8601 8602; do
        if ss -tln 2>/dev/null | grep -q ":${port} "; then
            echo -e "  Port ${port}: ${GREEN}● LISTENING${NC}"
        else
            echo -e "  Port ${port}: ${RED}○ FREE${NC}"
        fi
    done
    echo ""
    set -e # Aktifkan kembali errexit
}

show_help() {
    print_header
    echo ""
    echo "  Usage: bash RVM/start_rvm.sh <command>"
    echo ""
    echo "  Commands:"
    echo "    frontend       Mulai frontend server (di tmux 'rvm_frontend' Login Node)"
    echo "    backend        Mulai backend API (di tmux 'rvm_backend' + otomatis attach GPU)"
    echo "    stop           Hentikan semua sesi tmux frontend dan backend"
    echo "    stop_backend   Hentikan backend saja"
    echo "    stop_frontend  Hentikan frontend saja"
    echo "    status         Cek status port & sesi tmux"
    echo ""
}

case "${1:-help}" in
    frontend) start_frontend ;;
    backend)  start_backend ;;
    stop)     stop_services ;;
    stop_backend) stop_backend ;;
    stop_frontend) stop_frontend ;;
    status)   check_status ;;
    *)  show_help ;;
esac
