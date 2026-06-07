#!/bin/bash
# Interactive GPU Booking Menu

# Selesaikan lokasi asli script (mengatasi symlink)
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Navigasi ke direktori asli utils
cd "$SCRIPT_DIR" || exit 1

# Function to display squeue in a human‑readable table
pretty_squeue() {
    python3 "$SCRIPT_DIR/slurm/pretty_squeue.py"
}

# Function to display tmux sessions nicely with numbers
pretty_tmux_ls() {
    if ! tmux ls >/dev/null 2>&1; then
        echo "⚠️ Tidak ada sesi tmux yang aktif."
        return 1
    fi
    echo "╭────────────────────────────────────────╮"
    echo "│         DAFTAR SESI TMUX AKTIF         │"
    echo "╰────────────────────────────────────────╯"
    tmux ls | awk -F: '{ printf "  %d) 🟢 %-30s \n", NR, $1 }'
    echo "------------------------------------------"
    return 0
}

# Function to list available GPU nodes (ai2, ai3) and let user pick one
select_node() {
    echo "Pilih node GPU yang tersedia:"
    echo "1) ai2"
    echo "2) ai3"
    read -p "Nomor [1-2]: " node_choice
    case $node_choice in
        1) SELECTED_NODE="ai2" ;;
        2) SELECTED_NODE="ai3" ;;
        *) SELECTED_NODE="ai2" ;; # default fallback
    esac
}

# Function to cancel a job by its ID
cancel_job() {
    read -p "Masukkan JobID yang ingin dibatalkan: " jid
    if [ -n "$jid" ]; then
        scancel $jid && echo "Job $jid berhasil dibatalkan."
    else
        echo "JobID kosong, tidak ada yang dibatalkan."
    fi
}

# Function to move a job to another node (requeue + update node list)
move_job_node() {
    read -p "Masukkan JobID yang ingin dipindahkan: " jid
    if [ -z "$jid" ]; then
        echo "JobID kosong, operasi dibatalkan."
        return
    fi
    select_node
    target_node=$SELECTED_NODE
    echo "Memindahkan Job $jid ke node $target_node..."
    
    # Alur pemindahan node Slurm yang aman & teratur:
    scontrol hold $jid >/dev/null 2>&1
    scontrol requeue $jid >/dev/null 2>&1
    scontrol update JobId=$jid ReqNodeList=$target_node
    scontrol release $jid >/dev/null 2>&1
    
    echo "Permintaan requeue dan pindah node dikirim ke scheduler."
}

# Function to generate a unique tmux session name for GPU booking
generate_gpu_session_name() {
    base="gpu_booking"
    if tmux has-session -t "$base" 2>/dev/null; then
        # Append timestamp to create unique name
        echo "${base}_$(date +%s)"
    else
        echo "$base"
    fi
}

# Function to manage Cloudflare Tunnel on Master Node
manage_cloudflare_tunnel() {
    # Ambil konfigurasi dari config_shared.py dan .env
    local cf_bin=$(grep -E "^CLOUDFLARE_BIN\s*=" "${SCRIPT_DIR}/../config_shared.py" | head -n 1 | cut -d'"' -f2)
    local cf_token=$(grep -E "^CLOUDFLARE_TUNNEL_TOKEN\s*=" "${SCRIPT_DIR}/../.env" | head -n 1 | cut -d'=' -f2)

    # Fallback jika tidak ditemukan
    if [ -z "$cf_bin" ]; then
        cf_bin="/data/users/g6717500336/singularity/cloudflared"
    fi
    if [ -z "$cf_token" ]; then
        cf_token=""
    fi

    # Definisikan path log dinamis berbasis binary path
    local cf_log="${cf_bin}.log"

    # Pastikan file binary dapat dieksekusi
    if [ -f "$cf_bin" ] && [ ! -x "$cf_bin" ]; then
        chmod +x "$cf_bin"
    fi

    while true; do
        clear
        echo "============================================="
        echo "       Manajemen Cloudflare Tunnel"
        echo "============================================="
        
        local session_active=0
        local process_active=0
        
        if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
            session_active=1
        fi
        if pgrep -u $USER cloudflared >/dev/null; then
            process_active=1
        fi
        
        echo -n "Status Sesi TMUX: "
        if [ $session_active -eq 1 ]; then
            echo -e "\033[92m🟢 AKTIF (cloudflare_tunnel)\033[0m"
        else
            echo -e "\033[91m🔴 TIDAK AKTIF\033[0m"
        fi
        
        echo -n "Status Proses  : "
        if [ $process_active -eq 1 ]; then
            echo -e "\033[92m🟢 RUNNING\033[0m"
        else
            echo -e "\033[91m🔴 TIDAK AKTIF\033[0m"
        fi
        echo "---------------------------------------------"
        echo "1. 🚀 Jalankan Cloudflare Tunnel (Background)"
        echo "2. 🛑 Hentikan Cloudflare Tunnel"
        echo "3. 📋 Cek Status & Log Terowongan"
        echo "4. 💻 Masuk / Attach ke Sesi TMUX Tunnel"
        echo "Enter untuk kembali ke menu utama"
        echo "============================================="
        read -p "Pilih aksi [1-4]: " cf_choice
        
        if [ -z "$cf_choice" ]; then
            break
        fi
        
        case $cf_choice in
            1)
                if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                    echo "⚠️ Terowongan Cloudflare sudah berjalan dalam sesi tmux 'cloudflare_tunnel'."
                else
                    echo "Memulai Cloudflare Tunnel di background (Sesi TMUX: cloudflare_tunnel)..."
                    # Menggunakan parameter optimasi konektivitas --edge-ip-version dan --retries serta pipe tee ke berkas log
                    tmux new-session -d -s cloudflare_tunnel "$cf_bin tunnel --protocol http2 --edge-ip-version 4 --retries 10 --no-autoupdate run --token $cf_token 2>&1 | tee -a \"$cf_log\""
                    sleep 2
                    if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                        echo "✅ Sesi tmux 'cloudflare_tunnel' berhasil dibuat."
                        echo "💡 Log dialihkan ke berkas: $cf_log"
                    else
                        echo "❌ Gagal membuat sesi tmux 'cloudflare_tunnel'. Pastikan path binary valid."
                    fi
                fi
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            2)
                echo "Menghentikan Terowongan Cloudflare..."
                if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                    tmux kill-session -t cloudflare_tunnel
                    echo "✅ Sesi tmux 'cloudflare_tunnel' berhasil dihentikan."
                else
                    echo "⚠️ Sesi tmux 'cloudflare_tunnel' tidak aktif."
                fi
                
                # Pembersihan proses sisa
                if pgrep -u $USER cloudflared >/dev/null; then
                    pkill -u $USER cloudflared
                    echo "✅ Proses cloudflared sisa berhasil dihentikan."
                fi
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            3)
                echo "============================================="
                echo "          Log Cloudflare Tunnel"
                echo "============================================="
                if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                    echo "Mengambil 20 baris log terakhir dari sesi tmux:"
                    echo "---------------------------------------------"
                    tmux capture-pane -t cloudflare_tunnel -p -S -20 2>/dev/null || echo "⚠️ Tidak dapat mengambil output log tmux pane."
                    echo "---------------------------------------------"
                    echo "💡 Log lengkap tersimpan di: $cf_log"
                else
                    echo "⚠️ Sesi tmux 'cloudflare_tunnel' tidak aktif. Tidak ada log untuk ditampilkan."
                fi
                echo "============================================="
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            4)
                if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                    echo "Memasuki sesi tmux 'cloudflare_tunnel'. Tekan Ctrl+B lalu D untuk keluar (detach) tanpa mematikan terowongan."
                    sleep 2
                    tmux attach -t cloudflare_tunnel
                else
                    echo "⚠️ Sesi tmux 'cloudflare_tunnel' tidak aktif."
                    read -p "Tekan Enter untuk melanjutkan..."
                fi
                ;;
            *)
                echo "❌ Pilihan tidak valid."
                sleep 1
                ;;
        esac
    done
}

manage_web_rvm() {
    local start_rvm_script="/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/RVM/start_rvm.sh"

    while true; do
        clear
        echo "============================================="
        echo "          Manajemen Web RVM"
        echo "============================================="
        
        # Tampilkan status port dan sesi tmux RVM
        if [ -f "$start_rvm_script" ]; then
            bash "$start_rvm_script" status
        else
            echo -e "\033[91m❌ Error: Launcher '$start_rvm_script' tidak ditemukan!\033[0m"
        fi
        
        echo "============================================="
        echo "Sub Menu Web RVM:"
        echo "1. 🚀 Start All Services (Frontend & Backend)"
        echo "2. 🛑 Stop All Services (Matikan Frontend & Backend)"
        echo "3. 🔄 Restart All Services"
        echo "4. 📱 Start Frontend Only"
        echo "5. 🛑 Stop Frontend Only"
        echo "6. 🚀 Start Layanan backend Flask"
        echo "7. 🛑 Stop Layanan backend Flask"
        echo "8. 🔄 Restart Layanan backend Flask"
        echo "9. 📋 Lihat Layanan Berjalan (Refresh Status)"
        echo "0. 🔙 Kembali ke Menu Utama"
        echo "============================================="
        read -p "Pilih aksi [0-9]: " rvm_choice

        if [ -z "$rvm_choice" ] || [ "$rvm_choice" == "0" ]; then
            break
        fi

        case $rvm_choice in
            1)
                echo "Menghentikan semua service lama..."
                bash "$start_rvm_script" stop
                echo "Memulai Frontend..."
                bash "$start_rvm_script" frontend
                echo "Memulai Backend (tmux rvm_backend & attach GPU)..."
                bash "$start_rvm_script" backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            2)
                echo "Menghentikan semua service RVM..."
                bash "$start_rvm_script" stop
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            3)
                echo "Melakukan restart semua service RVM..."
                bash "$start_rvm_script" stop
                sleep 1
                bash "$start_rvm_script" frontend
                bash "$start_rvm_script" backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            4)
                echo "Memulai Frontend..."
                bash "$start_rvm_script" frontend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            5)
                echo "Menghentikan Frontend..."
                bash "$start_rvm_script" stop_frontend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            6)
                echo "Memulai Layanan backend Flask..."
                bash "$start_rvm_script" backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            7)
                echo "Menghentikan Layanan backend Flask..."
                bash "$start_rvm_script" stop_backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            8)
                echo "Melakukan restart Layanan backend Flask..."
                bash "$start_rvm_script" stop_backend
                sleep 1
                bash "$start_rvm_script" backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            9)
                echo "Mengecek/Lihat status Layanan Berjalan..."
                bash "$start_rvm_script" status
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            *)
                echo -e "\033[91m❌ Pilihan tidak valid.\033[0m"
                sleep 1
                ;;
        esac
    done
}

manage_aspri_ai() {
    local aspri_dir="/data/users/g6717500336/singularity/ollama"
    while true; do
        clear
        
        # Deteksi status secara real-time
        local active_job=$(squeue -u $USER -n vnot -h -o "%i %N %t" | head -n 1)
        local job_id=$(echo $active_job | cut -d' ' -f1)
        local node_name=$(echo $active_job | cut -d' ' -f2)
        local job_state=$(echo $active_job | cut -d' ' -f3)
        
        local fallback_url="TIDAK AKTIF"
        local active_port="N/A"
        local ollama_running="MATI"
        
        if [ -n "$job_id" ]; then
            local ollama_active=$(ssh -o StrictHostKeyChecking=no ${node_name} "ps -u $USER -f | grep ollama | grep -v grep" 2>/dev/null | grep -v "sbatch_aspri" || echo "")
            if [ -n "$ollama_active" ]; then
                ollama_running="RUNNING"
                active_port=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2)
                active_port=${active_port:-N/A}
                fallback_url=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "${aspri_dir}/logs/tunnel_sbatch.log" | tail -n 1)
                fallback_url=${fallback_url:-TIDAK AKTIF}
            else
                ollama_running="MATI / STANDBY"
            fi
        fi
        
        # Deteksi status kesehatan AspriAI Core & Desk secara real-time
        local core_status="Unhealthy"
        local env_file="/data/users/g6717500336/singularity/AspriAI/.env"
        local cf_id=""
        local cf_secret=""
        if [ -f "$env_file" ]; then
            cf_id=$(grep -E "^CF-Access-Client-Id=" "$env_file" | cut -d'=' -f2 | tr -d '\r' | tr -d '\n')
            cf_secret=$(grep -E "^CF-Access-Client-Secret=" "$env_file" | cut -d'=' -f2 | tr -d '\r' | tr -d '\n')
        fi
        
        if [ -n "$cf_id" ] && [ -n "$cf_secret" ]; then
            local core_http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
                -H "CF-Access-Client-Id: $cf_id" \
                -H "CF-Access-Client-Secret: $cf_secret" \
                https://backend-ollama.penelitian.my.id/v1/health --max-time 2)
            if [ "$core_http_code" = "200" ]; then
                core_status="Healthy"
            fi
        else
            local core_http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:11434/v1/health --max-time 2)
            if [ "$core_http_code" = "200" ]; then
                core_status="Healthy"
            fi
        fi
        
        local desk_status="Unhealthy"
        local desk_http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST https://xyz.penelitian.my.id/v1/health --max-time 2)
        if [ "$desk_http_code" = "200" ]; then
            desk_status="Healthy"
        fi

        # Cetak Header & Status Utama
        echo "============================================="
        echo "       Manajemen Layanan AspriAI"
        echo "============================================="
        echo "Status Saat Ini:"
        if [ -z "$job_id" ]; then
            echo -e "• Sewa GPU (Slurm)  : \033[91mTIDAK AKTIF\033[0m"
            echo -e "• Ollama Server     : \033[91mMATI\033[0m (Butuh Booking GPU)"
            echo -e "• Port Compute Node : \033[91mN/A\033[0m"
        else
            echo -e "• Sewa GPU (Slurm)  : \033[92mAKTIF\033[0m (Job ID: ${job_id} di Node: ${node_name})"
            if [ "$ollama_running" = "RUNNING" ]; then
                echo -e "• Ollama Server     : \033[92mRUNNING\033[0m (di compute node ${node_name})"
                echo -e "• Port Compute Node : \033[96m${active_port}\033[0m"
            else
                echo -e "• Ollama Server     : \033[93m${ollama_running}\033[0m"
                echo -e "• Port Compute Node : \033[91mN/A\033[0m"
            fi
        fi
        
        if [ "$core_status" = "Healthy" ]; then
            echo -e "• AspriAI Core : \033[92mHealthy\033[0m (POST ke Endpoint https://backend-ollama.penelitian.my.id/v1/health)"
        else
            echo -e "• AspriAI Core : \033[91mUnhealthy\033[0m (POST ke Endpoint https://backend-ollama.penelitian.my.id/v1/health)"
        fi
        
        if [ "$desk_status" = "Healthy" ]; then
            echo -e "• AspriAI Desk : \033[92mHealthy\033[0m (POST ke Endpoint https://xyz.penelitian.my.id/v1/health)"
        else
            echo -e "• AspriAI Desk : \033[91mUnhealthy\033[0m (POST ke Endpoint https://xyz.penelitian.my.id/v1/health)"
        fi
        
        # Cetak Status Ollama
        echo "============================================="
        echo "       Manajemen Layanan Ollama"
        echo "============================================="
        if [ "$ollama_running" = "RUNNING" ]; then
            echo -e "• URL : \033[92mhttps://backend-ollama.penelitian.my.id\033[0m"
            if [ "$fallback_url" != "TIDAK AKTIF" ]; then
                echo -e "• URL Fallback : \033[92m${fallback_url}\033[0m"
            else
                echo -e "• URL Fallback : \033[91m${fallback_url}\033[0m"
            fi
            echo -e "• Port : \033[96m11434\033[0m"
        else
            echo -e "• URL : https://backend-ollama.penelitian.my.id (MATI)"
            echo -e "• URL Fallback : \033[91m${fallback_url}\033[0m"
            echo -e "• Port : 11434"
        fi
        
        # Cetak Status ComfyUI (Placeholder)
        echo "============================================="
        echo "       Manajemen Layanan ComfUI"
        echo "============================================="
        echo -e "• URL : \033[93mhttps://backend-xyz.penelitian.my.id\033[0m (masih placeholder)"
        echo -e "• URL Fallback : \033[93mhttps://xyz.trycloudflare.com\033[0m"
        echo -e "• Port : \033[93mxxxx\033[0m"
        
        echo "---------------------------------------------"
        echo "Sub Menu AspriAI:"
        echo "1. Jalankan AspriAI Core"
        echo "2. Stop AspriAI Core"
        echo "3. Restart AspriAI Core"
        echo "---------------------------------------------"
        echo "Modul AspriAI:"
        echo "4. Ollama"
        echo "5. Chat via Ollama"
        echo "6. ComfUI"
        echo "7. {AI Lainnya belum saya fikirkan}"
        echo "Enter untuk kembali ke menu utama"
        echo "---------------------------------------------"
        read -p "Pilih aksi [1-7]: " main_choice
        
        if [ -z "$main_choice" ]; then
            break
        fi
        
        case $main_choice in
            1)
                # 1. Jalankan AspriAI Core
                echo "Meluncurkan AspriAI Core (FastAPI Proxy)..."
                local is_core_running=$(pgrep -f "uvicorn main:app --host 127.0.0.1 --port 11434" || echo "")
                if [ -n "$is_core_running" ]; then
                    echo -e "\033[93mℹ️ AspriAI Core sudah berjalan.\033[0m"
                else
                    nohup bash /data/users/g6717500336/singularity/AspriAI/aspri-core/run_gateway.sh > /data/users/g6717500336/singularity/AspriAI/aspri-core/gateway.log 2>&1 &
                    sleep 2
                    echo -e "\033[92m✅ AspriAI Core berhasil diluncurkan di background!\033[0m"
                fi
                
                # 2. Jalankan Server Ollama secara otomatis jika GPU aktif
                if [ -n "$job_id" ]; then
                    local current_job_state=$(squeue -j $job_id -h -o "%t" 2>/dev/null | tr -d ' ' || echo "")
                    if [ "$current_job_state" = "R" ]; then
                        echo "Memeriksa status Server Ollama di compute node ${node_name}..."
                        local is_ollama_running_node=$(ssh -o StrictHostKeyChecking=no ${node_name} "ps -u $USER -f | grep ollama | grep -v grep" 2>/dev/null | grep -v "sbatch_aspri" || echo "")
                        if [ -z "$is_ollama_running_node" ]; then
                            echo "Meluncurkan Server Ollama secara otomatis di node ${node_name}..."
                            ssh -o StrictHostKeyChecking=no ${node_name} "nohup bash ${aspri_dir}/sbatch_aspri_service.sh > ${aspri_dir}/logs/nohup_runner.log 2>&1 &"
                            echo "⏳ Menunggu inisiasi Ollama & Cloudflare Tunnel..."
                            local wait_sec=0
                            while [ $wait_sec -lt 15 ]; do
                                local check_port=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2 || echo "")
                                local check_url=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "${aspri_dir}/logs/tunnel_sbatch.log" | tail -n 1 || echo "")
                                if [ -n "$check_port" ] && [ -n "$check_url" ]; then
                                    break
                                fi
                                sleep 2
                                wait_sec=$((wait_sec + 2))
                            done
                            echo -e "\033[92m✅ Server Ollama & Tunnel berhasil diinisiasi secara otomatis!\033[0m"
                        else
                            echo -e "\033[93mℹ️ Server Ollama sudah berjalan di node ${node_name}.\033[0m"
                        fi
                    else
                        echo -e "\033[93m⏳ Sewa GPU (Job ID: ${job_id}) masih dalam status ${current_job_state}. Server Ollama akan aktif saat job RUNNING.\033[0m"
                    fi
                else
                    echo -e "\033[91m⚠️ Sewa GPU tidak aktif. Server Ollama tidak dapat dijalankan secara otomatis.\033[0m"
                fi
                sleep 1.5
                ;;
            2)
                echo "Menghentikan AspriAI Core..."
                pkill -f "uvicorn main:app --host 127.0.0.1 --port 11434" || true
                echo -e "\033[92m✅ AspriAI Core berhasil dihentikan.\033[0m"
                sleep 1.5
                ;;
            3)
                echo "Restarting AspriAI Core..."
                pkill -f "uvicorn main:app --host 127.0.0.1 --port 11434" || true
                sleep 2
                nohup bash /data/users/g6717500336/singularity/AspriAI/aspri-core/run_gateway.sh > /data/users/g6717500336/singularity/AspriAI/aspri-core/gateway.log 2>&1 &
                sleep 2
                echo -e "\033[92m✅ AspriAI Core berhasil di-restart!\033[0m"
                sleep 1.5
                ;;
            4)
                # Submenu Ollama
                while true; do
                    # Ambil status real-time khusus submenu Ollama
                    local active_job_sub=$(squeue -u $USER -n vnot -h -o "%i %N %t" | head -n 1)
                    local job_id_sub=$(echo $active_job_sub | cut -d' ' -f1)
                    local node_name_sub=$(echo $active_job_sub | cut -d' ' -f2)
                    local job_state_sub=$(echo $active_job_sub | cut -d' ' -f3)
                    
                    local fallback_url_sub="TIDAK AKTIF"
                    local active_port_sub="N/A"
                    local ollama_running_sub="MATI"
                    
                    if [ -n "$job_id_sub" ]; then
                        local ollama_active_sub=$(ssh -o StrictHostKeyChecking=no ${node_name_sub} "ps -u $USER -f | grep ollama | grep -v grep" 2>/dev/null | grep -v "sbatch_aspri" || echo "")
                        if [ -n "$ollama_active_sub" ]; then
                            ollama_running_sub="RUNNING"
                            active_port_sub=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2)
                            active_port_sub=${active_port_sub:-N/A}
                            fallback_url_sub=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "${aspri_dir}/logs/tunnel_sbatch.log" | tail -n 1)
                            fallback_url_sub=${fallback_url_sub:-TIDAK AKTIF}
                        else
                            ollama_running_sub="MATI / STANDBY"
                        fi
                    fi
                    
                    clear
                    echo "============================================="
                    echo "       Manajemen Layanan AspriAI"
                    echo "============================================="
                    echo "Status Saat Ini:"
                    if [ -z "$job_id_sub" ]; then
                        echo -e "• Sewa GPU (Slurm)  : \033[91mTIDAK AKTIF\033[0m"
                        echo -e "• Ollama Server     : \033[91mMATI\033[0m (Butuh Booking GPU)"
                        echo -e "• Port Compute Node : \033[91mN/A\033[0m"
                    else
                        echo -e "• Sewa GPU (Slurm)  : \033[92mAKTIF\033[0m (Job ID: ${job_id_sub} di Node: ${node_name_sub})"
                        if [ "$ollama_running_sub" = "RUNNING" ]; then
                            echo -e "• Ollama Server     : \033[92mRUNNING\033[0m (di compute node ${node_name_sub})"
                            echo -e "• Port Compute Node : \033[96m${active_port_sub}\033[0m"
                        else
                            echo -e "• Ollama Server     : \033[93m${ollama_running_sub}\033[0m"
                            echo -e "• Port Compute Node : \033[91mN/A\033[0m"
                        fi
                    fi
                    
                    echo "============================================="
                    echo "       Manajemen Layanan Ollama"
                    echo "============================================="
                    if [ "$ollama_running_sub" = "RUNNING" ]; then
                        echo -e "• URL : \033[92mhttps://backend-ollama.penelitian.my.id\033[0m"
                        if [ "$fallback_url_sub" != "TIDAK AKTIF" ]; then
                            echo -e "• URL Fallback : \033[92m${fallback_url_sub}\033[0m"
                        else
                            echo -e "• URL Fallback : \033[91m${fallback_url_sub}\033[0m"
                        fi
                        echo -e "• Port : \033[96m11434\033[0m"
                    else
                        echo -e "• URL : https://backend-ollama.penelitian.my.id (MATI)"
                        echo -e "• URL Fallback : \033[91m${fallback_url_sub}\033[0m"
                        echo -e "• Port : 11434"
                    fi
                    echo "---------------------------------------------"
                    echo "Sub Menu Ollama:"
                    echo "1. 🚀 Jalankan Server Ollama (di Compute Node aktif)"
                    echo "2. 🛑 Hentikan Server Ollama (Sewa GPU tetap aktif)"
                    echo "3. 🔄 Restart Server Ollama"
                    echo "4. 📋 Lihat Log Runtime Ollama (Log & Tunnel)"
                    echo "5. ⚙️ Instal Ulang Modul (setup.sh --install)"
                    echo "Enter untuk kembali ke Manajemen Layanan AspriAI"
                    echo "---------------------------------------------"
                    read -p "Pilih aksi [1-5]: " ollama_choice
                    
                    if [ -z "$ollama_choice" ]; then
                        break
                    fi
                    
                    case $ollama_choice in
                        1)
                            if [ -z "$job_id_sub" ]; then
                                echo -e "\033[91m⚠️ Anda belum menyewa GPU! Silakan jalankan menu utama Opsi 1 terlebih dahulu.\033[0m"
                                sleep 2
                            elif [ "$job_state_sub" != "R" ]; then
                                echo -e "\033[93m⏳ Job booking GPU Anda masih dalam antrean (PENDING). Tunggu hingga RUNNING.\033[0m"
                                sleep 2
                            else
                                # Pastikan server belum running
                                local is_running=$(ssh -o StrictHostKeyChecking=no ${node_name_sub} "ps -u $USER -f | grep ollama | grep -v grep" 2>/dev/null | grep -v "sbatch_aspri" || echo "")
                                if [ ! -z "$is_running" ]; then
                                    echo -e "\033[93mℹ️ Server Ollama sudah berjalan di node ${node_name_sub}.\033[0m"
                                    sleep 1
                                else
                                    echo "Meluncurkan Server Ollama modular di node ${node_name_sub}..."
                                    ssh -o StrictHostKeyChecking=no ${node_name_sub} "nohup bash ${aspri_dir}/sbatch_aspri_service.sh > ${aspri_dir}/logs/nohup_runner.log 2>&1 &"
                                    echo "⏳ Menunggu inisiasi Ollama & Cloudflare Tunnel..."
                                    local wait_sec=0
                                    while [ $wait_sec -lt 15 ]; do
                                        local check_port=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2 || echo "")
                                        local check_url=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "${aspri_dir}/logs/tunnel_sbatch.log" | tail -n 1 || echo "")
                                        if [ -n "$check_port" ] && [ -n "$check_url" ]; then
                                            break
                                        fi
                                        sleep 2
                                        wait_sec=$((wait_sec + 2))
                                    done
                                    echo -e "\033[92m✅ Server Ollama & Tunnel berhasil diluncurkan!\033[0m"
                                    sleep 1.5
                                fi
                            fi
                            ;;
                        2)
                            if [ -z "$job_id_sub" ] || [ "$job_state_sub" != "R" ]; then
                                echo -e "\033[91m⚠️ Tidak ada server aktif di compute node yang bisa dihentikan.\033[0m"
                                sleep 1.5
                            else
                                echo "Menghentikan proses server Ollama di node ${node_name_sub}..."
                                ssh -o StrictHostKeyChecking=no ${node_name_sub} "pkill -u \$USER -f ollama-0.24.sif || true; pkill -u \$USER -f cloudflared || true; pkill -u \$USER -f 'ssh -o StrictHostKeyChecking=no'" 2>/dev/null || true
                                pkill -f "11434:localhost" 2>/dev/null || true
                                echo -e "\033[92m✅ Server Ollama di node ${node_name_sub} berhasil dihentikan.\033[0m"
                                sleep 1.5
                            fi
                            ;;
                        3)
                            if [ -z "$job_id_sub" ] || [ "$job_state_sub" != "R" ]; then
                                echo -e "\033[91m⚠️ Tidak ada server aktif di compute node yang bisa di-restart.\033[0m"
                                sleep 1.5
                            else
                                echo "Melakukan restart Server Ollama di node ${node_name_sub}..."
                                echo "1. Menghentikan proses server lama..."
                                ssh -o StrictHostKeyChecking=no ${node_name_sub} "pkill -u \$USER -f ollama-0.24.sif || true; pkill -u \$USER -f cloudflared || true; pkill -u \$USER -f 'ssh -o StrictHostKeyChecking=no'" 2>/dev/null || true
                                pkill -f "11434:localhost" 2>/dev/null || true
                                sleep 3
                                
                                echo "2. Meluncurkan Server Ollama kembali..."
                                ssh -o StrictHostKeyChecking=no ${node_name_sub} "nohup bash ${aspri_dir}/sbatch_aspri_service.sh > ${aspri_dir}/logs/nohup_runner.log 2>&1 &"
                                echo "⏳ Menunggu inisiasi Ollama & Cloudflare Tunnel..."
                                local wait_sec=0
                                while [ $wait_sec -lt 15 ]; do
                                    local check_port=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2 || echo "")
                                    local check_url=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" "${aspri_dir}/logs/tunnel_sbatch.log" | tail -n 1 || echo "")
                                    if [ -n "$check_port" ] && [ -n "$check_url" ]; then
                                        break
                                    fi
                                    sleep 2
                                    wait_sec=$((wait_sec + 2))
                                done
                                echo -e "\033[92m✅ Server Ollama berhasil di-restart dan diinisiasi!\033[0m"
                                sleep 1.5
                            fi
                            ;;
                        4)
                            clear
                            echo "=== RUNTIME OLLAMA LOG ==="
                            if [ -f "${aspri_dir}/logs/ollama_sbatch.log" ]; then
                                tail -n 25 "${aspri_dir}/logs/ollama_sbatch.log"
                            else
                                echo "Log Ollama belum tersedia."
                            fi
                            echo -e "\n=== TUNNEL LOG ==="
                            if [ -f "${aspri_dir}/logs/tunnel_sbatch.log" ]; then
                                tail -n 10 "${aspri_dir}/logs/tunnel_sbatch.log"
                            else
                                echo "Log Tunnel belum tersedia."
                            fi
                            echo ""
                            read -p "Tekan Enter untuk kembali..."
                            ;;
                        5)
                            echo "Menginisiasi ulang modul Ollama..."
                            if [ -f "${aspri_dir}/setup.sh" ]; then
                                bash "${aspri_dir}/setup.sh" --install
                            else
                                echo -e "\033[91m❌ Berkas setup.sh tidak ditemukan di ${aspri_dir}!\033[0m"
                            fi
                            read -p "Tekan Enter untuk melanjutkan..."
                            ;;
                        *)
                            echo "Pilihan tidak valid!"
                            sleep 1
                            ;;
                    esac
                done
                ;;
            5)
                # Submenu Chat via Ollama
                while true; do
                    clear
                    echo "============================================="
                    echo "       Sub Menu Chat via Ollama"
                    echo "============================================="
                    
                    # Cek real-time status Ollama
                    local active_job_chat=$(squeue -u $USER -n vnot -h -o "%i %N %t" | head -n 1)
                    local job_id_chat=$(echo $active_job_chat | cut -d' ' -f1)
                    local node_name_chat=$(echo $active_job_chat | cut -d' ' -f2)
                    
                    local ollama_running_chat=0
                    if [ -n "$job_id_chat" ]; then
                        local ollama_active_chat=$(ssh -o StrictHostKeyChecking=no ${node_name_chat} "ps -u $USER -f | grep ollama | grep -v grep" 2>/dev/null | grep -v "sbatch_aspri" || echo "")
                        if [ -n "$ollama_active_chat" ]; then
                            ollama_running_chat=1
                        fi
                    fi
                    
                    if [ $ollama_running_chat -eq 0 ]; then
                        echo -e "\033[91m⚠️ Server Ollama tidak aktif! Silakan jalankan Server Ollama terlebih dahulu.\033[0m"
                        echo "============================================="
                        read -p "Tekan Enter untuk kembali..."
                        break
                    fi
                    
                    # Dapatkan port compute node dinamis
                    local active_port_chat=$(grep -oE "Port: [0-9]+" "${aspri_dir}/logs/ollama_sbatch.log" | tail -n 1 | cut -d' ' -f2 || echo "N/A")
                    
                    # Memetakan model terinstall secara dinamis menggunakan python
                    local local_model_list=()
                    local external_model_list=()
                    while read -r line; do
                        if [ -n "$line" ]; then
                            local model_name=$(echo "$line" | cut -d'|' -f1)
                            local model_src=$(echo "$line" | cut -d'|' -f2)
                            if [ "$model_src" = "local" ]; then
                                local_model_list+=("$model_name")
                            else
                                external_model_list+=("$model_name")
                            fi
                        fi
                    done < <(python3 -c "
import urllib.request, json, os

manifests_dir = '${aspri_dir}/models/manifests'

try:
    response = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
    data = json.loads(response.read().decode())
    for m in data.get('models', []):
        model_name = m.get('name', '')
        if not model_name:
            continue
            
        # Parse name and tag
        if ':' in model_name:
            name, tag = model_name.split(':', 1)
        else:
            name, tag = model_name, 'latest'
            
        if '/' in name:
            parts = name.split('/', 1)
            namespace, mname = parts[0], parts[1]
        else:
            namespace, mname = 'library', name
            
        manifest_path = os.path.join(manifests_dir, 'registry.ollama.ai', namespace, mname, tag)
        
        if os.path.exists(manifest_path):
            print(f'{model_name}|local')
        else:
            print(f'{model_name}|external')
except Exception:
    pass
" 2>/dev/null)
                    
                    echo "Model yang Terinstall (Local):"
                    local idx=1
                    if [ ${#local_model_list[@]} -eq 0 ]; then
                        echo -e "  \033[90m(Tidak ada model lokal yang terinstall)\033[0m"
                    else
                        for model in "${local_model_list[@]}"; do
                            echo -e "${idx}) 🤖 ${model}"
                            idx=$((idx + 1))
                        done
                    fi
                    echo "---------------------------------------------"
                    echo "0) 📥 Download Model Baru (ollama pull)"
                    echo "L) 🌐 Lihat Model External"
                    echo "Enter untuk kembali ke Manajemen Layanan AspriAI"
                    echo "---------------------------------------------"
                    
                    local max_choice=$((idx - 1))
                    if [ $max_choice -eq 0 ]; then
                        read -p "Pilih aksi [0, L]: " chat_choice
                    else
                        read -p "Pilih aksi [0-$max_choice, L]: " chat_choice
                    fi
                    
                    if [ -z "$chat_choice" ]; then
                        break
                    fi
                    
                    if [ "$chat_choice" = "0" ]; then
                        echo "============================================="
                        echo "           Download Model Ollama"
                        echo "============================================="
                        echo "💡 Nama model populer: llama3, qwen2.5, mistral, gemma2, phi3"
                        read -p "Masukkan nama model yang ingin diunduh (contoh: qwen2.5:7b): " model_to_pull
                        
                        if [ -n "$model_to_pull" ]; then
                            echo -e "\nMemulai unduhan model \033[92m${model_to_pull}\033[0m..."
                            echo -e "Menghubungkan ke compute node \033[96m${node_name_chat}\033[0m..."
                            sleep 1
                            
                            # Jalankan pull interaktif via ssh tty allocation
                            ssh -t -o StrictHostKeyChecking=no ${node_name_chat} "export OLLAMA_HOST=127.0.0.1:${active_port_chat}; export SINGULARITYENV_OLLAMA_HOST=127.0.0.1:${active_port_chat}; singularity exec --bind ${aspri_dir}/models:/root/.ollama ${aspri_dir}/ollama-0.24.sif ollama pull ${model_to_pull}"
                            
                            echo -e "\nProses unduhan model selesai."
                            read -p "Tekan Enter untuk memuat ulang daftar model..."
                        else
                            echo "⚠️ Nama model tidak boleh kosong."
                            sleep 1.5
                        fi
                    elif [[ "$chat_choice" =~ ^[lL]$ ]]; then
                        clear
                        echo "============================================="
                        echo "       Daftar Model Eksternal (Port Clash)"
                        echo "============================================="
                        if [ ${#external_model_list[@]} -eq 0 ]; then
                            echo "Tidak ada model eksternal yang terdeteksi."
                        else
                            local ext_idx=1
                            for ext_model in "${external_model_list[@]}"; do
                                echo "${ext_idx}) 🌐 ${ext_model}"
                                ext_idx=$((ext_idx + 1))
                            done
                            echo "---------------------------------------------"
                            echo "💡 Model eksternal ini milik pengguna lain di server Slurm."
                            echo "   Untuk menggunakannya, Anda harus mendownloadnya secara lokal"
                            echo "   menggunakan Opsi 0."
                        fi
                        echo "============================================="
                        read -p "Tekan Enter untuk kembali..."
                    elif [[ "$chat_choice" =~ ^[0-9]+$ ]] && [ "$chat_choice" -ge 1 ] && [ "$chat_choice" -le "$max_choice" ]; then
                        local selected_model="${local_model_list[$((chat_choice - 1))]}"
                        
                        echo -e "\nMemulai sesi chat interaktif dengan \033[92m${selected_model}\033[0m..."
                        echo -e "Hubungkan ke compute node \033[96m${node_name_chat}\033[0m di port \033[96m${active_port_chat}\033[0m..."
                        echo -e "Gunakan \033[93m/exit\033[0m atau \033[93mCtrl+D\033[0m untuk keluar dari sesi chat.\n"
                        sleep 1
                        
                        # Jalankan interactive chat via ssh tty allocation
                        ssh -t -o StrictHostKeyChecking=no ${node_name_chat} "export OLLAMA_HOST=127.0.0.1:${active_port_chat}; export SINGULARITYENV_OLLAMA_HOST=127.0.0.1:${active_port_chat}; singularity exec --bind ${aspri_dir}/models:/root/.ollama ${aspri_dir}/ollama-0.24.sif ollama run ${selected_model}"
                        
                        echo -e "\nSesi chat dengan \033[92m${selected_model}\033[0m selesai."
                        read -p "Tekan Enter untuk kembali ke daftar model..."
                    else
                        echo "Pilihan tidak valid!"
                        sleep 1
                    fi
                done
                ;;
            6)
                # Placeholder ComfyUI
                clear
                echo "============================================="
                echo "       Manajemen Layanan ComfUI"
                echo "============================================="
                echo -e "ℹ️ Layanan ComfUI belum dikonfigurasi atau diinstal."
                echo -e "  Status saat ini masih berupa placeholder."
                echo "============================================="
                read -p "Tekan Enter untuk kembali ke Manajemen Layanan AspriAI..."
                ;;
            7)
                echo -e "\033[93mℹ️ Pilihan ini belum diimplementasikan.\033[0m"
                sleep 1.5
                ;;
            *)
                echo "Pilihan tidak valid!"
                sleep 1
                ;;
        esac
    done
}

while true; do
    clear
    echo "============================================="
    echo "     Sistem Pintar Booking GPU Slurm"
    echo "============================================="
    echo "1. 🚀 Jalankan Booking GPU (Background)"
    echo "2. 💻 Masuk / Attach ke Node GPU (Interactive)"
    echo "3. 🛑 Batalkan / Hentikan Booking GPU"
    echo "4. 📋 Cek Status Antrean (squeue)"
    echo "5. 🗑️ Manajemen Sesi TMUX"
    echo "6. ☁️ Manajemen Cloudflare Tunnel"
    echo "7. 🖥️ Web RVM"
    echo "8. 🤖 Manajemen AspriAI (Ollama & WebUI)"
    echo -e "\033[91m0. ❌ Keluar Menu\033[0m"
    echo "============================================="
    read -p "Pilih menu [0-8]: " pilihan

    case $pilihan in
        1)
            # Memastikan daemon python book_gpu.py benar-benar berjalan (bukan proses tmux yatim yang memuat string book_gpu.py)
            daemon_running=0
            for pid in $(pgrep -u $USER -f "book_gpu.py" 2>/dev/null); do
                if [ -f "/proc/$pid/exe" ] && [[ "$(readlink "/proc/$pid/exe" 2>/dev/null)" == *"python"* ]]; then
                    daemon_running=1
                    break
                fi
            done

            if [ $daemon_running -eq 1 ]; then
                echo "=================================================================="
                echo "⚠️  DAEMON SUDAH BERJALAN!"
                echo "   Sistem mendeteksi bahwa daemon booking GPU (book_gpu.py) sudah aktif."
                echo "   Tidak perlu menjalankan ulang untuk mencegah Double Job."
                echo "=================================================================="
            else
                echo "=================================================================="
                echo "ℹ️  INFO TMUX DAEMON BOOKING:"
                echo "   Sesi tmux ini menjalankan daemon 'book_gpu.py' di background."
                echo "   Daemon ini berfungsi melakukan monitoring job, Telegram alerting,"
                echo "   serta auto-rebooking otomatis jika job Slurm mati/timeout."
                echo "   ⚠️ JANGAN matikan/kill sesi tmux ini agar fitur rebook tetap aktif!"
                echo "=================================================================="
                echo "Memulai Booking GPU dalam sesi tmux..."
                SESSION_NAME=$(generate_gpu_session_name)
                tmux new-session -d -s "$SESSION_NAME" "source /data/programs/anaconda3/bin/activate && conda activate yolo_env && python book_gpu.py"
                echo "✅ Proses berjalan di background pada sesi tmux: $SESSION_NAME."
                echo "   Anda akan menerima notifikasi Telegram saat status Running."
            fi
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        2)
            echo "Mencoba attach ke node GPU..."
            ./attach_gpu.sh
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        3)
            echo "Mencari Job vnot Anda..."
            JOBID=$(squeue -u $USER -n vnot -h -o %i | head -n 1)
            if [ -z "$JOBID" ]; then
                echo "Tidak ada job booking GPU yang aktif atau mengantre."
            else
                echo "Membatalkan Job ID: $JOBID"
                scancel $JOBID
                echo "Booking berhasil dihentikan."
            fi
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        4)
            # Show pretty squeue and submenu actions
            pretty_squeue
            echo "Aksi pada antrian:"
            echo "1) Hapus antrian (cancel job)"
            echo "2) Pindah node (requeue to ai2/ai3)"
            read -p "Pilih aksi [1-2] (atau Enter untuk kembali): " q_choice
            case $q_choice in
                1) cancel_job ;;
                2) move_job_node ;;
                *) echo "Kembali ke menu utama..." ;;
            esac
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        5)
            while true; do
                clear
                echo "============================================="
                echo "       Manajemen Sesi TMUX"
                echo "============================================="
                echo "⚠️ Peringatan Sesi Penting:"
                echo "   - 'gpu_booking'      : Daemon monitoring Slurm & Telegram. JANGAN di-kill!"
                echo "   - 'cloudflare_tunnel': Terowongan konektivitas luar Master Node. JANGAN di-kill!"
                echo "   (Membunuh sesi di atas akan mematikan layanan background terkait)"
                echo "---------------------------------------------"
                pretty_tmux_ls
                has_sessions=$?
                
                echo "Sub Menu TMUX:"
                echo "1. Masuk ke Sesi Tmux (Attach)"
                echo "2. Hapus Sesi Tmux (Kill)"
                echo "3. Tambah Sesi Baru (Create)"
                read -p "Pilih aksi [1-3] (Enter untuk batal): " action_choice
                
                if [ -z "$action_choice" ]; then
                    break
                elif [ "$action_choice" == "1" ]; then
                    if [ $has_sessions -ne 0 ]; then
                        echo "⚠️ Tidak ada sesi yang aktif untuk dimasuki."
                        sleep 1
                        continue
                    fi
                    total_sessions=$(tmux ls | wc -l)
                    read -p "Pilih nomor sesi untuk MASUK [1-$total_sessions] (Enter untuk batal): " sel_choice
                    if [[ "$sel_choice" =~ ^[0-9]+$ ]] && [ "$sel_choice" -ge 1 ] && [ "$sel_choice" -le "$total_sessions" ]; then
                        session_name=$(tmux ls | sed -n "${sel_choice}p" | cut -d: -f1)
                        echo "Memasuki sesi tmux '$session_name'..."
                        tmux attach -t "$session_name"
                    fi
                elif [ "$action_choice" == "2" ]; then
                    if [ $has_sessions -ne 0 ]; then
                        echo "⚠️ Tidak ada sesi yang aktif untuk dihapus."
                        sleep 1
                        continue
                    fi
                    total_sessions=$(tmux ls | wc -l)
                    read -p "Pilih nomor sesi untuk DIHAPUS [1-$total_sessions] (Enter untuk batal): " sel_choice
                    if [[ "$sel_choice" =~ ^[0-9]+$ ]] && [ "$sel_choice" -ge 1 ] && [ "$sel_choice" -le "$total_sessions" ]; then
                        session_name=$(tmux ls | sed -n "${sel_choice}p" | cut -d: -f1)
                        tmux kill-session -t "$session_name"
                        echo "✅ Sesi tmux '$session_name' berhasil dihapus."
                        sleep 1
                    fi
                elif [ "$action_choice" == "3" ]; then
                    read -p "Masukkan nama sesi baru (Enter untuk default): " new_session_name
                    if [ -z "$new_session_name" ]; then
                        new_session_name="session_$(date +%s)"
                    fi
                    tmux new-session -d -s "$new_session_name"
                    echo "✅ Sesi tmux '$new_session_name' berhasil dibuat di background."
                    read -p "Apakah Anda ingin langsung masuk (attach) ke sesi baru ini? (y/n): " go_choice
                    if [ "$go_choice" == "y" ] || [ "$go_choice" == "Y" ]; then
                        tmux attach -t "$new_session_name"
                    fi
                else
                    echo "❌ Aksi tidak valid."
                    sleep 1
                fi
            done
            ;;
        6)
            manage_cloudflare_tunnel
            ;;
        7)
            manage_web_rvm
            ;;
        8)
            manage_aspri_ai
            ;;
        0)
            echo "Terima kasih telah menggunakan sistem ini."
            exit 0
            ;;
        *)
            echo "Pilihan tidak valid!"
            read -p "Tekan Enter untuk kembali..."
            ;;
    esac
done
