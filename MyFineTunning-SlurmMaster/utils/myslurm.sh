#!/bin/bash
# Interactive GPU Booking Menu

# Navigasi ke direktori utils
cd "$(dirname "$0")"

# Function to display squeue in a human‑readable table
pretty_squeue() {
    python3 "$(dirname "$0")/slurm/pretty_squeue.py"
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
    # Ambil konfigurasi dari config_shared.py
    local cf_bin=$(grep -E "^CLOUDFLARE_BIN\s*=" ../config_shared.py | head -n 1 | cut -d'"' -f2)
    local cf_token=$(grep -E "^CLOUDFLARE_TUNNEL_TOKEN\s*=" ../config_shared.py | head -n 1 | cut -d'"' -f2)

    # Fallback jika tidak ditemukan
    if [ -z "$cf_bin" ]; then
        cf_bin="/data/users/g6717500336/singularity/cloudflared"
    fi
    if [ -z "$cf_token" ]; then
        cf_token="eyJhIjoiNDgzMWNmYzhiMDgxODc0NDNiZTI3YmI4OGMxNWQ4ZjIiLCJ0IjoiMDI3MDBjMGUtYTBlYS00NjhiLThhYmQtMTk2MTlhZmZlNThlIiwicyI6IlltRm1NbVprT1RVdFlqUTFaUzAwTUdFMExXSmlZakF0WmpGallXTTRNREZsTm1RdyJ9"
    fi

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
        if pgrep -f "cloudflared.*tunnel.*run" >/dev/null; then
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
                    tmux new-session -d -s cloudflare_tunnel "$cf_bin tunnel --protocol http2 --no-autoupdate run --token $cf_token"
                    sleep 2
                    if tmux has-session -t cloudflare_tunnel 2>/dev/null; then
                        echo "✅ Sesi tmux 'cloudflare_tunnel' berhasil dibuat."
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
                if pgrep -f "cloudflared.*tunnel.*run" >/dev/null; then
                    pkill -f "cloudflared.*tunnel.*run"
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
        echo "1. 🚀 Start All Services (Stop -> Start Frontend & Backend)"
        echo "2. 🛑 Stop All Services (Matikan Frontend & Backend)"
        echo "3. 🔄 Restart All Services"
        echo "4. 📱 Start Frontend Only"
        echo "5. ⚙️ Start Backend Only"
        echo "0. 🔙 Kembali ke Menu Utama"
        echo "============================================="
        read -p "Pilih aksi [0-5]: " rvm_choice

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
                echo "Memulai Backend..."
                bash "$start_rvm_script" backend
                read -p "Tekan Enter untuk melanjutkan..."
                ;;
            *)
                echo -e "\033[91m❌ Pilihan tidak valid.\033[0m"
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
    echo -e "\033[91m0. ❌ Keluar Menu\033[0m"
    echo "============================================="
    read -p "Pilih menu [0-7]: " pilihan

    case $pilihan in
        1)
            if pgrep -f "book_gpu.py" >/dev/null; then
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
