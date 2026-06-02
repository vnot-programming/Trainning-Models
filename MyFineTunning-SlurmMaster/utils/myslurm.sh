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
    echo -e "\033[91m0. ❌ Keluar Menu\033[0m"
    echo "============================================="
    read -p "Pilih menu [0-5]: " pilihan

    case $pilihan in
        1)
            echo "Memulai Booking GPU dalam sesi tmux..."
            SESSION_NAME=$(generate_gpu_session_name)
            tmux new-session -d -s "$SESSION_NAME" "source /data/programs/anaconda3/bin/activate && conda activate yolo_env && python book_gpu.py"
            echo "Proses berjalan di background pada sesi $SESSION_NAME. Anda akan menerima notifikasi Telegram saat status Running."
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        2)
            echo "Mencoba attach ke node GPU..."
            ./attach_gpu.sh
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        3)
            echo "Mencari Job VnoT-Train Anda..."
            JOBID=$(squeue -u $USER -n VnoT-Train -h -o %i | head -n 1)
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
                if pretty_tmux_ls; then
                    total_sessions=$(tmux ls | wc -l)
                    echo "Sub Menu TMUX:"
                    echo "1. Masuk ke Sesi Tmux (Attach)"
                    echo "2. Hapus Sesi Tmux (Kill)"
                    read -p "Pilih aksi [1-2] (Enter untuk batal): " action_choice
                    
                    if [ -z "$action_choice" ]; then
                        break
                    elif [ "$action_choice" == "1" ]; then
                        read -p "Pilih nomor sesi untuk MASUK [1-$total_sessions] (Enter untuk batal): " sel_choice
                        if [[ "$sel_choice" =~ ^[0-9]+$ ]] && [ "$sel_choice" -ge 1 ] && [ "$sel_choice" -le "$total_sessions" ]; then
                            session_name=$(tmux ls | sed -n "${sel_choice}p" | cut -d: -f1)
                            echo "Memasuki sesi tmux '$session_name'..."
                            tmux attach -t "$session_name"
                        fi
                    elif [ "$action_choice" == "2" ]; then
                        read -p "Pilih nomor sesi untuk DIHAPUS [1-$total_sessions] (Enter untuk batal): " sel_choice
                        if [[ "$sel_choice" =~ ^[0-9]+$ ]] && [ "$sel_choice" -ge 1 ] && [ "$sel_choice" -le "$total_sessions" ]; then
                            session_name=$(tmux ls | sed -n "${sel_choice}p" | cut -d: -f1)
                            tmux kill-session -t "$session_name"
                            echo "✅ Sesi tmux '$session_name' berhasil dihapus."
                            sleep 1
                        fi
                    else
                        echo "❌ Aksi tidak valid."
                        sleep 1
                    fi
                else
                    read -p "Tekan Enter untuk kembali ke menu..."
                    break
                fi
            done
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
