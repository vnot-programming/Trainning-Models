#!/bin/bash
# Interactive GPU Booking Menu

# Navigasi ke direktori utils
cd "$(dirname "$0")"

while true; do
    clear
    echo "============================================="
    echo "     Sistem Pintar Booking GPU Slurm"
    echo "============================================="
    echo "1. 🚀 Jalankan Booking GPU (Background)"
    echo "2. 💻 Masuk / Attach ke Node GPU (Interactive)"
    echo "3. 🛑 Batalkan / Hentikan Booking GPU"
    echo "4. 📋 Cek Status Antrean (squeue)"
    echo "5. ❌ Keluar Menu"
    echo "============================================="
    read -p "Pilih menu [1-5]: " pilihan

    case $pilihan in
        1)
            echo "Memulai Booking GPU dalam sesi tmux..."
            tmux new-session -d -s gpu_booking "source /data/programs/anaconda3/bin/activate && conda activate yolo_env && python book_gpu.py"
            echo "Proses berjalan di background. Anda akan menerima notifikasi Telegram saat status Running."
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        2)
            echo "Mencoba attach ke node GPU..."
            ./attach_gpu.sh
            echo ""
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        3)
            echo "Mencari Job VnoT-Train Anda..."
            # Cek job yang Running maupun Pending
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
            echo "Menampilkan status antrean Anda:"
            squeue -u $USER
            read -p "Tekan Enter untuk kembali ke menu..."
            ;;
        5)
            echo "Terima kasih telah menggunakan sistem ini."
            exit 0
            ;;
        *)
            echo "Pilihan tidak valid!"
            read -p "Tekan Enter untuk kembali..."
            ;;
    esac
done
