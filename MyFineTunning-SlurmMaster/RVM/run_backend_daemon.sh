#!/bin/bash
# ==============================================================================
# run_backend_daemon.sh — Watchdog (Self-Healing) RVM Backend
# ==============================================================================
# Script ini akan terus berjalan (Loop) di dalam tmux `rvm_backend` pada Login Node.
# Tugas utamanya adalah memastikan bahwa jika koneksi ke Compute Node terputus (GPU dibunuh Slurm),
# ia akan menunggu GPU baru disewa oleh tmux `gpu_booking`, lalu merestart Reverse Tunnel dan Flask API.
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ACTIVATE="/data/programs/anaconda3/bin/activate"
CONDA_ENV="yolo_env"
BACKEND_PORT=8502

echo "============================================================"
echo "🛡️ RVM Backend Auto-Resume Daemon Aktif."
echo "============================================================"

# Fungsi untuk mendeteksi Job GPU aktif
get_active_job() {
    # Ambil JOBID (bukan nama Node)
    squeue -u $USER -n vnot -t R -h -o "%i" 2>/dev/null | head -n 1
}

while true; do
    JOBID=$(get_active_job)
    
    if [ -z "$JOBID" ] || [ "$JOBID" == "" ] || [ "$JOBID" == "None" ]; then
        echo -ne "⏳ Menunggu GPU tersedia (Sewa GPU belum aktif atau sedang PENDING)... \r"
        sleep 5
        continue
    fi
    
    echo -e "\n✅ Job GPU ($JOBID) tersedia! Menghubungkan ke Node via srun..."
    
    # Menjalankan eksekusi langsung ke dalam Slurm step.
    # Jika Job dibunuh, srun akan otomatis exit dan skrip ini kembali ke atas loop.
    srun --overlap --jobid=$JOBID bash -c "
        source $CONDA_ACTIVATE $CONDA_ENV
        cd $ROOT_DIR
        
        # 4. Bangun Reverse Tunnel ke Master Node
        ssh -o StrictHostKeyChecking=no -N -f -R ${BACKEND_PORT}:localhost:${BACKEND_PORT} slurmmaster
        
        # 5. Jalankan Backend RVM secara BLOCKING
        echo \"🚀 Menjalankan RVM Backend Flask Server di \$HOSTNAME...\"
        python -u RVM/backend/visual_eval_api.py 2>&1 | tee RVM/backend/logs/backend.log
    "
    
    # Jika perintah ssh di atas selesai (karena error atau node mati), berikan jeda.
    echo -e "\n⚠️ Koneksi ke Compute Node terputus atau Server API Berhenti!"
    echo "🔄 Daemon akan mencoba mencari GPU baru dan menyambung ulang dalam 5 detik..."
    sleep 5
done
