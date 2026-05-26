#!/bin/bash

# Cari JOBID dari job bernama VnoT-Train milik user saat ini yang sedang Running
JOBID=$(squeue -u $USER -n VnoT-Train -t R -h -o %i | head -n 1)

if [ -z "$JOBID" ]; then
    echo "❌ Tidak ada job 'VnoT-Train' yang sedang berjalan."
    echo "💡 Jalankan terlebih dahulu: sbatch book_gpu.batch"
    exit 1
fi

echo "✅ Menemukan Job Booking GPU yang sedang berjalan: $JOBID"
echo "Menghubungkan (attach) ke Node..."
echo "Ketik 'exit' jika sudah selesai (Job utama tetap akan menahan GPU)."

# Menentukan direktori project
PROJECT_DIR="/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster"

# Attach ke terminal bash di dalam alokasi job tersebut
# Otomatis: load .bashrc, pindah ke project dir, dan aktifkan conda yolo_env
srun --jobid=$JOBID --pty bash --rcfile <(cat <<EOF
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
cd "$PROJECT_DIR"
source /data/programs/anaconda3/bin/activate yolo_env
EOF
)
