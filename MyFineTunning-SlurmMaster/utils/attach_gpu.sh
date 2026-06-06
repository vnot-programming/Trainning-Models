#!/bin/bash

# Cari JOBID dari job bernama vnot milik user saat ini yang sedang Running
JOBID=$(squeue -u $USER -n vnot -t R -h -o %i | head -n 1)

if [ -z "$JOBID" ]; then
    echo "❌ Tidak ada job 'vnot' yang sedang berjalan."
    echo "💡 Jalankan terlebih dahulu: sbatch book_gpu.batch"
    exit 1
fi

echo "✅ Menemukan Job Booking GPU yang sedang berjalan: $JOBID"
echo "Menghubungkan (attach) ke Node..."
echo "Ketik 'exit' jika sudah selesai (Job utama tetap akan menahan GPU)."

# Menentukan direktori project
PROJECT_DIR="/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster"

# Buat file startup rc sementara di shared NFS agar terbaca oleh compute node
TMP_RC="$PROJECT_DIR/utils/.attach_rc"
cat <<EOF > "$TMP_RC"
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
cd "$PROJECT_DIR"
source /data/programs/anaconda3/bin/activate yolo_env
EOF

# Attach ke terminal bash di dalam alokasi job tersebut dengan rcfile di NFS
srun --overlap --jobid=$JOBID --pty bash --rcfile "$TMP_RC"
