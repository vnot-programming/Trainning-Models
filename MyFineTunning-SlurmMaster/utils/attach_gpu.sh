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

# Attach ke terminal bash di dalam alokasi job tersebut
srun --jobid=$JOBID --pty bash
