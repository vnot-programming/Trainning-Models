#!/bin/bash
#SBATCH --job-name=VnoT-Train
#SBATCH --output=slurm_logs/booking_output_%j.log
#SBATCH --error=slurm_logs/booking_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

mkdir -p slurm_logs
echo "=========================================================="
echo "GPU Booking Active on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="
echo ""
echo "Cara masuk ke node ini tanpa antre:"
echo "  ./attach_gpu.sh"
echo ""
echo "Menahan node agar tidak tertutup..."
sleep infinity