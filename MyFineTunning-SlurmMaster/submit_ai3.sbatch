#!/bin/bash
# Untuk Melihat status queue
# squeue -j 6448 
#SBATCH --job-name=MyFineTuning_ai3
#SBATCH --output=slurm_logs/ai3_output_%j.log
#SBATCH --error=slurm_logs/ai3_error_%j.log
#SBATCH --partition=gpu
#SBATCH --nodelist=ai3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

mkdir -p slurm_logs

echo "=========================================================="
echo "Job Started on $(hostname) at $(date)"
echo "Target Node: ai3"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Aktivasi Conda Environment (Python Host 3.6 terlalu usang untuk PyTorch modern)
# Memuat path anaconda3 yang sesuai pada cluster ini
source /data/programs/anaconda3/bin/activate
conda activate yolo_env || echo "[WARNING] conda env 'yolo_env' tidak ditemukan! Pastikan telah disetup."

# Jalankan pipeline dengan GPU index 0 (karena Slurm sudah mengisolasi GPU via cgroups)
python run_pipeline_parallel.py --gpus 0

echo "=========================================================="
echo "Job Finished at $(date)"
echo "=========================================================="
