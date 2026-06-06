#!/bin/bash
#SBATCH --job-name=VnoT-Train100
#SBATCH --output=slurm_logs/smart_output_%j.log
#SBATCH --error=slurm_logs/smart_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=ai3

mkdir -p slurm_logs
echo "=========================================================="
echo "Job Started on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Aktivasi Conda Environment
source /data/programs/anaconda3/bin/activate
conda activate yolo_env

# 1. Setup Workspace & Dataset (Ekuivalen dengan bagian akhir setup.sh)
python main.py

# 2. Jalankan Orchestrator Multi-GPU Parallel Pipeline
python run_pipeline_parallel.py --gpus 0

# 3. Otomatis Upload Hasil ke Google Drive
bash rclone_sync.sh upload

echo "=========================================================="
echo "Job Finished at $(date)"
echo "=========================================================="