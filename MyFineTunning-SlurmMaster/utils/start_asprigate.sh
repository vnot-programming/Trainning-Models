#!/bin/bash

# start_asprigate.sh
# Script untuk menjalankan proksi AspriGate di dalam tmux

# Pindah ke direktori utama
cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster

echo "Mengaktifkan environment conda yolo_env..."
source /data/programs/anaconda3/bin/activate yolo_env

echo "Memastikan pustaka FastAPI & Uvicorn & httpx tersedia..."
pip install -q fastapi uvicorn httpx aiohttp jinja2 python-multipart

# Eksekusi uvicorn di port utama (Login Node). 
# Karena ini dijalankan via tmux, kita cukup run langsung.
# Proksi ini akan memonitor Port 8501, 8502, dan 19095 dengan satu proses uvicorn.
echo "Menjalankan AspriGate Proxy..."
python utils/asprigate_proxy.py
