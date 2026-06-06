#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slurm_orchestrator.py
======================
Smart Slurm Job Submitter & Real-time Telegram Monitor.
- Secara cerdas mendeteksi kesehatan node (mengabaikan DRAIN/DOWN, memilih node idle/mixed).
- Mensubmit job ke Slurm secara terisolasi.
- Mengirim notifikasi Telegram:
  1. Pengingat antrean setiap 30 menit.
  2. Saat job masuk fase eksekusi (Running).
  3. Saat job selesai (Success/Failed).

# Menjalankan CONDA:
source /data/programs/anaconda3/bin/activate
conda activate yolo_env || echo "[WARNING] conda env 'yolo_env' tidak ditemukan! Pastikan telah disetup."

# Melihat daftar antrean Slurm Anda yang sesungguhnya:
squeue -u g6717500336

# Memeriksa status antrean job komputasi Anda di Slurm, silakan gunakan nomor Job ID 6449:
# Mengecek status antrean di Slurm
squeue -j 6449

"""

import os
import sys
import time
import subprocess
import re
import socket
from telegram_utils import send_telegram_msg

# Definisikan direktori kerja
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SBATCH_TEMPLATE_PATH = os.path.join(_THIS_DIR, "submit_smart_temp.sbatch")

def get_cmd_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""

def scan_slurm_nodes():
    """Mendeteksi node yang sehat dan bermasalah di partisi gpu."""
    output = get_cmd_output("scontrol show node")
    nodes = {}
    current_node = None
    
    for line in output.split('\n'):
        if line.startswith("NodeName="):
            parts = line.split()
            current_node = parts[0].split('=')[1]
            nodes[current_node] = {'name': current_node, 'state': '', 'gres': '', 'alloc_gpu': 0}
        
        if current_node:
            if "State=" in line:
                match = re.search(r'State=([A-Z\+]+)', line)
                if match:
                    nodes[current_node]['state'] = match.group(1)
            if "Gres=" in line:
                match = re.search(r'Gres=.*?gpu:(\d+)', line)
                if match:
                    nodes[current_node]['gres'] = match.group(1)
            if "AllocTRES=" in line:
                match = re.search(r'gres/gpu=(\d+)', line)
                if match:
                    nodes[current_node]['alloc_gpu'] = int(match.group(1))

    healthy_nodes = []
    draining_nodes = []
    
    for name, data in nodes.items():
        state = data['state'].upper()
        if "DRAIN" in state or "DOWN" in state or "FAIL" in state or "REBOOT" in state:
            draining_nodes.append(name)
        else:
            total = int(data['gres']) if data['gres'].isdigit() else 0
            alloc = data['alloc_gpu']
            data['free_gpu'] = total - alloc
            healthy_nodes.append(data)
            
    return healthy_nodes, draining_nodes

def generate_sbatch_script(target_node=None, exclude_nodes=None):
    """Membuat file sbatch dinamis berdasarkan kesehatan cluster saat ini."""
    sbatch_path = os.path.join(_THIS_DIR, "submit_smart_run.sbatch")
    
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=VnoT-Train100",
        "#SBATCH --output=slurm_logs/smart_output_%j.log",
        "#SBATCH --error=slurm_logs/smart_error_%j.log",
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G"
    ]
    
    if target_node:
        lines.append(f"#SBATCH --nodelist={target_node}")
    elif exclude_nodes:
        lines.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")
        
    lines.extend([
        "",
        "mkdir -p slurm_logs",
        "echo \"==========================================================\"",
        "echo \"Job Started on $(hostname) at $(date)\"",
        "echo \"Allocated GPUs: $CUDA_VISIBLE_DEVICES\"",
        "echo \"==========================================================\"",
        "",
        "# Aktivasi Conda Environment",
        "source /data/programs/anaconda3/bin/activate",
        "conda activate yolo_env",
        "",
        "# 1. Setup Workspace & Dataset (Ekuivalen dengan bagian akhir setup.sh)",
        "python main.py",
        "",
        "# 2. Jalankan Orchestrator Multi-GPU Parallel Pipeline",
        "python run_pipeline_parallel.py --gpus 0",
        "",
        "# 3. Otomatis Upload Hasil ke Google Drive",
        "bash rclone_sync.sh upload",
        "",
        "echo \"==========================================================\"",
        "echo \"Job Finished at $(date)\"",
        "echo \"==========================================================\""
    ])
    
    with open(sbatch_path, "w") as f:
        f.write("\n".join(lines))
    return sbatch_path

def get_job_state(job_id):
    """Membaca status job Slurm."""
    output = get_cmd_output(f"squeue -j {job_id} -h -o '%t %r'")
    if not output:
        # Job mungkin sudah selesai, cek via sacct
        sacct_output = get_cmd_output(f"sacct -j {job_id} -h -o 'State,ExitCode' | head -n 1")
        if sacct_output:
            parts = sacct_output.split()
            if len(parts) >= 2:
                return parts[0], parts[1] # misal: COMPLETED, 0:0
        return "UNKNOWN", ""
        
    parts = output.split()
    state = parts[0]
    reason = " ".join(parts[1:]) if len(parts) > 1 else ""
    return state, reason

def monitor_job(job_id, target_desc):
    """Fungsi monitoring utama dengan Telegram Alerting."""
    send_telegram_msg(
        f"📝 <b>Job Submitted Successfully</b>\n"
        f"Job ID: <code>{job_id}</code>\n"
        f"Rencana Rute: {target_desc}\n"
        f"Status: <code>PENDING</code> (Menunggu Antrean)"
    )
    
    last_notif_time = time.time()
    notif_interval = 1800  # 30 menit (1800 detik)
    started = False
    
    while True:
        state, detail = get_job_state(job_id)
        
        # 1. Kasus Job Sedang Menunggu di Antrean (Pending)
        if state == "PD":
            current_time = time.time()
            if current_time - last_notif_time >= notif_interval:
                send_telegram_msg(
                    f"⏳ <b>Queue Status Update</b>\n"
                    f"Job ID: <code>{job_id}</code> masih menunggu di antrean.\n"
                    f"Alasan: <code>{detail}</code>"
                )
                last_notif_time = current_time
                
        # 2. Kasus Job Mulai Masuk Proses (Running)
        elif state == "R" and not started:
            started = True
            node_running = get_cmd_output(f"squeue -j {job_id} -h -o '%N'")
            send_telegram_msg(
                f"🚀 <b>Processing Started!</b>\n"
                f"Job ID: <code>{job_id}</code> telah aktif.\n"
                f"Node Komputasi: <code>{node_running}</code>"
            )
            
        # 3. Kasus Job Selesai (Success atau Failed)
        elif state in ["COMPLETED", "SUCCESS", "FAILED", "TIMEOUT", "CANCELLED", "UNKNOWN"]:
            if state in ["COMPLETED", "SUCCESS"]:
                send_telegram_msg(
                    f"✅ <b>Processing Finished!</b>\n"
                    f"Job ID: <code>{job_id}</code> sukses diselesaikan.\n"
                    f"Silakan periksa folder hasil evaluasi & grid perbandingan."
                )
            else:
                send_telegram_msg(
                    f"❌ <b>Processing Interrupted/Failed!</b>\n"
                    f"Job ID: <code>{job_id}</code> berakhir dengan status: <code>{state}</code>\n"
                    f"Exit Code/Detail: <code>{detail}</code>"
                )
            break
            
        time.sleep(15)  # Cek status setiap 15 detik

def main():
    print("✧ BIO-DIGITAL SLURM SMART ORCHESTRATOR ✧")
    print("Memindai topologi kluster secara real-time...")
    
    # 1. Pindai Kesehatan Node
    healthy, draining = scan_slurm_nodes()
    
    target_node = None
    exclude_list = draining.copy()
    
    # Rekomendasikan Node Idle Terlebih Dahulu
    idle_nodes = [n for n in healthy if n['free_gpu'] > 0]
    if idle_nodes:
        # Pilih node dengan free GPU terbanyak
        best_node = max(idle_nodes, key=lambda x: x['free_gpu'])
        target_node = best_node['name']
        desc = f"Di-dispatch langsung ke Node <b>{target_node}</b> (Memiliki {best_node['free_gpu']} GPU Kosong)"
    else:
        desc = f"Antrean Global (Menggunakan Node Sehat, Mengecualikan Node DRAIN: {', '.join(exclude_list) if exclude_list else 'None'})"

    print(f"Rute Terpilih: {desc}")
    
    # 2. Hasilkan File SBATCH Dinamis
    sbatch_script = generate_sbatch_script(target_node=target_node, exclude_nodes=exclude_list)
    print(f"Mengompilasi script Slurm: {sbatch_script}")
    
    # 3. Submit ke Slurm
    submit_res = get_cmd_output(f"sbatch {sbatch_script}")
    match = re.search(r'Submitted batch job (\d+)', submit_res)
    
    if match:
        job_id = match.group(1)
        print(f"Berhasil mensubmit Job ID: {job_id}")
        
        # JALANKAN MONITORING DAEMON
        # Mengubah script monitoring berjalan secara background/foreground sesuai kebutuhan
        print("Memulai pemantauan background daemon Telegram...")
        monitor_job(job_id, desc)
    else:
        print("❌ Gagal mengirim job ke Slurm Master.")
        send_telegram_msg("❌ <b>Smart Submit Gagal</b>\nSlurm Master menolak pengiriman job.")

if __name__ == "__main__":
    main()
