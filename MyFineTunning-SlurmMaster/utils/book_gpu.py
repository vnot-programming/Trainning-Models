#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
book_gpu.py
===========
Smart Slurm GPU Booking & Real-time Telegram Monitor.
- Secara cerdas mendeteksi kesehatan node (mengabaikan DRAIN/DOWN, memilih node idle).
- Mensubmit job booking (sleep infinity) ke Slurm.
- Mengirim notifikasi Telegram saat antrean, berjalan, atau selesai.

Cara Menjalankan:
    python book_gpu.py
    
atau via tmux agar proses monitoring tidak terputus:
    tmux new-session -d -s gpu_booking "python book_gpu.py"
"""

import os
import sys
import time
import subprocess
import re

# We need telegram_utils, since it's in MyFineTunning-SlurmMaster
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

try:
    from telegram_utils import send_telegram_msg
except ImportError:
    def send_telegram_msg(msg, force=False):
        print(f"[Telegram Mock] {msg}")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_cmd_output(cmd, timeout=30):
    try:
        return subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
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

def generate_booking_sbatch(target_node=None, exclude_nodes=None):
    """Membuat file sbatch booking dinamis."""
    sbatch_path = os.path.join(_THIS_DIR, "submit_booking_run.sbatch")
    
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=VnoT-Train",
        "#SBATCH --output=slurm_logs/booking_output_%j.log",
        "#SBATCH --error=slurm_logs/booking_error_%j.log",
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
        "echo \"GPU Booking Active on $(hostname) at $(date)\"",
        "echo \"Job ID: $SLURM_JOB_ID\"",
        "echo \"Allocated GPUs: $CUDA_VISIBLE_DEVICES\"",
        "echo \"==========================================================\"",
        "echo \"\"",
        "echo \"Cara masuk ke node ini tanpa antre:\"",
        "echo \"  ./attach_gpu.sh\"",
        "echo \"\"",
        "echo \"Menahan node agar tidak tertutup...\"",
        "sleep infinity"
    ])
    
    with open(sbatch_path, "w") as f:
        f.write("\n".join(lines))
    return sbatch_path

def get_job_state(job_id):
    """Membaca status job Slurm."""
    output = get_cmd_output(f"squeue -j {job_id} -h -o '%t %r'")
    if not output:
        sacct_output = get_cmd_output(f"sacct -j {job_id} -h -o 'State,ExitCode' | head -n 1")
        if sacct_output:
            parts = sacct_output.split()
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
            elif len(parts) == 1:
                return parts[0], ""
        return "UNKNOWN", ""
        
    parts = output.split()
    state = parts[0]
    reason = " ".join(parts[1:]) if len(parts) > 1 else ""
    return state, reason

def monitor_job(job_id, target_desc):
    """Fungsi monitoring utama dengan Telegram Alerting."""
    send_telegram_msg(
        f"📝 <b>GPU Booking Submitted</b>\n"
        f"Job ID: <code>{job_id}</code>\n"
        f"Rencana Rute: {target_desc}\n"
        f"Status: <code>PENDING</code> (Menunggu Antrean)"
    )
    
    last_notif_time = time.time()
    notif_interval = 1800  # 30 menit
    started = False
    
    while True:
        state, detail = get_job_state(job_id)
        
        if state == "PD":
            current_time = time.time()
            if current_time - last_notif_time >= notif_interval:
                send_telegram_msg(
                    f"⏳ <b>GPU Booking Queue Update</b>\n"
                    f"Job ID: <code>{job_id}</code> masih menunggu di antrean.\n"
                    f"Alasan: <code>{detail}</code>"
                )
                last_notif_time = current_time
                
        elif state == "R" and not started:
            started = True
            node_running = get_cmd_output(f"squeue -j {job_id} -h -o '%N'")
            send_telegram_msg(
                f"🚀 <b>GPU Booking Active!</b>\n"
                f"Job ID: <code>{job_id}</code> telah berjalan.\n"
                f"Node Komputasi: <code>{node_running}</code>\n\n"
                f"Gunakan <code>./attach_gpu.sh</code> (di dalam utils) untuk masuk ke shell."
            )
            
        elif state in ["CG", "COMPLETING"]:
            pass  # Sedang proses transisi penutupan, tunggu saja
        elif state != "R":
            # Jika bukan PD, bukan R, dan bukan CG, berarti job sudah mati (TIMEOUT+, CANCELLED, FAILED, OUT_OF_ME, dll)
            if "COMPLETED" in state or "SUCCESS" in state:
                send_telegram_msg(
                    f"✅ <b>GPU Booking Finished!</b>\n"
                    f"Job ID: <code>{job_id}</code> selesai atau ditutup."
                )
                return False
            else:
                send_telegram_msg(
                    f"❌ <b>GPU Booking Interrupted!</b>\n"
                    f"Job ID: <code>{job_id}</code> berakhir dengan status: <code>{state}</code>\n"
                    f"Exit Code/Detail: <code>{detail}</code>\n\n"
                    f"🔄 <b>Sistem True-Daemon Auto-Resume akan merekonstruksi ulang booking GPU Anda segera!</b>"
                )
                return True
            
        time.sleep(15)

def main():
    print("✧ SMART SLURM GPU BOOKING ✧")
    
    while True:
        print("Memindai topologi kluster secara real-time...")
        
        healthy, draining = scan_slurm_nodes()
        target_node = None
        exclude_list = draining.copy()
        
        idle_nodes = [n for n in healthy if n['free_gpu'] > 0]
        if idle_nodes:
            best_node = max(idle_nodes, key=lambda x: x['free_gpu'])
            target_node = best_node['name']
            desc = f"Di-dispatch langsung ke Node <b>{target_node}</b> (Memiliki {best_node['free_gpu']} GPU Kosong)"
        else:
            desc = f"Antrean Global (Menggunakan Node Sehat, Mengecualikan Node DRAIN: {', '.join(exclude_list) if exclude_list else 'None'})"

        print(f"Rute Terpilih: {desc}")
        
        sbatch_script = generate_booking_sbatch(target_node=target_node, exclude_nodes=exclude_list)
        print(f"Mengompilasi script Slurm: {sbatch_script}")
        
        submit_res = get_cmd_output(f"sbatch {sbatch_script}")
        match = re.search(r'Submitted batch job (\d+)', submit_res)
        
        if match:
            job_id = match.group(1)
            print(f"Berhasil mensubmit Job ID: {job_id}")
            print("Memulai pemantauan background daemon Telegram...")
            should_resume = monitor_job(job_id, desc)
            
            if not should_resume:
                break
            
            print("⏳ Menunggu 5 detik sebelum auto-resume...")
            time.sleep(5)
        else:
            print("❌ Gagal mengirim job ke Slurm Master.")
            send_telegram_msg("❌ <b>Smart Submit Gagal</b>\nSlurm Master menolak pengiriman GPU Booking.")
            break

if __name__ == "__main__":
    main()
