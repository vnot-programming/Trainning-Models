#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
Menjalankan training pipeline secara BERURUTAN (sequential):
  YOLO8 → YOLO9 → YOLO11 → MaskRCNN → Hybrid Evaluation

Setiap model:
  1. Dijalankan dalam tmux session terpisah
  2. Script ini menunggu sampai tmux session selesai (training done)
  3. Membersihkan GPU memory sebelum memulai model berikutnya

Cara pakai:
  source Trainning-Models/MyFineTunning-dev/.venv/bin/activate && python3 run_pipeline.py 2>&1 | tee PipelineReport.log

Atau via tmux (recommended):
  tmux new-session -d -s run_pipeline "source .venv/bin/activate && python3 run_pipeline.py 2>&1 | tee run_pipeline.log"

  

Opsi:
  --skip yolo8,yolo9     ← Skip model tertentu
  --only maskrcnn        ← Jalankan hanya model tertentu

Contoh:
Ran command: `tmux kill-session -t yolo8training`
Edited config_shared.py
Viewed main.py:4-23
Edited main.py
Created run_pipeline.py

Script `run_pipeline.py` sudah dibuat! Berikut cara pakainya:

### Cara Jalankan (via tmux, recommended):
```bash
tmux new-session -d -s run_pipeline "python3 run_pipeline.py 2>&1 | tee run_pipeline.log"
```

### Fitur:
| Fitur | Penjelasan |
|---|---|
| **Sequential** | YOLO8 → YOLO9 → YOLO11 → MaskRCNN → Hybrid berurutan |
| **Memory Cleanup** | `torch.cuda.empty_cache()` + gc + cek zombie proses di antara setiap model |
| **GPU Monitoring** | Cetak status VRAM setiap 30 detik selama training |
| **Telegram** | Notifikasi saat setiap model selesai dan ringkasan akhir |
| **Ringkasan** | Tabel durasi setiap model di akhir pipeline |

### Opsi Tambahan:
```bash
# Skip model tertentu
python3 run_pipeline.py --skip yolo8,yolo9

# Jalankan hanya model tertentu
python3 run_pipeline.py --only maskrcnn
```

### Monitor:
```bash
tmux attach -t pipeline     # Lihat output pipeline
tmux ls                      # Lihat semua session aktif
nvitop                       # Monitor GPU real-time
```

"""

import subprocess
import time
import os
import sys
import gc
import argparse
from datetime import datetime, timedelta

# ==============================================================================
# KONFIGURASI PIPELINE
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VENV_ACTIVATE = f"source {os.path.join(BASE_DIR, '.venv', 'bin', 'activate')}"

TRAINING_JOBS = [
    {
        "name": "yolo8",
        "session": "yolo8training",
        "workdir": f"{BASE_DIR}/yolo/yolo8",
        "script": "main.py",
        "logfile": "yolo8training.log",
        "args": "",
    },
    {
        "name": "yolo9",
        "session": "yolo9training",
        "workdir": f"{BASE_DIR}/yolo/yolo9",
        "script": "main.py",
        "logfile": "yolo9training.log",
        "args": "",
    },
    {
        "name": "yolo11",
        "session": "yolo11training",
        "workdir": f"{BASE_DIR}/yolo/yolo11",
        "script": "main.py",
        "logfile": "yolo11training.log",
        "args": "",
    },
    {
        "name": "maskrcnn",
        "session": "masktraining",
        "workdir": f"{BASE_DIR}/mask-r-cnn",
        "script": "train_multigpu.py",
        "logfile": "masktraining.log",
        "args": "",
    },
    {
        "name": "hybrid",
        "session": "hybrideval",
        "workdir": f"{BASE_DIR}/hybrid",
        "script": "main.py",
        "logfile": "hybrideval.log",
        "args": "",
    },
]

# Interval polling (detik) — seberapa sering cek apakah tmux session masih aktif
POLL_INTERVAL = 30


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{timestamp()}] [Pipeline] {msg}", flush=True)


def send_telegram(msg):
    """Kirim notifikasi Telegram (silent fail jika tidak dikonfigurasi)."""
    try:
        sys.path.insert(0, BASE_DIR)
        from telegram_utils import send_telegram_msg
        send_telegram_msg(msg)
    except Exception:
        pass


def get_gpu_memory():
    """Ambil penggunaan VRAM setiap GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        gpu_info = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpu_info.append({
                    "index": int(parts[0]),
                    "used_mb": int(parts[1]),
                    "total_mb": int(parts[2]),
                    "free_mb": int(parts[3]),
                })
        return gpu_info
    except Exception as e:
        log(f"⚠️  Gagal baca GPU info: {e}")
        return []


def print_gpu_status(label=""):
    """Cetak status VRAM semua GPU."""
    gpu_info = get_gpu_memory()
    if not gpu_info:
        return
    header = f"GPU Status{f' ({label})' if label else ''}"
    log(header)
    for gpu in gpu_info:
        pct = (gpu['used_mb'] / gpu['total_mb'] * 100) if gpu['total_mb'] > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"    GPU {gpu['index']}: [{bar}] {gpu['used_mb']:>6} / {gpu['total_mb']:>6} MiB ({pct:.0f}%)", flush=True)


def flush_gpu_memory():
    """Bersihkan GPU memory secara agresif."""
    log("🧹 Membersihkan GPU memory...")

    # 1. Python garbage collection
    gc.collect()

    # 2. PyTorch CUDA cache (jika torch tersedia di environment ini)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            log("   ✅ torch.cuda.empty_cache() berhasil")
    except ImportError:
        pass

    # 3. Kill proses GPU zombie yang mungkin tertinggal
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if pids:
            log(f"   ⚠️  Ditemukan {len(pids)} proses GPU masih berjalan: {pids}")
            log(f"   ℹ️  Proses akan dibiarkan (bukan milik pipeline ini)")
        else:
            log("   ✅ Tidak ada proses GPU yang tertinggal")
    except Exception:
        pass

    # 4. Tunggu sebentar agar memory benar-benar dibebaskan oleh driver
    time.sleep(5)

    print_gpu_status("Setelah Cleanup")


def is_tmux_session_alive(session_name):
    """Cek apakah tmux session masih aktif."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True
    )
    return result.returncode == 0


def kill_tmux_session(session_name):
    """Kill tmux session jika masih ada."""
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True
    )


def start_training(job):
    """Mulai training dalam tmux session baru."""
    session = job["session"]
    workdir = job["workdir"]
    script = job["script"]
    logfile = job["logfile"]
    extra_args = job.get("args", "")

    # Kill session lama jika ada
    kill_tmux_session(session)
    time.sleep(1)

    # Bangun command untuk tmux
    script_cmd = f'python -u {script}'
    if extra_args:
        script_cmd += f' {extra_args}'

    cmd = (
        f'{VENV_ACTIVATE} && '
        f'cd {workdir} && '
        f'{script_cmd} 2>&1 | tee {logfile}'
    )

    tmux_cmd = ["tmux", "new-session", "-d", "-s", session, cmd]

    log(f"🚀 Memulai: {session}")
    log(f"   Dir   : {workdir}")
    log(f"   Script: {script} {extra_args}")
    log(f"   Log   : {workdir}/{logfile}")

    result = subprocess.run(tmux_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"❌ Gagal memulai tmux session: {result.stderr}")
        return False

    # Verifikasi session berjalan
    time.sleep(2)
    if is_tmux_session_alive(session):
        log(f"   ✅ Session '{session}' berjalan")
        return True
    else:
        log(f"   ❌ Session '{session}' langsung mati — cek log: {workdir}/{logfile}")
        return False


def wait_for_completion(job):
    """Tunggu sampai tmux session selesai (polling)."""
    session = job["session"]
    name = job["name"]
    start_time = time.time()

    log(f"⏳ Menunggu {name} selesai (polling setiap {POLL_INTERVAL}s)...")

    while is_tmux_session_alive(session):
        elapsed = timedelta(seconds=int(time.time() - start_time))
        # Cetak status setiap poll
        gpu_info = get_gpu_memory()
        if gpu_info:
            max_used = max(g['used_mb'] for g in gpu_info)
            avg_used = sum(g['used_mb'] for g in gpu_info) // len(gpu_info)
            print(
                f"    [{timestamp()}] {name} masih berjalan... "
                f"(elapsed: {elapsed}, GPU avg: {avg_used}MiB, max: {max_used}MiB)",
                flush=True
            )
        else:
            print(f"    [{timestamp()}] {name} masih berjalan... (elapsed: {elapsed})", flush=True)

        time.sleep(POLL_INTERVAL)

    elapsed = timedelta(seconds=int(time.time() - start_time))
    log(f"✅ {name} selesai! (durasi: {elapsed})")

    return elapsed


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sequential Training Pipeline")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated model names to skip (e.g. yolo8,yolo9)")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated model names to run exclusively")
    args = parser.parse_args()

    skip_list = [s.strip().lower() for s in args.skip.split(",") if s.strip()]
    only_list = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    # Filter jobs berdasarkan --skip / --only
    jobs = TRAINING_JOBS
    if only_list:
        jobs = [j for j in jobs if j["name"].lower() in only_list]
    elif skip_list:
        jobs = [j for j in jobs if j["name"].lower() not in skip_list]

    if not jobs:
        log("❌ Tidak ada model yang akan dijalankan. Cek --skip atau --only.")
        sys.exit(1)

    # Header
    print("=" * 65, flush=True)
    print("  🏗️  Sequential Training Pipeline", flush=True)
    print(f"  Started: {timestamp()}", flush=True)
    print(f"  Models : {', '.join(j['name'].upper() for j in jobs)}", flush=True)
    print("=" * 65, flush=True)
    print(flush=True)

    send_telegram(
        f"🏗️ <b>Training Pipeline Started</b>\n"
        f"Models: {', '.join(j['name'].upper() for j in jobs)}\n"
        f"Time: <code>{timestamp()}</code>"
    )

    print_gpu_status("Sebelum Pipeline")
    print(flush=True)

    results = []
    pipeline_start = time.time()

    for idx, job in enumerate(jobs, 1):
        name = job["name"].upper()
        separator = "=" * 65
        print(flush=True)
        print(separator, flush=True)
        print(f"  [{idx}/{len(jobs)}] {name}", flush=True)
        print(separator, flush=True)
        print(flush=True)

        # Mulai training
        success = start_training(job)
        if not success:
            results.append({"name": name, "status": "GAGAL START", "duration": "-"})
            send_telegram(f"❌ <b>{name}</b> gagal dimulai!")
            continue

        print_gpu_status(f"{name} — Awal Training")

        # Tunggu selesai
        duration = wait_for_completion(job)
        results.append({"name": name, "status": "SELESAI", "duration": str(duration)})

        send_telegram(
            f"✅ <b>{name}</b> selesai!\n"
            f"Durasi: <code>{duration}</code>\n"
            f"Progress: {idx}/{len(jobs)}"
        )

        # Bersihkan memory sebelum model berikutnya
        if idx < len(jobs):
            print(flush=True)
            flush_gpu_memory()
            print(flush=True)

    # ==============================================================================
    # RINGKASAN AKHIR
    # ==============================================================================
    pipeline_duration = timedelta(seconds=int(time.time() - pipeline_start))

    print(flush=True)
    print("=" * 65, flush=True)
    print("  📊 RINGKASAN PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print(f"  {'Model':<12} {'Status':<15} {'Durasi'}", flush=True)
    print(f"  {'-'*12} {'-'*15} {'-'*20}", flush=True)
    for r in results:
        print(f"  {r['name']:<12} {r['status']:<15} {r['duration']}", flush=True)
    print(f"\n  Total Pipeline: {pipeline_duration}", flush=True)
    print(f"  Selesai pada  : {timestamp()}", flush=True)
    print("=" * 65, flush=True)

    # Ringkasan Telegram
    summary_lines = "\n".join(
        f"  {r['name']}: {r['status']} ({r['duration']})" for r in results
    )
    send_telegram(
        f"📊 <b>Pipeline Selesai!</b>\n"
        f"Total: <code>{pipeline_duration}</code>\n\n"
        f"<pre>{summary_lines}</pre>"
    )

    print_gpu_status("Akhir Pipeline")

    # # ==============================================================================
    # # UPLOAD KE GDRIVE
    # # ==============================================================================
    # print(flush=True)
    # print("=" * 65, flush=True)
    # print("  ☁️ MEMULAI UPLOAD KE GDRIVE", flush=True)
    # print("=" * 65, flush=True)
    # send_telegram("☁️ <b>Memulai Upload ke GDrive...</b>\nMenjalankan sinkronisasi via RClone.")
    
    # upload_result = subprocess.run(["bash", "rclone_sync.sh", "upload"], cwd=BASE_DIR)
    
    # if upload_result.returncode == 0:
    #     send_telegram("🏁 <b>Upload Selesai!</b>\nSemua data hasil pipeline & kompresi berhasil diamankan ke cloud.")
    # else:
    #     send_telegram("⚠️ <b>Upload Selesai dengan Error!</b>\nMohon periksa log RClone untuk melihat detail gagalnya upload.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ [Pipeline] Dibatalkan oleh pengguna (KeyboardInterrupt)!")
        print("  🧹 Membersihkan sesi tmux yang tertinggal...")
        for job in JOBS:
            subprocess.run(["tmux", "kill-session", "-t", job["name"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  🧹 Membersihkan GPU memory...")
        flush_gpu_memory()
        print("✅ Pipeline dihentikan secara aman.")
        sys.exit(130)
