#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
Menjalankan training pipeline secara BERURUTAN (sequential):
  YOLO8 → YOLO9 → YOLO11 → MaskRCNN

Setiap model:
  1. Dijalankan dalam tmux session terpisah
  2. Script ini menunggu sampai tmux session selesai (training done)
  3. Cooldown + verifikasi GPU idle sebelum model berikutnya
  4. Kill proses GPU zombie jika VRAM tidak turun

Cara pakai (WAJIB dalam tmux agar tahan terhadap terminal close):

  tmux new-session -d -s pipeline \
    "source ~/Computer-Vision/MyFineTunning-dev/.venv/bin/activate && \
     cd ~/Computer-Vision/MyFineTunning-dev && \
     python3 run_pipeline.py 2>&1 | tee PipelineReport.log"

Monitor:
  tmux attach -t pipeline

Jika listrik mati / pipeline crash:
  Jalankan ulang perintah yang sama — model yang sudah selesai (best.pt)
  otomatis di-skip, yang terputus (last.pt) otomatis resume.

Opsi:
  --skip yolo8,yolo9     ← Skip model tertentu (skip seluruh eksekusi termasuk eval)
  --only maskrcnn        ← Jalankan hanya model tertentu
"""

import subprocess
import time
import os
import sys
import gc
import signal
import argparse
from datetime import datetime, timedelta

# ==============================================================================
# PATH SETUP — Import dari config_shared (TIDAK ADA HARDCODED PATH)
# ==============================================================================
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from config_shared import (
    _FINETUNING_ROOT, PIPELINE_JOBS, VENV_ACTIVATE_PATH,
    GPU_IDLE_THRESHOLD_MIB, GPU_CLEANUP_TIMEOUT,
    GPU_CLEANUP_POLL_SEC, GPU_COOLDOWN_SEC,
)

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


# ==============================================================================
# GPU MEMORY MANAGEMENT — nvidia-smi based (bukan torch.cuda.empty_cache)
# ==============================================================================
def kill_gpu_zombies():
    """
    Kill semua proses yang menahan GPU VRAM (kecuali proses pipeline sendiri).
    Menggunakan nvidia-smi untuk deteksi PID → SIGKILL.
    """
    log("   🔪 Mencoba kill proses GPU zombie...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        my_pid = str(os.getpid())
        killed = 0

        for pid in pids:
            if pid == my_pid:
                continue
            log(f"   🔪 Killing GPU process PID={pid}")
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass  # Proses sudah mati
            except PermissionError:
                # Fallback: gunakan kill command
                subprocess.run(["kill", "-9", pid], capture_output=True)
                killed += 1

        if killed:
            time.sleep(5)  # Tunggu driver release VRAM
            log(f"   ✅ Killed {killed} proses GPU zombie")
        else:
            log("   ℹ️ Tidak ada proses GPU zombie ditemukan")
    except Exception as e:
        log(f"   ⚠️ Gagal kill GPU zombies: {e}")


def ensure_gpu_idle(threshold_mib=GPU_IDLE_THRESHOLD_MIB,
                    timeout=GPU_CLEANUP_TIMEOUT,
                    poll_sec=GPU_CLEANUP_POLL_SEC):
    """
    Tunggu sampai SEMUA GPU memiliki VRAM terpakai ≤ threshold_mib.

    Flow:
      1. Poll nvidia-smi setiap poll_sec detik
      2. Jika setelah 30s masih tinggi → kill proses GPU zombie
      3. Jika setelah timeout masih tinggi → force proceed + Telegram warning

    Returns:
        True jika GPU idle tercapai, False jika timeout (force proceed).
    """
    log(f"🧹 Verifikasi GPU idle (threshold: ≤{threshold_mib} MiB, timeout: {timeout}s)...")

    # Pembersihan ringan di proses pipeline (untuk jaga-jaga)
    gc.collect()

    start = time.time()
    kill_attempted = False

    while time.time() - start < timeout:
        gpu_info = get_gpu_memory()
        if not gpu_info:
            log("   ⚠️ Tidak bisa baca GPU info, skip verifikasi")
            return True

        max_used = max(g['used_mb'] for g in gpu_info)

        if max_used <= threshold_mib:
            log(f"   ✅ GPU idle! (max used: {max_used} MiB ≤ {threshold_mib} MiB)")
            print_gpu_status("GPU Idle — Verified")
            return True

        elapsed = int(time.time() - start)
        log(f"   ⏳ GPU belum idle: {max_used} MiB (elapsed: {elapsed}s)")

        # Setelah 30 detik masih tidak idle, coba kill zombie proses
        if elapsed >= 30 and not kill_attempted:
            kill_attempted = True
            kill_gpu_zombies()

        time.sleep(poll_sec)

    # Timeout — force proceed dengan warning
    gpu_info = get_gpu_memory()
    max_used = max(g['used_mb'] for g in gpu_info) if gpu_info else "?"
    log(f"   ⚠️ TIMEOUT ({timeout}s)! GPU masih: {max_used} MiB. Force proceed...")
    send_telegram(
        f"⚠️ <b>GPU cleanup timeout!</b>\n"
        f"VRAM: <code>{max_used} MiB</code>\n"
        f"Force proceed ke model berikutnya."
    )
    print_gpu_status("GPU Cleanup — Timeout")
    return False


# ==============================================================================
# TMUX SESSION MANAGEMENT
# ==============================================================================
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

    # Kill session lama jika ada
    kill_tmux_session(session)
    time.sleep(1)

    # Bangun command untuk tmux
    cmd = (
        f'source {VENV_ACTIVATE_PATH} && '
        f'cd {workdir} && '
        f'python -u {script} 2>&1 | tee {logfile}'
    )

    tmux_cmd = ["tmux", "new-session", "-d", "-s", session, cmd]

    log(f"🚀 Memulai: {session}")
    log(f"   Dir   : {workdir}")
    log(f"   Script: {script}")
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
    jobs = list(PIPELINE_JOBS)  # Copy dari config_shared
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
    print(f"  Root   : {_FINETUNING_ROOT}", flush=True)
    print(f"  GPU Idle Threshold : ≤{GPU_IDLE_THRESHOLD_MIB} MiB", flush=True)
    print(f"  GPU Cleanup Timeout: {GPU_CLEANUP_TIMEOUT}s", flush=True)
    print(f"  Cooldown           : {GPU_COOLDOWN_SEC}s", flush=True)
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

        # Bersihkan GPU sebelum model berikutnya
        if idx < len(jobs):
            print(flush=True)
            log(f"💤 Cooldown {GPU_COOLDOWN_SEC}s sebelum verifikasi GPU...")
            time.sleep(GPU_COOLDOWN_SEC)
            ensure_gpu_idle()
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


if __name__ == "__main__":
    main()
