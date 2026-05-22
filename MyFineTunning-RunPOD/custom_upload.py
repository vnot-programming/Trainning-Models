#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_upload.py
================
Upload file-file training ke Google Drive menggunakan rclone.

Cara pakai:
    python3 custom_upload.py

    # Background (recommended):
    tmux new-session -d -s upload "cd /root/Trainning-Models/MyFineTunning-dev && source .venv/bin/activate && python3 custom_upload.py 2>&1 | tee Custom_UploadReport.log"

Prerequisite:
    rclone harus sudah dikonfigurasi dengan remote Google Drive.
    Jalankan `rclone config` jika belum.
"""

import subprocess
import sys
import os
import time
import shutil
from datetime import datetime, timedelta

# ==============================================================================
# KONFIGURASI
# ==============================================================================
RCLONE_REMOTE = "gdrive"  # Nama remote rclone (ubah sesuai konfigurasi)
GDRIVE_BASE   = "gdrive-backup/runpod/20260502_134510"

DATA_DIR = "/root/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260502_134510"

# File-file .tar.gz yang akan diupload
TAR_FILES = [
    f"{DATA_DIR}/runs/yolov9m.tar.gz",
    f"{DATA_DIR}/runs/yolov9c_seg.tar.gz",
    f"{DATA_DIR}/runs/yolov8m.tar.gz",
    f"{DATA_DIR}/runs/yolov8m_seg.tar.gz",
    f"{DATA_DIR}/runs/yolo11l.tar.gz",
    f"{DATA_DIR}/runs/yolo11l_seg.tar.gz",
]

# Folder-folder yang akan diupload
FOLDERS = [
    f"{DATA_DIR}/image_samples",
    f"{DATA_DIR}/reports",
    f"{DATA_DIR}/visuals",
]

# ==============================================================================
# WARNA TERMINAL
# ==============================================================================
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, color=C.CYAN):
    print(f"{C.DIM}[{timestamp()}]{C.RESET} {color}{msg}{C.RESET}", flush=True)


def human_size(nbytes):
    """Format bytes ke human-readable (KB, MB, GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def get_path_size(path):
    """Hitung total ukuran file atau folder."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


# ==============================================================================
# RCLONE UPLOAD DENGAN PROGRESS
# ==============================================================================
def check_rclone():
    """Validasi rclone terinstall dan remote terkonfigurasi."""
    if not shutil.which("rclone"):
        log("❌ rclone tidak ditemukan. Install: curl https://rclone.org/install.sh | sudo bash", C.RED)
        sys.exit(1)

    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    remotes = [r.strip().rstrip(":") for r in result.stdout.strip().split("\n") if r.strip()]

    if RCLONE_REMOTE not in remotes:
        log(f"❌ Remote '{RCLONE_REMOTE}' tidak ditemukan di rclone config.", C.RED)
        log(f"   Remote yang tersedia: {remotes or '(kosong)'}", C.RED)
        log(f"   Jalankan: rclone config", C.YELLOW)
        print()
        log("📋 Panduan singkat setup rclone Google Drive:", C.YELLOW)
        print(f"""
    1. Jalankan: rclone config
    2. Pilih: n (New remote)
    3. Name : {RCLONE_REMOTE}
    4. Type : drive (Google Drive)
    5. Ikuti instruksi OAuth (bisa pakai --auto jika ada browser,
       atau copy-paste token dari mesin lain)
    6. Selesai, lalu jalankan ulang script ini.
""")
        sys.exit(1)

    log(f"✅ rclone remote '{RCLONE_REMOTE}' ditemukan", C.GREEN)


def upload_item(local_path, remote_path, item_label, idx, total):
    """
    Upload satu file/folder ke Google Drive dengan progress bar.

    Untuk file   → rclone copyto
    Untuk folder → rclone copy
    """
    if not os.path.exists(local_path):
        log(f"⚠️  SKIP — tidak ditemukan: {local_path}", C.YELLOW)
        return False, 0, 0

    size = get_path_size(local_path)
    is_dir = os.path.isdir(local_path)
    kind = "📁" if is_dir else "📦"

    # Header
    print(flush=True)
    print(f"{C.BOLD}{C.CYAN}{'━' * 70}{C.RESET}", flush=True)
    print(f"  {kind} [{idx}/{total}] {C.BOLD}{item_label}{C.RESET}", flush=True)
    print(f"  {C.DIM}Size: {human_size(size)} | Dest: {RCLONE_REMOTE}:{remote_path}{C.RESET}", flush=True)
    print(f"{C.CYAN}{'━' * 70}{C.RESET}", flush=True)

    # Bangun command rclone
    full_remote = f"{RCLONE_REMOTE}:{remote_path}"

    if is_dir:
        cmd = [
            "rclone", "copy",
            local_path, full_remote,
            "--progress",
            "--stats", "2s",
            "--stats-one-line",
            "--transfers", "4",
            "--checkers", "8",
            "--log-level", "NOTICE",
        ]
    else:
        cmd = [
            "rclone", "copyto",
            local_path, full_remote,
            "--progress",
            "--stats", "2s",
            "--stats-one-line",
            "--transfers", "4",
            "--log-level", "NOTICE",
        ]

    start_time = time.time()

    # Jalankan rclone dengan real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_progress_line = ""
    try:
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            # rclone progress line biasanya mengandung "Transferred:" atau "%"
            if "Transferred:" in line or "ETA" in line or "%" in line:
                last_progress_line = line.strip()
                # Print ke terminal (pakai \r agar tidak spam baris baru)
                print(f"\r  {C.GREEN}{last_progress_line}{C.RESET}".ljust(100), end="", flush=True)
            elif "ERROR" in line.upper():
                print(f"\n  {C.RED}{line}{C.RESET}", flush=True)
            else:
                if any(kw in line for kw in ["Checks:", "Elapsed", "Transferred"]):
                    print(f"\r  {C.DIM}{line.strip()}{C.RESET}".ljust(100), end="", flush=True)
                else:
                    print(f"\n  {C.DIM}{line}{C.RESET}", flush=True)

    except KeyboardInterrupt:
        process.terminate()
        log("⚠️  Upload dibatalkan oleh user (Ctrl+C)", C.YELLOW)
        return False, size, time.time() - start_time

    process.wait()
    elapsed = time.time() - start_time

    print(flush=True)  # New line setelah progress

    if process.returncode == 0:
        speed = size / elapsed if elapsed > 0 else 0
        log(f"✅ Berhasil! ({human_size(size)} dalam {timedelta(seconds=int(elapsed))}, "
            f"avg: {human_size(speed)}/s)", C.GREEN)
        return True, size, elapsed
    else:
        log(f"❌ Gagal! (exit code: {process.returncode})", C.RED)
        return False, size, elapsed


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║           📤  Custom Upload to Google Drive (rclone)                ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
  {C.DIM}Destination: {RCLONE_REMOTE}:{GDRIVE_BASE}{C.RESET}
  {C.DIM}Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}
""")

    # 1. Validasi rclone
    check_rclone()

    # 2. Validasi file-file yang ada
    all_items = []

    # tar.gz files → upload ke root folder
    for fpath in TAR_FILES:
        fname = os.path.basename(fpath)
        remote_dest = f"{GDRIVE_BASE}/{fname}"
        all_items.append((fpath, remote_dest, fname))

    # Folder → upload ke subfolder dengan nama sama
    for fpath in FOLDERS:
        folder_name = os.path.basename(fpath)
        remote_dest = f"{GDRIVE_BASE}/{folder_name}"
        all_items.append((fpath, remote_dest, f"{folder_name}/"))

    # Hitung total size
    total_size = 0
    valid_count = 0
    for local, _remote, _label in all_items:
        if os.path.exists(local):
            total_size += get_path_size(local)
            valid_count += 1

    log(f"📊 Total: {valid_count} item, {human_size(total_size)} untuk diupload", C.CYAN)

    if valid_count == 0:
        log("❌ Tidak ada file/folder yang ditemukan untuk diupload!", C.RED)
        sys.exit(1)

    # 3. Upload satu per satu
    results = []
    total_uploaded = 0
    total_time = 0
    pipeline_start = time.time()

    for idx, (local, remote, label) in enumerate(all_items, 1):
        success, size, elapsed = upload_item(local, remote, label, idx, len(all_items))
        results.append({
            "label": label,
            "success": success,
            "size": size,
            "elapsed": elapsed,
        })
        if success:
            total_uploaded += size
            total_time += elapsed

    # 4. Ringkasan
    pipeline_elapsed = time.time() - pipeline_start
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║                       📊  RINGKASAN UPLOAD                         ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
""", flush=True)

    # Tabel hasil
    print(f"  {C.BOLD}{'Item':<30} {'Status':<12} {'Size':>10} {'Durasi':>10}{C.RESET}", flush=True)
    print(f"  {'─' * 30} {'─' * 12} {'─' * 10} {'─' * 10}", flush=True)

    success_count = 0
    fail_count = 0
    for r in results:
        if r["success"]:
            status = f"{C.GREEN}✅ OK{C.RESET}"
            success_count += 1
        elif not os.path.exists(r.get("local", "")):
            status = f"{C.YELLOW}⚠️ SKIP{C.RESET}"
        else:
            status = f"{C.RED}❌ FAIL{C.RESET}"
            fail_count += 1

        dur = str(timedelta(seconds=int(r["elapsed"]))) if r["elapsed"] > 0 else "-"
        sz = human_size(r["size"]) if r["size"] > 0 else "-"
        print(f"  {r['label']:<30} {status:<22} {sz:>10} {dur:>10}", flush=True)

    avg_speed = total_uploaded / total_time if total_time > 0 else 0
    print(f"""
  {C.DIM}{'─' * 65}{C.RESET}
  {C.BOLD}Total Uploaded : {human_size(total_uploaded)}{C.RESET}
  {C.BOLD}Total Waktu    : {timedelta(seconds=int(pipeline_elapsed))}{C.RESET}
  {C.BOLD}Avg Speed      : {human_size(avg_speed)}/s{C.RESET}
  {C.BOLD}Berhasil       : {success_count}/{len(results)}{C.RESET}
""", flush=True)

    if fail_count > 0:
        log(f"⚠️  {fail_count} upload gagal. Cek error di atas.", C.RED)
        sys.exit(1)
    else:
        log(f"🎉 Semua upload berhasil ke {RCLONE_REMOTE}:{GDRIVE_BASE}", C.GREEN)

    # Verifikasi isi folder di GDrive
    print(f"\n{C.CYAN}📂 Isi folder di Google Drive:{C.RESET}")
    subprocess.run(
        ["rclone", "lsd", f"{RCLONE_REMOTE}:{GDRIVE_BASE}"],
        timeout=30,
    )
    print()
    subprocess.run(
        ["rclone", "ls", f"{RCLONE_REMOTE}:{GDRIVE_BASE}",
         "--max-depth", "1"],
        timeout=30,
    )


if __name__ == "__main__":
    main()
