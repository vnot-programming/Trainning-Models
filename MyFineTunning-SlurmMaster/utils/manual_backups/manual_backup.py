#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_backup.py
================
Script backup manual interaktif:
  1. User memilih file/folder yang akan dikompresi
  2. Kompresi ke format .tar.gz
  3. Upload ke Google Drive via rclone
  4. Notifikasi Telegram setiap 5 menit

Cara pakai:
    python3 manual_backup.py

    tmux new-session -s manual_backup
    python3 /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/manual_backups/manual_backup.py 2>&1 | tee manual_backup.log
    
    # Atau via tmux (otomatis):
    # Script akan membuat sesi tmux 'manual_backup' secara otomatis.

Prerequisite:
    - rclone sudah dikonfigurasi dengan remote 'gdrive'
    - pip install requests python-dotenv (opsional, fallback manual tersedia)
"""

import subprocess
import sys
import os
import time
import shutil
import socket
import tarfile
import signal
import threading
from datetime import datetime, timedelta

# ==============================================================================
# TELEGRAM NOTIFICATION
# ==============================================================================
def _load_dotenv():
    """Memuat variabel .env dari utils/manual_backups/.env terlebih dahulu, lalu fallback ke root project."""
    env_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/.env",
    ]
    env_path = None
    for candidate in env_candidates:
        if os.path.exists(candidate):
            env_path = candidate
            break

    if not env_path:
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # Fallback manual jika python-dotenv belum terinstall
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram(message: str):
    """Kirim pesan teks ke Telegram Bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️  Telegram tidak dikonfigurasi (BOT_TOKEN/CHAT_ID kosong)", C.YELLOW)
        return False

    hostname = socket.gethostname()
    prefix = f"<b>[Manual Backup @ {hostname}]</b>\n"
    full_msg = prefix + message

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": full_msg,
        "parse_mode": "HTML",
    }

    try:
        import requests
        response = requests.post(url, json=payload, timeout=15)
        return response.status_code == 200
    except ImportError:
        # Fallback menggunakan urllib jika requests tidak tersedia
        import urllib.request
        import json
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception:
            return False
    except Exception:
        return False


# ==============================================================================
# KONFIGURASI
# ==============================================================================
RCLONE_REMOTE = "gdrive"

# Kamus bulan Bahasa Indonesia untuk format folder Google Drive
BULAN_INDO = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

# Interval notifikasi Telegram (5 menit = 300 detik)
TELEGRAM_INTERVAL_SEC = 300

# Direktori temporary untuk menyimpan file tar.gz sebelum upload
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".manual_backup_tmp")

# Log file
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_backup.log")


# ==============================================================================
# WARNA TERMINAL
# ==============================================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    WHITE = "\033[97m"


# ==============================================================================
# UTILITAS
# ==============================================================================
def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, color=C.CYAN):
    text = f"[{timestamp()}] {msg}"
    print(f"{C.DIM}[{timestamp()}]{C.RESET} {color}{msg}{C.RESET}", flush=True)
    # Tulis juga ke log file
    try:
        with open(LOG_FILE, "a") as f:
            # Strip ANSI codes untuk log file
            f.write(text + "\n")
    except Exception:
        pass


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


def get_hostname():
    """Ambil hostname sistem."""
    return socket.gethostname()


def get_timestamp_id():
    """Buat timestamp ID untuk penamaan file (contoh: 20260609_154430)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_gdrive_date_folder():
    """Buat nama folder tanggal format Indonesia (contoh: 09_Juni_2026)."""
    now = datetime.now()
    hari = now.strftime("%d")
    bulan = BULAN_INDO[now.month]
    tahun = now.strftime("%Y")
    return f"{hari}_{bulan}_{tahun}"


def get_gdrive_base_path():
    """
    Bangun base path tujuan Google Drive.
    Format: (hostname)-backup/(timestamp format 09_Juni_2026)
    """
    hostname = get_hostname()
    date_folder = get_gdrive_date_folder()
    return f"{hostname}-backup/{date_folder}"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    hostname = get_hostname()
    env_type = "Slurm HPC Node" if "SLURM_JOB_ID" in os.environ else "Server"
    gdrive_path = get_gdrive_base_path()

    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║              📦  Manual Backup Tool (Compress + Upload)             ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
  {C.DIM}Hostname     : {hostname} ({env_type}){C.RESET}
  {C.DIM}GDrive Dest  : {RCLONE_REMOTE}:{gdrive_path}{C.RESET}
  {C.DIM}Waktu Mulai  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}
  {C.DIM}Log File     : {LOG_FILE}{C.RESET}
""")


# ==============================================================================
# VALIDASI
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
        log(f"   Remote tersedia: {remotes or '(kosong)'}", C.RED)
        log(f"   Jalankan: rclone config", C.YELLOW)
        sys.exit(1)

    log(f"✅ rclone remote '{RCLONE_REMOTE}' ditemukan", C.GREEN)


# ==============================================================================
# INTERAKTIF: PILIH FILE/FOLDER
# ==============================================================================
def interactive_select():
    """
    Menu interaktif untuk user memilih file/folder yang akan dibackup.
    Returns: list of absolute paths
    """
    targets = []

    while True:
        # Tampilkan header
        print(f"\n{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
        if targets:
            print(f"  {C.BOLD}📋 Item yang akan dibackup ({len(targets)} item):{C.RESET}")
            total_size = 0
            for i, t in enumerate(targets, 1):
                kind = "📁 FOLDER" if os.path.isdir(t) else "📄 FILE  "
                size = get_path_size(t)
                total_size += size
                print(f"    {C.GREEN}{i}.{C.RESET} [{kind}] {t} {C.DIM}({human_size(size)}){C.RESET}")
            print(f"    {C.DIM}{'─' * 50}{C.RESET}")
            print(f"    {C.BOLD}Total: {human_size(total_size)}{C.RESET}")
        else:
            print(f"  {C.DIM}📋 Belum ada item yang ditambahkan.{C.RESET}")

        print(f"{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
        print(f"  {C.BOLD}1.{C.RESET} 📄 Tambah File    {C.DIM}(bisa banyak, 1 path per baris){C.RESET}")
        print(f"  {C.BOLD}2.{C.RESET} 📁 Tambah Folder  {C.DIM}(bisa banyak, 1 path per baris){C.RESET}")
        print(f"  {C.BOLD}3.{C.RESET} ✅ Selesai & Lanjut Kompresi")
        print(f"  {C.BOLD}4.{C.RESET} ❌ Batalkan & Keluar")
        print()

        choice = input(f"  {C.BOLD}Pilihan (1-4): {C.RESET}").strip()

        if choice == "1":
            print(f"  {C.DIM}Paste path file (1 per baris). Tekan Enter kosong untuk selesai:{C.RESET}")
            added = 0
            while True:
                path = input(f"  📄 ").strip().strip("'\"")
                if not path:
                    break
                path = os.path.expanduser(path)
                abs_path = os.path.abspath(path)
                if not os.path.exists(path):
                    log(f"❌ Tidak ditemukan: {path}", C.RED)
                elif not os.path.isfile(path):
                    log(f"❌ Bukan file: {path}", C.RED)
                elif abs_path in targets:
                    log(f"⚠️  Sudah ada di daftar: {path}", C.YELLOW)
                else:
                    targets.append(abs_path)
                    added += 1
                    log(f"✅ +FILE: {os.path.basename(path)} ({human_size(get_path_size(path))})", C.GREEN)
            if added > 0:
                log(f"📊 {added} file ditambahkan.", C.CYAN)

        elif choice == "2":
            print(f"  {C.DIM}Paste path folder (1 per baris). Tekan Enter kosong untuk selesai:{C.RESET}")
            added = 0
            while True:
                path = input(f"  📁 ").strip().strip("'\"")
                if not path:
                    break
                path = os.path.expanduser(path)
                abs_path = os.path.abspath(path)
                if not os.path.exists(path):
                    log(f"❌ Tidak ditemukan: {path}", C.RED)
                elif not os.path.isdir(path):
                    log(f"❌ Bukan folder: {path}", C.RED)
                elif abs_path in targets:
                    log(f"⚠️  Sudah ada di daftar: {path}", C.YELLOW)
                else:
                    targets.append(abs_path)
                    added += 1
                    log(f"✅ +FOLDER: {os.path.basename(path)} ({human_size(get_path_size(path))})", C.GREEN)
            if added > 0:
                log(f"📊 {added} folder ditambahkan.", C.CYAN)

        elif choice == "3":
            if not targets:
                log("⚠️  Daftar masih kosong! Tambahkan file/folder terlebih dahulu.", C.YELLOW)
                continue
            break

        elif choice == "4":
            log("👋 Dibatalkan oleh user.", C.YELLOW)
            sys.exit(0)

        else:
            log("⚠️  Pilihan tidak valid. Masukkan 1-4.", C.YELLOW)

    return targets


# ==============================================================================
# KOMPRESI TAR.GZ
# ==============================================================================
def compress_to_targz(targets, output_path):
    """
    Kompresi daftar file/folder ke format .tar.gz.
    Menampilkan progress per-item.
    """
    log(f"📦 Memulai kompresi ke: {os.path.basename(output_path)}", C.CYAN)
    total_items = len(targets)

    try:
        with tarfile.open(output_path, "w:gz") as tar:
            for idx, item in enumerate(targets, 1):
                basename = os.path.basename(item)
                kind = "📁" if os.path.isdir(item) else "📄"
                size = get_path_size(item)
                print(
                    f"  {kind} [{idx}/{total_items}] Menambahkan: {basename} "
                    f"({human_size(size)})...",
                    end="",
                    flush=True,
                )
                tar.add(item, arcname=basename)
                print(f" {C.GREEN}✅{C.RESET}", flush=True)

        final_size = os.path.getsize(output_path)
        log(f"✅ Kompresi selesai: {os.path.basename(output_path)} ({human_size(final_size)})", C.GREEN)
        return True
    except Exception as e:
        log(f"❌ Gagal kompresi: {e}", C.RED)
        return False


# ==============================================================================
# PENAMAAN FILE
# ==============================================================================
def get_archive_name():
    """
    Tanya user untuk penamaan file arsip.
    Returns: nama file tanpa ekstensi .tar.gz
    """
    ts = get_timestamp_id()

    print(f"\n{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
    print(f"  {C.BOLD}Penamaan File Arsip:{C.RESET}")
    print(f"  {C.BOLD}1.{C.RESET} 📝 Custom name (format: {ts}-<nama_anda>)")
    print(f"  {C.BOLD}2.{C.RESET} 📋 Default (format: {ts})")
    print()

    choice = input(f"  {C.BOLD}Pilihan (1/2): {C.RESET}").strip()

    if choice == "1":
        custom = input(f"  📝 Masukkan nama custom (tanpa ekstensi): ").strip()
        if custom:
            # Sanitasi: hilangkan karakter berbahaya dari nama file
            safe_name = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in custom
            )
            name = f"{ts}-{safe_name}"
        else:
            log("⚠️  Nama kosong, menggunakan default.", C.YELLOW)
            name = ts
    else:
        name = ts

    log(f"📛 Nama arsip: {name}.tar.gz", C.CYAN)
    return name


# ==============================================================================
# UPLOAD KE GOOGLE DRIVE (RCLONE)
# ==============================================================================
def upload_to_gdrive(local_path, remote_base):
    """
    Upload file ke Google Drive menggunakan rclone copyto.
    Menampilkan progress real-time.
    """
    filename = os.path.basename(local_path)
    file_size = os.path.getsize(local_path)
    full_remote = f"{RCLONE_REMOTE}:{remote_base}/{filename}"

    log(f"📤 Mengupload: {filename} ({human_size(file_size)})", C.CYAN)
    log(f"   Tujuan: {full_remote}", C.DIM)

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

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            if "Transferred:" in line or "ETA" in line or "%" in line:
                print(
                    f"\r  {C.GREEN}{line.strip()}{C.RESET}".ljust(100),
                    end="",
                    flush=True,
                )
            elif "ERROR" in line.upper():
                print(f"\n  {C.RED}{line}{C.RESET}", flush=True)
            else:
                if any(kw in line for kw in ["Checks:", "Elapsed", "Transferred"]):
                    print(
                        f"\r  {C.DIM}{line.strip()}{C.RESET}".ljust(100),
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        process.terminate()
        log("⚠️  Upload dibatalkan oleh user (Ctrl+C)", C.YELLOW)
        return False, time.time() - start_time

    process.wait()
    elapsed = time.time() - start_time
    print(flush=True)

    if process.returncode == 0:
        speed = file_size / elapsed if elapsed > 0 else 0
        log(
            f"✅ Upload berhasil! ({human_size(file_size)} dalam "
            f"{timedelta(seconds=int(elapsed))}, avg: {human_size(speed)}/s)",
            C.GREEN,
        )
        return True, elapsed
    else:
        log(f"❌ Upload gagal! (exit code: {process.returncode})", C.RED)
        return False, elapsed


# ==============================================================================
# TELEGRAM HEARTBEAT (BACKGROUND THREAD)
# ==============================================================================
class TelegramHeartbeat:
    """
    Thread background yang mengirim notifikasi Telegram setiap 5 menit
    selama proses backup berjalan.
    """

    def __init__(self, interval_sec=TELEGRAM_INTERVAL_SEC):
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self._status = "Memulai..."
        self._start_time = time.time()

    def set_status(self, status):
        self._status = status

    def _run(self):
        while not self._stop_event.is_set():
            elapsed = timedelta(seconds=int(time.time() - self._start_time))
            msg = (
                f"⏱ <b>Heartbeat (setiap 5 menit)</b>\n"
                f"Status: {self._status}\n"
                f"Durasi berjalan: {elapsed}"
            )
            send_telegram(msg)
            # Tunggu interval, tapi bisa di-interrupt oleh stop_event
            self._stop_event.wait(self._interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)


# ==============================================================================
# VERIFIKASI UPLOAD
# ==============================================================================
def verify_upload(remote_base):
    """Verifikasi isi folder di Google Drive setelah upload."""
    log("📂 Memverifikasi isi folder di Google Drive...", C.CYAN)
    full_remote = f"{RCLONE_REMOTE}:{remote_base}"

    print(f"\n{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
    print(f"  📂 Isi {full_remote}:")
    print(f"{C.CYAN}{'━' * 60}{C.RESET}")

    try:
        result = subprocess.run(
            ["rclone", "ls", full_remote, "--max-depth", "1"],
            capture_output=True, text=True, timeout=30,
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"    {C.GREEN}{line.strip()}{C.RESET}")
        else:
            print(f"    {C.YELLOW}(folder kosong atau tidak ditemukan){C.RESET}")
    except Exception as e:
        log(f"⚠️  Gagal verifikasi: {e}", C.YELLOW)


# ==============================================================================
# CLEANUP
# ==============================================================================
def cleanup(temp_dir, tar_path=None):
    """Bersihkan file temporary."""
    try:
        if tar_path and os.path.exists(tar_path):
            os.remove(tar_path)
            log(f"🧹 File temporary dihapus: {tar_path}", C.DIM)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            log(f"🧹 Folder temporary dihapus: {temp_dir}", C.DIM)
    except Exception as e:
        log(f"⚠️  Gagal cleanup: {e}", C.YELLOW)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Inisialisasi log file
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Manual Backup Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")

    print_banner()

    # Step 1: Validasi rclone
    check_rclone()

    # Step 2: Deteksi lingkungan
    hostname = get_hostname()
    gdrive_base = get_gdrive_base_path()
    env_type = "Slurm HPC Node" if "SLURM_JOB_ID" in os.environ else "Server"
    log(f"📍 Lingkungan: {env_type} | Hostname: {hostname}", C.CYAN)
    log(f"📂 Google Drive tujuan: {RCLONE_REMOTE}:{gdrive_base}", C.CYAN)

    # Kirim notifikasi awal
    send_telegram(
        f"🚀 <b>Manual Backup Dimulai</b>\n"
        f"Lingkungan: {env_type}\n"
        f"Tujuan GDrive: <code>{RCLONE_REMOTE}:{gdrive_base}</code>"
    )

    # ══════════════════════════════════════════════════════════════
    # LOOP UTAMA: Siklus backup berulang hingga user memilih keluar
    # ══════════════════════════════════════════════════════════════
    session_count = 0

    while True:
        session_count += 1
        if session_count > 1:
            # Refresh gdrive base path (tanggal bisa berubah jika melewati tengah malam)
            gdrive_base = get_gdrive_base_path()
            print(f"\n{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
            log(f"🔄 Sesi Backup #{session_count}", C.CYAN)
            log(f"📂 Google Drive tujuan: {RCLONE_REMOTE}:{gdrive_base}", C.CYAN)

        # Step 3: Interaktif pilih file/folder
        targets = interactive_select()
        total_source_size = sum(get_path_size(t) for t in targets)
        log(f"📊 Total {len(targets)} item dipilih, ukuran: {human_size(total_source_size)}", C.CYAN)

        # Step 4: Tentukan nama file arsip
        archive_name = get_archive_name()

        # Step 5: Kompresi
        os.makedirs(TEMP_DIR, exist_ok=True)
        tar_path = os.path.join(TEMP_DIR, f"{archive_name}.tar.gz")

        # Mulai heartbeat Telegram
        heartbeat = TelegramHeartbeat()
        heartbeat.set_status(f"🗜 Mengompresi {len(targets)} item...")
        heartbeat.start()

        send_telegram(
            f"🗜 <b>Memulai Kompresi</b>\n"
            f"Item: {len(targets)} file/folder\n"
            f"Ukuran sumber: {human_size(total_source_size)}\n"
            f"Output: <code>{archive_name}.tar.gz</code>"
        )

        compress_start = time.time()
        success = compress_to_targz(targets, tar_path)

        if not success:
            heartbeat.stop()
            send_telegram(f"❌ <b>Kompresi GAGAL!</b>\nArsip: {archive_name}.tar.gz")
            cleanup(TEMP_DIR, tar_path)
            # Tanya apakah mau coba lagi setelah gagal
            retry = input(f"\n  {C.BOLD}Kompresi gagal. Lakukan backup lagi? (y/n): {C.RESET}").strip().lower()
            if retry in ("y", "yes", "ya"):
                continue
            else:
                break

        compress_elapsed = time.time() - compress_start
        compressed_size = os.path.getsize(tar_path)
        ratio = (1 - compressed_size / total_source_size) * 100 if total_source_size > 0 else 0

        send_telegram(
            f"✅ <b>Kompresi Selesai</b>\n"
            f"File: <code>{archive_name}.tar.gz</code>\n"
            f"Ukuran: {human_size(total_source_size)} → {human_size(compressed_size)} "
            f"(hemat {ratio:.1f}%)\n"
            f"Durasi: {timedelta(seconds=int(compress_elapsed))}"
        )

        # Step 6: Upload ke Google Drive
        heartbeat.set_status(
            f"📤 Mengupload {archive_name}.tar.gz ({human_size(compressed_size)})..."
        )

        send_telegram(
            f"📤 <b>Memulai Upload</b>\n"
            f"File: <code>{archive_name}.tar.gz</code> ({human_size(compressed_size)})\n"
            f"Tujuan: <code>{RCLONE_REMOTE}:{gdrive_base}</code>"
        )

        upload_success, upload_elapsed = upload_to_gdrive(tar_path, gdrive_base)

        # Step 7: Hentikan heartbeat
        heartbeat.stop()

        if upload_success:
            # Verifikasi
            verify_upload(gdrive_base)

            total_elapsed = compress_elapsed + upload_elapsed
            send_telegram(
                f"🎉 <b>Backup Selesai!</b>\n"
                f"File: <code>{archive_name}.tar.gz</code>\n"
                f"Ukuran: {human_size(compressed_size)}\n"
                f"Lokasi GDrive: <code>{RCLONE_REMOTE}:{gdrive_base}/{archive_name}.tar.gz</code>\n"
                f"Total durasi: {timedelta(seconds=int(total_elapsed))}"
            )

            # Ringkasan akhir
            print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║                     🎉  BACKUP BERHASIL!                            ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
  {C.BOLD}Arsip        :{C.RESET} {archive_name}.tar.gz
  {C.BOLD}Ukuran Asli  :{C.RESET} {human_size(total_source_size)}
  {C.BOLD}Ukuran Arsip :{C.RESET} {human_size(compressed_size)} (hemat {ratio:.1f}%)
  {C.BOLD}Lokasi GDrive:{C.RESET} {RCLONE_REMOTE}:{gdrive_base}/{archive_name}.tar.gz
  {C.BOLD}Kompresi     :{C.RESET} {timedelta(seconds=int(compress_elapsed))}
  {C.BOLD}Upload       :{C.RESET} {timedelta(seconds=int(upload_elapsed))}
  {C.BOLD}Total        :{C.RESET} {timedelta(seconds=int(total_elapsed))}
""")
        else:
            send_telegram(
                f"❌ <b>Upload GAGAL!</b>\n"
                f"File: <code>{archive_name}.tar.gz</code>\n"
                f"File lokal masih tersedia di: <code>{tar_path}</code>"
            )
            log(f"⚠️  File arsip masih tersedia di: {tar_path}", C.YELLOW)
            log("   Anda bisa upload manual: ", C.YELLOW)
            log(f"   rclone copyto {tar_path} {RCLONE_REMOTE}:{gdrive_base}/{archive_name}.tar.gz", C.YELLOW)

        # Step 8: Cleanup temporary files
        if upload_success:
            cleanup(TEMP_DIR, tar_path)

        # ══════════════════════════════════════════════════════════
        # Tanya apakah ingin melakukan backup lagi
        # ══════════════════════════════════════════════════════════
        print(f"\n{C.BOLD}{C.CYAN}{'━' * 60}{C.RESET}")
        again = input(f"  {C.BOLD}🔄 Lakukan backup lagi? (y/n): {C.RESET}").strip().lower()
        if again not in ("y", "yes", "ya"):
            break

    log("✅ Semua sesi backup selesai. Script ditutup.", C.GREEN)
    send_telegram(f"👋 <b>Manual Backup Ditutup</b>\nTotal sesi: {session_count}")


if __name__ == "__main__":
    main()

