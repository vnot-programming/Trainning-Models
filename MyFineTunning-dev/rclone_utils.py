# -*- coding: utf-8 -*-
"""
rclone_utils.py
===============
Helper Python untuk sinkronisasi hasil training ke Google Drive via rclone.

Struktur folder di GDrive:
    gdrive-backup/
    └── {hostname}/
        └── {project_name}-{timestamp}/
            ├── weights/
            ├── reports/
            └── visuals/

Cara pakai:
    from rclone_utils import upload_results, download_weights

    # Upload semua hasil training
    upload_results()

    # Download weights yang sudah ada di GDrive ke local
    download_weights()
"""

import os
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from telegram_utils import send_telegram_msg

# Root project
_PROJECT_ROOT = Path(__file__).resolve().parent

# Baca config dari .env atau environment
def _load_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

RCLONE_BIN    = os.environ.get("RCLONE_BIN",    str(Path.home() / ".local/bin/rclone"))
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
RCLONE_DEST   = os.environ.get("RCLONE_DEST",   "gdrive-backup")
HOSTNAME      = socket.gethostname()


# ==============================================================================
# HELPER — Baca workspace_id (timestamp sesi training)
# ==============================================================================
def _get_workspace_timestamp() -> str:
    ws_id_file = _PROJECT_ROOT / ".workspace_id"
    if ws_id_file.exists():
        return ws_id_file.read_text().strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_remote_folder() -> str:
    """
    Susun path remote GDrive secara dinamis:
        gdrive-backup/{hostname}/MyFineTunning-{timestamp}
    """
    ts     = _get_workspace_timestamp()
    folder = f"{RCLONE_REMOTE}:{RCLONE_DEST}/{HOSTNAME}/MyFineTunning-{ts}"
    return folder


def _rclone_available() -> bool:
    return Path(RCLONE_BIN).exists() or _cmd_exists("rclone")


def _cmd_exists(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


def _run_rclone(args: list[str], label: str = "") -> bool:
    """Jalankan rclone command dan return True jika berhasil."""
    bin_path = RCLONE_BIN if Path(RCLONE_BIN).exists() else "rclone"
    cmd = [bin_path] + args
    print(f"[RClone] {label or ' '.join(args[:3])}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[RClone] ✅ Berhasil.")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"[RClone] ❌ Gagal (code {result.returncode}):")
            print(result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        print("[RClone] ❌ Timeout setelah 5 menit.")
        return False
    except Exception as e:
        print(f"[RClone] ❌ Error: {e}")
        return False


# ==============================================================================
# PUBLIC API
# ==============================================================================

def upload_results(verbose: bool = True) -> bool:
    """
    Upload semua hasil training ke Google Drive.
    Folder target: gdrive-backup/{hostname}/MyFineTunning-{timestamp}/

    Konten yang di-upload:
      - data-files/reports/  → CSV laporan evaluasi
      - data-files/visuals/  → PNG visualisasi
      - MyFineTunning-*/runs/ → Folder run berisi best.pt

    Returns True jika semua upload berhasil.
    """
    if not _rclone_available():
        print("[RClone] ⚠️  rclone tidak ditemukan di PATH. Install dulu.")
        return False

    remote = _get_remote_folder()
    print(f"\n[RClone] 📤 Upload ke: {remote}")
    print(f"[RClone]    Hostname: {HOSTNAME}")
    
    send_telegram_msg(f"☁️ <b>RClone Upload Started</b>\nHost: <code>{HOSTNAME}</code>\nDest: <code>{RCLONE_DEST}</code>")

    success = True

    # 1. Upload data-files (reports + visuals)
    data_files_dir = _PROJECT_ROOT / "data-files"
    if data_files_dir.exists():
        ok = _run_rclone(
            ["copy", str(data_files_dir), f"{remote}/data-files", "--progress"],
            label=f"Upload data-files/ → {remote}/data-files"
        )
        success = success and ok

    # 2. Upload runs/ dari setiap workspace folder (best.pt, last.pt)
    for ws_dir in sorted(_PROJECT_ROOT.glob("MyFineTunning-*")):
        if ws_dir.is_dir():
            runs_dir = ws_dir / "runs"
            if runs_dir.exists():
                ok = _run_rclone(
                    ["copy", str(runs_dir), f"{remote}/runs",
                     "--include", "*.pt",
                     "--include", "*.csv",
                     "--include", "*.png",
                     "--progress"],
                    label=f"Upload runs/ dari {ws_dir.name}"
                )
                success = success and ok

    if success:
        print(f"\n[RClone] 🎉 Upload selesai → {remote}")
        send_telegram_msg(f"✅ <b>RClone Upload Finished</b>\nHost: <code>{HOSTNAME}</code>\nFolder: <code>{remote.split(':')[-1]}</code>")
    else:
        print(f"\n[RClone] ⚠️  Upload selesai dengan beberapa error.")
        send_telegram_msg(f"⚠️ <b>RClone Upload Finished with Errors</b>\nHost: <code>{HOSTNAME}</code>")

    return success


def download_weights(target_dir: str | None = None) -> bool:
    """
    Download weights (best.pt) dari GDrive ke direktori lokal.
    Berguna saat memulai instance RunPOD baru.

    target_dir: folder tujuan (default: WORKSPACE_DIR/runs/)
    """
    if not _rclone_available():
        print("[RClone] ⚠️  rclone tidak ditemukan.")
        return False

    remote = _get_remote_folder()
    local  = target_dir or str(_PROJECT_ROOT / "downloads" / "weights")
    Path(local).mkdir(parents=True, exist_ok=True)

    print(f"\n[RClone] 📥 Download weights dari: {remote}/runs")
    send_telegram_msg(f"📥 <b>RClone Download Started</b>\nHost: <code>{HOSTNAME}</code>")
    
    ok = _run_rclone(
        ["copy", f"{remote}/runs", local,
         "--include", "*.pt",
         "--progress"],
        label=f"Download *.pt → {local}"
    )
    
    if ok:
        send_telegram_msg(f"✅ <b>RClone Download Finished</b>\nHost: <code>{HOSTNAME}</code>")
    else:
        send_telegram_msg(f"❌ <b>RClone Download Failed</b>\nHost: <code>{HOSTNAME}</code>")
        
    return ok


def list_remote(subpath: str = "") -> None:
    """Tampilkan isi folder GDrive (untuk debugging)."""
    if not _rclone_available():
        print("[RClone] ⚠️  rclone tidak ditemukan.")
        return
    remote = f"{RCLONE_REMOTE}:{RCLONE_DEST}/{HOSTNAME}"
    if subpath:
        remote += f"/{subpath}"
    print(f"\n[RClone] 📋 List: {remote}")
    _run_rclone(["lsd", remote], label=f"List {remote}")


def get_status() -> dict:
    """Return dict berisi info konfigurasi RClone saat ini."""
    return {
        "rclone_bin":    RCLONE_BIN,
        "rclone_remote": RCLONE_REMOTE,
        "rclone_dest":   RCLONE_DEST,
        "hostname":      HOSTNAME,
        "remote_folder": _get_remote_folder(),
        "available":     _rclone_available(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  rclone_utils.py — Status & Test")
    print("=" * 60)
    status = get_status()
    for k, v in status.items():
        print(f"  {k:20s}: {v}")
    print()
    list_remote()
