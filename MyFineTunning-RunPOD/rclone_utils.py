# -*- coding: utf-8 -*-
"""
rclone_utils.py
===============
Helper Python untuk sinkronisasi hasil training ke Google Drive via rclone.

Alur Upload:
  1. Cek apakah arsip sudah ada di GDrive. Jika ada, SKIP kompresi.
  2. Jika belum ada: Kompres folder MyFineTunning-xxxx → rclone_local/{name}.tar.zst
  3. Upload file kompresi ke GDrive: fine_models_archive/
  4. DELAY jika file > 1GB (untuk menghindari rate limit GDrive)
  5. Hapus file kompresi lokal
  6. Upload satu per satu (skip jika identik): reports/, visuals/, image_samples/
  7. Upload best.pt setiap model (skip jika identik) → GDrive: fine_models/
"""

import os
import shutil
import socket
import subprocess
import threading
import time
import sys
from pathlib import Path
from datetime import datetime

# Root project
_PROJECTROOT = Path(__file__).resolve().parent

# ── Load .env ──────────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = _PROJECTROOT / ".env"
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
RCLONE_LOCAL  = _PROJECTROOT / "rclone_local"

# Interval laporan Telegram untuk operasi panjang (5 menit)
_PROGRESS_INTERVAL = 300


# ==============================================================================
# HELPER — Telegram
# ==============================================================================
def _tg(msg: str) -> None:
    """Kirim pesan ke Telegram (non-blocking, tidak error jika gagal)."""
    try:
        from telegram_utils import send_telegram_msg
        send_telegram_msg(msg)
    except Exception:
        pass


# ==============================================================================
# HELPER — Delay & Countdown
# ==============================================================================
def _apply_smart_delay(seconds: int, reason: str = "Rate Limit Protection") -> None:
    """
    Menjalankan delay dengan hitungan mundur di log dan notifikasi Telegram.
    """
    msg = f"⏳ <b>Smart Delay Activated</b>\nReason: <code>{reason}</code>\nDuration: <b>{seconds}s</b>"
    print(f"\n[RClone] 🕒 {reason}. Waiting for {seconds} seconds...")
    _tg(msg)
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = int(seconds - elapsed)
        if remaining <= 0:
            break
        
        sys.stdout.write(f"\r         Continuing in {remaining}s...   ")
        sys.stdout.flush()
        
        if remaining > 0 and remaining % 60 == 0 and remaining != seconds:
            _tg(f"⏳ <b>Still Waiting...</b>\nRemaining: <b>{remaining}s</b>")
            
        time.sleep(1)
    
    print(f"\r[RClone] ✅ Wait finished. Resuming process...          \n")


# ==============================================================================
# HELPER — Workspace ID
# ==============================================================================
def _get_workspace_timestamp() -> str:
    ws_id_file = _PROJECTROOT / ".workspace_id"
    if ws_id_file.exists():
        return ws_id_file.read_text().strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_remote_folder() -> str:
    ts = _get_workspace_timestamp()
    return f"{RCLONE_REMOTE}:{RCLONE_DEST}/{HOSTNAME}/MyFineTunning-{ts}"


def _workspace_dirs() -> list[Path]:
    """Temukan semua folder MyFineTunning-* di data-files/."""
    data_files = _PROJECTROOT / "data-files"
    if not data_files.exists():
        return []
    return sorted(data_files.glob("MyFineTunning-*"))


# ==============================================================================
# HELPER — rclone
# ==============================================================================
def _rclone_cmd() -> str:
    if Path(RCLONE_BIN).exists():
        return RCLONE_BIN
    if shutil.which("rclone"):
        return "rclone"
    return None


def _remote_file_exists(remote_path: str) -> bool:
    """Cek apakah file ada di remote (menggunakan lsf)."""
    cmd_bin = _rclone_cmd()
    if not cmd_bin: return False
    
    # Ambil folder dan nama file
    parts = remote_path.rsplit("/", 1)
    if len(parts) < 2: return False
    folder, filename = parts[0], parts[1]
    
    try:
        # lsf mengembalikan list file di folder tersebut
        res = subprocess.run([cmd_bin, "lsf", folder], capture_output=True, text=True)
        if res.returncode == 0:
            files = res.stdout.splitlines()
            return filename in files or (filename + "/") in files
    except:
        pass
    return False


def _run_rclone(args: list[str], label: str = "",
                progress_msg_fn=None) -> bool:
    cmd_bin = _rclone_cmd()
    if not cmd_bin: return False

    cmd = [cmd_bin] + args
    print(f"[RClone] ▶ {label or ' '.join(cmd[:4])}")
    print(f"         {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

        stop_event = threading.Event()
        def _heartbeat():
            start = time.time()
            while not stop_event.is_set():
                time.sleep(_PROGRESS_INTERVAL)
                if stop_event.is_set(): break
                elapsed = int(time.time() - start)
                m, s = divmod(elapsed, 60)
                msg = progress_msg_fn(m, s) if progress_msg_fn else \
                      f"⏳ <b>RClone Berjalan...</b>\nTask: {label}\nElapsed: {m}m {s}s"
                _tg(msg)

        if progress_msg_fn:
            t = threading.Thread(target=_heartbeat, daemon=True)
            t.start()

        for line in proc.stdout:
            print(line, end="")

        proc.wait()
        stop_event.set()

        if proc.returncode == 0:
            print(f"[RClone] ✅ Selesai: {label}")
            return True
        else:
            print(f"[RClone] ❌ Gagal (exit {proc.returncode}): {label}")
            if proc.returncode == -9: print("         (Proses dihentikan paksa/SIGKILL)")
            return False
    except Exception as e:
        print(f"[RClone] ❌ Exception: {e}")
        return False


# ==============================================================================
# HELPER — Kompresi
# ==============================================================================
def _best_compressor() -> tuple[str, str]:
    if shutil.which("zstd"): return ".tar.zst", "zstd"
    if shutil.which("pigz"): return ".tar.gz", "pigz"
    return ".tar.gz", "gzip"


def compress_workspace(ws_dir: Path, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix, compressor = _best_compressor()
    out_file = out_dir / f"{ws_dir.name}{suffix}"

    print(f"[Compress] Mengompres {ws_dir.name} dengan {compressor}...")
    _tg(f"🗜 <b>Kompresi Dimulai</b>\nFolder: <code>{ws_dir.name}</code>")

    try:
        if compressor == "zstd":
            cmd = f"tar --use-compress-program='zstd -T0 -19' -cf '{out_file}' -C '{ws_dir.parent}' '{ws_dir.name}'"
        else:
            flag = "-I pigz" if compressor == "pigz" else "-z"
            cmd = f"tar {flag} -cf '{out_file}' -C '{ws_dir.parent}' '{ws_dir.name}'"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            _tg(f"❌ <b>Kompresi Gagal</b>")
            return None

        size_mb = out_file.stat().st_size / 1e6
        _tg(f"✅ <b>Kompresi Selesai</b>\nSize: <b>{size_mb:.1f} MB</b>")
        return out_file
    except Exception as e:
        _tg(f"❌ <b>Kompresi Error</b>\n<code>{e}</code>")
        return None


# ==============================================================================
# PUBLIC API — UPLOAD
# ==============================================================================
def upload_results(verbose: bool = True) -> bool:
    cmd_bin = _rclone_cmd()
    if not cmd_bin: return False

    remote = _get_remote_folder()
    ws_dirs = _workspace_dirs()
    if not ws_dirs: return False

    print(f"\n[RClone] 📤 Upload ke: {remote}")
    _tg(f"☁️ <b>RClone Upload Dimulai</b>\nHost: <code>{HOSTNAME}</code>")

    overall_success = True
    suffix, _ = _best_compressor()

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 1: Kompresi & Upload Arsip (Skip if exists)
    # ═══════════════════════════════════════════════════════════════════════
    for ws_dir in ws_dirs:
        archive_name = f"{ws_dir.name}{suffix}"
        remote_archive_path = f"{remote}/fine_models_archive/{archive_name}"
        
        print(f"\n[Check] Memeriksa arsip di GDrive: {archive_name}")
        if _remote_file_exists(remote_archive_path):
            print(f"[RClone] ⏭  SKIP: Arsip '{archive_name}' sudah ada di GDrive.")
            _tg(f"⏭ <b>Skip Arsip</b>\n<code>{archive_name}</code> sudah ada di cloud.")
            continue

        archive_path = compress_workspace(ws_dir, RCLONE_LOCAL)
        if not archive_path:
            overall_success = False
            continue

        file_size_gb = archive_path.stat().st_size / 1e9
        ok = _run_rclone(
            ["copy", str(archive_path), f"{remote}/fine_models_archive", "--progress", "--stats-one-line"],
            label=f"Upload arsip {archive_path.name}",
            progress_msg_fn=lambda m, s: (f"⏳ <b>Upload Arsip Berjalan...</b>\nElapsed: <b>{m}m {s}s</b>")
        )

        if ok:
            archive_path.unlink(missing_ok=True)
            if file_size_gb > 1.0:
                _apply_smart_delay(180, reason=f"Post-Upload Cooldown ({file_size_gb:.1f}GB file)")
        else:
            overall_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 2: Upload Folder (reports, visuals, image_samples)
    # ═══════════════════════════════════════════════════════════════════════
    for ws_dir in ws_dirs:
        for folder_name in ["reports", "visuals", "image_samples"]:
            folder = ws_dir / folder_name
            if not folder.exists(): continue
            ok = _run_rclone(
                ["copy", str(folder), f"{remote}/{folder_name}", "--progress", "--stats-one-line"],
                label=f"Upload {folder_name}/"
            )
            overall_success = overall_success and ok

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 3: Upload best.pt per model
    # ═══════════════════════════════════════════════════════════════════════
    model_count = 0
    for ws_dir in ws_dirs:
        runs_dir = ws_dir / "runs"
        if not runs_dir.exists(): continue
        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir(): continue
            best_pt = model_dir / "weights" / "best.pt"
            if not best_pt.exists(): continue

            dest_name = f"{model_dir.name}-best.pt"
            ok = _run_rclone(
                ["copyto", str(best_pt), f"{remote}/fine_models/{dest_name}", "--progress"],
                label=f"Upload {dest_name}"
            )
            if ok: model_count += 1
            overall_success = overall_success and ok

    _tg(f"🏁 <b>RClone Upload Selesai</b>\nOverall: {'✅' if overall_success else '⚠️'}")
    return overall_success


def download_weights(target_dir: str | None = None, auto_extract: bool = True) -> bool:
    cmd_bin = _rclone_cmd()
    if not cmd_bin: return False
    remote = _get_remote_folder()
    dl_dir = Path(target_dir) if target_dir else RCLONE_LOCAL / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    ok = _run_rclone([cmd_bin, "copy", f"{remote}/fine_models_archive", str(dl_dir), "--progress"], label="Download arsip")
    if ok and auto_extract:
        archives = list(dl_dir.glob("*.tar.zst")) + list(dl_dir.glob("*.tar.gz"))
        extract_dest = _PROJECTROOT / "data-files"
        for archive in archives:
            cmd = f"tar --use-compress-program='zstd -d -T0' -xf '{archive}' -C '{extract_dest}'" if ".zst" in archive.name else f"tar -xzf '{archive}' -C '{extract_dest}'"
            if subprocess.run(cmd, shell=True).returncode == 0: archive.unlink()
    return ok

def get_status() -> dict:
    return {"remote_folder": _get_remote_folder(), "hostname": HOSTNAME}

if __name__ == "__main__":
    print(get_status())
