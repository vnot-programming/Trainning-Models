# -*- coding: utf-8 -*-
"""
rclone_utils.py
===============
Helper Python untuk sinkronisasi hasil training ke Google Drive via rclone.

Alur Upload:
  1. Kompres seluruh folder MyFineTunning-xxxx → rclone_local/{name}.tar.zst
  2. Upload file kompresi ke GDrive: fine_models_archive/
  3. Hapus file kompresi lokal
  4. Upload satu per satu: reports/, visuals/, image_samples/ → GDrive
  5. Upload best.pt setiap model → GDrive: fine_models/

Struktur GDrive:
  gdrive-backup/
  └── {hostname}/
      └── MyFineTunning-{timestamp}/
          ├── reports/
          ├── visuals/
          ├── image_samples/
          ├── fine_models/         ← {model}-best.pt
          └── fine_models_archive/ ← file .tar.zst (arsip penuh)

Cara pakai:
    from rclone_utils import upload_results, download_weights
"""

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

# Root project
_PROJECT_ROOT = Path(__file__).resolve().parent

# ── Load .env ──────────────────────────────────────────────────────────────────
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
RCLONE_LOCAL  = _PROJECT_ROOT / "rclone_local"

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
# HELPER — Workspace ID
# ==============================================================================
def _get_workspace_timestamp() -> str:
    ws_id_file = _PROJECT_ROOT / ".workspace_id"
    if ws_id_file.exists():
        return ws_id_file.read_text().strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_remote_folder() -> str:
    ts = _get_workspace_timestamp()
    return f"{RCLONE_REMOTE}:{RCLONE_DEST}/{HOSTNAME}/MyFineTunning-{ts}"


def _workspace_dirs() -> list[Path]:
    """Temukan semua folder MyFineTunning-* di data-files/."""
    data_files = _PROJECT_ROOT / "data-files"
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


def _run_rclone(args: list[str], label: str = "",
                progress_msg_fn=None) -> bool:
    """
    Jalankan rclone command. Jika progress_msg_fn disediakan, kirim Telegram
    setiap _PROGRESS_INTERVAL detik selama proses berjalan.
    """
    cmd_bin = _rclone_cmd()
    if not cmd_bin:
        print("[RClone] ❌ rclone tidak ditemukan.")
        return False

    cmd = [cmd_bin] + args
    print(f"[RClone] ▶ {label or ' '.join(cmd[:4])}")
    print(f"         {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

        # Thread untuk heartbeat Telegram
        stop_event = threading.Event()
        def _heartbeat():
            start = time.time()
            while not stop_event.is_set():
                time.sleep(_PROGRESS_INTERVAL)
                if stop_event.is_set():
                    break
                elapsed = int(time.time() - start)
                m, s = divmod(elapsed, 60)
                msg = progress_msg_fn(m, s) if progress_msg_fn else \
                      f"⏳ <b>RClone Berjalan...</b>\nTask: {label}\nElapsed: {m}m {s}s"
                _tg(msg)

        if progress_msg_fn:
            t = threading.Thread(target=_heartbeat, daemon=True)
            t.start()

        output_lines = []
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)

        proc.wait()
        stop_event.set()

        if proc.returncode == 0:
            print(f"[RClone] ✅ Selesai: {label}")
            return True
        else:
            print(f"[RClone] ❌ Gagal (exit {proc.returncode}): {label}")
            return False

    except Exception as e:
        print(f"[RClone] ❌ Exception: {e}")
        return False


# ==============================================================================
# HELPER — Kompresi
# ==============================================================================
def _best_compressor() -> tuple[str, str]:
    """
    Pilih kompressor terbaik yang tersedia.
    Prioritas: zstd > pigz > gzip
    Mengembalikan (suffix, tar_flag_or_pipe)
    """
    if shutil.which("zstd"):
        return ".tar.zst", "zstd"
    if shutil.which("pigz"):
        return ".tar.gz", "pigz"
    return ".tar.gz", "gzip"


def compress_workspace(ws_dir: Path, out_dir: Path) -> Path | None:
    """
    Kompres ws_dir ke out_dir/{ws_dir.name}.tar.{ext}.
    Menggunakan kompressor terbaik yang tersedia.
    Mengembalikan path file hasil kompresi, atau None jika gagal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix, compressor = _best_compressor()
    out_file = out_dir / f"{ws_dir.name}{suffix}"

    print(f"[Compress] Mengompres {ws_dir.name} dengan {compressor}...")
    print(f"           Sumber: {ws_dir}")
    print(f"           Target: {out_file}")

    _tg(f"🗜 <b>Kompresi Dimulai</b>\nFolder: <code>{ws_dir.name}</code>\nMetode: <code>{compressor}</code>")

    try:
        if compressor == "zstd":
            # tar + zstd (level 19 = max, multi-thread otomatis)
            cmd = [
                "tar", "--use-compress-program=zstd -T0 -19",
                "-cf", str(out_file),
                "-C", str(ws_dir.parent),
                ws_dir.name
            ]
            # Gunakan shell=True agar --use-compress-program bisa pakai argumen
            result = subprocess.run(
                f"tar --use-compress-program='zstd -T0 -19' "
                f"-cf '{out_file}' -C '{ws_dir.parent}' '{ws_dir.name}'",
                shell=True, capture_output=True, text=True
            )
        else:
            # tar + pigz/gzip
            flag = "-I pigz" if compressor == "pigz" else "-z"
            result = subprocess.run(
                f"tar {flag} -cf '{out_file}' -C '{ws_dir.parent}' '{ws_dir.name}'",
                shell=True, capture_output=True, text=True
            )

        if result.returncode != 0:
            print(f"[Compress] ❌ Gagal:\n{result.stderr}")
            _tg(f"❌ <b>Kompresi Gagal</b>\n<code>{result.stderr[:200]}</code>")
            return None

        size_mb = out_file.stat().st_size / 1e6
        print(f"[Compress] ✅ Selesai: {out_file.name} ({size_mb:.1f} MB)")
        _tg(f"✅ <b>Kompresi Selesai</b>\nFile: <code>{out_file.name}</code>\nUkuran: <b>{size_mb:.1f} MB</b>")
        return out_file

    except Exception as e:
        print(f"[Compress] ❌ Exception: {e}")
        _tg(f"❌ <b>Kompresi Error</b>\n<code>{e}</code>")
        return None


# ==============================================================================
# PUBLIC API — UPLOAD
# ==============================================================================
def upload_results(verbose: bool = True) -> bool:
    """
    Upload hasil training ke Google Drive.

    Urutan:
    1. Kompres data-files/MyFineTunning-* → rclone_local/
    2. Upload arsip ke GDrive: fine_models_archive/
    3. Hapus arsip lokal
    4. Upload reports/, visuals/, image_samples/ satu per satu
    5. Upload best.pt per model ke fine_models/{model}-best.pt
    """
    cmd_bin = _rclone_cmd()
    if not cmd_bin:
        print("[RClone] ⚠️  rclone tidak ditemukan di PATH. Install dulu.")
        _tg("❌ <b>Upload Gagal</b>\nrclone tidak ditemukan di sistem.")
        return False

    remote = _get_remote_folder()
    ws_dirs = _workspace_dirs()

    if not ws_dirs:
        print("[RClone] ⚠️  Tidak ada folder MyFineTunning-* di data-files/")
        _tg("⚠️ <b>Upload</b>\nTidak ada workspace yang ditemukan.")
        return False

    print(f"\n[RClone] 📤 Upload ke: {remote}")
    _tg(f"☁️ <b>RClone Upload Dimulai</b>\nHost: <code>{HOSTNAME}</code>\nDest: <code>{remote.split(':',1)[-1]}</code>\nWorkspace: {len(ws_dirs)} folder")

    overall_success = True

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 1: Kompresi & Upload Arsip
    # ═══════════════════════════════════════════════════════════════════════
    for ws_dir in ws_dirs:
        ws_name = ws_dir.name
        print(f"\n{'='*60}")
        print(f"  [1/3] Kompresi: {ws_name}")
        print(f"{'='*60}")

        archive_path = compress_workspace(ws_dir, RCLONE_LOCAL)
        if not archive_path:
            overall_success = False
            continue

        # Upload arsip
        print(f"\n  Mengupload arsip ke GDrive...")
        _tg(f"📤 <b>Upload Arsip</b>\nFile: <code>{archive_path.name}</code>")

        ok = _run_rclone(
            ["copy", str(archive_path),
             f"{remote}/fine_models_archive",
             "--progress", "--stats-one-line"],
            label=f"Upload arsip {archive_path.name}",
            progress_msg_fn=lambda m, s: (
                f"⏳ <b>Upload Arsip Berjalan...</b>\n"
                f"File: <code>{archive_path.name}</code>\n"
                f"Elapsed: <b>{m}m {s}s</b>"
            )
        )

        # Hapus arsip lokal setelah upload
        if ok:
            print(f"  🗑️  Menghapus arsip lokal: {archive_path.name}")
            archive_path.unlink(missing_ok=True)
            _tg(f"🗑 <b>Arsip Lokal Dihapus</b>\n<code>{archive_path.name}</code>")
        else:
            print(f"  ⚠️  Upload arsip gagal, file lokal dipertahankan.")
            overall_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 2: Upload Folder (reports, visuals, image_samples)
    # ═══════════════════════════════════════════════════════════════════════
    for ws_dir in ws_dirs:
        ws_name = ws_dir.name
        for folder_name in ["reports", "visuals", "image_samples"]:
            folder = ws_dir / folder_name
            if not folder.exists():
                print(f"[RClone] ⏭  Skip {folder_name}/ (tidak ditemukan)")
                continue

            print(f"\n{'='*60}")
            print(f"  [2/3] Upload {folder_name}/ dari {ws_name}")
            print(f"{'='*60}")
            _tg(f"📁 <b>Upload Folder</b>\n<code>{ws_name}/{folder_name}/</code>")

            ok = _run_rclone(
                ["copy", str(folder),
                 f"{remote}/{folder_name}",
                 "--progress", "--stats-one-line"],
                label=f"Upload {folder_name}/",
                progress_msg_fn=lambda m, s, fn=folder_name: (
                    f"⏳ <b>Upload {fn}/ Berjalan...</b>\n"
                    f"Elapsed: <b>{m}m {s}s</b>"
                )
            )
            if ok:
                _tg(f"✅ <b>Upload {folder_name}/ Selesai</b>")
            else:
                _tg(f"❌ <b>Upload {folder_name}/ Gagal</b>")
                overall_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # TAHAP 3: Upload best.pt per model → fine_models/{model}-best.pt
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  [3/3] Upload best.pt per model → fine_models/")
    print(f"{'='*60}")

    model_count = 0
    for ws_dir in ws_dirs:
        runs_dir = ws_dir / "runs"
        if not runs_dir.exists():
            continue

        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            best_pt = model_dir / "weights" / "best.pt"
            if not best_pt.exists():
                continue

            dest_name = f"{model_dir.name}-best.pt"
            print(f"  Uploading {model_dir.name}/best.pt → fine_models/{dest_name}")

            # rclone copyto untuk rename saat upload
            ok = _run_rclone(
                ["copyto", str(best_pt),
                 f"{remote}/fine_models/{dest_name}",
                 "--progress"],
                label=f"Upload {dest_name}"
            )
            if ok:
                model_count += 1
                _tg(f"✅ <b>Model Terupload</b>\n<code>fine_models/{dest_name}</code>")
            else:
                _tg(f"❌ <b>Upload Model Gagal</b>\n<code>{dest_name}</code>")
                overall_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # RINGKASAN
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    if overall_success:
        print(f"✅ Semua upload selesai → {remote}")
        _tg(
            f"🎉 <b>RClone Upload SELESAI</b>\n"
            f"Host: <code>{HOSTNAME}</code>\n"
            f"Folder: <code>{remote.split(':',1)[-1]}</code>\n"
            f"Model terupload: <b>{model_count}</b> file\n"
            f"Struktur:\n"
            f"  📊 reports/\n"
            f"  🖼 visuals/\n"
            f"  📷 image_samples/\n"
            f"  🤖 fine_models/ ({model_count} model)\n"
            f"  🗜 fine_models_archive/"
        )
    else:
        print(f"⚠️  Upload selesai dengan beberapa error.")
        _tg(f"⚠️ <b>RClone Upload Selesai dengan Error</b>\nHost: <code>{HOSTNAME}</code>")

    return overall_success


# ==============================================================================
# PUBLIC API — DOWNLOAD
# ==============================================================================
def download_weights(target_dir: str | None = None,
                     auto_extract: bool = True) -> bool:
    """
    Download & ekstrak arsip dari GDrive ke lokal.

    auto_extract=True  → Otomatis ekstrak .tar.zst/.tar.gz setelah download
    auto_extract=False → Hanya download saja tanpa ekstrak
    """
    cmd_bin = _rclone_cmd()
    if not cmd_bin:
        print("[RClone] ⚠️  rclone tidak ditemukan.")
        _tg("❌ <b>Download Gagal</b>\nrclone tidak ditemukan.")
        return False

    remote      = _get_remote_folder()
    dl_dir      = Path(target_dir) if target_dir else RCLONE_LOCAL / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RClone] 📥 Download dari: {remote}")
    _tg(f"📥 <b>RClone Download Dimulai</b>\nHost: <code>{HOSTNAME}</code>\nSumber: <code>{remote.split(':',1)[-1]}</code>")

    # 1. Download arsip
    archive_remote = f"{remote}/fine_models_archive"
    print(f"\n[RClone] Mengunduh arsip dari fine_models_archive/...")
    ok = _run_rclone(
        [cmd_bin if cmd_bin != "rclone" else "rclone",
         "copy", archive_remote, str(dl_dir),
         "--progress", "--stats-one-line"],
        label="Download arsip",
        progress_msg_fn=lambda m, s: (
            f"⏳ <b>Download Arsip Berjalan...</b>\n"
            f"Elapsed: <b>{m}m {s}s</b>"
        )
    )

    if not ok:
        _tg("❌ <b>Download Arsip Gagal</b>")
        return False

    _tg(f"✅ <b>Download Selesai</b>\nDisimpan di: <code>{dl_dir}</code>")

    # 2. Ekstrak otomatis
    if auto_extract:
        archives = list(dl_dir.glob("*.tar.zst")) + list(dl_dir.glob("*.tar.gz"))
        if not archives:
            print("[Extract] Tidak ada arsip yang ditemukan untuk diekstrak.")
            return True

        extract_dest = _PROJECT_ROOT / "data-files"
        extract_dest.mkdir(parents=True, exist_ok=True)

        for archive in archives:
            print(f"\n[Extract] Mengekstrak: {archive.name} → {extract_dest}")
            _tg(f"📦 <b>Ekstraksi Dimulai</b>\nFile: <code>{archive.name}</code>")

            if archive.suffix == ".zst" or archive.name.endswith(".tar.zst"):
                cmd = f"tar --use-compress-program='zstd -d -T0' -xf '{archive}' -C '{extract_dest}'"
            else:
                cmd = f"tar -xzf '{archive}' -C '{extract_dest}'"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[Extract] ✅ Selesai: {archive.name}")
                _tg(f"✅ <b>Ekstraksi Selesai</b>\n<code>{archive.name}</code>\nDi: <code>data-files/</code>")
                archive.unlink()  # Hapus arsip setelah ekstrak
            else:
                print(f"[Extract] ❌ Gagal:\n{result.stderr}")
                _tg(f"❌ <b>Ekstraksi Gagal</b>\n<code>{result.stderr[:200]}</code>")
                return False

    return True


# ==============================================================================
# PUBLIC API — STATUS
# ==============================================================================
def list_remote(subpath: str = "") -> None:
    """Tampilkan isi folder GDrive."""
    cmd_bin = _rclone_cmd()
    if not cmd_bin:
        print("[RClone] ⚠️  rclone tidak ditemukan.")
        return
    remote = f"{RCLONE_REMOTE}:{RCLONE_DEST}/{HOSTNAME}"
    if subpath:
        remote += f"/{subpath}"
    print(f"\n[RClone] 📋 List: {remote}")
    subprocess.run([cmd_bin, "lsd", remote])


def get_status() -> dict:
    """Return dict info konfigurasi RClone."""
    return {
        "rclone_bin":    RCLONE_BIN,
        "rclone_remote": RCLONE_REMOTE,
        "rclone_dest":   RCLONE_DEST,
        "hostname":      HOSTNAME,
        "remote_folder": _get_remote_folder(),
        "available":     _rclone_cmd() is not None,
        "rclone_local":  str(RCLONE_LOCAL),
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
