# -*- coding: utf-8 -*-
from __future__ import annotations
"""
upload_utils.py
===============
Utilitas Python untuk melakukan kompresi dan pemindahan file-file hasil 
training secara lokal tanpa proses upload remote (ditunda sementara).

Mendukung deteksi dinamis terhadap workspace aktif dan kompresi tingkat tinggi.

Struktur Path, Folder, dan Berkas Terkait:
------------------------------------------
1. Direktori Sumber (Workspace Aktif):
   - Path Root Workspace : `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/data-files/MyFineTunning-<WS_ID>/`
   - Folder File Weights  : `runs/` (Berisi weights per model dan arsip model tar.gz)
   - Folder Logs/Jurnal   : `logs/` (Catatan log training & evaluasi)
   - Folder Gambar Sampel : `image_samples/` (10 sampel gambar visualisasi)
   - Folder Laporan Hasil : `reports/` (Laporan matriks kuantitatif CSV & visualisasi komparatif)

2. Direktori Tujuan Pengarsipan Lokal:
   - Path Root Pengarsipan: `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/datas/`
   - Target Folder Model Archived: `datas/models_archived/` ⬅️ Berisi file model tar.gz hasil pemindahan dari `runs/`
   - Target Folder Model Weights : `datas/models/`          ⬅️ Berisi bobot terbaik disalin sebagai `<model>-best.pt`
   - Target Arsip Tar Logs       : `datas/logs.tar.gz`       ⬅️ Hasil kompresi folder `logs/`
   - Target Arsip Tar Samples    : `datas/image_samples.tar.gz` ⬅️ Hasil kompresi folder `image_samples/`
   - Target Arsip Tar Reports    : `datas/reports.tar.gz`    ⬅️ Hasil kompresi folder `reports/`
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Root project
_PROJECTROOT = Path(__file__).resolve().parents[1]


def get_active_workspace() -> Path:
    """
    Mendeteksi workspace aktif secara dinamis.
    1. Membaca file `.workspace_id` di root project.
    2. Jika tidak ada, mencari folder `MyFineTunning-*` yang paling baru di `data-files/`.
    3. Jika tidak ditemukan, memunculkan ValueError dengan deskripsi detail.
    """
    ws_id_file = _PROJECTROOT / ".workspace_id"
    data_files_dir = _PROJECTROOT / "data-files"
    
    workspace_name = ""
    
    if ws_id_file.exists():
        ws_id = ws_id_file.read_text().strip()
        if ws_id:
            workspace_name = f"MyFineTunning-{ws_id}"
            ws_path = data_files_dir / workspace_name
            if ws_path.exists() and ws_path.is_dir():
                return ws_path
            print(f"[Warning] Workspace dari .workspace_id ({workspace_name}) tidak ditemukan di {data_files_dir}.")

    # Fallback: Cari folder MyFineTunning-* yang paling baru
    if data_files_dir.exists() and data_files_dir.is_dir():
        folders = sorted(
            [d for d in data_files_dir.glob("MyFineTunning-*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if folders:
            latest_ws = folders[0]
            print(f"[Info] Menggunakan workspace terbaru secara otomatis: {latest_ws.name}")
            return latest_ws

    # Jika semua langkah gagal, keluarkan pesan error terperinci
    err_msg = (
        "Gagal mendeteksi Workspace aktif!\n"
        "Penyebab Kemungkinan:\n"
        f"  1. File '.workspace_id' di {_PROJECTROOT} tidak ada atau isinya kosong.\n"
        f"  2. Direktori 'data-files/' di {_PROJECTROOT} tidak ada atau kosong.\n"
        "  3. Tidak ada subdirektori dengan pola nama 'MyFineTunning-*' di dalam 'data-files/'.\n"
        "Solusi: Pastikan training pipeline telah berjalan setidaknya sekali dan membuat folder hasil di 'data-files/'."
    )
    raise ValueError(err_msg)


def move_archived_models(ws_dir: Path, dest_dir: Path) -> list[str]:
    """
    Langkah 1: Memindahkan file-file .tar.gz model dari folder 'runs/' ke 'datas/models_archived/'.
    Mengembalikan daftar file yang berhasil dipindahkan.
    """
    runs_dir = ws_dir / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(f"[Warning] Direktori runs tidak ditemukan di {runs_dir}. Langkah 1 dilewati.")
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    moved_files = []

    # Memindai seluruh file .tar.gz di folder runs
    tar_files = list(runs_dir.glob("*.tar.gz"))
    if not tar_files:
        print("[Info] Tidak ada file .tar.gz yang ditemukan di folder runs/.")
        return []

    print(f"\n[Langkah 1] Memindahkan {len(tar_files)} file .tar.gz ke {dest_dir}...")
    for tar_path in tar_files:
        try:
            dest_path = dest_dir / tar_path.name
            size_mb = tar_path.stat().st_size / 1e6
            shutil.move(str(tar_path), str(dest_path))
            print(f"  ✅ [Move] {tar_path.name} ({size_mb:.2f} MB) -> {dest_path}")
            moved_files.append(tar_path.name)
        except Exception as e:
            print(f"  ❌ [Move Gagal] Gagal memindahkan {tar_path.name}. Penyebab: {e}")

    return moved_files


def copy_best_weights(ws_dir: Path, dest_dir: Path) -> list[str]:
    """
    Langkah 2: Menyalin weights 'best.pt' per model ke 'datas/models/' dengan nama '{model}-best.pt'.
    Mengembalikan daftar file yang berhasil disalin.
    """
    runs_dir = ws_dir / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []

    # Iterasi setiap subdirektori model di dalam runs/
    model_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print("[Info] Tidak ada subdirektori model yang ditemukan di folder runs/.")
        return []

    print(f"\n[Langkah 2] Menyalin file 'best.pt' ke {dest_dir}...")
    for model_dir in model_dirs:
        model_name = model_dir.name
        best_pt = model_dir / "weights" / "best.pt"
        
        if not best_pt.exists():
            # Abaikan dengan warning log jika model tidak/belum memiliki best.pt
            print(f"  ⚠️  [Weights Lewat] Model '{model_name}' tidak memiliki file 'weights/best.pt'.")
            continue

        try:
            dest_name = f"{model_name}-best.pt"
            dest_path = dest_dir / dest_name
            shutil.copy2(str(best_pt), str(dest_path))
            size_mb = best_pt.stat().st_size / 1e6
            print(f"  ✅ [Copy] {model_name}/weights/best.pt ({size_mb:.2f} MB) -> {dest_name}")
            copied_files.append(dest_name)
        except Exception as e:
            print(f"  ❌ [Copy Gagal] Gagal menyalin best.pt untuk model '{model_name}'. Penyebab: {e}")

    return copied_files


def _best_compressor() -> tuple[str, list[str]]:
    """
    Menemukan kompresor sistem terbaik yang tersedia.
    Mendukung 'pigz' untuk multi-threaded gzip, fallback ke 'gzip' standar.
    """
    if shutil.which("pigz"):
        return "pigz", ["tar", "-I", "pigz", "-cf"]
    if shutil.which("tar"):
        return "gzip", ["tar", "-czf"]
    return "python", []


def compress_directory(source_dir: Path, dest_archive: Path) -> bool:
    """
    Mengompresi direktori tertentu menggunakan utilitas tar sistem atau modul python tarfile.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"  ❌ [Kompres Gagal] Direktori sumber {source_dir} tidak ditemukan.")
        return False

    compressor_name, cmd_prefix = _best_compressor()
    start_time = time.perf_counter()

    try:
        if compressor_name != "python":
            # Menggunakan perintah sistem tar untuk kecepatan maksimal
            cmd = cmd_prefix + [str(dest_archive), "-C", str(source_dir.parent), source_dir.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                elapsed = time.perf_counter() - start_time
                size_mb = dest_archive.stat().st_size / 1e6
                print(f"  ✅ [Kompres] {source_dir.name}/ -> {dest_archive.name} ({size_mb:.2f} MB) | Durasi: {elapsed:.2f}s (via {compressor_name})")
                return True
            else:
                print(f"  ⚠️  [Kompres Command Gagal] Gagal menggunakan {compressor_name}. Code: {result.returncode}. Detail: {result.stderr}")
        
        # Fallback ke modul python tarfile bawaan jika perintah tar sistem gagal
        import tarfile
        print(f"  [Info] Menggunakan modul python 'tarfile' bawaan untuk mengompresi {source_dir.name}...")
        with tarfile.open(dest_archive, "w:gz") as tar:
            tar.add(str(source_dir), arcname=source_dir.name)
        
        elapsed = time.perf_counter() - start_time
        size_mb = dest_archive.stat().st_size / 1e6
        print(f"  ✅ [Kompres Fallback] {source_dir.name}/ -> {dest_archive.name} ({size_mb:.2f} MB) | Durasi: {elapsed:.2f}s (via tarfile)")
        return True

    except Exception as e:
        print(f"  ❌ [Kompres Gagal] Gagal mengompresi {source_dir.name}. Penyebab: {e}")
        return False


def run_local_archiver() -> bool:
    """
    Mengkoordinasikan seluruh alur pemindahan dan kompresi hasil training secara lokal.
    """
    t_start = time.perf_counter()
    
    print("=" * 70)
    print("🚀 MEMULAI PROSES PENGARSIPAN DAN KOMPRESI LOKAL")
    print("=" * 70)

    try:
        ws_dir = get_active_workspace()
    except ValueError as e:
        print(f"❌ [ERROR KRITIS] {e}")
        return False

    print(f"📁 Workspace Aktif : {ws_dir.name}")
    print(f"📍 Path Direktori   : {ws_dir}")
    print("-" * 70)

    # Direktori tujuan
    datas_dir = _PROJECTROOT / "datas"
    models_archived_dir = datas_dir / "models_archived"
    models_dir = datas_dir / "models"

    # Buat direktori dasar jika belum ada
    datas_dir.mkdir(parents=True, exist_ok=True)

    # 1. Langkah Pertama: Pindahkan semua file .tar.gz model
    moved_list = move_archived_models(ws_dir, models_archived_dir)

    # 2. Langkah Kedua: Copy semua weights best.pt ke models/{model}-best.pt
    copied_list = copy_best_weights(ws_dir, models_dir)

    # 3, 4, 5. Langkah Ketiga, Keempat, Kelima: Kompresi folder logs, image_samples, reports
    print("\n[Langkah 3-5] Mengompresi folder-folder pendukung...")
    
    compress_jobs = [
        ("logs", datas_dir / "logs.tar.gz"),
        ("image_samples", datas_dir / "image_samples.tar.gz"),
        ("reports", datas_dir / "reports.tar.gz")
    ]

    compressed_count = 0
    for folder_name, dest_archive in compress_jobs:
        src_path = ws_dir / folder_name
        if src_path.exists() and src_path.is_dir():
            if compress_directory(src_path, dest_archive):
                compressed_count += 1
        else:
            print(f"  ⚠️  [Lewat] Folder '{folder_name}' tidak ditemukan di workspace. Kompresi dilewati.")

    elapsed_total = time.perf_counter() - t_start
    
    print("\n" + "=" * 70)
    print("🏁 PROSES PENGARSIPAN LOKAL SELESAI")
    print("=" * 70)
    print(f"⏱️ Total Durasi    : {elapsed_total:.2f} detik")
    print(f"📦 File Dipindahkan : {len(moved_list)} file model (.tar.gz)")
    print(f"🎯 File Bobot Copy : {len(copied_list)} file weights (-best.pt)")
    print(f"🗜️ Folder Kompresi : {compressed_count} folder berhasil dikompresi")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    success = run_local_archiver()
    sys.exit(0 if success else 1)
