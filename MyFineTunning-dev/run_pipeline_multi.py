# -*- coding: utf-8 -*-
"""
run_pipeline_multi.py
=====================
Pipeline orchestrator — menjalankan semua eval_multigpu.py secara SEKUENSIAL
menggunakan virtual environment .venv yang tersedia di direktori ini.

Urutan eksekusi:
  1. yolo/yolo8/eval_multigpu.py
  2. yolo/yolo9/eval_multigpu.py
  3. yolo/yolo11/eval_multigpu.py
  4. mask-r-cnn/eval_multigpu.py
  5. hybrid/eval_multigpu.py

Cara menjalankan:
    # Langsung (tanpa aktivasi venv manual):
    python run_pipeline_multi.py

    # Background via tmux:
    tmux new-session -d -s pipeline \\
      "python /home/my/Trainning-Models/MyFineTunning-dev/run_pipeline_multi.py \
       2>&1 | tee run_pipeline_multi.log"

    # Lanjut hanya model tertentu (skip yang sudah selesai):
    python run_pipeline_multi.py --skip yolo8,yolo9

Argumen:
    --skip   Nama model yang dilewati, dipisah koma.
             Pilihan: yolo8, yolo9, yolo11, maskrcnn, hybrid
    --gpus   GPU yang digunakan, diteruskan ke setiap skrip.
             Contoh: '0,1' atau 'all' (default). 

Log:
    run_pipeline_multi.log  (jika dijalankan dengan tee)
"""

import os
import sys
import time
import argparse
import subprocess
import datetime

# ── Konfigurasi utama ──────────────────────────────────────────────────────────
_THIS_DIR  = os.path.abspath(os.path.dirname(__file__))
_VENV_PYTHON = os.path.join(_THIS_DIR, ".venv", "bin", "python")

# Daftar skrip dalam urutan eksekusi
_PIPELINE = [
    {
        "name":    "yolo8",
        "label":   "YOLOv8m  (Det + Seg)",
        "script":  os.path.join(_THIS_DIR, "yolo", "yolo8",   "eval_multigpu.py"),
        "cwd":     os.path.join(_THIS_DIR, "yolo", "yolo8"),
    },
    {
        "name":    "yolo9",
        "label":   "YOLOv9m  (Det + Seg)",
        "script":  os.path.join(_THIS_DIR, "yolo", "yolo9",   "eval_multigpu.py"),
        "cwd":     os.path.join(_THIS_DIR, "yolo", "yolo9"),
    },
    {
        "name":    "yolo11",
        "label":   "YOLO11m  (Det + Seg)",
        "script":  os.path.join(_THIS_DIR, "yolo", "yolo11",  "eval_multigpu.py"),
        "cwd":     os.path.join(_THIS_DIR, "yolo", "yolo11"),
    },
    {
        "name":    "maskrcnn",
        "label":   "Mask R-CNN",
        "script":  os.path.join(_THIS_DIR, "mask-r-cnn",      "eval_multigpu.py"),
        "cwd":     os.path.join(_THIS_DIR, "mask-r-cnn"),
    },
    {
        "name":    "hybrid",
        "label":   "Hybrid (YOLO11m + SAM2)",
        "script":  os.path.join(_THIS_DIR, "hybrid",          "eval_multigpu.py"),
        "cwd":     os.path.join(_THIS_DIR, "hybrid"),
    },
]

# ── ANSI Colors ────────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_DIM    = "\033[2m"

def _c(text, *codes): return "".join(codes) + text + _RESET
def _hr(char="═", width=70): return char * width


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:   return f"{h}j {m}m {s}s"
    if m:   return f"{m}m {s}s"
    return f"{s}s"


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _print_header():
    print(_c(_hr(), _BOLD, _CYAN))
    print(_c("  MultiGPU Evaluation Pipeline", _BOLD, _CYAN))
    print(_c("  Semua model dievaluasi secara sekuensial", _DIM))
    print(_c(_hr(), _BOLD, _CYAN))
    print(f"  {'Venv Python':<18}: {_VENV_PYTHON}")
    print(f"  {'Working Dir':<18}: {_THIS_DIR}")
    print(f"  {'Mulai':<18}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_c(_hr(), _BOLD, _CYAN))


def _print_plan(pipeline: list, skip_names: set):
    print("\n  RENCANA EKSEKUSI:")
    for i, step in enumerate(pipeline, 1):
        status = _c("SKIP", _YELLOW) if step["name"] in skip_names else _c("RUN ", _GREEN)
        print(f"    [{status}] {i}. {step['label']}")
    print()


def _print_step_start(idx: int, total: int, label: str):
    print()
    print(_c(_hr("─"), _CYAN))
    print(_c(f"  LANGKAH {idx}/{total}: {label}", _BOLD, _CYAN))
    print(_c(f"  Mulai: {_now()}", _DIM))
    print(_c(_hr("─"), _CYAN))


def _print_step_result(label: str, rc: int, elapsed: float):
    dur = _fmt_duration(elapsed)
    if rc == 0:
        mark = _c("✅ SELESAI", _GREEN, _BOLD)
    else:
        mark = _c(f"❌ GAGAL (exit code: {rc})", _RED, _BOLD)
    print(_c(_hr("─"), _CYAN))
    print(f"  {mark} — {label} — {dur}")
    print(_c(_hr("─"), _CYAN))


def _run_step(step: dict, gpus: str) -> tuple[int, float]:
    """
    Jalankan satu eval_multigpu.py menggunakan venv python.
    Returns (returncode, elapsed_seconds).
    """
    cmd = [_VENV_PYTHON, "-u", step["script"], "--gpus", gpus]

    print(f"  {_c('CMD', _DIM)}: {' '.join(cmd)}")
    print(f"  {_c('CWD', _DIM)}: {step['cwd']}")
    print()

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=step["cwd"],
            stdout=sys.stdout,          # stream langsung ke terminal/log
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
        )
        rc = proc.wait()
    except FileNotFoundError as e:
        print(_c(f"\n  ❌ ERROR: {e}", _RED))
        rc = -1
    except KeyboardInterrupt:
        print(_c("\n  ⚠️  Interrupted oleh user (Ctrl+C).", _YELLOW))
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass
        rc = -2
        raise
    elapsed = time.perf_counter() - t0
    return rc, elapsed


def _print_summary(results: list):
    print()
    print(_c(_hr(), _BOLD, _CYAN))
    print(_c("  RINGKASAN AKHIR", _BOLD, _CYAN))
    print(_c(_hr(), _BOLD, _CYAN))

    total_time   = sum(r["elapsed"] for r in results)
    success_cnt  = sum(1 for r in results if r["rc"] == 0)
    failed_cnt   = sum(1 for r in results if r["rc"] != 0 and r["rc"] != "SKIP")
    skipped_cnt  = sum(1 for r in results if r["rc"] == "SKIP")

    col_w = 28
    print(f"  {'Model':<{col_w}} {'Status':<20} {'Durasi':>10}")
    print(f"  {'─'*col_w} {'─'*20} {'─'*10}")

    for r in results:
        if r["rc"] == "SKIP":
            status  = _c("DILEWATI", _YELLOW)
            dur_str = "   —"
        elif r["rc"] == 0:
            status  = _c("BERHASIL", _GREEN)
            dur_str = _fmt_duration(r["elapsed"])
        else:
            status  = _c(f"GAGAL ({r['rc']})", _RED)
            dur_str = _fmt_duration(r["elapsed"])
        print(f"  {r['label']:<{col_w}} {status:<30} {dur_str:>10}")

    print(f"  {'─'*col_w} {'─'*20} {'─'*10}")
    print(f"  {'Total waktu':<{col_w}} {'':<20} {_fmt_duration(total_time):>10}")
    print()
    print(f"  ✅ Berhasil : {success_cnt}")
    if failed_cnt:
        print(_c(f"  ❌ Gagal    : {failed_cnt}", _RED))
    if skipped_cnt:
        print(_c(f"  ⏭  Dilewati : {skipped_cnt}", _YELLOW))
    print(_c(_hr(), _BOLD, _CYAN))


# ── Validasi environment ───────────────────────────────────────────────────────
def _validate_env():
    errors = []

    if not os.path.isfile(_VENV_PYTHON):
        errors.append(
            f"Venv python tidak ditemukan: {_VENV_PYTHON}\n"
            f"  Pastikan .venv sudah dibuat: python -m venv .venv && .venv/bin/pip install ..."
        )

    for step in _PIPELINE:
        if not os.path.isfile(step["script"]):
            errors.append(f"Script tidak ditemukan: {step['script']}")

    if errors:
        print(_c("\n  ❌ VALIDASI GAGAL:", _RED, _BOLD))
        for e in errors:
            print(f"    • {e}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MultiGPU Evaluation Pipeline — jalankan semua eval_multigpu.py secara sekuensial"
    )
    parser.add_argument(
        "--skip", type=str, default="",
        metavar="MODEL[,MODEL...]",
        help=(
            "Model yang dilewati, pisah koma. "
            "Pilihan: yolo8, yolo9, yolo11, maskrcnn, hybrid. "
            "Contoh: --skip yolo8,maskrcnn"
        ),
    )
    parser.add_argument(
        "--gpus", type=str, default="all",
        help="GPU yang digunakan: '0', '0,1', 'all'. Default: 'all' (semua GPU).",
    )
    args = parser.parse_args()

    skip_names = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    gpus       = args.gpus.strip()

    _validate_env()
    _print_header()
    _print_plan(_PIPELINE, skip_names)

    pipeline_to_run = [s for s in _PIPELINE]
    results = []
    total_start = time.perf_counter()
    interrupted = False

    for idx, step in enumerate(pipeline_to_run, 1):
        total = len(pipeline_to_run)

        if step["name"] in skip_names:
            print(f"  {_c('⏭  SKIP', _YELLOW)} [{idx}/{total}] {step['label']}")
            results.append({"label": step["label"], "rc": "SKIP", "elapsed": 0})
            continue

        _print_step_start(idx, total, step["label"])
        try:
            rc, elapsed = _run_step(step, gpus)
        except KeyboardInterrupt:
            results.append({"label": step["label"], "rc": -2, "elapsed": 0})
            interrupted = True
            break

        _print_step_result(step["label"], rc, elapsed)
        results.append({"label": step["label"], "rc": rc, "elapsed": elapsed})

        if rc != 0:
            print(_c(
                f"\n  ⚠️  {step['label']} gagal (exit {rc}). "
                "Pipeline tetap dilanjutkan ke model berikutnya.\n",
                _YELLOW
            ))

    total_wall = time.perf_counter() - total_start
    _print_summary(results)

    if interrupted:
        print(_c("  Pipeline dihentikan oleh user.\n", _YELLOW))
        sys.exit(130)

    failed = [r for r in results if isinstance(r["rc"], int) and r["rc"] != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
