# -*- coding: utf-8 -*-
"""
run_pipeline_parallel.py
========================
Parallel Multi-GPU Training & Evaluation Scheduler (Scopus Q1/Q2 Compliance)

Sistem akan memetakan jumlah GPU secara otomatis, mendistribusikan beban 
kerja secara cerdas (1 GPU = 1 Model), dan menjalankan evaluasi otomatis 
pada GPU yang sedang idle. Semua konfigurasi membaca dari `config_shared.py`.

PANDUAN EKSEKUSI AGAR TIDAK TERPUTUS (WORKFLOW TMUX YANG BENAR):

1. Buat Sesi TMUX di Login Node (JANGAN MASUK NODE AI DULU):
   tmux new-session -s training_pipeline

2. Di dalam TMUX, sambungkan (attach) ke Node GPU:
   cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils && ./attach_gpu.sh

3. Setelah masuk Node GPU, jalankan training:
   source /data/programs/anaconda3/bin/activate yolo_env
   cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && LOG_DIR=$(python3 -c "import config_shared, os; print(os.path.join(config_shared.WORKSPACE_DIR, 'logs'))") && mkdir -p "$LOG_DIR" && python3 run_pipeline_parallel.py 2>&1 | tee "$LOG_DIR/run_pipeline_parallel.log"
   
4. Detach dari TMUX untuk meninggalkan proses (Aman jika terminal ditutup):
   Tekan Ctrl+b, lalu tekan d. atau 
   di mac Tekan tombol control (^)+b tekan tombol d (untuk detach).
   (Untuk kembali melihat proses nanti, jalankan: tmux attach -t training_pipeline)

Catatan: Argumen --gpus bersifat opsional. Jika tidak diisi, sistem akan 
menggunakan seluruh GPU yang tersedia secara otomatis.
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from pathlib import Path
import torch

# ── Konfigurasi utama ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _THIS_DIR)

from config_shared import WORKSPACE_DIR, EARLY_STOPPING_PATIENCE, PARALLEL_TRAINING
from telegram_utils import send_telegram_msg

_VENV_PYTHON = sys.executable  # Gunakan interpreter saat ini (misal conda yolo_env)

# Folder Log terpusat di dalam Workspace agar workspace tetap bersih
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── ANSI Colors ────────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_DIM    = "\033[2m"

def _c(text, *codes): return "".join(codes) + str(text) + _RESET
def _hr(char="═", width=75): return char * width

def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

class ParallelScheduler:
    def __init__(self, gpu_list: list, skip_models: set, eval_only: bool = False):
        self.gpu_list = gpu_list
        self.skip_models = skip_models
        self.eval_only = eval_only
        self.gpu_status = {gpu: None for gpu in gpu_list}  # gpu_id -> task_id or None
        
        # Inisialisasi daftar tugas (Task Registry)
        self.tasks = {}
        self._build_task_registry()
        
    def _build_task_registry(self):
        # 1. Training Tasks
        train_specs = [
            {"id": "train_yolo8",    "name": "yolo8",    "label": "YOLOv8m/x Train",    "cwd": os.path.join(_THIS_DIR, "yolo", "yolo8"),     "script": "main.py",          "args": ["--skip-eval"], "device_arg": "--device"},
            {"id": "train_yolo9",    "name": "yolo9",    "label": "YOLOv9m/e Train",    "cwd": os.path.join(_THIS_DIR, "yolo", "yolo9"),     "script": "main.py",          "args": ["--skip-eval"], "device_arg": "--device"},
            {"id": "train_yolo10",   "name": "yolo10",   "label": "YOLOv10m/x Train",   "cwd": os.path.join(_THIS_DIR, "yolo", "yolov10"),   "script": "main.py",          "args": ["--skip-eval"], "device_arg": "--device"},
            {"id": "train_yolo11",   "name": "yolo11",   "label": "YOLO11n/l/x Train",  "cwd": os.path.join(_THIS_DIR, "yolo", "yolo11"),   "script": "main.py",          "args": ["--skip-eval"], "device_arg": "--device"},
            {"id": "train_maskrcnn", "name": "maskrcnn", "label": "Mask R-CNN Train",    "cwd": os.path.join(_THIS_DIR, "mask-r-cnn"),        "script": "train_multigpu.py","args": ["--skip-eval"], "device_arg": "--gpus"},
            # Paper 3 — RT-DETR: Vision Transformer untuk benchmark Edge vs YOLO
            {"id": "train_rtdetr",   "name": "rtdetr",   "label": "RT-DETR-L Train",    "cwd": os.path.join(_THIS_DIR, "rtdetr"),            "script": "train_rtdetr.py",  "args": ["--skip-eval"], "device_arg": "--device"},
        ]
        
        # 2. Evaluation Tasks
        # generate_report_single_model.py adalah skrip evaluasi sentral (pengganti semua eval_multigpu.py)
        # Dipanggil dengan --family <model> untuk menjalankan COCOeval + generate visual
        _UTILS_DIR = os.path.join(_THIS_DIR, "utils")
        eval_specs = [
            {"id": "eval_yolo8",    "name": "yolo8",    "label": "YOLOv8 Eval",         "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "yolo8"],    "device_arg": "--gpus", "deps": ["train_yolo8"]},
            {"id": "eval_yolo9",    "name": "yolo9",    "label": "YOLOv9 Eval",         "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "yolo9"],    "device_arg": "--gpus", "deps": ["train_yolo9"]},
            {"id": "eval_yolo10",   "name": "yolo10",   "label": "YOLOv10 Eval",        "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "yolov10"],  "device_arg": "--gpus", "deps": ["train_yolo10"]},
            {"id": "eval_yolo11",   "name": "yolo11",   "label": "YOLO11 Eval",         "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "yolo11"],   "device_arg": "--gpus", "deps": ["train_yolo11"]},
            {"id": "eval_maskrcnn", "name": "maskrcnn", "label": "Mask R-CNN Eval",     "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "maskrcnn"], "device_arg": "--gpus", "deps": ["train_maskrcnn"]},
            {"id": "eval_hybrid",   "name": "hybrid",   "label": "Hybrid Pipeline Eval","cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "hybrid"],   "device_arg": "--gpus", "deps": ["train_yolo11"]},
            # Paper 3 — RT-DETR evaluation (menggunakan family key yang sama dengan FAMILY_VARIANTS)
            {"id": "eval_rtdetr",   "name": "rtdetr",   "label": "RT-DETR-L Eval",      "cwd": _UTILS_DIR, "script": "generate_report_single_model.py", "args": ["--family", "rtdetr"],   "device_arg": "--gpus", "deps": ["train_rtdetr"]},
        ]
        
        # 3. Global Compilation (Langkah penutup setelah semua evaluasi selesai)
        # kompilasi_ALL_*.csv sudah di-append oleh masing-masing generate_report_single_model.py
        # Tidak perlu skrip tambahan — cukup tunggu semua eval selesai.
        global_eval = {
            "id": "eval_global_multigpu",
            "name": "global_eval",
            "label": "Global Compilation (ALL CSV)",
            "cwd": _UTILS_DIR,
            "script": "generate_report_single_model.py",
            "args": ["--family", "all"],
            "device_arg": "--gpus",
            # RT-DETR dimasukkan ke dalam dependensi global agar CSV ALL mencakup metrik Paper 3
            "deps": ["eval_yolo8", "eval_yolo9", "eval_yolo10", "eval_yolo11", "eval_maskrcnn", "eval_hybrid", "eval_rtdetr"]
        }
        
        # 4. New Method Evaluations (Tambahan khusus Scopus Q1)
        new_eval_specs = [
            {"id": "eval_new_std", "name": "global_eval", "label": "New Method Std Eval", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "standar_evaluation-new_method.py", "args": [], "device_arg": "--gpus", "deps": ["eval_global_multigpu"]},
            {"id": "eval_new_std_vis", "name": "global_eval", "label": "New Method Std Visuals", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "standar_evaluation_visuals-new_method.py", "args": [], "device_arg": "--gpus", "deps": ["eval_new_std"]},
            {"id": "eval_new_gld", "name": "global_eval", "label": "New Method Golden Eval", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "golden_evaluation-new_method.py", "args": [], "device_arg": "--gpus", "deps": ["eval_new_std_vis"]},
            {"id": "eval_new_gld_vis", "name": "global_eval", "label": "New Method Golden Visuals", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "golden_evaluation_visuals-new_method.py", "args": [], "device_arg": "--gpus", "deps": ["eval_new_gld"]},
            {"id": "eval_new_hybrid_sota", "name": "global_eval", "label": "New Method Hybrid SOTA Eval", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "evaluation_hybrid_sota.py", "args": [], "device_arg": "--gpus", "deps": ["eval_new_gld_vis"]}
        ]
        # 5. Post Processing Tasks (Grid Generator & Upload/Archive)
        post_specs = [
            {"id": "post_grid", "name": "global_eval", "label": "Generate Final Grid", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "generate_comparison_grid.py", "args": [], "device_arg": "", "deps": ["eval_new_hybrid_sota"]},
            {"id": "post_upload", "name": "global_eval", "label": "Local Archive & Upload", "cwd": os.path.join(_THIS_DIR, "utils"), "script": "upload_utils.py", "args": [], "device_arg": "", "deps": ["post_grid"]}
        ]
        
        new_eval_specs.extend(post_specs)
        
        # Daftarkan Training Tasks
        for spec in train_specs:
            is_skipped = self.eval_only or (spec["name"] in self.skip_models)
            self.tasks[spec["id"]] = {
                "id": spec["id"],
                "label": spec["label"],
                "type": "train",
                "cwd": spec["cwd"],
                "script": spec["script"],
                "args": spec["args"],
                "device_arg": spec["device_arg"],
                "dependencies": [],
                "state": "SKIPPED" if is_skipped else "PENDING",
                "proc": None,
                "gpu": None,
                "start_time": None,
                "elapsed": 0.0,
                "logfile": os.path.join(LOG_DIR, f"{spec['id']}.log")
            }
            
        # Daftarkan Evaluation Tasks
        for spec in eval_specs:
            # Jika eval-only aktif, evaluasi tidak di-skip. Jika tidak, ikuti aturan skip model.
            is_skipped = False if self.eval_only else (spec["name"] in self.skip_models)
            self.tasks[spec["id"]] = {
                "id": spec["id"],
                "label": spec["label"],
                "type": "eval",
                "cwd": spec["cwd"],
                "script": spec["script"],
                "args": spec["args"],
                "device_arg": spec["device_arg"],
                "dependencies": spec["deps"],
                "state": "SKIPPED" if is_skipped else "PENDING",
                "proc": None,
                "gpu": None,
                "start_time": None,
                "elapsed": 0.0,
                "logfile": os.path.join(LOG_DIR, f"{spec['id']}.log")
            }
            
        # Daftarkan Global Evaluation Task
        # Global eval hanya aktif jika ada model yang ditraining / dievaluasi
        run_any = any(self.tasks[tid]["state"] == "PENDING" for tid in ["train_yolo8", "train_yolo9", "train_yolo10", "train_yolo11", "train_maskrcnn", "train_rtdetr"])
        self.tasks[global_eval["id"]] = {
            "id": global_eval["id"],
            "label": global_eval["label"],
            "type": "global_eval",
            "cwd": global_eval["cwd"],
            "script": global_eval["script"],
            "args": global_eval["args"],
            "device_arg": global_eval["device_arg"],
            "dependencies": [dep for dep in global_eval["deps"] if self.tasks[dep]["state"] != "SKIPPED"],
            "state": "PENDING" if run_any else "SKIPPED",
            "proc": None,
            "gpu": None,
            "start_time": None,
            "elapsed": 0.0,
            "logfile": os.path.join(LOG_DIR, f"{global_eval['id']}.log")
        }

        # Daftarkan New Method Evaluations
        for spec in new_eval_specs:
            self.tasks[spec["id"]] = {
                "id": spec["id"],
                "label": spec["label"],
                "type": "global_eval",
                "cwd": spec["cwd"],
                "script": spec["script"],
                "args": spec["args"],
                "device_arg": spec["device_arg"],
                "dependencies": spec["deps"],
                "state": "PENDING",  # Diubah dari SKIPPED menjadi PENDING agar ikut dieksekusi
                "proc": None,
                "gpu": None,
                "start_time": None,
                "elapsed": 0.0,
                "logfile": os.path.join(LOG_DIR, f"{spec['id']}.log")
            }

    def print_plan(self):
        print(_c("\n  PARALLEL SCHEDULE PLAN:", _BOLD, _CYAN))
        print(f"  {'Task ID':<25} {'Type':<12} {'Status':<15} {'Dependencies'}")
        print(f"  {'-'*25} {'-'*12} {'-'*15} {'-'*30}")
        for tid, t in self.tasks.items():
            state_color = _GREEN if t["state"] == "PENDING" else _YELLOW
            deps_str = ", ".join(t["dependencies"]) if t["dependencies"] else "None"
            print(f"  {tid:<25} {t['type']:<12} {_c(t['state'], state_color):<25} {deps_str}")
        print("\n" + _c(_hr(), _BOLD, _CYAN))

    def run(self):
        send_telegram_msg(
            f"🚀 <b>Parallel Multi-GPU Pipeline Started</b>\n"
            f"Workspace: <code>{os.path.basename(WORKSPACE_DIR)}</code>\n"
            f"Active GPUs: <code>{self.gpu_list}</code>\n"
            f"Patience early stopping: <code>{EARLY_STOPPING_PATIENCE} epochs</code>"
        )
        
        total_start = time.perf_counter()
        print(_c("\n[Scheduler] Memulai orchestrator loop...", _BOLD, _GREEN))
        
        try:
            while True:
                # 1. Periksa proses aktif
                self._poll_active_processes()
                
                # 2. Perbarui status tugas jika dependensi gagal
                self._update_failed_dependencies()
                
                # 3. Periksa apakah semua tugas selesai
                if self._all_tasks_completed():
                    break
                    
                # 3. Cari GPU bebas dan tugaskan task ready
                self._dispatch_ready_tasks()
                
                # 4. Tampilkan HUD Status Ringkas
                self._print_hud()
                
                time.sleep(3.0)  # Polling interval 3 detik
                
        except KeyboardInterrupt:
            print(_c("\n⚠️  Interrupted oleh user! Mematikan semua sub-proses aktif...", _RED, _BOLD))
            self._kill_all_active_processes()
            sys.exit(130)
            
        total_elapsed = time.perf_counter() - total_start
        self._print_final_summary(total_elapsed)

    def _poll_active_processes(self):
        """Periksa status setiap proses yang sedang berjalan."""
        for tid, t in self.tasks.items():
            if t["state"] == "RUNNING" and t["proc"] is not None:
                rc = t["proc"].poll()
                if rc is not None:
                    # Proses selesai!
                    t["elapsed"] = time.perf_counter() - t["start_time"]
                    gpu_id = t["gpu"]
                    
                    # Bebaskan GPU
                    if gpu_id is not None:
                        self.gpu_status[gpu_id] = None
                        t["gpu"] = None
                        
                    if rc == 0:
                        t["state"] = "SUCCESS"
                        print(_c(f"\n[Scheduler] ✅ Task {t['label']} selesai sukses di GPU {gpu_id} ({_fmt_duration(t['elapsed'])})", _GREEN, _BOLD))
                        send_telegram_msg(f"✅ <b>Task Success</b>\nTask: <code>{t['label']}</code>\nGPU: <code>{gpu_id}</code>\nDurasi: <code>{_fmt_duration(t['elapsed'])}</code>")
                    else:
                        t["state"] = "FAILED"
                        print(_c(f"\n[Scheduler] ❌ Task {t['label']} gagal (exit code: {rc}) di GPU {gpu_id}", _RED, _BOLD))
                        send_telegram_msg(f"❌ <b>Task Failed</b>\nTask: <code>{t['label']}</code>\nGPU: <code>{gpu_id}</code>\nExit Code: <code>{rc}</code>")

    def _update_failed_dependencies(self):
        """Membatalkan (SKIPPED/FAILED) tugas yang memiliki dependensi gagal."""
        changed = True
        while changed:
            changed = False
            for tid, t in self.tasks.items():
                if t["state"] == "PENDING":
                    # Cek dependensi
                    for dep in t["dependencies"]:
                        dep_state = self.tasks[dep]["state"]
                        if dep_state in ["FAILED", "SKIPPED"]:
                            # Jika dep gagal atau di-skip, task ini juga di-skip
                            print(_c(f"\n[Scheduler] ⏭  Task {t['label']} otomatis dilewati karena dependensi ({dep}) {dep_state}", _YELLOW))
                            t["state"] = "SKIPPED"
                            changed = True
                            break

    def _dispatch_ready_tasks(self):
        """Temukan task yang siap dijalankan dan jalankan pada GPU bebas."""
        # 1. Cari GPU yang menganggur
        free_gpus = [gpu for gpu, status in self.gpu_status.items() if status is None]
        if not free_gpus:
            return  # Tidak ada GPU bebas
            
        for gpu_id in free_gpus:
            # Temukan task yang siap dijalankan (ready)
            ready_task = self._get_next_ready_task(gpu_id)
            if ready_task:
                # Alokasikan GPU ke task
                self.gpu_status[gpu_id] = ready_task["id"]
                ready_task["gpu"] = gpu_id
                ready_task["state"] = "RUNNING"
                ready_task["start_time"] = time.perf_counter()
                
                # Susun argumen perintah
                # Format arguments dengan menyisipkan device/gpus
                cmd = [_VENV_PYTHON, "-u", ready_task["script"]]
                if ready_task["device_arg"]:
                    cmd.extend([ready_task["device_arg"], str(gpu_id)])

                # Jika global eval, berikan daftar semua GPU aktif
                if ready_task["id"] == "eval_global_multigpu":
                    gpus_str = ",".join(map(str, self.gpu_list))
                    cmd = [_VENV_PYTHON, "-u", ready_task["script"], "--gpus", gpus_str]
                    
                cmd.extend(ready_task["args"])
                
                # Tulis log terpisah
                log_file = open(ready_task["logfile"], "w", encoding="utf-8")
                
                print(_c(f"\n[Scheduler] 🚀 Memulai {ready_task['label']} pada GPU {gpu_id}", _CYAN, _BOLD))
                print(f"  Logfile : {ready_task['logfile']}")
                print(f"  Command : {' '.join(cmd)}")
                
                # Launch subprocess
                ready_task["proc"] = subprocess.Popen(
                    cmd,
                    cwd=ready_task["cwd"],
                    stdout=log_file,
                    stderr=log_file,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Telegram Notification
                send_telegram_msg(f"🚀 <b>Task Started</b>\nTask: <code>{ready_task['label']}</code>\nGPU: <code>{gpu_id}</code>\nLog: <code>{os.path.basename(ready_task['logfile'])}</code>")

    def _get_next_ready_task(self, gpu_id):
        """Dapatkan tugas berikutnya yang siap dijalankan."""
        # Prioritas Pemilihan:
        # 1. EVALUATION Tasks dari model yang sudah sukses ditraining (agar GPU tidak menganggur)
        # 2. TRAINING Tasks yang belum jalan
        # 3. GLOBAL EVALUATION Task
        
        # Cek Evaluation Tasks
        for tid, t in self.tasks.items():
            if t["type"] == "eval" and t["state"] == "PENDING":
                if self._dependencies_met(t):
                    return t
                    
        # Cek Training Tasks
        for tid, t in self.tasks.items():
            if t["type"] == "train" and t["state"] == "PENDING":
                if self._dependencies_met(t):
                    return t
                    
        # Cek Global Evaluation Tasks (Termasuk New Method Evaluations)
        for tid, t in self.tasks.items():
            if t["type"] == "global_eval" and t["state"] == "PENDING":
                if self._dependencies_met(t):
                    # Pastikan tidak ada task lain yang sedang berjalan (karena butuh all GPUs / eksklusif)
                    any_other_running = any(ot["state"] == "RUNNING" for otid, ot in self.tasks.items() if otid != tid)
                    if not any_other_running:
                        return t
                    
        return None

    def _dependencies_met(self, task):
        """Periksa apakah seluruh dependensi tugas telah sukses diselesaikan."""
        for dep_id in task["dependencies"]:
            dep_task = self.tasks[dep_id]
            if dep_task["state"] != "SUCCESS" and dep_task["state"] != "SKIPPED":
                return False
        return True

    def _all_tasks_completed(self):
        """Periksa apakah seluruh tugas telah mencapai status akhir."""
        terminal_states = {"SUCCESS", "FAILED", "SKIPPED"}
        return all(t["state"] in terminal_states for t in self.tasks.values())

    def _kill_all_active_processes(self):
        for tid, t in self.tasks.items():
            if t["state"] == "RUNNING" and t["proc"] is not None:
                try:
                    t["proc"].terminate()
                    t["proc"].wait(timeout=5)
                except Exception:
                    pass

    def _print_hud(self):
        """Tampilkan dashboard status HUD di terminal."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + _c(_hr("─"), _DIM))
        print(f"📡  HUD PIPELINE PARALEL | {now_str}")
        print(_c(_hr("─"), _DIM))
        
        # 1. Tampilkan status GPU
        gpu_line = []
        for gpu, task_id in self.gpu_status.items():
            if task_id:
                task_label = self.tasks[task_id]["label"]
                gpu_line.append(f"GPU {gpu}: {_c(task_label, _GREEN, _BOLD)}")
            else:
                gpu_line.append(f"GPU {gpu}: {_c('IDLE', _DIM)}")
        print("  | " + " | ".join(gpu_line) + " |")
        
        # 2. Ringkasan Tugas
        print("\n  Status Tugas:")
        for tid, t in self.tasks.items():
            if t["state"] == "PENDING":
                state_str = _c("PENDING", _YELLOW)
            elif t["state"] == "RUNNING":
                elapsed = time.perf_counter() - t["start_time"]
                state_str = _c(f"RUNNING ({_fmt_duration(elapsed)})", _GREEN, _BOLD)
            elif t["state"] == "SUCCESS":
                state_str = _c(f"SUCCESS ({_fmt_duration(t['elapsed'])})", _GREEN)
            elif t["state"] == "SKIPPED":
                state_str = _c("SKIPPED", _DIM)
            else:
                state_str = _c("FAILED", _RED, _BOLD)
            print(f"    • {t['label']:<28} : {state_str}")
        print(_c(_hr("─"), _DIM))

    def _print_final_summary(self, total_elapsed: float):
        print("\n" + _c(_hr("═"), _BOLD, _CYAN))
        print(_c("  RINGKASAN PARALLEL PIPELINE UTAMA (Scopus Q1/Q2 Standard)", _BOLD, _CYAN))
        print(_c(_hr("═"), _BOLD, _CYAN))
        
        success_cnt = sum(1 for t in self.tasks.values() if t["state"] == "SUCCESS")
        failed_cnt = sum(1 for t in self.tasks.values() if t["state"] == "FAILED")
        skipped_cnt = sum(1 for t in self.tasks.values() if t["state"] == "SKIPPED")
        
        col_w = 30
        print(f"  {'Task Label':<{col_w}} {'Status':<25} {'Durasi':>10}")
        print(f"  {'─'*col_w} {'─'*25} {'─'*10}")
        
        for tid, t in self.tasks.items():
            if t["state"] == "SUCCESS":
                status = _c("BERHASIL", _GREEN)
                dur_str = _fmt_duration(t["elapsed"])
            elif t["state"] == "SKIPPED":
                status = _c("DILEWATI", _DIM)
                dur_str = "   —"
            elif t["state"] == "FAILED":
                status = _c("GAGAL", _RED, _BOLD)
                dur_str = _fmt_duration(t["elapsed"])
            else:
                status = _c("PENDING", _YELLOW)
                dur_str = "   —"
            print(f"  {t['label']:<{col_w}} {status:<35} {dur_str:>10}")
            
        print(f"  {'─'*col_w} {'─'*25} {'─'*10}")
        print(f"  {'Total Waktu Pipeline':<{col_w}} {'':<25} {_fmt_duration(total_elapsed):>10}")
        print()
        print(f"  ✅ Sukses   : {success_cnt}")
        if failed_cnt:
            print(_c(f"  ❌ Gagal    : {failed_cnt}", _RED))
        if skipped_cnt:
            print(_c(f"  ⏭  Dilewati : {skipped_cnt}", _YELLOW))
        print(_c(_hr("═"), _BOLD, _CYAN))
        
        send_telegram_msg(
            f"🏁 <b>Parallel Multi-GPU Pipeline Finished!</b>\n"
            f"Total Waktu: <code>{_fmt_duration(total_elapsed)}</code>\n"
            f"Sukses: <code>{success_cnt}/{len(self.tasks)}</code>"
        )


def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-GPU Training & Evaluation Orchestrator")
    parser.add_argument(
        "--gpus", type=str, default="default",
        help="GPU index yang akan digunakan (misal '0,1'). 'default' membaca config_shared."
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Daftar model yang di-skip trainingnya (misal 'yolo8,yolo9')."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Hanya jalankan proses evaluasi, lewati proses training."
    )
    args = parser.parse_args()
    
    # Resolusi GPU list
    if args.gpus.strip().lower() == "default":
        # Import default list from config_shared if defined, or auto-detect
        try:
            from config_shared import PARALLEL_GPUS
            gpu_str = PARALLEL_GPUS
        except ImportError:
            gpu_str = "0,1"
    else:
        gpu_str = args.gpus
        
    gpu_list = [int(g.strip()) for g in gpu_str.split(",") if g.strip()]
    skip_models = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    
    # Validasi CUDA
    if not torch.cuda.is_available() or not gpu_list:
        print(_c("❌ CUDA tidak tersedia atau GPU list kosong. Scheduler membutuhkan GPU aktif.", _RED, _BOLD))
        sys.exit(1)
        
    n_avail = torch.cuda.device_count()
    for g in gpu_list:
        if g >= n_avail:
            print(_c(f"❌ GPU {g} tidak tersedia (sistem hanya memiliki {n_avail} GPU).", _RED, _BOLD))
            sys.exit(1)
            
    print(_c(_hr(), _BOLD, _CYAN))
    print(_c("  Parallel Multi-GPU Training & Evaluation Pipeline Scheduler", _BOLD, _CYAN))
    print(_c("  Skenario 1 GPU = 1 Model (Standard 2026)", _DIM))
    print(_c(_hr(), _BOLD, _CYAN))
    print(f"  GPUs Available    : {n_avail}")
    print(f"  GPUs Targeted     : {gpu_list}")
    print(f"  Models to Skip    : {list(skip_models) if skip_models else 'None'}")
    print(f"  Workspace Directory: {WORKSPACE_DIR}")
    print(f"  Python Interpreter: {_VENV_PYTHON}")
    print(_c(_hr(), _BOLD, _CYAN))
    # 2. Inisialisasi Scheduler
    scheduler = ParallelScheduler(gpu_list=gpu_list, skip_models=skip_models, eval_only=args.eval_only)
    scheduler.print_plan()
    scheduler.run()


if __name__ == "__main__":
    main()
