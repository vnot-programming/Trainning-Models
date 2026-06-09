# -*- coding: utf-8 -*-
"""
visual_eval_api.py — Backend API Evaluasi Visual (GPU-Powered + Queue)
======================================================================
Flask API dengan GPU Inference Queue. Hanya 1 inferensi berjalan di GPU
pada satu waktu — request lain mengantri secara FIFO.

MENGAPA PERLU QUEUE:
    GPU V100 32GB hanya bisa handle 1 model inference at a time. Jika >2 user
    mengirim request bersamaan tanpa queue → CUDA OOM / crash.
    Queue memastikan request diproses satu per satu (thread-safe).

ARSITEKTUR:
    HTTP Request → Flask → GPU Queue (FIFO) → Worker Thread → GPU Inference
    Compute Node (GPU) ←— SSH Reverse Tunnel ←— Login Node ←— Cloudflare Tunnel
    Diakses publik: https://backend-rvm.penelitian.my.id

CARA MENJALANKAN (di compute node setelah attach_gpu.sh):
    source /data/programs/anaconda3/bin/activate yolo_env
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster
    python RVM/backend/visual_eval_api.py &
    ssh -N -f -R 8502:localhost:8502 slurmmaster
"""

import os
import sys
import time
import uuid
import gc
import traceback
import threading
import queue as queue_module
from datetime import datetime
from collections import OrderedDict

# Sisipkan root project ke sys.path untuk import config_shared
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, request, jsonify
from config_shared import (
    ROOT, WORKSPACE_DIR, MODELS_DIR, NUM_CLASSES,
    EVAL_API_HOST, EVAL_API_BACKEND_PORT, EVAL_API_MAX_UPLOAD_MB,
    EVAL_API_ALLOWED_EXTENSIONS, EVAL_API_FRONTEND_URL,
    EVAL_API_DEBUG, EVAL_CONF, EVAL_IOU, get_output_dir,
)

# ==============================================================================
# DEVICE DETECTION — Deteksi GPU otomatis, fallback ke CPU
# ==============================================================================
import torch as _torch

if _torch.cuda.is_available():
    _DEVICE = "cuda:0"
    _DEVICE_NAME = _torch.cuda.get_device_name(0)
    print(f"[GPU] ✅ CUDA tersedia: {_DEVICE_NAME}")
else:
    _DEVICE = "cpu"
    _DEVICE_NAME = "CPU"
    print("[GPU] ⚠️  CUDA tidak tersedia, menggunakan CPU (inferensi lebih lambat)")

# ==============================================================================
# FLASK APP INITIALIZATION
# ==============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = EVAL_API_MAX_UPLOAD_MB * 1024 * 1024

def _update_class_mapping():
    """
    Sinkronisasi otomatis mapping kelas (class_mapping.json) dari data.yaml segmentasi.
    MENGAPA: Menjamin Mask R-CNN selalu sinkron dengan dataset latih terbaru tanpa hardcoding.
    """
    yaml_path = os.path.join(ROOT, "datasets", "training_seg", "data.yaml")
    json_path = os.path.join(ROOT, "RVM", "backend", "class_mapping.json")
    
    default_mapping = {
        "0": "background",
        "1": "dishwasher",
        "2": "milk",
        "3": "mineral",
        "4": "non_mineral",
        "5": "not_empty",
        "6": "soda",
        "7": "yogurt"
    }
    
    try:
        import yaml
        import json
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            names = data.get("names", [])
            if names:
                mapping = {0: "background"}
                for idx, name in enumerate(names):
                    mapping[idx + 1] = name
                with open(json_path, 'w') as f:
                    json.dump(mapping, f, indent=2)
                print(f"[Mapping] ✅ class_mapping.json terupdate dari {yaml_path}", flush=True)
                return
        
        if os.path.exists(json_path):
            print(f"[Mapping] ℹ️ class_mapping.json sudah ada (menggunakan file yang ada)", flush=True)
            return
            
        with open(json_path, 'w') as f:
            json.dump(default_mapping, f, indent=2)
        print("[Mapping] ⚠️ data.yaml tidak ditemukan, menulis default class_mapping.json", flush=True)
    except Exception as e:
        print(f"[Mapping] ❌ Gagal mengupdate class mapping: {e}", flush=True)
        if not os.path.exists(json_path):
            import json
            with open(json_path, 'w') as f:
                json.dump(default_mapping, f, indent=2)

# Jalankan update mapping otomatis saat startup
_update_class_mapping()

# Direktori sementara untuk upload
_UPLOAD_DIR = os.path.join(ROOT, "RVM", "backend", "_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

# Direktori log
_LOG_DIR = os.path.join(ROOT, "RVM", "backend", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


# ==============================================================================
# GPU INFERENCE QUEUE — Antrian inferensi agar GPU tidak overload
# ==============================================================================
# MENGAPA: GPU V100 32GB hanya bisa handle 1 inference sekaligus.
# Jika 3 user hit /api/evaluate bersamaan tanpa queue → CUDA OOM.
# Queue memproses FIFO, 1 job at a time, thread-safe.

_GPU_QUEUE = queue_module.Queue()  # FIFO queue untuk job inferensi
_JOB_RESULTS = OrderedDict()       # {job_id: result_dict} — hasil yang sudah selesai
_JOB_RESULTS_LOCK = threading.Lock()
_JOB_RESULTS_MAX = 100             # Simpan max 100 hasil terakhir, cleanup otomatis
_QUEUE_STATS = {
    "total_processed": 0,
    "total_errors": 0,
    "current_job": None,            # job_id yang sedang diproses
    "worker_alive": False,
}


class InferenceJob:
    """
    Representasi satu request evaluasi yang masuk antrian GPU.
    MENGAPA class: Agar setiap job punya ID unik, event signaling untuk
    notifikasi selesai, dan metadata tracking (waktu masuk, posisi antrian).
    """
    def __init__(self, job_id, img_path, selected_models, conf, iou, image_name,
                 img_width, img_height):
        self.job_id = job_id
        self.img_path = img_path
        self.selected_models = selected_models
        self.conf = conf
        self.iou = iou
        self.image_name = image_name
        self.img_width = img_width
        self.img_height = img_height
        self.submitted_at = time.time()
        self.completed_event = threading.Event()  # Signal selesai
        self.result = None  # Akan diisi oleh worker


def _gpu_worker():
    """
    Background thread tunggal yang mengambil job dari queue dan menjalankan
    inferensi di GPU satu per satu. Thread ini hidup selama server berjalan.

    MENGAPA 1 worker: GPU hanya 1, menjalankan >1 inferensi paralel
    menyebabkan VRAM bentrok. FIFO = adil dan aman.
    """
    _QUEUE_STATS["worker_alive"] = True
    print("[Queue Worker] ✅ GPU worker thread dimulai — siap menerima job.")

    while True:
        try:
            # Blocking get — tunggu job masuk
            job = _GPU_QUEUE.get(timeout=None)

            if job is None:
                # Poison pill — shutdown signal
                print("[Queue Worker] 🛑 Shutdown signal diterima.")
                break

            _QUEUE_STATS["current_job"] = job.job_id
            print(f"[Queue Worker] ▶ Memproses job {job.job_id} "
                  f"({len(job.selected_models)} model, image: {job.image_name})")

            t_start = time.perf_counter()
            results = []

            for model_key in job.selected_models:
                result = _run_inference(model_key, job.img_path, job.conf, job.iou)
                results.append(result)

            t_total_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # Compose final result
            job.result = {
                "status": "success",
                "job_id": job.job_id,
                "image_name": job.image_name,
                "image_size": [job.img_width, job.img_height],
                "confidence_threshold": job.conf,
                "iou_threshold": job.iou,
                "total_models_evaluated": len(results),
                "total_time_ms": t_total_ms,
                "queue_wait_ms": round((t_start - job.submitted_at) * 1000, 2),
                "device": _DEVICE,
                "results": results,
            }

            # Simpan result ke cache
            with _JOB_RESULTS_LOCK:
                _JOB_RESULTS[job.job_id] = job.result
                # Cleanup cache lama jika melebihi batas
                while len(_JOB_RESULTS) > _JOB_RESULTS_MAX:
                    _JOB_RESULTS.popitem(last=False)

            _QUEUE_STATS["total_processed"] += 1
            _QUEUE_STATS["current_job"] = None

            print(f"[Queue Worker] ✅ Job {job.job_id} selesai "
                  f"({t_total_ms:.0f}ms, {len(results)} model)")

            # Cleanup file temporary
            try:
                os.remove(job.img_path)
            except OSError:
                pass

            # Signal bahwa job selesai — unblock request thread yang menunggu
            job.completed_event.set()
            _GPU_QUEUE.task_done()

        except Exception as e:
            traceback.print_exc()
            _QUEUE_STATS["total_errors"] += 1
            _QUEUE_STATS["current_job"] = None

            if 'job' in locals() and job is not None:
                job.result = {
                    "status": "error",
                    "job_id": job.job_id,
                    "message": f"GPU worker error: {e}",
                    "results": [],
                }
                with _JOB_RESULTS_LOCK:
                    _JOB_RESULTS[job.job_id] = job.result
                try:
                    os.remove(job.img_path)
                except OSError:
                    pass
                job.completed_event.set()
                _GPU_QUEUE.task_done()


# Start GPU worker thread — daemon=True agar mati bersama main process
_worker_thread = threading.Thread(target=_gpu_worker, daemon=True, name="gpu-worker")
_worker_thread.start()


# ==============================================================================
# MODEL REGISTRY — Daftar semua model yang tersedia beserta metadata
# ==============================================================================
def _build_model_registry():
    """
    Membangun registry model berdasarkan folder runs/ di workspace aktif.
    MENGAPA: Agar frontend tahu model apa saja yang tersedia tanpa hardcode.
    Dipanggil sekali saat startup — registry di-cache di memori.
    """
    registry = {}

    # --- 1. YOLO Models (Detection & Segmentation) ---
    yolo_models = {
        "yolov8m": {"family": "YOLOv8", "task": "detection"},
        "yolov8m_seg": {"family": "YOLOv8", "task": "segmentation"},
        "yolov8x": {"family": "YOLOv8", "task": "detection"},
        "yolov8x_seg": {"family": "YOLOv8", "task": "segmentation"},
        "yolov9m": {"family": "YOLOv9", "task": "detection"},
        "yolov9c_seg": {"family": "YOLOv9", "task": "segmentation"},
        "yolov9e": {"family": "YOLOv9", "task": "detection"},
        "yolov9e_seg": {"family": "YOLOv9", "task": "segmentation"},
        "yolov10m": {"family": "YOLOv10", "task": "detection"},
        "yolov10x": {"family": "YOLOv10", "task": "detection"},
        "yolo11n": {"family": "YOLO11", "task": "detection"},
        "yolo11n_seg": {"family": "YOLO11", "task": "segmentation"},
        "yolo11l": {"family": "YOLO11", "task": "detection"},
        "yolo11l_seg": {"family": "YOLO11", "task": "segmentation"},
        "yolo11x": {"family": "YOLO11", "task": "detection"},
        "yolo11x_seg": {"family": "YOLO11", "task": "segmentation"},
    }

    for model_key, meta in yolo_models.items():
        weights_path = os.path.join(
            get_output_dir(model_key), "weights", "best.pt"
        )
        if os.path.exists(weights_path):
            size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2)
            registry[model_key] = {
                "key": model_key,
                "display_name": model_key.replace("_", "-"),
                "family": meta["family"],
                "task": meta["task"],
                "type": "yolo",
                "weights_path": weights_path,
                "weights_size_mb": size_mb,
            }

    # --- 2. Mask R-CNN ---
    maskrcnn_path = os.path.join(
        get_output_dir("maskrcnn"), "weights", "best.pt"
    )
    if os.path.exists(maskrcnn_path):
        size_mb = round(os.path.getsize(maskrcnn_path) / (1024 * 1024), 2)
        registry["maskrcnn"] = {
            "key": "maskrcnn",
            "display_name": "Mask R-CNN (ResNet-50 FPN V2)",
            "family": "Mask R-CNN",
            "task": "segmentation",
            "type": "maskrcnn",
            "weights_path": maskrcnn_path,
            "weights_size_mb": size_mb,
        }

    # --- 3. RT-DETR ---
    rtdetr_path = os.path.join(
        get_output_dir("rtdetr_l"), "weights", "best.pt"
    )
    if os.path.exists(rtdetr_path):
        size_mb = round(os.path.getsize(rtdetr_path) / (1024 * 1024), 2)
        registry["rtdetr_l"] = {
            "key": "rtdetr_l",
            "display_name": "RT-DETR-L",
            "family": "RT-DETR",
            "task": "detection",
            "type": "yolo",
            "weights_path": rtdetr_path,
            "weights_size_mb": size_mb,
        }

    # --- 4. Hybrid Models (YOLO + SAM2 / MobileSAM) ---
    sam2_path = os.path.join(MODELS_DIR, "sam2.1_t.pt")
    mobilesam_path = os.path.join(MODELS_DIR, "mobile_sam.pt")

    for yolo_key, yolo_meta in yolo_models.items():
        yolo_weights = os.path.join(
            get_output_dir(yolo_key), "weights", "best.pt"
        )
        if not os.path.exists(yolo_weights):
            continue

        # Hybrid SAM2
        if os.path.exists(sam2_path):
            hybrid_key = f"hybrid_{yolo_key}_sam2"
            registry[hybrid_key] = {
                "key": hybrid_key,
                "display_name": f"{yolo_key.replace('_', '-')}+SAM2.1_t",
                "family": "Hybrid SAM2",
                "task": "hybrid_segmentation",
                "type": "hybrid_sam2",
                "weights_path": yolo_weights,
                "sam_weights_path": sam2_path,
                "weights_size_mb": round(
                    os.path.getsize(yolo_weights) / (1024 * 1024), 2
                ),
            }

        # Hybrid MobileSAM
        if os.path.exists(mobilesam_path):
            hybrid_key_m = f"hybrid_{yolo_key}_mobilesam"
            registry[hybrid_key_m] = {
                "key": hybrid_key_m,
                "display_name": f"{yolo_key.replace('_', '-')}+MobileSAM",
                "family": "Hybrid MobileSAM",
                "task": "hybrid_segmentation",
                "type": "hybrid_mobilesam",
                "weights_path": yolo_weights,
                "sam_weights_path": mobilesam_path,
                "weights_size_mb": round(
                    os.path.getsize(yolo_weights) / (1024 * 1024), 2
                ),
            }

    return registry


# Cache registry saat startup agar tidak rebuild setiap request
MODEL_REGISTRY = _build_model_registry()


# ==============================================================================
# MODEL LRU CACHE — Mencegah Re-load Model dan Fragmentasi VRAM CUDA
# ==============================================================================
class ModelCache:
    """
    LRU Cache untuk menampung model-model yang sudah dimuat di GPU VRAM.
    Maksimal menampung 6 model aktif secara bersamaan agar tidak CUDA OOM.
    Mempercepat inferensi multi-model secara dramatis dari disk I/O.
    """
    def __init__(self, max_size=6):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get_yolo(self, path):
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
            
            from ultralytics import YOLO
            print(f"[ModelCache] 💾 Loading YOLO/RT-DETR dari {path} ke VRAM...", flush=True)
            model = YOLO(path)
            model.to(_DEVICE)
            
            self.cache[path] = model
            self._cleanup()
            return model

    def get_sam(self, path):
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
            
            from ultralytics import SAM
            print(f"[ModelCache] 💾 Loading SAM dari {path} ke VRAM...", flush=True)
            model = SAM(path)
            model.to(_DEVICE)
            
            self.cache[path] = model
            self._cleanup()
            return model

    def get_maskrcnn(self, path):
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
            
            import torch
            maskrcnn_dir = os.path.join(ROOT, "mask-r-cnn")
            if maskrcnn_dir not in sys.path:
                sys.path.insert(0, maskrcnn_dir)
            from maskrcnn_builder import build_model
            
            print(f"[ModelCache] 💾 Loading Mask R-CNN dari {path} ke VRAM...", flush=True)
            device = torch.device(_DEVICE)
            model = build_model(device=device, num_classes=NUM_CLASSES + 1)
            state_dict = torch.load(path, map_location=device, weights_only=False)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(device).eval()
            
            self.cache[path] = model
            self._cleanup()
            return model

    def _cleanup(self):
        while len(self.cache) > self.max_size:
            old_path, old_model = self.cache.popitem(last=False)
            print(f"[ModelCache] 🗑️ Mengeluarkan model {old_path} dari VRAM cache.", flush=True)
            del old_model
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

_MODEL_CACHE = ModelCache(max_size=6)


# ==============================================================================
# INFERENCE FUNCTIONS
# ==============================================================================
def _infer_yolo(model_info, img_path, conf, iou):
    """Inferensi menggunakan Ultralytics YOLO (juga kompatibel untuk RT-DETR)."""
    t0 = time.perf_counter()
    model = _MODEL_CACHE.get_yolo(model_info["weights_path"])
    results = model.predict(
        img_path, conf=conf, iou=iou, verbose=False, device=_DEVICE
    )
    t_ms = round((time.perf_counter() - t0) * 1000, 2)

    detections = []
    if results and len(results) > 0:
        r = results[0]
        names = r.names or {}

        if r.boxes is not None:
            for i in range(len(r.boxes)):
                det = {
                    "class": names.get(int(r.boxes.cls[i]), "unknown"),
                    "confidence": round(float(r.boxes.conf[i]), 4),
                    "bbox": [round(float(v), 1) for v in r.boxes.xyxy[i].tolist()],
                    "mask_area": None,
                }
                if r.masks is not None and i < len(r.masks.data):
                    mask_np = r.masks.data[i].cpu().numpy()
                    det["mask_area"] = int((mask_np > 0.5).sum())
                    if r.masks.xy is not None and i < len(r.masks.xy):
                        det["segment"] = [[float(pt[0]), float(pt[1])] for pt in r.masks.xy[i]]
                detections.append(det)

    return {
        "model": model_info["display_name"],
        "model_key": model_info["key"],
        "model_type": model_info["task"],
        "family": model_info["family"],
        "inference_time_ms": t_ms,
        "detections": detections,
    }


def _infer_maskrcnn(model_info, img_path, conf):
    """Inferensi menggunakan TorchVision Mask R-CNN."""
    import torch
    import cv2

    t0 = time.perf_counter()
    device = torch.device(_DEVICE)
    model = _MODEL_CACHE.get_maskrcnn(model_info["weights_path"])

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {
            "model": model_info["display_name"],
            "model_key": model_info["key"],
            "model_type": "segmentation",
            "family": "Mask R-CNN",
            "inference_time_ms": 0,
            "detections": [],
            "error": f"Gagal membaca gambar: {img_path}",
        }
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        predictions = model([img_tensor])

    t_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred = predictions[0]
    detections = []
    
    # Baca mapping secara dinamis dari class_mapping.json
    json_path = os.path.join(ROOT, "RVM", "backend", "class_mapping.json")
    class_names = {}
    try:
        import json
        with open(json_path, 'r') as f:
            mapping_data = json.load(f)
            class_names = {int(k): v for k, v in mapping_data.items()}
    except Exception as e:
        print(f"[Inference] ⚠️ Gagal memuat class_mapping.json: {e}, menggunakan fallback", flush=True)
        class_names = {
            0: "background", 1: "dishwasher", 2: "milk", 3: "mineral",
            4: "non_mineral", 5: "not_empty", 6: "soda", 7: "yogurt",
        }

    scores = pred.get("scores", torch.tensor([]))
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf:
            continue
        label_id = int(pred["labels"][i])
        bbox = [round(float(v), 1) for v in pred["boxes"][i].tolist()]
        mask_area = None
        segment = None
        if "masks" in pred and i < len(pred["masks"]):
            import numpy as np
            mask_np = pred["masks"][i, 0].cpu().numpy()
            mask_area = int((mask_np > 0.5).sum())
            binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                segment = [[float(pt[0][0]), float(pt[0][1])] for pt in largest_contour]
        detections.append({
            "class": class_names.get(label_id, f"class_{label_id}"),
            "confidence": round(score, 4),
            "bbox": bbox,
            "mask_area": mask_area,
            "segment": segment,
        })

    return {
        "model": model_info["display_name"],
        "model_key": model_info["key"],
        "model_type": "segmentation",
        "family": "Mask R-CNN",
        "inference_time_ms": t_ms,
        "detections": detections,
    }


def _infer_hybrid(model_info, img_path, conf, iou):
    """Inferensi Hybrid: YOLO prompt generator → SAM2/MobileSAM segmentasi."""
    t0 = time.perf_counter()

    yolo_model = _MODEL_CACHE.get_yolo(model_info["weights_path"])
    yolo_results = yolo_model.predict(
        img_path, conf=conf, iou=iou, verbose=False, device=_DEVICE
    )

    if not yolo_results or len(yolo_results) == 0 or yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
        return {
            "model": model_info["display_name"],
            "model_key": model_info["key"],
            "model_type": "hybrid_segmentation",
            "family": model_info["family"],
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "detections": [],
        }

    r = yolo_results[0]
    names = r.names or {}
    bboxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    sam_model = _MODEL_CACHE.get_sam(model_info["sam_weights_path"])
    sam_results = sam_model.predict(
        img_path, bboxes=bboxes.tolist(), verbose=False, device=_DEVICE
    )

    detections = []
    for i in range(len(bboxes)):
        mask_area = None
        segment = None
        if (
            sam_results and len(sam_results) > 0
            and sam_results[0].masks is not None
            and i < len(sam_results[0].masks.data)
        ):
            mask_np = sam_results[0].masks.data[i].cpu().numpy()
            mask_area = int((mask_np > 0.5).sum())
            if sam_results[0].masks.xy is not None and i < len(sam_results[0].masks.xy):
                segment = [[float(pt[0]), float(pt[1])] for pt in sam_results[0].masks.xy[i]]
        detections.append({
            "class": names.get(int(classes[i]), "unknown"),
            "confidence": round(float(confs[i]), 4),
            "bbox": [round(float(v), 1) for v in bboxes[i].tolist()],
            "mask_area": mask_area,
            "segment": segment,
        })

    t_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "model": model_info["display_name"],
        "model_key": model_info["key"],
        "model_type": "hybrid_segmentation",
        "family": model_info["family"],
        "inference_time_ms": t_ms,
        "detections": detections,
    }


def _run_inference(model_key, img_path, conf, iou):
    """Dispatcher inferensi berdasarkan tipe model."""
    model_info = MODEL_REGISTRY.get(model_key)
    if model_info is None:
        print(f"  [Inference] ❌ Model '{model_key}' tidak ditemukan di registry.", flush=True)
        return {
            "model": model_key,
            "model_key": model_key,
            "error": f"Model '{model_key}' tidak ditemukan di registry.",
            "detections": [],
        }

    mtype = model_info["type"]
    print(f"  [Inference] 🚀 Memulai {model_key} ({mtype}) | Conf={conf} | IoU={iou}", flush=True)
    try:
        if mtype == "yolo":
            res = _infer_yolo(model_info, img_path, conf, iou)
        elif mtype == "maskrcnn":
            res = _infer_maskrcnn(model_info, img_path, conf)
        elif mtype in ("hybrid_sam2", "hybrid_mobilesam"):
            res = _infer_hybrid(model_info, img_path, conf, iou)
        else:
            res = {
                "model": model_info["display_name"],
                "model_key": model_key,
                "error": f"Tipe model tidak dikenali: {mtype}",
                "detections": [],
            }
        
        det_len = len(res.get("detections", []))
        err = res.get("error", None)
        status_msg = f"❌ Error: {err}" if err else f"✅ {det_len} deteksi"
        print(f"  [Inference] 🏁 Selesai {model_key} | {status_msg} | Waktu: {res.get('inference_time_ms', 0)}ms", flush=True)
        return res
    except Exception as e:
        traceback.print_exc()
        print(f"  [Inference] ❌ Exception pada {model_key}: {e}", flush=True)
        return {
            "model": model_info.get("display_name", model_key),
            "model_key": model_key,
            "error": str(e),
            "inference_time_ms": 0,
            "detections": [],
        }


# ==============================================================================
# CORS MIDDLEWARE
# ==============================================================================
@app.after_request
def _add_cors_headers(response):
    """Wildcard CORS karena frontend dan backend di subdomain berbeda."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, CF-Access-Client-Id, CF-Access-Client-Secret"
    return response


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check — server status, GPU info, queue stats."""
    gpu_info = {}
    if _torch.cuda.is_available():
        gpu_info = {
            "name": _torch.cuda.get_device_name(0),
            "vram_total_gb": round(_torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
        try:
            free, total = _torch.cuda.mem_get_info(0)
            gpu_info["vram_free_gb"] = round(free / 1e9, 2)
            gpu_info["vram_used_gb"] = round((total - free) / 1e9, 2)
        except Exception:
            pass

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": len(MODEL_REGISTRY),
        "workspace": WORKSPACE_DIR,
        "device": _DEVICE,
        "device_name": _DEVICE_NAME,
        "gpu": gpu_info,
        "config": {
            "eval_conf": EVAL_CONF,
            "eval_iou": EVAL_IOU,
        },
        "queue": {
            "pending": _GPU_QUEUE.qsize(),
            "processing": _QUEUE_STATS["current_job"],
            "total_processed": _QUEUE_STATS["total_processed"],
            "total_errors": _QUEUE_STATS["total_errors"],
            "worker_alive": _worker_thread.is_alive(),
        },
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """Daftar seluruh model tersedia — dipakai frontend untuk dropdown."""
    models_list = []
    for key, info in sorted(MODEL_REGISTRY.items()):
        models_list.append({
            "key": info["key"],
            "display_name": info["display_name"],
            "family": info["family"],
            "task": info["task"],
            "type": info["type"],
            "weights_size_mb": info.get("weights_size_mb", "N/A"),
        })

    families = {}
    for m in models_list:
        fam = m["family"]
        if fam not in families:
            families[fam] = []
        families[fam].append(m)

    return jsonify({
        "status": "success",
        "total_models": len(models_list),
        "families": families,
        "models": models_list,
    })


@app.route("/api/queue/status", methods=["GET"])
def queue_status():
    """
    Status antrian GPU — dipakai frontend untuk menampilkan posisi antrian.
    Frontend bisa poll endpoint ini setiap 2 detik saat menunggu.
    """
    return jsonify({
        "status": "success",
        "queue_size": _GPU_QUEUE.qsize(),
        "currently_processing": _QUEUE_STATS["current_job"],
        "total_processed": _QUEUE_STATS["total_processed"],
        "total_errors": _QUEUE_STATS["total_errors"],
        "worker_alive": _worker_thread.is_alive(),
    })


@app.route("/api/evaluate", methods=["POST", "OPTIONS"])
def evaluate():
    """
    Upload gambar + pilih model → masuk antrian GPU → tunggu → return JSON.

    Request (multipart/form-data):
        - image: File gambar (required)
        - models: Comma-separated model keys, atau "all" (default: yolo11l)
        - conf: Confidence threshold (default dari config_shared)
        - iou: IoU threshold (default dari config_shared)

    Response (JSON):
        { status, job_id, queue_position, image_name, results: [...] }

    FLOW:
        1. Validasi upload → simpan file temporary
        2. Buat InferenceJob → masukkan ke GPU queue
        3. Tunggu job selesai (blocking, max timeout 300s)
        4. Return hasil
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    # --- Validasi file upload ---
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Field 'image' tidak ditemukan. Kirim gambar via multipart/form-data.",
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Nama file kosong.",
        }), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in EVAL_API_ALLOWED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": f"Ekstensi '{ext}' tidak diizinkan. Gunakan: {', '.join(sorted(EVAL_API_ALLOWED_EXTENSIONS))}",
        }), 400

    # --- Parse parameter ---
    models_param = request.form.get("models", "yolo11l")
    conf = float(request.form.get("conf", EVAL_CONF))
    iou = float(request.form.get("iou", EVAL_IOU))

    if models_param.strip().lower() == "all":
        selected_models = list(MODEL_REGISTRY.keys())
    else:
        selected_models = [
            m.strip() for m in models_param.split(",") if m.strip()
        ]

    if not selected_models:
        return jsonify({
            "status": "error",
            "message": "Tidak ada model yang dipilih.",
        }), 400

    # --- Simpan file sementara ---
    job_id = uuid.uuid4().hex[:12]
    unique_name = f"{job_id}{ext}"
    tmp_path = os.path.join(_UPLOAD_DIR, unique_name)
    try:
        file.save(tmp_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal menyimpan file upload: {e}",
        }), 500

    # --- Dapatkan dimensi gambar ---
    try:
        from PIL import Image as PILImage
        with PILImage.open(tmp_path) as pil_img:
            img_width, img_height = pil_img.size
    except Exception:
        img_width, img_height = 0, 0

    # --- Buat job dan masukkan ke antrian ---
    job = InferenceJob(
        job_id=job_id,
        img_path=tmp_path,
        selected_models=selected_models,
        conf=conf,
        iou=iou,
        image_name=file.filename,
        img_width=img_width,
        img_height=img_height,
    )

    queue_position = _GPU_QUEUE.qsize() + 1  # +1 karena belum masuk
    _GPU_QUEUE.put(job)

    print(f"[Queue] 📥 Job {job_id} masuk antrian "
          f"(posisi: {queue_position}, models: {len(selected_models)})")

    # --- Tunggu job selesai (blocking dengan timeout) ---
    # Timeout 300 detik (5 menit) — cukup untuk inferensi 49 model sekaligus
    timeout_seconds = 300
    completed = job.completed_event.wait(timeout=timeout_seconds)

    if not completed:
        # Timeout — job masih dalam antrian atau sedang diproses
        return jsonify({
            "status": "timeout",
            "job_id": job_id,
            "message": f"Request timeout setelah {timeout_seconds}s. "
                       f"Job mungkin masih dalam antrian atau sedang diproses. "
                       f"Cek /api/queue/status untuk status terkini.",
        }), 408

    # --- Return hasil ---
    if job.result is None:
        return jsonify({
            "status": "error",
            "job_id": job_id,
            "message": "Job selesai tapi tidak ada hasil — kemungkinan GPU worker error.",
            "results": [],
        }), 500

    return jsonify(job.result)


@app.route("/api/result/<job_id>", methods=["GET"])
def get_result(job_id):
    """
    Ambil hasil evaluasi berdasarkan job_id.
    Berguna jika request timeout tapi job sebenarnya sudah selesai.
    """
    with _JOB_RESULTS_LOCK:
        result = _JOB_RESULTS.get(job_id)

    if result is None:
        return jsonify({
            "status": "not_found",
            "message": f"Job '{job_id}' tidak ditemukan. "
                       f"Mungkin sudah expired atau belum selesai.",
        }), 404

    return jsonify(result)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  🔬 Visual Evaluation API — GPU-Powered + Queue Backend")
    print("=" * 60)
    print(f"  Host     : {EVAL_API_HOST}")
    print(f"  Port     : {EVAL_API_BACKEND_PORT}")
    print(f"  Models   : {len(MODEL_REGISTRY)} tersedia")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print(f"  Device   : {_DEVICE} ({_DEVICE_NAME})")
    print(f"  Queue    : FIFO, 1 GPU worker, max timeout 300s")
    print(f"  Publik   : https://backend-rvm.penelitian.my.id")
    print("=" * 60)

    for fam in sorted(set(m["family"] for m in MODEL_REGISTRY.values())):
        models_in_fam = [
            m["display_name"]
            for m in MODEL_REGISTRY.values()
            if m["family"] == fam
        ]
        print(f"  [{fam}] {len(models_in_fam)} model(s)")

    print("=" * 60)
    print("  ⚠️  Jangan lupa SSH reverse tunnel:")
    print(f"  ssh -N -f -R {EVAL_API_BACKEND_PORT}:localhost:{EVAL_API_BACKEND_PORT} slurmmaster")
    print("=" * 60)

    app.run(
        host=EVAL_API_HOST,
        port=EVAL_API_BACKEND_PORT,
        debug=EVAL_API_DEBUG,
        use_reloader=False,
        threaded=True,
    )
