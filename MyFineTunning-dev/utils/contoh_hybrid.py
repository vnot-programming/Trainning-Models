# Langkah 0: Instalasi dan Pengecekan Ultralytics
!pip install ultralytics
import ultralytics
ultralytics.checks() # Memastikan semuanya terinstal dengan baik

# Langkah 1: Impor Library
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO, SAM # Impor YOLO dan SAM dari ultralytics
from pathlib import Path

# --- Konfigurasi ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
yolo_model_name = 'yolo11x.pt' # Jika Anda yakin model ini ada dan valid
sam_model_name = "sam2.1_l.pt" # Jika Anda yakin model ini ada dan valid
local_image_path = "4.jpg" # Gunakan /home/my/Trainning-Models/MyFineTunning-dev/data-files/MyFineTunning-20260505_034341/image_samples

# --- Langkah 2: Muat Model ---

# Muat Model YOLO
try:
    yolo_model = YOLO(yolo_model_name)
    yolo_model.to(DEVICE)
    print(f"Successfully loaded YOLO model: {yolo_model_name}")
except Exception as e:
    print(f"Error loading YOLO model '{yolo_model_name}': {e}")
    print("Pastikan nama model YOLO sudah benar atau file .pt ada di direktori kerja.")
    yolo_model = None

# Muat Model SAM dari Ultralytics
# Pilih model SAM yang diinginkan: 'sam_b.pt', 'sam2_b.pt', 'sam2.1_b.pt'
try:
    sam_model_ultralytics = SAM(sam_model_name)
    # Model SAM dari Ultralytics mungkin juga menangani perpindahan ke device secara otomatis
    # atau bisa dipindahkan dengan sam_model_ultralytics.to(DEVICE) jika ada metodenya.
    # Berdasarkan dokumentasi, pemanggilan `model(source)` akan otomatis menggunakan device.
    print(f"Successfully loaded SAM model from Ultralytics: {sam_model_name}")
    sam_model_ultralytics.info() # Opsional, menampilkan info model
except Exception as e:
    print(f"Error loading SAM model from Ultralytics '{sam_model_name}': {e}")
    sam_model_ultralytics = None

# --- Langkah 3: Fungsi Pembantu untuk Visualisasi (Tetap sama, tapi kita akan menggunakan fitur .show() dari Ultralytics) ---
# Kita mungkin tidak memerlukan fungsi show_mask dan show_box secara manual
# karena objek `results` dari Ultralytics SAM memiliki metode `.show()`

# --- Langkah 4: Proses Gambar ---

def process_image_yolo_ultralytics_sam(image_source, yolo_model_instance, sam_model_instance, confidence_threshold=0.25, is_url=False):
    """
    Memproses gambar: Deteksi dengan YOLO, lalu segmentasi objek yang terdeteksi dengan SAM dari Ultralytics.
    Menampilkan hasil berdampingan.
    """
    if not yolo_model_instance:
        print("YOLO model not loaded. Aborting.")
        return
    if not sam_model_instance:
        print("Ultralytics SAM model not loaded. Aborting.")
        return

    current_image_path = image_source
    if is_url:
        print(f"Attempting to download image from URL: {image_source}")
        downloaded_path = download_image(image_source, "temp_image_for_processing.jpg") # Gunakan nama file sementara yang lebih jelas
        if not downloaded_path:
            return
        current_image_path = downloaded_path

    print(f"Processing image from path: {current_image_path}")

    # Untuk menyimpan output teks
    yolo_output_text = []
    sam_output_text = []

    # 1. Deteksi dengan YOLO
    print("Running YOLO detection...")
    try:
        yolo_results_list = yolo_model_instance(current_image_path, device=DEVICE)
    except Exception as e:
        yolo_output_text.append(f"Error during YOLO inference: {e}")
        print(yolo_output_text[-1])
        if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
             Path(current_image_path).unlink(missing_ok=True)
        return

    if not yolo_results_list:
        yolo_output_text.append("No results from YOLO detection.")
        print(yolo_output_text[-1])
        if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
             Path(current_image_path).unlink(missing_ok=True)
        return

    yolo_result = yolo_results_list[0]

    yolo_speed_info = yolo_result.speed
    yolo_output_text.append(f"YOLO Results for: {Path(current_image_path).name}")
    # PERBAIKAN DI SINI: Gunakan orig_shape atau orig_img.shape
    yolo_output_text.append(f"Original image shape: {yolo_result.orig_shape} (H, W)")
    # Jika Anda ingin bentuk gambar yang mungkin telah di-resize untuk inferensi (meskipun orig_shape lebih standar untuk ditampilkan):
    # yolo_output_text.append(f"Processed image shape (from orig_img): {yolo_result.orig_img.shape}")
    yolo_output_text.append(f"Speed: {yolo_speed_info['preprocess']:.1f}ms preprocess, "
                            f"{yolo_speed_info['inference']:.1f}ms inference, "
                            f"{yolo_speed_info['postprocess']:.1f}ms postprocess per image.")

    # Dapatkan gambar hasil plot YOLO untuk ditampilkan di subplot
    # Metode .plot() mengembalikan array numpy dari gambar yang diplot
    yolo_plotted_image = yolo_result.plot() # Jangan tampilkan langsung, simpan arraynya

    detected_boxes_for_sam = []
    class_names_for_sam = []
    boxes = yolo_result.boxes
    num_yolo_detections_above_thresh = 0
    for box_obj in boxes:
        conf = box_obj.conf[0].cpu().numpy()
        if conf >= confidence_threshold:
            num_yolo_detections_above_thresh +=1
            xyxy = box_obj.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box_obj.cls[0].cpu().numpy())
            class_name = yolo_model_instance.names[cls_id]
            yolo_output_text.append(f"  - {class_name} (conf: {conf:.2f}) at [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            detected_boxes_for_sam.append(xyxy)
            class_names_for_sam.append(class_name)
    yolo_output_text.insert(2, f"Detections (conf >= {confidence_threshold}): {num_yolo_detections_above_thresh} objects")


    if not detected_boxes_for_sam:
        yolo_output_text.append("\nNo objects detected by YOLO with sufficient confidence for SAM prompting.")
        print("\n".join(yolo_output_text)) # Cetak semua info YOLO
        if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
             Path(current_image_path).unlink(missing_ok=True)
        return

    # 2. Segmentasi dengan SAM dari Ultralytics
    # sam_output_text.append(f"\nSAM Results (using {len(detected_boxes_for_sam)} YOLO prompts):")
    print(f"\nRunning Ultralytics SAM segmentation with {len(detected_boxes_for_sam)} bounding box prompts...")
    try:
        sam_results_list = sam_model_instance(current_image_path, bboxes=detected_boxes_for_sam, device=DEVICE)
    except Exception as e:
        sam_output_text.append(f"Error during SAM inference: {e}")
        print(sam_output_text[-1])
        if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
             Path(current_image_path).unlink(missing_ok=True)
        return

    if not sam_results_list:
        sam_output_text.append("No results from SAM segmentation.")
        print(sam_output_text[-1])
        if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
             Path(current_image_path).unlink(missing_ok=True)
        return

    sam_result_with_prompts = sam_results_list[0]
    sam_plotted_image = sam_result_with_prompts.plot() # Jangan tampilkan conf/label SAM, fokus pada mask

    sam_speed_info = sam_result_with_prompts.speed
    sam_output_text.append(f"SAM Results (using {len(detected_boxes_for_sam)} YOLO prompts):")
    # PERBAIKAN DI SINI: Gunakan orig_shape atau orig_img.shape
    sam_output_text.append(f"Original image shape: {sam_result_with_prompts.orig_shape} (H, W)")
    # sam_output_text.append(f"Processed image shape (from orig_img): {sam_result_with_prompts.orig_img.shape}")
    sam_output_text.append(f"Speed: {sam_speed_info['preprocess']:.1f}ms preprocess, "
                           f"{sam_speed_info['inference']:.1f}ms inference, "
                           f"{sam_speed_info['postprocess']:.1f}ms postprocess per image.")


    if sam_result_with_prompts.masks is not None:
        sam_output_text.append(f"Number of masks generated by SAM: {len(sam_result_with_prompts.masks.data)}")
        for i, mask_tensor in enumerate(sam_result_with_prompts.masks.data):
            sam_output_text.append(f"  - Mask for prompt {i+1} (YOLO class: {class_names_for_sam[i]}): shape {mask_tensor.shape}")
    else:
        sam_output_text.append("SAM did not produce any masks.")

    # --- Membuat Tampilan Berdampingan ---
    # fig, axs = plt.subplots(1, 2, figsize=(20, 10)) # 1 baris, 2 kolom
    # Ukuran default yang lebih kecil
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Atau ukuran yang lebih spesifik jika Anda tahu dimensi target di laporan
    # Misalnya, jika laporan Anda lebarnya sekitar 6 inci untuk dua gambar berdampingan
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # Sesuaikan sesuai kebutuhan

    # Subplot 1: Hasil YOLO
    axs[0].imshow(yolo_plotted_image)
    axs[0].axis('off')
    axs[0].set_title("YOLO Detections", fontsize=16)

    # Tambahkan teks informasi YOLO di bawah gambar YOLO
    # Kita akan menggunakan fig.text untuk menempatkan teks di luar sumbu subplot agar lebih rapi
    # Koordinat untuk fig.text adalah (x, y) dalam fraksi dari figur (0-1)

    # Subplot 2: Hasil SAM
    axs[1].imshow(sam_plotted_image)
    axs[1].axis('off')
    axs[1].set_title("SAM Segmentation (Prompted by YOLO)", fontsize=16)

    plt.tight_layout(rect=[0, 0.15, 1, 0.95]) # Sisakan ruang di bawah untuk teks

    # Menampilkan teks informasi di bawah plot
    fig.text(0.02, 0.20, "\n".join(yolo_output_text), ha='left', va='top', fontsize=10, wrap=True,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))
    fig.text(0.52, 0.20, "\n".join(sam_output_text), ha='left', va='top', fontsize=10, wrap=True,
             bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))

    plt.suptitle(f"YOLO + SAM Processing for: {Path(current_image_path).name}", fontsize=20, y=0.98)
    output_image_path = "hasil_yolo_sam.png"
    plt.savefig(output_image_path, dpi=360) # Coba dengan DPI 150 atau 200
    print(f"Gambar hasil disimpan di: {output_image_path}")
    plt.show()

    # Hapus gambar yang diunduh jika bersifat sementara
    if is_url and Path(current_image_path).name == "temp_image_for_processing.jpg":
        Path(current_image_path).unlink(missing_ok=True)
        print(f"Temporary image {current_image_path} removed.")

# --- Langkah 5: Jalankan Proses ---
# (Kode pemanggilan fungsi tetap sama)
if yolo_model and sam_model_ultralytics:
    # Ganti dengan path gambar lokal Anda yang valid
    if Path(local_image_path).exists():
         process_image_yolo_ultralytics_sam(local_image_path, yolo_model, sam_model_ultralytics, confidence_threshold=0.7, is_url=False)
    else:
        print(f"File gambar lokal '{local_image_path}' tidak ditemukan. Harap unggah terlebih dahulu.")
else:
    print("Skipping image processing as one or both models could not be loaded.")