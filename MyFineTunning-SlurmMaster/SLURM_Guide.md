# Panduan Pintar Booking GPU (Slurm)

> **⚠️ ATURAN WAJIB (MANDATORY RULE): DILARANG MENGEKSEKUSI SCRIPT PYTHON SECARA LANGSUNG DI LOGIN NODE!**
> Semua eksekusi script `python` (sekalipun hanya untuk *setup* atau mengunduh model seperti `config_shared.py`) **WAJIB** dilakukan melalui Slurm. 
> AI Agent maupun Manusia dilarang keras memicu `python <script>.py` secara langsung di terminal utama. Selalu mulai interaksi menggunakan menu `/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh` atau pastikan Anda sudah masuk ke node GPU via `attach_gpu.sh` sebelum menjalankan *script* apa pun.
Dokumen ini menjelaskan cara menggunakan fitur **Smart GPU Booking** untuk melakukan iterasi *development* & *debugging* tanpa harus mengantre Slurm berulang-ulang setiap kali menjalankan skrip.

## Membuka Menu Interaktif:
```bash
/data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils/myslurm.sh
```

## Daftar Skrip
- `book_gpu.py`: Submitter pintar yang mengalokasikan GPU, memfilter node yang error/DRAIN secara otomatis, dan memonitor status lewat Telegram.
- `attach_gpu.sh`: Helper bash script yang akan meluncurkan `srun` agar Anda seketika (instant) berada di dalam node komputasi yang Anda pegang (hold).

## Alur Kerja (Workflow)

### 1. Booking GPU
Anda hanya perlu menjalankan *booking daemon* ini **SATU KALI SAJA**. 
Script ini akan mem-booking GPU hingga Anda membatalkannya sendiri (lewat `scancel`).

Jika Anda ingin menutup terminal Anda tanpa menghentikan monitoring Telegram, gunakan *tmux*:
```bash
tmux new-session -d -s booking "source /data/programs/anaconda3/bin/activate && conda activate yolo_env && cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils && python book_gpu.py"
```

Atau jalankan secara langsung:
```bash
cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
python book_gpu.py
```
> **Note:** Anda akan mendapat pesan Telegram jika job Anda sudah masuk fase **Running**.

### 2. Masuk / Attach ke Node
Ketika notifikasi Telegram menunjukkan "GPU Booking Active!" atau statusnya `R` di `squeue`, sambungkan terminal interaktif Anda ke node tersebut:

```bash
cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
./attach_gpu.sh
```

**Hasilnya:**
Anda akan seketika berada di dalam shell bash node komputasi (misal: `@ai2` atau `@ai3`) **lengkap dengan akses GPU**.

### 3. Iterasi Pengembangan Tanpa Antre
1. Aktifkan *conda environment* di shell node tersebut:
   ```bash
   conda activate yolo_env
   ```
2. Jalankan pengujian atau eksperimen Anda:
   ```bash
   python main.py
   ```
3. Terjadi *error*? **Biarkan terminal Anda tetap hidup.**
4. Edit skrip Anda (di Editor / VSCode).
5. Ulangi eksekusi skrip python tersebut. Tidak akan ada antrean lagi karena GPU tersebut masih "disewa" oleh sesi `book_gpu` Anda.

### 4. Mengakhiri Sesi & Melepaskan GPU
- Jika Anda mengetik `exit` di `attach_gpu.sh`, **GPU masih Anda tahan** dan belum kembali ke server. Anda bisa melakukan `./attach_gpu.sh` lagi nanti.
- Jika eksperimen Anda **telah tuntas** dan Anda ingin mengembalikan GPU, Anda wajib membatalkannya dari Slurm:
  ```bash
  scancel <JOBID_ANDA>
  ```
  *(Cek Job ID Anda menggunakan `squeue -u $USER` atau cek dari notifikasi Telegram Anda).*
