#!/bin/bash

# Script untuk menjalankan Ollama via Singularity dan mengeksposnya menggunakan Cloudflared
# Pastikan Anda menjalankan script ini di dalam node Slurm yang telah di-booking (misal: @ai2 / @ai3)

: <<'CARA_PAKAI'

==============================================================================
PANDUAN llm API
==============================================================================
cd utils
chmod +x run_llm_api.sh

cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster && LOG_DIR=$(python3 -c "import config_shared, os; print(os.path.join(config_shared.WORKSPACE_DIR, 'logs'))") && mkdir -p "$LOG_DIR" && utils/run_llm_api.sh 2>&1 | tee "$LOG_DIR/run_llm_api.log"

./run_llm_api.sh
==============================================================================
CARA_PAKAI

OLLAMA_IMAGE="ollama.sif"
OLLAMA_PORT=11434

echo "======================================================"
echo "  Mempersiapkan Lingkungan Ollama + Cloudflared"
echo "======================================================"

# 1. Mengecek ketersediaan Port
echo "[1] Memeriksa ketersediaan Port $OLLAMA_PORT..."
# Kita menggunakan perintah 'ss' untuk mengecek port yang sedang listen
while ss -tuln | grep -q ":$OLLAMA_PORT "; do
    echo "    Port $OLLAMA_PORT sedang digunakan oleh proses lain."
    # Jika terpakai (misal oleh user lain), kita cari port acak antara 10000-60000
    OLLAMA_PORT=$(shuf -i 10000-60000 -n 1)
    echo "    Mencoba port alternatif: $OLLAMA_PORT..."
done
echo "    -> Menggunakan Port $OLLAMA_PORT yang berstatus KOSONG/OPEN."

# 2. Download Cloudflared
if [ ! -f "cloudflared" ]; then
    echo "[2] Mengunduh binary Cloudflared (Linux AMD64)..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
else
    echo "[2] Binary Cloudflared sudah tersedia."
fi

# 3. Tarik (Pull) Image Ollama via Singularity
if [ ! -f "$OLLAMA_IMAGE" ]; then
    echo "[3] Menarik container Ollama dari Docker Hub via Singularity..."
    singularity pull $OLLAMA_IMAGE docker://ollama/ollama:latest
else
    echo "[3] Container Ollama ($OLLAMA_IMAGE) sudah tersedia."
fi

# 4. Menjalankan Ollama Server di Background
echo "[4] Menjalankan Ollama Server di Port $OLLAMA_PORT (--nv untuk GPU)..."
# Mematikan proses Ollama milik kita sendiri yang mungkin masih menggantung
pkill -u $USER -f "ollama serve" || true

# Kita melempar variabel OLLAMA_HOST ke dalam container singularity menggunakan SINGULARITYENV_
export SINGULARITYENV_OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

nohup singularity exec --nv $OLLAMA_IMAGE ollama serve > ollama.log 2>&1 &

# Tunggu sampai port benar-benar terbuka (Ollama siap merespons)
echo "    Menunggu Ollama server siap merespons API..."
while ! nc -z 127.0.0.1 $OLLAMA_PORT; do
  sleep 1
done
echo "    -> Ollama Server RUNNING! Log dapat dilihat pada: ollama.log"

# 5. Mengekspos API via Cloudflared
echo "[5] Membuka Tunnel Publik via Cloudflare (Mengekspos Port $OLLAMA_PORT)..."
echo "    Silakan cari URL publik Anda (berakhiran .trycloudflare.com) pada log di bawah ini:"
echo "======================================================"

# Menjalankan Cloudflared di foreground agar log tunnel terlihat
./cloudflared tunnel --url http://127.0.0.1:$OLLAMA_PORT
