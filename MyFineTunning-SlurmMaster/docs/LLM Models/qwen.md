# 🤖 Analisis Komprehensif Ekosistem Model Qwen (Hingga Mei 2026)

## 1. Pendahuluan & Latar Belakang

**Qwen** (singkatan dari **Tongyi Qianwen**, dikembangkan oleh tim Alibaba Cloud / DAMO Academy) telah memantapkan dirinya sebagai salah satu keluarga Model Bahasa Besar (LLM) open-source paling berpengaruh secara global. Sejak rilis pertamanya, Qwen menonjol karena kemampuannya yang sangat kuat dalam tugas-tugas multibahasa (terutama Inggris, Mandarin, dan bahasa Asia Tenggara seperti Indonesia), penalaran matematika, pemecahan masalah coding, serta pemrosesan multimodal.

Hingga **Mei 2026**, ekosistem Qwen telah matang melalui beberapa iterasi generasi utama (Qwen 1.0, Qwen 1.5, Qwen 2.0, Qwen 2.5, dan eksperimen penalaran khusus QwQ). Karakteristik utama yang membuat keluarga model Qwen sangat diminati oleh para peneliti dan praktisi adalah performa tingkat komersial (commercial-grade) yang dapat dijalankan secara lokal dengan efisiensi komputasi yang tinggi.

---

## 2. Peta Evolusi & Silsilah Model Qwen

Evolusi Qwen dapat dipetakan ke dalam beberapa fase penting sebagai berikut:

### A. Qwen (Generasi Pertama)
*   **Varian Utama:** Qwen-7B, Qwen-14B, Qwen-72B.
*   **Karakteristik:** Menggunakan arsitektur Transformer decoder-only dasar. Memperkenalkan tokenizer dengan kosakata yang sangat besar (151.936 token), yang sangat efisien untuk memproses teks non-Inggris.
*   **Keterbatasan:** Jendela konteks yang masih terbatas (8k token) dan belum terintegrasi secara bawaan di library Hugging Face Transformers (memerlukan opsi `trust_remote_code=True`).

### B. Qwen1.5 (Generasi Transisi)
*   **Varian Utama:** 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B, serta model Mixture-of-Experts (MoE-A2.7B).
*   **Karakteristik:** Integrasi resmi ke dalam kode inti Hugging Face Transformers (`Qwen2` model class). Peningkatan stabilitas pelatihan, perluasan jendela konteks dasar hingga 32k token, dan performa instruksi yang lebih mulus.

### C. Qwen2 (Generasi Lompatan Performa)
*   **Varian Utama:** 0.5B, 1.5B, 7B, 57B-A14B (MoE), dan 72B.
*   **Karakteristik:** Memperkenalkan **Grouped-Query Attention (GQA)** untuk efisiensi memori pada model menengah ke atas (mulai dari 7B). Konteks window ditingkatkan secara drastis hingga **128k token**. Performa matematika, coding, dan pemahaman instruksi multibahasa mengalami peningkatan masif yang menyaingi Llama-3.

### D. Qwen2.5 (Generasi Kematangan & Spesialisasi)
Dirilis pada akhir 2024 dan terus disempurnakan hingga awal 2026. Merupakan versi paling matang dan standar industri saat ini:
*   **Varian General:** 0.5B, 1.5B, 3B, 7B, 14B, 32B, dan 72B.
*   **Qwen2.5-Coder (0.5B, 1.5B, 3B, 7B, 14B, 32B):** Seri model khusus pemrograman yang sangat dominan di dunia open-source. Varian 32B-Instruct mendekati kemampuan GPT-4o dalam evaluasi HumanEval dan tugas rekayasa perangkat lunak nyata.
*   **Qwen2.5-Math (1.5B, 7B, 72B):** Dioptimalkan secara khusus untuk menyelesaikan persoalan matematika tingkat sekolah hingga kompetisi olimpiade menggunakan teknik penyelarasan khusus dan penalaran multi-langkah.
*   **Qwen2.5-VL (Vision-Language):** Model pengolah gambar dan video yang canggih, mendukung resolusi dinamis, pemahaman dokumen (OCR tingkat lanjut), deteksi objek spasial (grounding), dan analisis konten video dinamis berbasis waktu.

### E. QwQ (Seri Penalaran / Reasoning)
*   **Varian Utama:** QwQ-32B-Preview (dan model inkrementalnya hingga 2026).
*   **Karakteristik:** Seri model khusus yang dirancang untuk mensimulasikan proses berpikir mendalam (*deep thinking/reasoning*) dengan memicu *Chain-of-Thought* (CoT) sebelum memberikan jawaban akhir. Sangat unggul dalam matematika teoretis, logika kompleks, dan debugging kode pemrograman yang rumit.

---

## 3. Matriks Spesifikasi Teknis & Analisis Kebutuhan Resource

Berikut adalah tabel spesifikasi teknis dan analisis kebutuhan VRAM untuk menjalankan model seri **Qwen2.5 / QwQ** pada fase inferensi:

| Nama Model | Jumlah Parameter | Arsitektur Attention | Konteks Default | VRAM (FP16 / BF16) | VRAM Kuantisasi (INT4 / GGUF) | Skenario Rekomendasi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B** | ~490M | MHA | 32k / 128k | ~1.2 GB | ~0.4 GB | Edge devices, IoT, klasifikasi teks super cepat |
| **Qwen2.5-1.5B** | ~1.54B | MHA | 32k / 128k | ~3.5 GB | ~1.2 GB | Perangkat mobile, ekstraksi entitas, RAG ringan |
| **Qwen2.5-3B** | ~3.09B | GQA | 32k / 128k | ~7.0 GB | ~2.2 GB | Chatbot lokal ringan, asisten pribadi pada PC |
| **Qwen2.5-7B** | ~7.61B | GQA | 128k | ~16.0 GB | ~5.5 GB | RAG standar perusahaan, agen mandiri multi-fungsi |
| **Qwen2.5-14B** | ~14.7B | GQA | 128k | ~30.0 GB | ~9.5 GB | Terjemahan dokumen tebal, analisis teks kompleks |
| **Qwen2.5-32B** | ~32.5B | GQA | 128k | ~68.0 GB | **~20.5 GB** | **Pemrograman advanced (Coder), penalaran tingkat tinggi** |
| **QwQ-32B** | ~32.5B | GQA | 32k / 128k | ~68.0 GB | **~21.5 GB** | **Pemecahan logika kompleks, riset ilmiah, coding debug** |
| **Qwen2.5-72B** | ~72.7B | GQA | 128k | ~148.0 GB | ~45.0 GB | SOTA lokal penuh, membutuhkan multi-GPU (2x V100/A100) |

> [!NOTE]
> Perhitungan VRAM di atas adalah estimasi untuk inferensi dengan panjang konteks sedang (~4k-8k token). Penggunaan context window penuh hingga 128k token akan secara signifikan meningkatkan konsumsi VRAM karena ukuran alokasi *KV Cache*.

---

## 4. Evaluasi Kelayakan pada Server Slurm AI_KU_V100 (V100 32GB VRAM)

Berdasarkan batasan infrastruktur yang dimiliki pengguna:
*   **GPU:** Tesla V100 32GB VRAM (Single GPU per Job).
*   **RAM Node:** Max 64GB (setelah ditingkatkan di `book_gpu.py`).
*   **CPU Cores:** Max 8 Cores.
*   **QoS Slurm:** `normal` / `gpu` (1 GPU max).

Berikut adalah analisis kelayakan deployment seri model Qwen pada hardware ini:

### A. Varian Model 7B & 14B (Sangat Direkomendasikan & Cepat)
*   **Qwen2.5-7B-Instruct / Coder:** Dapat dijalankan tanpa kuantisasi (FP16/BF16) dengan performa sangat cepat (latency rendah, throughput tinggi). Konsumsi VRAM sekitar ~16GB, menyisakan ruang VRAM yang sangat luas untuk context window panjang.
*   **Qwen2.5-14B-Instruct / Coder:** Dapat berjalan pada FP16 (~30GB VRAM), namun sangat berisiko Out-Of-Memory (OOM) jika context window melebihi 2k-4k token. Direkomendasikan dijalankan menggunakan kuantisasi **INT8 (~12GB VRAM)** atau **INT4 (~9.5GB VRAM)** untuk kestabilan penuh di bawah beban kerja berat.

### B. Varian Model 32B (Sweet Spot untuk Kemampuan Penalaran Tinggi)
*   Model **Qwen2.5-32B-Instruct**, **Qwen2.5-Coder-32B-Instruct**, dan **QwQ-32B-Preview** adalah model terbaik untuk dijalankan di GPU V100 32GB Anda.
*   **Strategi Kuantisasi:** Model-model ini tidak akan muat di VRAM 32GB jika dijalankan pada format FP16 mentah (butuh >65GB VRAM). Namun, jika dikuantisasi ke **INT4 (format GGUF atau GPTQ/AWQ)**, kebutuhan VRAM turun menjadi **~20.5 GB hingga ~22 GB**.
*   **Sisa VRAM (~10 GB):** Ruang sisa ini sangat ideal dan aman untuk menangani ekspansi *KV Cache* hingga panjang konteks sekitar 8k - 16k token tanpa memicu OOM.

### C. Varian Model 72B (Tidak Layak untuk GPU Tunggal)
*   Model 72B membutuhkan minimal ~45GB VRAM pada kuantisasi INT4 yang paling agresif sekalipun.
*   Menjalankan model 72B pada satu GPU Tesla V100 32GB akan memicu transfer memori ke RAM sistem (CPU offloading), yang akan merusak throughput (kecepatan generasi turun drastis menjadi <1 token per detik, tidak praktis untuk digunakan).

---

## 5. Panduan Praktis Deployment Qwen2.5-32B / QwQ-32B via Ollama & Singularity

Untuk memanfaatkannya di klaster Slurm Anda dengan tunnel aman Cloudflared, Anda dapat menerapkan workflow berikut yang sepenuhnya berjalan di ruang user (tanpa akses root).

### A. Persiapan Script Start Server (`run_llm_api.sh`)
Pastikan container Singularity untuk Ollama telah diunduh atau buat skrip untuk memicunya. Di bawah ini adalah contoh logika integrasi:

```bash
#!/bin/bash
# run_llm_api.sh - Menjalankan Ollama + Cloudflared di Node GPU Slurm

# 1. Konfigurasi Port Dinamis (Mencegah konflik port di node komputasi)
OLLAMA_PORT=$(shuf -i 18000-18999 -n 1)
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_MODELS="/data/users/g6717500336/.ollama/models"

echo "[INFO] Menjalankan Ollama pada port: ${OLLAMA_PORT}"

# 2. Jalankan Ollama Server menggunakan Singularity di latar belakang
# Pastikan Anda memetakan folder data ollama ke area disk Anda agar model tidak terhapus
singularity exec --nv \
    --bind /data/users/g6717500336/.ollama:/root/.ollama \
    /data/users/g6717500336/containers/ollama.sif \
    ollama serve > logs/ollama_server.log 2>&1 &

# Tunggu Ollama Server melakukan inisialisasi (~5 detik)
sleep 5

# 3. Unduh model Qwen/QwQ pilihan
# Contoh: Mengunduh Qwen2.5-Coder 32B dalam format kuantisasi default (biasanya Q4_K_M)
echo "[INFO] Menarik model qwen2.5-coder:32b..."
singularity exec --nv \
    --bind /data/users/g6717500336/.ollama:/root/.ollama \
    /data/users/g6717500336/containers/ollama.sif \
    ollama run qwen2.5-coder:32b "Halo" > /dev/null

# 4. Ekspos Port via Cloudflared Tunnel
# Menggunakan cloudflared binary lokal yang dijalankan di node komputasi
echo "[INFO] Memulai eksposur Cloudflared Tunnel..."
/data/users/g6717500336/programs/cloudflared tunnel --url http://127.0.0.1:${OLLAMA_PORT} > logs/tunnel.log 2>&1 &

echo "[SUCCESS] Pipeline LLM API aktif!"
```

### B. Cara Attach & Memantau Proses via TMUX (Sesuai Aturan 3.1)
1.  Buka sesi TMUX baru di Login Node:
    ```bash
    tmux new-session -s qwen_service
    ```
2.  Masuk ke Node GPU yang telah dialokasikan (misalnya `ai2`/`ai3`):
    ```bash
    cd /data/users/g6717500336/Trainning-Models/MyFineTunning-SlurmMaster/utils
    ./attach_gpu.sh
    ```
3.  Jalankan skrip start server LLM:
    ```bash
    bash run_llm_api.sh
    ```
4.  Detakkan TMUX menggunakan shortcut `Ctrl+b` lalu `d` (atau `control+b` -> `d`). Sesi server LLM Anda akan tetap berjalan aman di background klaster Slurm selama durasi booking GPU Anda aktif.
5.  Untuk mematikan job Slurm secara bersih setelah selesai eksperimen:
    ```bash
    squeue -u $USER
    scancel <JOBID>
    ```

---

## 6. Ringkasan Rekomendasi Riset & Aplikasi Praktis

Untuk riset SOTA (Scopus Q1/Q2) yang sedang Anda lakukan di klaster Slurm ini, berikut adalah peta jalan pemanfaatan model Qwen:

1.  **Jika Tugas Anda adalah Rekayasa Perangkat Lunak & Otomasi Kode:**
    Gunakan **`qwen2.5-coder:32b`**. Model ini memiliki performa setara GPT-4o untuk pemahaman arsitektur kode, penulisan kode modular, dan pembuatan visualisasi data dari file CSV hasil eksperimen YOLO Anda.
2.  **Jika Tugas Anda adalah Evaluasi Logika / Analisis Ilmiah Tingkat Lanjut:**
    Gunakan **`qwq:32b`** (atau model penalaran sejenis yang didukung Ollama). Model ini akan melakukan dekonstruksi langkah-demi-langkah (CoT) sebelum menyajikan analisis statistik mAP model YOLO/SAM Anda, memberikan interpretasi hasil riset yang jauh lebih kaya dan akurat secara ilmiah dibandingkan model LLM standar.
3.  **Jika Anda memerlukan parsing visualisasi kualitatif (Gambar Evaluasi Bounding Box/Mask):**
    Gunakan **`qwen2.5-vl:7b`** (atau varian yang lebih tinggi jika tersedia dalam kuantisasi yang efisien). Model ini sangat berguna untuk melakukan visual QA terhadap hasil prediksi bounding box atau mask segmentasi YOLO/SAM Anda, mendeteksi secara kualitatif area di mana model YOLO mengalami *false positive* atau *miss-detection*.

---

## 7. Analisis Rilis Model Terbaru & Spezifikasi (Qwen3, Qwen3.6/3.7, & DeepSeek V4/V3.2)

Berdasarkan dokumentasi resmi Alibaba Cloud Model Studio dan rilis komunitas hingga **Mei 2026**, industri LLM telah kedatangan model-model mutakhir dari seri **Qwen3 (dan lanjutannya Qwen3.6/3.7)** serta seri **DeepSeek-V4 & V3.2**:

### A. Rilis Baru Keluarga Model Qwen (Generasi Qwen3, Qwen3.5, Qwen3.6 & Qwen3.7)
Keluarga model Qwen kini tidak hanya fokus pada peningkatan parameter dasar, tetapi juga memperkenalkan mode penalaran hybrid (*Thinking* vs *Non-thinking*) serta optimasi *Agentic Workflows* yang mendalam.

1.  **Qwen3.7-Plus:**
    *   **Deskripsi:** Dirilis pada akhir **Mei 2026**. Merupakan model agen multimodal interaktif (interactive agent) komersial utama yang dioptimalkan untuk pengoperasian GUI/CLI.
    *   **Fitur Utama:** Sangat kuat dalam penggunaan alat (*tool use*), penulisan kode modular, dan tugas-tugas otonom jangka panjang (*long-horizon execution*).
2.  **qwen3-vl-plus & qwen3-vl-flash:**
    *   **Deskripsi:** Model dari seri **Qwen3-VL** (Vision-Language). Varian **`plus`** merupakan model dengan performa tinggi pada tugas-tugas visual agent (navigasi GUI OS, analisis dokumen kompleks), sedangkan varian **`flash`** dioptimalkan untuk inferensi visual real-time berlatensi rendah.
3.  **Qwen3-VL-30B-A3B-Thinking:**
    *   **Deskripsi:** Model sparse **Mixture-of-Experts (MoE)** vision-language yang sangat revolusioner. Memiliki total parameter sebesar 30B, namun hanya menggunakan **3B active parameter** per token saat dijalankan.
    *   **Fitur Utama:** Varian **"Thinking"** ini memicu proses *Chain-of-Thought* (CoT) visual untuk memecahkan persoalan sains, matematika spasial, serta rekayasa kode berbasis input citra. Memiliki context window bawaan 256K hingga 1M token.
4.  **qwen3-vl-8b-thinking:**
    *   **Deskripsi:** Varian model *Visual Reasoning* yang lebih ringan (8B parameter) namun tetap dibekali kemampuan penalaran CoT visual untuk tugas-tugas yang membutuhkan efisiensi memori.
5.  **Qwen3.5-Omni-Flash:**
    *   **Deskripsi:** Bagian dari keluarga Qwen3.5-Omni yang dirancang untuk interaksi multimodal real-time berlatensi super rendah (mendukung input audio, gambar, video, dan teks dengan output audio/teks langsung secara native).
6.  **Qwen3.6-27B & Qwen3.6-Flash:**
    *   **Deskripsi:** Dirilis pada **April 2026**. Seri Qwen3.6 memperkenalkan fitur **Thinking Preservation** yang mampu mempertahankan konteks penalaran internal LLM di seluruh giliran percakapan (*multi-turn dialog*). Varian 27B menjadi model komputasi modular yang sangat unggul untuk tugas *Repository-level Coding*.
7.  **qwen-flash & qwen3.6-flash:**
    *   **Deskripsi:** Model high-throughput berbiaya rendah di DashScope API, dirancang untuk tugas penyimpulan dan RAG cepat dengan volume data besar.

### B. Rilis Baru Keluarga Model DeepSeek (Seri V4 & V3.2)
DeepSeek terus mendominasi pasar model open-weights berkinerja tinggi dengan merilis iterasi arsitektur baru:

1.  **DeepSeek-V3.2 (Desember 2025):**
    *   **Deskripsi:** Iterasi penyempurnaan dari V3 yang memperkenalkan **DeepSeek Sparse Attention (DSA)** untuk meningkatkan efisiensi komputasi pada jendela konteks 128K token, sangat menonjol di kompetisi matematika dan pemrograman (IMO/IOI).
2.  **DeepSeek-V4-Pro & DeepSeek-V4-Flash (April 2026):**
    *   **Deskripsi:** Arsitektur MoE generasi ke-4 dari DeepSeek yang mendukung **1 Juta Token Context Window** secara native.
    *   **DeepSeek-V4-Pro:** Memiliki total parameter sebesar 1.6T (49B active parameter per token). Dioptimalkan khusus untuk penalaran tingkat tinggi, manipulasi logika tingkat lanjut, dan rekayasa perangkat lunak skala besar.
    *   **DeepSeek-V4-Flash:** Versi efisien dengan total parameter 284B (13B active parameter per token). Dirancang untuk inferensi berkecepatan tinggi dengan biaya minimal namun tetap mempertahankan performa penalaran yang mendekati versi Pro.

---

## 8. Panduan Kelayakan Deployment Model Baru di GPU V100 32GB (Slurm)

Untuk infrastruktur riset Anda (1x Tesla V100 32GB VRAM, Max 64GB RAM):

1.  **Model yang Direkomendasikan Secara Lokal:**
    *   **`qwen3-vl-8b-thinking`** (Format GGUF/INT4): Sangat cocok untuk tugas RAG visual atau analisis citra kualitatif hasil eksperimen YOLO Anda.
    *   **`qwen3-vl-30b-a3b-thinking`** (Format INT4): Meskipun memiliki total parameter 30B (membutuhkan ~20GB memori untuk weights-nya), model MoE ini sangat efisien saat runtime karena hanya 3B active parameter yang diproses, menjadikannya pilihan yang sangat cerdas untuk penalaran multimodal visual yang cepat pada V100 32GB.
    *   **`qwen3.6-27B`** (Format INT4): Sangat layak untuk rekayasa kode lokal yang canggih dengan konsumsi VRAM ~18GB.
2.  **Model yang Harus Diakses via API (DashScope / DeepSeek API):**
    *   Model raksasa seperti `DeepSeek-V4-Pro` (1.6T), `Qwen3.7-Plus`, dan `Qwen3.5-Omni-Flash` wajib diakses melalui layanan cloud API (menggunakan Cloudflared Tunnel sebagai penghubung aman ke aplikasi lokal Anda) karena kebutuhan VRAM-nya yang jauh melampaui kapasitas 1x GPU V100 32GB.
