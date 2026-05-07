# Repository Guidelines - Trainning-Models

Panduan untuk agents yang bekerja di workspace Trainning-Models.

## Workspace Overview

Workspace ini berisi proyek **Computer Vision Training & Inference** dengan komponen:
- **MyFineTunning-dev/** - Pipeline pelatihan hybrid (YOLO, Mask R-CNN)
- **yolo/** - Implementasi YOLOv8, YOLOv9, YOLO11
- **mask-r-cnn/** - Implementasi Mask R-CNN dengan multi-GPU
- **hybrid/** - Arsitektur hybrid detection + segmentation
- **datasets/** - Dataset dari Roboflow (me-bottle-isempty)
- **models/** - Pre-trained weights (YOLOv8m, YOLOv9m, YOLO11m, dll)

## Agents Tersedia

### 1. CV Research Collaborator — Scopus Q1/Q2
- **File**: `.agents/cv-research-collaborator.agent.md`
- **Gunakan untuk**: Diskusi riset CV, review arsitektur model, evaluasi metodologi, draft jurnal
- **Bahasa**: Indonesia akademik (switch ke English untuk submission)
- **Ciri**: Brutal honesty, tantang asumsi, fokus Scopus Q1/Q2

### 2. Bio-Digital Design Reviewer
- **File**: `.agents/pro-design-review.agent.md`
- **Gunakan untuk**: Review UI/UX untuk kepatuhan Bio-Digital Minimalism 2026
- **Cek**: Typography (Inter/Outfit/Lora), 60fps animations, WCAG 2.2+, glassmorphism

## Rules & Instructions

### Global Instructions
- **Bio-Digital Minimalism 2026**: `.instructions/bio-digital-minimalism-2026.instructions.md`
  - Wajib untuk semua pembuatan UI/UX
  - 60fps animations (transform/opacity only)
  - Circadian-sync colors, WCAG 2.2+, essentialism

### Rules (di `.agents/rules/`)
1. **coding-standards.md** - Clean Code, DRY, KISS, Early Returns
2. **network-topology.md** - Topologi jaringan
3. **ui-ux-biodigital.md** - Bio-Digital UI/UX principles

## Skills Tersedia (di `.prompts/skills/`)

1. **pro-advanced-ui-ux** - Google UI/UX Expert, Bio-Digital Minimalism
2. **pro-circadian-js** - Circadian-sync color utility
3. **pro-performance-auditor** - Lighthouse audit, 60fps compliance, WCAG contrast

## Before Making Changes

Agents HARUS review:
1. **REVIEW.md** (jika ada) - Prioritas dan follow-up
2. **README.md** - Behavior dan commands yang diharapkan
3. **requirements.txt** / **setup.py** - Dependencies

## Project Structure

```
Trainning-Models/
├── MyFineTunning-dev/      # Main training pipeline
│   ├── main.py            # Entry point
│   ├── run_pipeline.py    # Pipeline orchestrator
│   ├── dataset_setup.py   # Dataset preparation
│   └── requirements.txt  # Python dependencies
├── yolo/                   # YOLO implementations
│   ├── yolo8/main.py
│   ├── yolo9/main.py
│   └── yolo11/main.py
├── mask-r-cnn/            # Mask R-CNN implementation
│   ├── train_multigpu.py
│   └── maskrcnn_trainer.py
├── hybrid/                # Hybrid architecture
│   └── main.py
├── datasets/              # Roboflow datasets
│   ├── me-bottle-isempty-ku3-8/
│   └── segpoligon-me-bottle-isempty3-7/
├── models/                # Pre-trained weights
└── data-files/            # Training outputs & reports
```

## Build & Run Commands

### Python Environment
```bash
cd MyFineTunning-dev
pip install -r requirements.txt
python main.py --help
```

### Training Pipeline
```bash
python run_pipeline.py --config config_shared.py
```

### GPU Fan Management
```bash
python gpu_fan_manager.py --set 70
```

## Toolchain

- **Python 3.8+** dengan CUDA untuk training
- **PyTorch** untuk Mask R-CNN
- **Ultralytics YOLO** untuk YOLOv8/v9/11
- **Roboflow** untuk dataset management
- **Telegram Bot** untuk notifikasi training

## Mandatory Verification

- Setelah perubahan kode Python: Jalankan `python -m py_compile <file>` untuk cek syntax
- Setelah perubahan konfigurasi: Review `config_shared.py`
- Untuk perubahan UI/UX: Gunakan **Bio-Digital Design Reviewer** agent
- Untuk diskusi riset: Gunakan **CV Research Collaborator** agent

## File Access Rules

**BACA & IKUTI** file `.agentignore` di root workspace. File yang terdaftar di sana **DILARANG** untuk dibaca atau diakses oleh agents.

## Dependency Policy

- Simpan `requirements.txt` di git
- Untuk perubahan dependencies: Update `requirements.txt` dan commit
- Hindari `pip install` manual tanpa update `requirements.txt`

## User-Facing Change Sync

- Saat menambah/mengubah commands atau settings, update `README.md`
- Untuk perubahan behavior yang terlihat user, buat catatan di `release_notes.md`
