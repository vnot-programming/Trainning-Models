# Release Notes: `MyFineTunning_20260505_000000`

## 🏷️ Version Details
- **Tag:** `v0.1.1`
- **Release Title:** `MyFineTunning_20260505_000000`
- **Date:** May 5, 2026
- **Status:** ✅ Release (Stable)

## 📝 Summary
This release includes major updates to the training infrastructure with hybrid approach integration, custom upload functionality, and codebase cleanup.

## 🚀 What's Changed
- **Hybrid Approach**: Added hybrid model implementation (YOLO11 + SAM2) with documentation and performance metrics.
- **Custom Upload**: New `custom_upload.py` module for flexible model upload workflows.
- **Code Cleanup**: Removed unnecessary cache files, logs, and deprecated scripts from mask-r-cnn module.
- **RClone Integration**: Added `rclone.conf.example` for cloud storage synchronization.
- **Updated Core Modules**:
  - `config_shared.py`: Enhanced configuration management.
  - `main.py`: Improved main training pipeline.
  - `run_pipeline.py`: Updated pipeline orchestration.
  - YOLO variants (yolo8, yolo9, yolo11): Synchronized updates across all versions.
- **Documentation**: Updated README with current project structure and workflow guidelines.

## 📊 New Features
- Hybrid model performance tracking (latency and mAP metrics).
- Enhanced training pipeline with better error handling.
- Streamlined project structure without redundant files.

## ⚠️ Notes for Developers
- This is a **Stable Release** version.
- All major features have been tested and validated.
- Please report any issues via GitHub Issues.

---
**GitHub Release Link:**
`https://github.com/vnot-programming/Trainning-Models/releases/tag/v0.1.1`
