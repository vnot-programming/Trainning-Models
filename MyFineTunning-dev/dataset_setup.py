# -*- coding: utf-8 -*-
"""
dataset_setup.py
================
Pengelolaan dataset: cek ketersediaan lokal,
jika tidak ada maka unduh dari Roboflow secara otomatis.

Credential Loading — urutan prioritas:
    1. File .env di root project  (lokal / server)
    2. Environment variable OS    (Docker, CI/CD)

Fungsi publik:
    load_api_key() -> str | None
    setup_detection_dataset(key) -> (dataset_location, yaml_path)
    setup_segmentation_dataset(key) -> (dataset_location, yaml_path)
    setup_all_datasets() -> dict
"""

import os
from pathlib import Path

# Root project = direktori file ini
_PROJECT_ROOT = Path(__file__).resolve().parent
_KEY_NAME     = "ROBOFLOW_KU_KEY1"

# Roboflow project info
_DET_WORKSPACE  = "wbc-laboratory"
_DET_PROJECT    = "me-bottle-isempty-ku3"
_DET_VERSION    = 7
_DET_FORMAT     = "yolov11"

_SEG_WORKSPACE  = "wbc-laboratory"
_SEG_PROJECT    = "segpoligon-me-bottle-isempty3"
_SEG_VERSION    = 5
_SEG_FORMAT     = "yolov11"


# ==============================================================================
# DATASETS_DIR — dari config_shared (jika sudah diimport), atau fallback manual
# ==============================================================================
def _get_datasets_dir() -> Path:
    try:
        from config_shared import DATASETS_DIR
        return Path(DATASETS_DIR)
    except ImportError:
        return _PROJECT_ROOT / "datasets"


# ==============================================================================
# CREDENTIAL — .env / OS env var
# ==============================================================================
def _load_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # Manual parse jika python-dotenv tidak terinstall
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_api_key() -> str | None:
    """
    Muat Roboflow API key:
      1. File .env di root project
      2. Environment variable OS

    Returns: str API key, atau None jika tidak ditemukan.
    """
    _load_dotenv()

    key = os.environ.get(_KEY_NAME)
    if key:
        print(f"✅ Roboflow API Key dimuat ({_KEY_NAME}).")
        return key

    print(
        f"⚠️  API Key '{_KEY_NAME}' tidak ditemukan.\n"
        f"   Buat file .env di root project:\n"
        f"   {_KEY_NAME}=your_key_here\n"
        f"   Atau: export {_KEY_NAME}=your_key_here"
    )
    return None


# ==============================================================================
# DATASET DETEKSI
# ==============================================================================
def setup_detection_dataset(key: str | None) -> tuple[str, str]:
    """
    Siapkan dataset deteksi (YOLOv11 format).
    Download ke DATASETS_DIR jika belum ada.

    Returns: (dataset_location, yaml_path)
    """
    datasets_dir = _get_datasets_dir()
    # Cek apakah dataset sudah ada (cari folder yang mengandung data.yaml)
    for entry in datasets_dir.iterdir() if datasets_dir.exists() else []:
        if entry.is_dir() and (entry / "data.yaml").exists():
            if _DET_PROJECT.split("-ku")[0] in entry.name.lower() or "isempty" in entry.name.lower():
                if "seg" not in entry.name.lower():
                    print(f"\n📁 Dataset Deteksi: ditemukan lokal → {entry}")
                    return str(entry), str(entry / "data.yaml")

    print(f"\n🌐 Dataset Deteksi: belum ada. Download dari Roboflow...")
    if not key:
        raise RuntimeError(
            f"❌ API Key tidak ditemukan. Set {_KEY_NAME} di .env atau environment variable."
        )

    try:
        from roboflow import Roboflow
        rf      = Roboflow(api_key=key)
        project = rf.workspace(_DET_WORKSPACE).project(_DET_PROJECT)
        dataset = project.version(_DET_VERSION).download(
            _DET_FORMAT,
            location=str(datasets_dir / f"{_DET_PROJECT}-{_DET_VERSION}"),
            overwrite=False,
        )
        location = dataset.location
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset deteksi: {e}") from e

    yaml_path = os.path.join(location, "data.yaml")
    print(f"   → {location}")
    return location, yaml_path


# ==============================================================================
# DATASET SEGMENTASI
# ==============================================================================
def setup_segmentation_dataset(key: str | None) -> tuple[str, str]:
    """
    Siapkan dataset segmentasi (YOLOv11 polygon format).
    Download ke DATASETS_DIR jika belum ada.

    Returns: (dataset_location, yaml_path)
    """
    datasets_dir = _get_datasets_dir()
    # Cek apakah dataset segmentasi sudah ada
    for entry in datasets_dir.iterdir() if datasets_dir.exists() else []:
        if entry.is_dir() and (entry / "data.yaml").exists():
            if "seg" in entry.name.lower() and "isempty" in entry.name.lower():
                print(f"\n📁 Dataset Segmentasi: ditemukan lokal → {entry}")
                return str(entry), str(entry / "data.yaml")

    print(f"\n🌐 Dataset Segmentasi: belum ada. Download dari Roboflow...")
    if not key:
        raise RuntimeError(
            f"❌ API Key tidak ditemukan. Set {_KEY_NAME} di .env atau environment variable."
        )

    try:
        from roboflow import Roboflow
        rf      = Roboflow(api_key=key)
        project = rf.workspace(_SEG_WORKSPACE).project(_SEG_PROJECT)
        dataset = project.version(_SEG_VERSION).download(
            _SEG_FORMAT,
            location=str(datasets_dir / f"{_SEG_PROJECT}-{_SEG_VERSION}"),
            overwrite=False,
        )
        location = dataset.location
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset segmentasi: {e}") from e

    yaml_path = os.path.join(location, "data.yaml")
    print(f"   → {location}")
    return location, yaml_path


# ==============================================================================
# SETUP SEMUA DATASET
# ==============================================================================
def setup_all_datasets() -> dict:
    """
    Load API key dan siapkan kedua dataset sekaligus.

    Returns: dict {det_location, det_yaml, seg_location, seg_yaml, key}
    """
    key = load_api_key()

    det_location, det_yaml = setup_detection_dataset(key)
    seg_location, seg_yaml = setup_segmentation_dataset(key)

    print(f"\n[Dataset] ✅ Deteksi    : {det_location}")
    print(f"[Dataset] ✅ Segmentasi : {seg_location}")

    return {
        "det_location": det_location,
        "det_yaml":     det_yaml,
        "seg_location": seg_location,
        "seg_yaml":     seg_yaml,
        "key":          key,
    }


if __name__ == "__main__":
    # Tes langsung
    print("=" * 60)
    print("  dataset_setup.py — Test Mode")
    print("=" * 60)
    result = setup_all_datasets()
    print("\n[Hasil]")
    for k, v in result.items():
        if k != "key":
            print(f"  {k}: {v}")
