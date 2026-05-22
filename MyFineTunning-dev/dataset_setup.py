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
    load_api_key() -> tuple[str | None, str | None]
    setup_detection_dataset(key) -> tuple[str, str]
    setup_segmentation_dataset(key) -> tuple[str, str]
    setup_coco_segmentation_dataset_unu(key_unu) -> str
    setup_all_datasets() -> dict

# Object Detection
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="F0VtV8b5YBdJHZbasy0w")
project = rf.workspace("wbc-laboratory").project("me-bottle-isempty-ku3")
version = project.version(8)
dataset = version.download("coco")

# Instance Segmentation
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="F0VtV8b5YBdJHZbasy0w")
project = rf.workspace("wbc-laboratory").project("segpoligon-me-bottle-isempty3")
version = project.version(7)
dataset = version.download("coco-segmentation")


# Golden Dataset Segmentation - For Evaluation
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tjeBcHkWc1oOc0oOv9kI")
project = rf.workspace("vnot").project("me-bottle-isempty-unu3-sem-seg")
dataset = project.version(1).download("coco-segmentation")

# Golden Dataset Detection - For Evaluation
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tjeBcHkWc1oOc0oOv9kI")
project = rf.workspace("vnot").project("me-bottle-isempty-unu3-sem-seg")
version = project.version(7)
dataset = version.download("coco-segmentation")

# Standar Dataset Detection - For Evaluation
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tjeBcHkWc1oOc0oOv9kI")
project = rf.workspace("vnot").project("me-bottle-isempty-ku3-h61lr")
version = project.version(2)
dataset = version.download("coco")

# Standar Dataset Segmentation - For Evaluation
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tjeBcHkWc1oOc0oOv9kI")
project = rf.workspace("vnot").project("me-bottle-isempty-ku3-h61lr-seg")
version = project.version(1)
dataset = version.download("coco-segmentation")
"""

import os
from pathlib import Path

# Root project = direktori file ini
_PROJECTROOT = Path(__file__).resolve().parent
_KEY_NAME     = "ROBOFLOW_KU_KEY1"
_KEY_NAME_UNU = "ROBOFLOW_UNU_KEY1"

# Roboflow project info
# _DET_WORKSPACE  = "wbc-laboratory"
# _DET_PROJECT    = "me-bottle-isempty-ku3"
# _DET_VERSION    = 7
# _DET_FORMAT     = "yolov11"

# _SEG_WORKSPACE  = "wbc-laboratory"
# _SEG_PROJECT    = "segpoligon-me-bottle-isempty3"
# _SEG_VERSION    = 5
# _SEG_FORMAT     = "yolov11"

_DET_WORKSPACE  = "wbc-laboratory"
_DET_PROJECT    = "me-bottle-isempty-ku3"
_DET_VERSION    = 8
_DET_FORMAT     = "yolov11"

_SEG_WORKSPACE  = "wbc-laboratory"
_SEG_PROJECT    = "segpoligon-me-bottle-isempty3"
_SEG_VERSION    = 7
_SEG_FORMAT     = "yolov11"


# ==============================================================================
# DATASETS_DIR — dari config_shared (jika sudah diimport), atau fallback manual
# ==============================================================================
def _get_datasets_dir() -> Path:
    try:
        from config_shared import DATASETS_DIR
        return Path(DATASETS_DIR)
    except ImportError:
        return _PROJECTROOT / "datasets"


# ==============================================================================
# CREDENTIAL — .env / OS env var
# ==============================================================================
def _load_dotenv() -> None:
    env_path = _PROJECTROOT / ".env"
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


def load_api_key() -> tuple[str | None, str | None]:
    """
    Muat Roboflow API key:
      1. File .env di root project
      2. Environment variable OS

    Returns: tuple(key_ku, key_unu), atau None pada masing-masing key jika tidak ditemukan.
    """
    _load_dotenv()

    key = os.environ.get(_KEY_NAME)
    key_unu = os.environ.get(_KEY_NAME_UNU)
    
    if key and key_unu:
        print(f"✅ Roboflow API Key dimuat ({_KEY_NAME}, {_KEY_NAME_UNU}).")
    else:
        if not key:
            print(f"⚠️  API Key '{_KEY_NAME}' tidak ditemukan.")
        if not key_unu:
            print(f"⚠️  API Key '{_KEY_NAME_UNU}' tidak ditemukan.")
            
    return key, key_unu


# ==============================================================================
# DATASET DETEKSI - Khusus Training
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
# DATASET SEGMENTASI - Khusus Training
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
# DATASET COCO SEGMENTASI (UNU) - Golden Dataset
# ==============================================================================
def setup_coco_segmentation_dataset_unu(key_unu: str | None) -> str:
    """
    Download dataset COCO segmentation dari workspace vnot.
    """
    import subprocess
    import sys
    
    # Auto-install roboflow setara dengan !pip install roboflow
    try:
        import roboflow
    except ImportError:
        print("📦 Menginstall roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])

    if not key_unu:
        raise RuntimeError(f"❌ API Key {_KEY_NAME_UNU} tidak ditemukan di .env")

    print(f"\n🌐 Dataset COCO Segmentasi (UNU): Download dari Roboflow...")
                
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key_unu)
        project = rf.workspace("vnot").project("me-bottle-isempty-unu3-sem-seg")
        
        datasets_dir = _get_datasets_dir()
        target_loc = str(datasets_dir / "golden_dataset_seg")
        
        dataset = project.version(7).download(
            "coco-segmentation",
            location=target_loc,
            overwrite=False
        )
        print(f"   → {dataset.location}")
        return dataset.location
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset COCO segmentasi UNU: {e}") from e


# ==============================================================================
# DATASET COCO DETECTION (UNU) - Golden Dataset
# ==============================================================================
def setup_coco_detection_dataset_unu(key_unu: str | None) -> str:
    """
    Download dataset COCO detection dari workspace vnot.
    Sesuai dengan snippet:
    !pip install roboflow
    key_unu = os.environ.get(_KEY_NAME_UNU)
    from roboflow import Roboflow
    rf = Roboflow(api_key=key_unu)
    project = rf.workspace("vnot").project("me-bottle-isempty-unu3-det")
    dataset = project.version(1).download("coco")
    """
    import subprocess
    import sys
    
    # Auto-install roboflow setara dengan !pip install roboflow
    try:
        import roboflow
    except ImportError:
        print("📦 Menginstall roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])

    if not key_unu:
        raise RuntimeError(f"❌ API Key {_KEY_NAME_UNU} tidak ditemukan di .env")

    print(f"\n🌐 Dataset COCO Detection (UNU): Download dari Roboflow...")
                
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key_unu)
        project = rf.workspace("vnot").project("me-bottle-isempty-unu3-det")
        
        datasets_dir = _get_datasets_dir()
        target_loc = str(datasets_dir / "golden_dataset_det")
        
        dataset = project.version(1).download(
            "coco",
            location=target_loc,
            overwrite=False
        )
        print(f"   → {dataset.location}")
        return dataset.location
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset COCO segmentasi UNU: {e}") from e

# ==============================================================================
# DATASET COCO DETECTION (UNU) - Standard Datasets
# ==============================================================================
def setup_h61lr_detection_dataset(key_unu: str | None) -> str:
    """
    Download dataset me-bottle-isempty-ku3-h61lr.
    """
    import subprocess
    import sys
    try:
        import roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])

    if not key_unu:
        raise RuntimeError(f"❌ API Key {_KEY_NAME_UNU} tidak ditemukan di .env")

    print(f"\n🌐 Dataset Detection (h61lr): Download dari Roboflow...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key_unu)
        project = rf.workspace("vnot").project("me-bottle-isempty-ku3-h61lr")
        
        datasets_dir = _get_datasets_dir()
        target_loc = str(datasets_dir / "standard_datasets_det")
        
        dataset = project.version(2).download(
            "coco",
            location=target_loc,
            overwrite=False
        )
        print(f"   → {dataset.location}")
        return os.path.join(dataset.location, "data.yaml")
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset h61lr detection: {e}") from e

# ==============================================================================
# DATASET COCO SEGMENTASI (UNU) - Standard Datasets
# ==============================================================================
def setup_h61lr_segmentation_dataset(key_unu: str | None) -> str:
    """
    Download dataset me-bottle-isempty-ku3-h61lr-seg.
    """
    import subprocess
    import sys
    try:
        import roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])

    if not key_unu:
        raise RuntimeError(f"❌ API Key {_KEY_NAME_UNU} tidak ditemukan di .env")

    print(f"\n🌐 Dataset Segmentation (h61lr): Download dari Roboflow...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key_unu)
        project = rf.workspace("vnot").project("me-bottle-isempty-ku3-h61lr-seg")
        
        datasets_dir = _get_datasets_dir()
        target_loc = str(datasets_dir / "standard_datasets_seg")
        
        dataset = project.version(1).download(
            "coco-segmentation",
            location=target_loc,
            overwrite=False
        )
        print(f"   → {dataset.location}")
        return os.path.join(dataset.location, "data.yaml")
    except Exception as e:
        raise RuntimeError(f"❌ Gagal download dataset h61lr segmentation: {e}") from e

# ==============================================================================
# SETUP SEMUA DATASET
# ==============================================================================
def setup_all_datasets() -> dict:
    """
    Load API key dan siapkan semua dataset sekaligus.

    Returns: dict {det_location, det_yaml, seg_location, seg_yaml, unu_location, key, key_unu}
    """
    key, key_unu = load_api_key()

    det_location, det_yaml = setup_detection_dataset(key)
    seg_location, seg_yaml = setup_segmentation_dataset(key)
    coco_seg_location = setup_coco_segmentation_dataset_unu(key_unu)
    coco_det_location = setup_coco_detection_dataset_unu(key_unu)
    h61lr_det_yaml = setup_h61lr_detection_dataset(key_unu)
    h61lr_seg_yaml = setup_h61lr_segmentation_dataset(key_unu)

    print(f"\n[Dataset] ✅ Deteksi    : {det_location}")
    print(f"[Dataset] ✅ Segmentasi : {seg_location}")
    print(f"[Dataset] ✅ Golden Seg   : {coco_seg_location}")
    print(f"[Dataset] ✅ Golden Det   : {coco_det_location}")
    print(f"[Dataset] ✅ Standart Det  : {h61lr_det_yaml}")
    print(f"[Dataset] ✅ Standart Seg  : {h61lr_seg_yaml}")

        # "det_location": det_location,
        # "det_yaml":     det_yaml,
        # "seg_location": seg_location,
        # "seg_yaml":     seg_yaml,
    return {
        "coco_seg_location": coco_seg_location,
        "coco_det_location": coco_det_location,
        "h61lr_det_yaml": h61lr_det_yaml,
        "h61lr_seg_yaml": h61lr_seg_yaml,
        "key":          key,
        "key_unu":      key_unu,
    }


if __name__ == "__main__":
    # Tes langsung
    print("=" * 60)
    print("  dataset_setup.py — Test Mode")
    print("=" * 60)
    result = setup_all_datasets()
    print("\n[Hasil]")
    for k, v in result.items():
        if k not in ("key", "key_unu"):
            print(f"  {k}: {v}")
