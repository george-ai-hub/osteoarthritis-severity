#!/usr/bin/env python3
"""
Model Download Script - Osteoarthritis Severity Classification
Downloads pre-trained models from Hugging Face Hub for the clinical application.
"""

import os
import requests
from pathlib import Path
from typing import Optional
import hashlib
from tqdm import tqdm
from zipfile import ZipFile
import io

# Repository configuration
REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "models"
DEPLOYMENT_DIR = MODELS_DIR / "deployment"

# Model URLs (individual files)
HUGGINGFACE_MODELS = {
    "efficientnet_best.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/efficientnet_best.pth",
        "size": "50MB",
        "description": "EfficientNet-B0 (Acc 89.09%)"
    },
    "regnet_best.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/regnet_best.pth", 
        "size": "45MB",
        "description": "RegNet-Y-800MF (Acc 88.64%)"
    },
    "densenet_best.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/densenet_best.pth",
        "size": "32MB", 
        "description": "DenseNet-121 (Acc 86.36%)"
    },
    "resnet_best.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/resnet_best.pth",
        "size": "98MB",
        "description": "ResNet-50 (Acc 88.18%)" 
    },
    "convnext_best.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/convnext_best.pth",
        "size": "113MB",
        "description": "ConvNeXt-Tiny (Acc 89.55%)"
    },
    "deployment/best_model_for_deployment.pth": {
        "url": "https://huggingface.co/george-ai-hub/osteoarthritis-models/resolve/main/best_model_for_deployment.pth",
        "size": "285MB", 
        "description": "Calibrated ensemble (Acc 90.45%, F1 0.9038, F2 0.9042)"
    }
}

# Dataset ZIP (single artifact with all model files)
HF_DATASET_ZIP_URL = (
    "https://huggingface.co/datasets/george-ai-hub/osteoarthritis-severity-models/resolve/main/osteoarthritis-models.zip"
)

def download_file(url: str, filepath: Path, description: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading {description}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)
                
        print(f"Downloaded: {filepath.name}")
        return True
        
    except Exception as e:
        print(f"Failed to download {filepath.name}: {str(e)}")
        return False


def _strip_common_top_folder(zf: ZipFile) -> Optional[str]:
    """Return common top-level folder name if all entries are under a single folder, else None."""
    names = [info.filename for info in zf.infolist() if info.filename and not info.filename.endswith("/")]
    if not names:
        return None
    top_parts = {name.split("/")[0] for name in names if "/" in name}
    if len(top_parts) == 1:
        return next(iter(top_parts))
    return None


def extract_zip_from_memory(zip_bytes: bytes, destination_dir: Path) -> None:
    """Extract a ZIP archive provided as bytes into destination_dir, flattening a single top folder (e.g., 'models/')."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        top = _strip_common_top_folder(zf)
        for info in zf.infolist():
            name = info.filename
            if not name:
                continue
            # Skip directory entries; we'll create dirs as needed
            is_dir = name.endswith("/")
            parts = name.split("/")
            if top and parts and parts[0].lower() == top.lower():
                parts = parts[1:]
            # Ignore empty paths that can occur from top-level folder entries
            if not parts or parts == ['']:
                continue
            target_path = destination_dir.joinpath(*parts)
            if is_dir:
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())


def download_and_extract_dataset_zip() -> bool:
    """Download HF dataset ZIP once and extract all models into models directory."""
    try:
        print("Attempting dataset ZIP download (all models at once)...")
        response = requests.get(HF_DATASET_ZIP_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024
        buffer = io.BytesIO()

        with tqdm(
            desc="osteoarthritis-models.zip",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        extract_zip_from_memory(buffer.getvalue(), MODELS_DIR)
        print("Dataset ZIP extracted to models directory")
        # If the ZIP contained a nested 'models/models' structure previously, it is now flattened.
        return True
    except Exception as e:
        print(f"Dataset ZIP download failed: {e}")
        return False

def check_model_exists(filepath: Path) -> bool:
    """Check if model file already exists."""
    return filepath.exists() and filepath.stat().st_size > 1000000  # At least 1MB

def download_models():
    """Download all required models."""
    print("Osteoarthritis Severity Classification - Model Download")
    print("=" * 60)
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    DEPLOYMENT_DIR.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(HUGGINGFACE_MODELS)
    
    for model_name, info in HUGGINGFACE_MODELS.items():
        model_path = MODELS_DIR / model_name
        
        if check_model_exists(model_path):
            print(f"Already exists: {model_name}")
            success_count += 1
            continue
            
        if download_file(info["url"], model_path, info["description"]):
            success_count += 1
        else:
            print(f"Consider manual download for: {model_name}")
    
    # If some models failed, try the dataset ZIP fallback once
    if success_count < total_count:
        print("\nSome individual downloads failed or are missing.")
        if download_and_extract_dataset_zip():
            # Re-count after extraction
            success_count = 0
            for model_name in HUGGINGFACE_MODELS.keys():
                if check_model_exists(MODELS_DIR / model_name):
                    success_count += 1
        else:
            print("Dataset ZIP fallback also failed.")

    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{total_count} models ready")
    
    if success_count == total_count:
        print("All models are present.")
        print("Run: streamlit run clinical_app_standalone.py")
    else:
        print("Some models are still missing.")
        print("You can download manually from the Hugging Face dataset ZIP and extract to models/")
    print("Source: https://huggingface.co/datasets/george-ai-hub/osteoarthritis-severity-models")

if __name__ == "__main__":
    download_models() 