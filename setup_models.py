#!/usr/bin/env python3
"""
Model Setup Script - Multiple Options for Getting Models
Provides several ways to obtain the required model files.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "models"

def check_models_exist():
    """Check if models are already downloaded."""
    required_models = [
        "efficientnet_best.pth",
        "regnet_best.pth", 
        "densenet_best.pth",
        "resnet_best.pth",
        "convnext_best.pth",
        "deployment/best_model_for_deployment.pth"
    ]
    
    existing = []
    missing = []
    
    for model in required_models:
        model_path = MODELS_DIR / model
        if model_path.exists() and model_path.stat().st_size > 1000000:
            existing.append(model)
        else:
            missing.append(model)
    
    return existing, missing

def main():
    print("Osteoarthritis Severity Classification - Model Setup")
    print("=" * 60)
    
    existing, missing = check_models_exist()
    
    if not missing:
        print("All models are ready!")
        print("Run: streamlit run clinical_app_standalone.py")
        return
    
    print(f"Status: {len(existing)}/{len(existing) + len(missing)} models ready")
    if existing:
        print(f"Found: {', '.join(existing)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    
    print("\nModel Download Options:")
    print("1. Hugging Face Hub (Recommended)")
    print("   python download_models.py")
    
    print("\n2. GitHub Releases")
    print("   Go to: https://github.com/george-ai-hub/osteoarthritis-severity/releases")
    print("   Download: osteoarthritis-models.zip")
    print("   Extract to: ./models/")
    
    print("\n3. Direct Download Links")
    print("   See README.md for individual model URLs")
    
    print("\n4. Train Your Own Models") 
    print("   Run the Jupyter notebooks in order:")
    print("   - 02_Multi_Class_HP_Search.ipynb")
    print("   - 03_Multi_Class_Model_HP_Selection.ipynb") 
    print("   - 04_Multi_Class_Full_Training_Ensemble.ipynb")
    
    print("\nFor quick start, use option 1:")
    print("   python download_models.py")

if __name__ == "__main__":
    main() 