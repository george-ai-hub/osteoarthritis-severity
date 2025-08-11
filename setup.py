#!/usr/bin/env python3
"""
Setup script for Osteoarthritis Severity Classification System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("Python 3.9+ required. Current version:", sys.version)
        print("Please upgrade Python and try again.")
        sys.exit(1)
    else:
        print(f"Python {sys.version.split()[0]} detected")

def check_gpu():
    """Check for CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA GPU detected: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("No CUDA GPU detected. Will use CPU (slower training).")
            return False
    except ImportError:
        print("PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"{requirements_file} not found!")
        sys.exit(1)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file, "--upgrade"
        ])
        print("All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        print("Try installing manually: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = ["models", "results", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"{directory}/ directory ready")

def download_sample_data():
    """Download sample X-ray images for testing"""
    print("\nSetting up sample data...")
    
    # Create sample data info file
    sample_info = """
# Sample Data for Testing

This directory will contain sample X-ray images for testing the application.

## Getting Started:
1. Add your own knee X-ray images to test the system
2. Use PNG or JPG format
3. Ensure images are clear anterior-posterior or lateral views

## Full Dataset:
- The complete dataset can be downloaded from Kaggle
- Search for "Digital Knee X-ray Images" by Gornale & Patravali
- Follow the data preparation notebook for full setup

## Quick Test:
- Upload any knee X-ray image through the web interface
- The demo model will provide classification results
"""
    
    with open("data/sample_data_info.txt", "w", encoding='utf-8') as f:
        f.write(sample_info)
    
    print("Sample data directory configured")

def test_installation():
    """Test if installation was successful"""
    print("\nTesting installation...")
    
    try:
        import streamlit
        import torch
        import torchvision
        import pandas
        import numpy
        import plotly
        print("Core packages imported successfully")
        
        # Test torch
        x = torch.randn(2, 3)
        print(f"PyTorch test successful: {x.shape}")
        
        # Test CUDA again after installation
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def check_launch_scripts():
    """Check if launch scripts exist"""
    print("\nChecking launch scripts...")
    
    scripts_found = []
    
    # Check for Windows scripts
    if os.path.exists("start_app_simple.bat"):
        scripts_found.append("start_app_simple.bat")
    if os.path.exists("start_app.bat"):
        scripts_found.append("start_app.bat")
    
    # Check for Unix scripts  
    if os.path.exists("start_app.sh"):
        scripts_found.append("start_app.sh")
    
    if scripts_found:
        print("Launch scripts found:")
        for script in scripts_found:
            print(f"   - {script}")
    else:
        print("No launch scripts found. You can start manually with: streamlit run clinical_app_standalone.py")

def main():
    """Main setup function"""
    print("Osteoarthritis Severity Classification Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Check for GPU
    has_gpu = check_gpu()
    
    # Step 3: Install requirements
    install_requirements()
    
    # Step 4: Create directories
    create_directories()
    
    # Step 5: Setup sample data
    download_sample_data()
    
    # Step 6: Test installation
    if not test_installation():
        print("\nInstallation test failed!")
        sys.exit(1)
    
    # Step 7: Check launch scripts
    check_launch_scripts()
    
    # Final message
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Launch the app:")
    if platform.system() == "Windows":
        if os.path.exists("start_app_simple.bat"):
            print("   - Double-click 'start_app_simple.bat' or")
        elif os.path.exists("start_app.bat"):
            print("   - Double-click 'start_app.bat' or")
    else:
        if os.path.exists("start_app.sh"):
            print("   - Run './start_app.sh' or")
    print("   - Run 'streamlit run clinical_app_standalone.py'")
    print("\n2. Open your browser to: http://localhost:8501")
    print("\n3. Upload a knee X-ray image and start analyzing!")
    
    if has_gpu:
        print("\nGPU detected! Training will be faster.")
    else:
        print("\nFor faster training, consider using a CUDA-compatible GPU.")
    
    print("\nFor research notebooks, run: jupyter lab")
    print("\nNeed help? Check README.md for complete setup and deployment guide")

if __name__ == "__main__":
    main() 