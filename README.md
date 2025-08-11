# Osteoarthritis Severity Classification System

An AI-powered medical imaging system for automated osteoarthritis severity assessment from knee X-rays using PyTorch deep learning, featuring clinical decision support and evidence-based treatment planning.

**⚠️ Educational & Research Project**: This software demonstrates machine learning techniques for medical image analysis. **NOT for clinical diagnosis.**

## Quick Start

**Launch application in 3 steps:**

### Step 1: Setup Project
```bash
git clone https://github.com/george-ai-hub/osteoarthritis-severity.git
cd osteoarthritis-severity
pip install -r requirements.txt

# Download pre-trained models (required for app functionality)
python download_models.py
```

### Step 2: Configure API (Optional)
For AI-powered treatment recommendations:
```bash
# Copy the template
# Windows:
copy .streamlit\secrets.toml.template .streamlit\secrets.toml
# Mac/Linux:
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Edit `.streamlit/secrets.toml` and add your OpenAI API key:
# OPENAI_API_KEY = "sk-..."
```

### Step 3: Launch Application
Choose your preferred launch method:

**Option A: Standalone App (Recommended)**
```bash
python -m streamlit run clinical_app_standalone.py
```

**Option B: Windows Launcher**
```bash
start_clinical_app.bat
```

### Step 4: Start Analyzing
- Upload X-ray images via the web interface
- Get automatic severity classification
- View Kellgren-Lawrence grading with confidence scores
- Generate AI-powered treatment recommendations (requires API key)

**Complete!** Open browser to `http://localhost:8501` and start analyzing X-rays immediately.

## ⚠️ **Model Download Required**

**Important**: This repository contains the source code, but you need to download the trained models separately to run the application.

### **Quick Model Setup**
```bash
# After cloning and installing requirements:
python download_models.py        # Download from Hugging Face (recommended)
# OR
python setup_models.py          # See all download options
```

**What happens**: Downloads ~600-650MB of pre-trained models to `osteoarthritis-severity/models/`
**No login required**: Public models accessible without Hugging Face account

**Alternative options:**
- **GitHub Releases**: Download `osteoarthritis-models.zip` from releases
- **Train Your Own**: Use the provided Jupyter notebooks
- **Manual Download**: Individual model links in documentation

## System Overview

### Key Features
- **Multi-class Classification**: 5-grade osteoarthritis severity assessment (Kellgren-Lawrence scale)
- **Clinical Interface**: Professional web-based application for healthcare settings
- **AI Treatment Planning**: GPT-powered evidence-based treatment recommendations
- **Research Tools**: Comprehensive Jupyter notebooks for model development
- **Deployment Ready**: Docker containerization and cloud deployment support
- **GPU Acceleration**: Optimized for NVIDIA RTX series graphics cards

### Test Set Performance
- Test set size: 220 samples

| Model | Acc% | F1 | F2 | ECE |
|-------|------|----|----|-----|
| EfficientNet-B0 | 89.09 | 0.8900 | 0.8905 | 0.1002 |
| RegNet-Y-800MF | 88.64 | 0.8861 | 0.8861 | 0.0987 |
| ResNet-50 | 88.18 | 0.8811 | 0.8814 | 0.0584 |
| DenseNet-121 | 86.36 | 0.8608 | 0.8609 | 0.0802 |
| ConvNeXt-Tiny | 89.55 | 0.8951 | 0.8952 | 0.0270 |
| Calibrated Ensemble | 90.45 | 0.9038 | 0.9042 | 0.1055 |

- Best individual model: ConvNeXt-Tiny (89.55% Acc)
- Original ensemble (pre-calibration): Acc 90.45% | F1 0.9038 | F2 0.9042 | ECE 0.0857 | Brier 0.0668
- Final calibrated ensemble: Acc 90.45% | F1 0.9038 | F2 0.9042 | Precision 0.9038 | Recall 0.9045 | ECE 0.1055 | Brier 0.0717 | Temperature 1.0888
- Calibration deltas: ECE -0.0198 (-23.2%), Brier -0.0049 (-7.3%)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: 4GB RAM minimum
- **Storage**: 2GB+ free space
- **CPU**: Multi-core processor recommended

### Recommended Configuration  
- **Python**: 3.9+
- **Memory**: 16GB+ RAM
- **GPU**: NVIDIA RTX 3070/4070 or better
- **VRAM**: 8GB+ for optimal performance
- **CUDA**: 11.8+

### GPU Acceleration Support
The system automatically detects and utilizes NVIDIA GPUs:
- **RTX 4070**: Optimized performance with ~2-3x faster inference
- **RTX 3070+**: Full feature support with GPU acceleration  
- **CPU Fallback**: Automatic fallback for non-GPU systems

## Installation & Configuration

### Method 1: Standard Installation
```bash
# Clone repository
git clone https://github.com/george-ai-hub/osteoarthritis-severity.git
cd osteoarthritis-severity

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run clinical_app_standalone.py
```

### Method 2: Conda Environment
```bash
# Create conda environment
conda env create -f environment_pytorch_cuda.yml
conda activate pytorch-cuda-osteoarthritis

# Launch application
streamlit run clinical_app_standalone.py
```

### API Configuration (Optional)

For AI treatment planning, create `.streamlit/secrets.toml`:
```toml
[api_keys]
openai_api_key = "your-openai-api-key-here"
```

**Windows users**: Use `copy .streamlit\secrets.toml.template .streamlit\secrets.toml`
**Mac/Linux users**: Use `cp .streamlit/secrets.toml.template .streamlit/secrets.toml`

⚠️ **Security**: Never commit your `secrets.toml` file to version control!

### Environment Variables
Create `.env` file for advanced configuration:
```env
# Model configuration
MODEL_PATH=./models/best_model.pth
DEVICE=cuda  # or 'cpu' for CPU-only

# Streamlit configuration  
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API configuration (optional)
OPENAI_API_KEY=your-api-key-here
MAX_TOKENS=1000
TEMPERATURE=0.3
```

## Deployment Options

### Local Network Access
```bash
# Allow network access for team sharing
streamlit run clinical_app_standalone.py --server.address 0.0.0.0

# Custom port
streamlit run clinical_app_standalone.py --server.port 8080

# Headless mode (no auto-browser)
streamlit run clinical_app_standalone.py --server.headless true
```

### Docker Deployment

**Quick Docker Setup**:
```bash
# Build and run with Docker Compose (easiest)
docker-compose up -d

# Access at http://localhost:8501
```

**Manual Docker**:
```bash
# Build the image
docker build -t osteoarthritis-app .

# Run the container
docker run -d \
  --name osteoarthritis-app \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  osteoarthritis-app
```

**Docker with GPU Support**:
```bash
# For NVIDIA GPUs
docker run -d \
  --name osteoarthritis-app \
  --gpus all \
  -p 8501:8501 \
  osteoarthritis-app
```

### Cloud Deployment

**Streamlit Cloud**:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

**Other Cloud Options**:
- **Heroku**: Container deployment
- **AWS/Azure/GCP**: Use container services (ECS, Container Instances, Cloud Run)

## Troubleshooting

### Common Issues & Solutions

**Port 8501 Already in Use**:
```bash
# Use different port
streamlit run clinical_app_standalone.py --server.port 8502

# Or kill existing process
# Windows: taskkill /F /IM streamlit.exe
# Mac/Linux: pkill -f streamlit
```

**Module Import Errors**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python environment
which python
pip list
```

**CUDA/GPU Issues**:
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install CPU-only PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Memory Issues**:
```bash
# For GPU memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**Streamlit Not Loading**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Reset configuration
rm -rf ~/.streamlit/

# Try incognito/private browser mode
```

### Verification Commands
```bash
# Check installations
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
streamlit --version

# Test GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"
```

## Model Download Options

### **Option 1: Hugging Face Hub (Recommended)**
**Best for**: Professional portfolios, automatic updates, version control
```bash
python download_models.py
```
- **No account required**: Public models accessible without login
- Automatic download with progress bars (~600-650MB total)
- Downloads to: `osteoarthritis-severity/models/`
- Version control and model cards  
- Professional ML model hosting standard
- Easy integration and updates

**For Maintainers Only:**
1. Upload your models to [Hugging Face Hub](https://huggingface.co/new)
2. Update URLs in `download_models.py`
3. End users simply run: `python download_models.py`

### **Option 2: GitHub Releases** 
**Best for**: Simple hosting, one-time setup
```bash
# Download from: https://github.com/george-ai-hub/osteoarthritis-severity/releases
# Extract osteoarthritis-models.zip to ./models/
```
- Simple to set up and use
- Version tracking through releases
- Manual upload required for each update
- Download size limits (~2GB per release)

### **Option 3: Cloud Storage Links**
**Best for**: Quick sharing, temporary hosting
- Google Drive, Dropbox, or OneDrive links
- Direct download URLs in documentation
- Users download and extract manually

### **Option 4: Train Your Own Models**
**Best for**: Learning, customization, full reproducibility
```bash
# Run notebooks in order:
jupyter lab notebooks/02_Multi_Class_HP_Search.ipynb
jupyter lab notebooks/03_Multi_Class_Model_HP_Selection.ipynb  
jupyter lab notebooks/04_Multi_Class_Full_Training_Ensemble.ipynb
```
- Full educational experience
- Complete reproducibility  
- Customizable hyperparameters
- Requires GPU and ~4-6 hours training time

## LLM Integration Setup (Optional)

To enable AI-powered treatment recommendations and documentation:

1. Get an API key from OpenAI (`https://platform.openai.com`)
2. Configure the key (pick one):
   - Environment variable:
     - Windows (PowerShell): `$env:OPENAI_API_KEY="your-api-key-here"`
     - Linux/Mac: `export OPENAI_API_KEY="your-api-key-here"`
   - Streamlit secrets: create `osteoarthritis-severity/.streamlit/secrets.toml` containing:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - In-app entry: enter temporarily in the app UI (session-only)
3. Install optional dependencies (for PDF generation and latest OpenAI client):
   ```bash
   pip install reportlab
   ```

Security: never commit `.streamlit/secrets.toml` or your API key to version control.

## Dataset & Performance

### Dataset Information
**Source**: [Digital Knee X-ray Images](https://doi.org/10.17632/t9ndx37v5h.1) by Gornale & Patravali (2020)

**Composition**:
- **Total Images**: ~3,000 knee X-ray images
- **Classification**: Kellgren-Lawrence grades (0-4)
- **Distribution**: Balanced across severity levels
- **Format**: PNG images, standardized resolution
- **Annotation**: Expert-validated severity labels

**Data Split**:
- **Training**: 75% (~1,400 images)
- **Validation**: 12.5% (~200 images)  
- **Testing**: 12.5% (~200 images)

### Citation
```bibtex
@data{gornale_patravali_2020,
  author = {Gornale, S. and Patravali, P.},
  title = {Digital Knee X-ray Images},
  year = {2020},
  publisher = {Mendeley Data},
  version = {V1},
  doi = {10.17632/t9ndx37v5h.1}
}
```

## Research & Documentation

### Research Notebooks
The `notebooks/` directory contains comprehensive research materials:

1. **01_Data_Preparation_PyTorch.ipynb** - Dataset loading and preprocessing
2. **02_HP_Search_Multi_Class_PyTorch.ipynb** - Hyperparameter optimization
3. **03_HP_Search_OvR_PyTorch.ipynb** - One-vs-Rest model tuning
4. **04_HP_Analysis_PyTorch.ipynb** - Performance analysis
5. **05_Model_Selection_Strategy.ipynb** - Model comparison and selection
6. **06_Full_Training_and_Final_Evaluation.ipynb** - Final model training
7. **07_Model_Deployment_and_Clinical_UI.ipynb** - Deployment implementation

**Additional**: **GPU_Optimization_Test_PyTorch.ipynb** - CUDA performance optimization

### Development Documentation
- **Clinical Workflow Guide**: `development/CLINICAL_WORKFLOW_GUIDE.md`
- **Mission Complete**: `development/MISSION_COMPLETE.md`
- **PyTorch Conversion**: `development/PyTorch_Conversion_Summary.md`

## Security & Clinical Use

⚠️ **Important Medical Disclaimer**: This system is for research and educational purposes. For clinical use:

### Clinical Requirements
- Obtain appropriate regulatory approvals
- Implement clinical validation protocols  
- Ensure HIPAA compliance for patient data
- Establish medical oversight and liability coverage
- Follow institutional AI/ML deployment guidelines

### Security Best Practices
- **API Security**: Store keys in environment variables, never in code
- **Data Encryption**: Secure file storage and transmission
- **Access Control**: Implement authentication for clinical use
- **Audit Logging**: Track all user interactions and predictions
- **Regular Updates**: Keep dependencies updated for security patches

## Performance Optimization

### Model Loading Optimization
```python
# Use model caching
@st.cache_resource
def load_model():
    return OsteoarthritisClassificationModel()
```

### Image Processing Optimization
```python
# Cache predictions
@st.cache_data
def predict_image(image_hash):
    # Prediction logic
    pass
```

### Mobile & Responsive Design
- Works on tablets and mobile devices
- Touch-friendly interface
- Optimized for medical workflows
- Image upload via camera (mobile)

## Quick Reference Commands

### Basic Commands
```bash
streamlit run clinical_app_standalone.py                    # Start app
streamlit run clinical_app_standalone.py --server.port 8080 # Custom port
streamlit cache clear                   # Clear cache
python setup.py                        # Automated setup
```

### Docker Commands
```bash
docker-compose up -d                    # Start with Docker
docker-compose down                     # Stop Docker services
docker logs osteoarthritis-app          # View logs
```

### Development Commands
```bash
jupyter lab                             # Start Jupyter for notebooks
pip install -r requirements.txt        # Install dependencies
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU
```

## Support & Contributing

### Getting Help
1. **Check this README** for setup and troubleshooting
2. **GitHub Issues** for bug reports and feature requests
3. **Discussions** for questions and community support

### Contributing
Contributions are welcome! Please:
- Fork the repository
- Create feature branches
- Add tests for new features
- Update documentation
- Submit pull requests

### Contact
For questions, issues, or collaboration opportunities, please open an issue on this repository.

## License & Important Disclaimers

This project is licensed under a **Educational and Research License** - see the [LICENSE.txt](LICENSE.txt) file for complete terms.

### Permitted Uses
- **Educational**: Learning, skill evaluation, academic use
- **Research**: Personal research and portfolio demonstration
- **Assessment**: Technical skill evaluation by employers/peers

### Prohibited Uses
- **Commercial Use**: No commercial distribution or profit-making
- **Medical Diagnosis**: No clinical decision-making or patient care
- **Healthcare Applications**: No deployment in medical settings

### ⚠️ Medical Disclaimer
**CRITICAL**: This is a research and educational project. It is **NOT** a medical device and **MUST NOT** be used for actual medical diagnosis, clinical decisions, or patient care. Always consult qualified healthcare professionals for medical advice.

### API Key Security
- API keys are stored locally in `osteoarthritis-severity/.streamlit/secrets.toml`
- Template provided for easy setup by other users
- Your personal API keys are never committed to version control

---

**Educational Achievement**: This osteoarthritis severity classification system demonstrates advanced machine learning skills including deep learning, medical imaging, and clinical AI applications!
