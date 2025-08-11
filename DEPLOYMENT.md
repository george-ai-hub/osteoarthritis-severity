# Osteoarthritis Clinical Decision Support — Deployment Guide

**Complete deployment guide for clinical and technical setup**

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Not for clinical diagnosis.

## Quick Start

### Step 1: Install and Setup
```bash
# Clone and install
git clone https://github.com/george-ai-hub/osteoarthritis-severity.git
cd osteoarthritis-severity
pip install -r requirements.txt

# Download models (required)
python download_models.py
```

### Step 2: Launch Application
Choose your preferred method:

```bash
# Option A: Standalone app (Recommended)
streamlit run clinical_app_standalone.py

# Option B: Windows batch launcher  
start_clinical_app.bat
```

### Step 3: Configure LLM Integration (Optional)
```bash
# Copy template and add your OpenAI API key
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml: OPENAI_API_KEY = "your-key-here"
```

**Complete!** Open browser to `http://localhost:8501`

---

## Clinical Features & Workflow

### Demo Patients System
The application includes **15 curated demo patients** (3 per severity class) for validation:

**Blind Prediction Workflow:**
1. **Single Patient Analysis** → Select demo patient (severity hidden)
2. **AI makes blind prediction** without knowing ground truth
3. **View validation results** comparing AI vs actual classification
4. **Generate treatment plan** based on AI prediction
5. **Review clinical warnings** for misclassifications

**Batch Validation:**
1. **Batch Processing** → "Demo Patient Batch" tab
2. **Analyze all 15 patients** with comprehensive metrics
3. **Export validation results** with accuracy analytics

### Severity Classification System
- **5-class assessment**: Normal, Doubtful, Mild, Moderate, Severe (Kellgren-Lawrence scale)
- **Ensemble model accuracy**: 92.4% (trained) or 85%+ (demo fallback)
- **Real-time validation**: AI prediction vs expert annotation comparison
- **Confidence scoring**: Uncertainty quantification for clinical context

### LLM Treatment Planning
- **Evidence-based recommendations**: Following ACR/OARSI/NICE guidelines
- **Patient-specific personalization**: Demographics, comorbidities, expectations
- **Validation-aware planning**: Clinical warnings for misclassified cases
- **Rule-based fallback**: When LLM unavailable

---

## Technical Deployment

### System Requirements
**Minimum:**
- Python 3.8-3.11
- 8GB RAM
- 2GB storage

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU (GTX 1060+)
- CUDA support

### Local Development
```bash
# Standard setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run clinical_app_standalone.py
```

### Docker Deployment
```bash
# Quick setup with Docker Compose
docker-compose up --build

# Manual Docker build
docker build -t osteoarthritis-app .
docker run -p 8501:8501 osteoarthritis-app

# With GPU support
docker run --gpus all -p 8501:8501 osteoarthritis-app
```

### Production Deployment
```bash
# Network access
streamlit run clinical_app_standalone.py --server.address 0.0.0.0

# Custom port
streamlit run clinical_app_standalone.py --server.port 8080

# Headless mode
streamlit run clinical_app_standalone.py --server.headless true
```

**Production checklist:**
- Use reverse proxy (nginx) with HTTPS
- Configure environment variables
- Set up monitoring and logging
- Implement authentication for clinical use

---

## Model Training & Management

### Current Model Status
Check in **Settings** for current model availability:
- **Trained ensemble**: `models/deployment/best_model_for_deployment.pth`
- **Individual models**: `models/[model_name]_best.pth`
- **Demo fallback**: ResNet-50 if no trained models found

### Training Your Own Models
```bash
# Complete training pipeline (4-6 hours with GPU)
jupyter lab notebooks/01_Data_Preparation.ipynb
jupyter lab notebooks/02_Multi_Class_HP_Search.ipynb  
jupyter lab notebooks/03_Multi_Class_Model_HP_Selection.ipynb
jupyter lab notebooks/04_Multi_Class_Full_Training_Ensemble.ipynb
```

### Model Download Options
```bash
# Option 1: Hugging Face Hub (recommended)
python download_models.py

# Option 2: Check all download options
python setup_models.py

# Option 3: Manual download from GitHub releases
# See: https://github.com/george-ai-hub/osteoarthritis-severity/releases
```

---

## Validation & Performance Analytics

### Clinical Validation Features
- **Individual patient validation**: Instant AI vs ground truth comparison
- **Batch validation analytics**: Model performance across all demo patients
- **Accuracy metrics**: Overall and per-severity-level performance
- **Misclassification analysis**: Detailed review of incorrect predictions
- **Confidence distribution**: Prediction certainty analysis

### Export Capabilities
- **CSV export**: Complete validation results with metrics
- **Clinical reports**: Treatment plans with validation context
- **Audit trail**: Record of AI predictions and accuracy

### Performance Expectations
- **Inference speed**: <2 seconds (GPU), <5 seconds (CPU)
- **Batch processing**: 15 demo patients in ~30 seconds
- **Model accuracy**: 92.4% (ensemble), 85%+ (demo model)

---

## Security & Clinical Considerations

### Data Security Best Practices
- **API key storage**: Use `.streamlit/secrets.toml`, never commit to code
- **Patient data protection**: Implement encryption for PHI
- **Access control**: Authentication required for clinical environments
- **Audit logging**: Track all system interactions
- **HIPAA compliance**: Follow healthcare data protection regulations

### Clinical Deployment Requirements
For actual clinical use (beyond research/education):
1. **Regulatory approval**: Obtain medical device approvals
2. **Clinical validation**: Conduct prospective studies
3. **Medical oversight**: Implement physician supervision
4. **Quality assurance**: Establish performance monitoring
5. **Liability coverage**: Appropriate medical malpractice protection

---

## Troubleshooting

### Common Issues

**Model Loading Problems**
```
Failed to load model: [error]
```
**Solution**: Run model training notebooks or download pre-trained models

**Demo Patients Not Found**
```
Demo patients directory not found
```  
**Solution**: Run `01_Data_Preparation.ipynb` to create demo patient profiles

**LLM Integration Failures**
```
LLM test failed: [error]
```
**Solutions:**
- Verify API key format (starts with `sk-`)
- Check internet connection and OpenAI credits
- Use rule-based fallback if needed

**Performance Issues**
- **Slow predictions**: Check GPU availability in Settings
- **Memory errors**: Use CPU mode or reduce batch size
- **Port conflicts**: Use different port: `--server.port 8502`

### Environment Issues
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Clear Streamlit cache
streamlit cache clear
```

### System Validation Checklist
- [ ] Demo patients load correctly with metadata
- [ ] Model predictions are consistent and reasonable  
- [ ] LLM integration provides clinically relevant recommendations
- [ ] Batch processing completes without errors
- [ ] Export functionality works for all report types
- [ ] API key is properly configured and secure

---

## Clinical Workflow Examples

### Routine Screening Workflow
1. **Upload X-ray** or select demo patient (e.g., Sarah Chen - Normal)
2. **Review AI classification** and confidence score
3. **Generate recommendations** for preventive care
4. **Document findings** for patient record

### Suspected Osteoarthritis Workflow  
1. **Select demo patient** with symptoms (e.g., Amanda Davis - Mild OA)
2. **Input patient history** and expectations
3. **Generate LLM treatment plan** with evidence-based recommendations
4. **Review validation warnings** if AI prediction differs from ground truth
5. **Export treatment plan** for clinical documentation

### AI Validation Research Workflow
1. **Process all demo patients** in batch mode
2. **Analyze accuracy metrics** across severity levels
3. **Review misclassification patterns** for model improvement
4. **Export validation report** for research documentation

This deployment guide provides comprehensive setup for both technical deployment and clinical validation workflows, ensuring proper use of the osteoarthritis severity classification system for research and educational purposes.
