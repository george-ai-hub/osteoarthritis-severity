# Models Directory

This directory contains trained deep learning models for osteoarthritis severity classification.

## Model Files

### Individual Model Weights
- `convnext_best.pth` - ConvNeXt-Tiny model
- `densenet_best.pth` - DenseNet-121 model  
- `efficientnet_best.pth` - EfficientNet-B0 model
- `regnet_best.pth` - RegNet-Y-800MF model
- `resnet_best.pth` - ResNet-50 model

### Deployment Model
- `deployment/best_model_for_deployment.pth` - Production-ready ensemble model

## Test Set Performance

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

## Usage

Models are automatically loaded by the clinical application. For manual loading:

```python
import torch

# Load individual model
model = torch.load('models/efficientnet_best.pth', map_location='cpu')

# Load deployment model  
deployment_model = torch.load('models/deployment/best_model_for_deployment.pth', map_location='cpu')
```

## Model Download

**Important**: Model files are not included in the Git repository due to size constraints (50-285MB each).

### Quick Setup
```bash
# From repository root:
python download_models.py     # Download from Hugging Face Hub
# OR
python setup_models.py       # See all download options
```

### Download Options
1. **Hugging Face Hub** (Recommended): `python download_models.py`
2. **GitHub Releases**: Download `osteoarthritis-models.zip` from releases
3. **Train Your Own**: Use the provided Jupyter notebooks (4-6 hours with GPU)
4. **Manual Download**: Individual model links (see main README.md)

## ⚠️ Important Notes

- **File Size**: Models are large (50-285MB) and excluded from Git commits
- **Purpose**: Educational demonstration of ML model development  
- **Not for Clinical Use**: Research and learning purposes only
- **GPU Recommended**: Best performance with CUDA-compatible GPUs
- **Required for App**: Models must be downloaded for the clinical application to function

## Model Development

See notebooks for complete model development pipeline:
- `osteoarthritis-severity/notebooks/02_Multi_Class_HP_Search.ipynb` - Hyperparameter optimization
- `osteoarthritis-severity/notebooks/03_Multi_Class_Model_HP_Selection.ipynb` - Model selection
- `osteoarthritis-severity/notebooks/04_Multi_Class_Full_Training_Ensemble.ipynb` - Final training