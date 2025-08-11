# Dataset Information

## Source
This project utilizes the publicly available **Digital Knee X-ray Images** dataset by Gornale & Patravali (2020).

- **Dataset**: Digital Knee X-ray Images
- **Authors**: Gornale & Patravali
- **Year**: 2020
- **DOI**: https://doi.org/10.17632/t9ndx37v5h.1
- **Platform**: Kaggle
- **License**: Public dataset for research use

## Dataset Overview
- **Total Images**: ~3,000 knee X-ray images
- **Classes**: 5 severity levels (Normal, Doubtful, Mild, Moderate, Severe)
- **Format**: PNG images
- **Views**: Anterior-posterior and lateral knee X-rays
- **Medical Standard**: Kellgren-Lawrence grading scale

## Quick Start

### For Testing (Sample Data)
1. Add your own knee X-ray images to this directory for testing
2. Use PNG or JPG format
3. Ensure images are clear anterior-posterior or lateral views
4. Upload through the web interface for classification

### For Full Development
1. Download the complete dataset from Kaggle
2. Search for "Digital Knee X-ray Images" by Gornale & Patravali
3. Follow the `01_Data_Preparation.ipynb` notebook for setup
4. The notebook handles data preprocessing and train/val/test splitting

## Directory Structure
```
data/
├── README.md (this file)
├── kaggle/                 # Original Kaggle dataset (after download)
├── consensus/              # Processed consensus dataset
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── demo_patients/      # Selected demo patients for LLM showcase
│   └── expert_disagreements/
└── sample_data_info.txt    # (deprecated - use this README)
```

## Data Preparation
The data preparation pipeline includes:
- Expert consensus analysis
- Demo patient selection (3 per class)
- Stratified train/val/test splits (75/12.5/12.5)
- Image preprocessing and quality checks
- Comprehensive metadata generation

## Citation
**This dataset is used under its public research license. Original authors:**
```
Gornale, S., & Patravali, P. (2020). Digital Knee X-ray Images. 
Mendeley Data, V1. https://doi.org/10.17632/t9ndx37v5h.1
```