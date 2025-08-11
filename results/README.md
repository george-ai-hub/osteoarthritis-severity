# 📊 Results Directory

This directory contains experimental results, analysis outputs, and performance visualizations from the osteoarthritis severity classification project.

## Directory Structure

```
results/
├── README.md (this file)
├── hp_search_per_model/          # Hyperparameter search results
│   ├── results_analysis.png      # Search analysis visualization  
│   ├── stage1_results.csv       # Initial hyperparameter results
│   └── stage2_results.csv       # Refined hyperparameter results
├── full_training/               # Final model training results
│   ├── ensemble_training_*/     # Timestamped training sessions
│   │   └── visualizations/      # Performance visualizations
│   └── model_selection_analysis.png  # Model comparison analysis
└── (Generated during training)
```

## Key Results

### Model Performance Summary (Test Set)
- **Best individual model**: ConvNeXt-Tiny — Acc 89.55%, F1 0.8951, F2 0.8952
- **Other individual models**: EfficientNet-B0 89.09%, RegNet-Y-800MF 88.64%, ResNet-50 88.18%, DenseNet-121 86.36%
- **Original ensemble (pre-calibration)**: Acc 90.45%, F1 0.9038, F2 0.9042, ECE 0.0857, Brier 0.0668
- **Final calibrated ensemble**: Acc 90.45%, F1 0.9038, F2 0.9042, Precision 0.9038, Recall 0.9045, ECE 0.1055, Brier 0.0717
- **Temperature scaling**: 1.0888
- **Calibration deltas**: ECE -0.0198 (-23.2%), Brier -0.0049 (-7.3%)

### Hyperparameter Optimization
- **Stage 1**: Broad parameter exploration across 5 model architectures
- **Stage 2**: Refined optimization for top-performing configurations
- **Search Space**: Learning rates, batch sizes, optimizers, schedulers

## Visualizations

The results include comprehensive visualizations:
- **Confusion Matrices**: Per-class performance analysis
- **Calibration Plots**: Model confidence assessment  
- **Training Curves**: Loss and accuracy progression
- **Performance Comparisons**: Cross-model analysis

## Analysis Files

### CSV Files
- **Stage 1 Results**: `hp_search_per_model/stage1_results.csv`
- **Stage 2 Results**: `hp_search_per_model/stage2_results.csv`
- Contain: Trial configurations, metrics, timestamps

### Visualization Files  
- **PNG Format**: High-quality plots for documentation
- **Timestamped**: Results linked to specific training sessions
- **Publication Ready**: Clean, professional formatting

## Reproducing Results

1. **Run Hyperparameter Search**: 
   ```bash
   # Execute notebooks in order
   jupyter lab osteoarthritis-severity/notebooks/02_Multi_Class_HP_Search.ipynb
   ```

2. **Model Selection**:
   ```bash
   jupyter lab osteoarthritis-severity/notebooks/03_Multi_Class_Model_HP_Selection.ipynb
   ```

3. **Final Training**:
   ```bash
   jupyter lab osteoarthritis-severity/notebooks/04_Multi_Class_Full_Training_Ensemble.ipynb
   ```

## ⚠️ Important Notes

- **Research Purpose**: Results demonstrate ML experimentation skills
- **Not Clinical**: Educational analysis, not for medical decisions
- **Timestamped**: Results preserve experimental timeline
- **Reproducible**: Complete pipeline documented in notebooks
