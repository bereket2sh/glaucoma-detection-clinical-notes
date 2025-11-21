# Project Structure - Reorganization Summary

## Overview
The project has been reorganized into a clean, professional structure following best practices for machine learning projects.

## Directory Structure

```
glaucoma-detection-clinical-notes/
├── data/                           # All data files
│   ├── raw/                       # Original dataset
│   │   └── clinical_notes.csv
│   └── processed/                 # Preprocessed data
│       ├── vocab.pkl              # Word vocabulary
│       ├── race_encoder.pkl       # Race label encoder
│       ├── *_processed.csv        # Processed DataFrames
│       └── *_dataset.pt           # PyTorch datasets
│
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # Neural network architectures
│   ├── data_preprocessing.py     # Data cleaning & tokenization
│   ├── train.py                  # Training pipeline
│   └── evaluate_fairness.py      # Fairness evaluation
│
├── outputs/                       # All generated outputs
│   ├── models/                   # Trained model checkpoints (.pt)
│   ├── figures/                  # Visualizations (.png)
│   ├── logs/                     # Training logs
│   └── *.csv, *.json             # Results and metrics
│
├── scripts/                       # Utility scripts
│   ├── check_status.py
│   ├── monitor_training.py
│   └── run_pipeline.py
│
├── docs/                          # Documentation
│   ├── GETTING_STARTED.md
│   ├── PROJECT_SUMMARY.md
│   ├── REFLECTION_TEMPLATE.md
│   ├── README.md (old)
│   ├── README_GITHUB.md
│   └── report_template.tex
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── run_pipeline.sh               # Complete pipeline script
└── README.md                     # Main documentation
```

## Key Changes

### 1. **Data Organization**
- **Before**: All data files in root directory
- **After**: Organized into `data/raw/` and `data/processed/`
- **Benefit**: Clear separation between original and derived data

### 2. **Source Code**
- **Before**: Python files scattered in root
- **After**: All source code in `src/` directory
- **Benefit**: Professional package structure, easier imports

### 3. **Outputs**
- **Before**: Generated files mixed with source code
- **After**: All outputs in `outputs/` with subdirectories
- **Benefit**: Easy to clean, exclude from git, locate results

### 4. **Documentation**
- **Before**: Multiple README files in root
- **After**: All docs in `docs/` directory
- **Benefit**: Cleaner root, better organization

### 5. **Scripts**
- **Before**: Utility scripts in root
- **After**: Separate `scripts/` directory
- **Benefit**: Clear distinction from main source code

## Updated File Paths

All file paths in the source code have been updated:

### data_preprocessing.py
- Input: `../data/raw/clinical_notes.csv`
- Output: `../data/processed/*.pkl`, `../data/processed/*.pt`, `../data/processed/*.csv`

### train.py
- Input: `../data/processed/*.pt`, `../data/processed/vocab.pkl`
- Output: `../outputs/models/*.pt`, `../outputs/*.json`, `../outputs/*.npy`

### evaluate_fairness.py
- Input: `../data/processed/*.pt`, `../outputs/models/*.pt`, `../outputs/*.npy`
- Output: `../outputs/figures/*.png`, `../outputs/*.csv`, `../outputs/*.json`

## Running the Project

### Option 1: Complete Pipeline
```bash
./run_pipeline.sh
```

### Option 2: Individual Steps
```bash
cd src
python data_preprocessing.py
python train.py
python evaluate_fairness.py
cd ..
```

## .gitignore Updates

Updated to work with new structure:
- Excludes `data/raw/*.csv` (large dataset)
- Excludes `outputs/models/*.pt` (large trained models)
- Excludes `outputs/logs/*.log` (log files)
- Keeps processed data for reproducibility

## Benefits of New Structure

1. **Professional**: Follows ML project best practices
2. **Scalable**: Easy to add new models, scripts, or data
3. **Clean**: Root directory has minimal files
4. **Navigable**: Clear hierarchy, easy to find files
5. **Maintainable**: Related files grouped together
6. **Git-friendly**: Easy to version control
7. **Reproducible**: Clear data flow and dependencies

## For GitHub

The new structure is ideal for GitHub:
- Clean root with README
- Professional organization
- Easy for others to understand and use
- Proper separation of concerns
