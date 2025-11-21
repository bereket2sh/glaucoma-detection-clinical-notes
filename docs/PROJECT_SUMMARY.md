# CSCE566 Data Mining - Final Project
## Glaucoma Detection from Clinical Notes

### Project Summary

This project implements and compares multiple deep learning architectures for detecting glaucoma from clinical text notes, with a specific focus on fairness evaluation across racial groups.

---

## ✅ Completed Tasks

### 1. **Data Exploration & Analysis** ✓
- **File**: `1_data_exploration.py`
- **Outputs**: 
  - `eda_visualizations.png` - 6-panel visualization dashboard
  - `glaucoma_rate_by_race.png` - Fairness-focused analysis
- **Key Findings**:
  - 10,000 clinical notes (balanced: 50.5% positive, 49.5% negative)
  - Average note length: 147 words
  - Racial distribution: White (76.9%), Black (14.9%), Asian (8.2%)
  - **Critical finding**: Black patients show 64.9% glaucoma rate vs White (47.9%) and Asian (48.7%)

### 2. **Data Preprocessing** ✓
- **File**: `2_data_preprocessing.py`
- **Outputs**:
  - `vocab.pkl` - Vocabulary (9,980 tokens)
  - `train/val/test_dataset.pt` - PyTorch datasets
  - `*_processed.csv` - Cleaned dataframes
- **Process**:
  - Text cleaning and normalization
  - Medical abbreviation expansion
  - Tokenization and vocabulary building
  - Sequence padding to 512 tokens

### 3. **Model Architectures** ✓
- **File**: `models.py`
- **Implemented Models**:
  1. **LSTM**: Bidirectional, 2-layer (3.7M parameters)
  2. **GRU**: Bidirectional, 2-layer (3.1M parameters)
  3. **Transformer**: 3 encoder layers, 8 heads (1.9M parameters)
  4. **Bonus**: 1D CNN (1.5M parameters)

### 4. **Training Pipeline** ✓
- **File**: `train_all_models.py`
- **Features**:
  - Automated training for all models
  - Early stopping with learning rate scheduling
  - Gradient clipping for stability
  - Comprehensive metrics logging
- **Hyperparameters**:
  - Batch size: 32
  - Learning rate: 0.001 (Adam)
  - Epochs: 10
  - Dropout: 0.3

### 5. **Fairness Evaluation Framework** ✓
- **File**: `5_fairness_evaluation.py`
- **Outputs**:
  - `model_comparison_table.csv` - Complete metrics table
  - `fairness_comparison.png` - Bar charts by race
  - `roc_curves_by_race.png` - ROC curves per subgroup
- **Metrics Evaluated**:
  - Overall: AUC, Sensitivity, Specificity, Accuracy
  - By Race: Same metrics for White, Black, Asian groups

### 6. **Documentation** ✓
- **Files**:
  - `README.md` - Complete project documentation
  - `requirements.txt` - All dependencies
  - `PROJECT_SUMMARY.md` - This file

---

## 📊 Expected Results Structure

After running `train_all_models.py`:

```
Results/
├── best_lstm_model.pt              # Best LSTM checkpoint
├── best_gru_model.pt               # Best GRU checkpoint
├── best_transformer_model.pt       # Best Transformer checkpoint
├── lstm_results.json               # Training history & metrics
├── gru_results.json
├── transformer_results.json
├── lstm_predictions.npy            # Test predictions
├── gru_predictions.npy
├── transformer_predictions.npy
├── *_labels.npy                    # Test labels
└── training_summary.csv            # Quick comparison
```

After running `5_fairness_evaluation.py`:

```
Fairness/
├── lstm_fairness_results.json      # Race-stratified metrics
├── gru_fairness_results.json
├── transformer_fairness_results.json
├── model_comparison_table.csv      # All models × all groups
├── fairness_comparison.png         # Bar charts
└── roc_curves_by_race.png         # ROC curves
```

---

## 🚀 Quick Start Guide

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run complete pipeline
python 1_data_exploration.py       # ~30 seconds
python 2_data_preprocessing.py     # ~1 minute
python train_all_models.py         # ~15-30 minutes (depends on GPU/CPU)
python 5_fairness_evaluation.py    # ~1 minute

# 3. View results
ls -lh *.png *.csv *.json best_*.pt
```

---

## 📈 Evaluation Metrics

### Required Metrics (Per Project Requirements):
1. ✅ **Overall AUC** - Area under ROC curve
2. ✅ **Sensitivity** - True positive rate (recall)
3. ✅ **Specificity** - True negative rate
4. ✅ **AUCs by Race**:
   - Asian group AUC
   - Black group AUC
   - White group AUC

### Additional Metrics:
- Accuracy, Precision, F1-Score
- Confusion matrix components (TP, FP, TN, FN)
- Training time and model size

---

## 🔬 Model Comparison Table (Template)

| Model | Group | N | AUC | Sensitivity | Specificity |
|-------|-------|---|-----|-------------|-------------|
| LSTM | Overall | 2000 | TBD | TBD | TBD |
| LSTM | White | 1537 | TBD | TBD | TBD |
| LSTM | Black | 305 | TBD | TBD | TBD |
| LSTM | Asian | 158 | TBD | TBD | TBD |
| GRU | Overall | 2000 | TBD | TBD | TBD |
| GRU | White | 1537 | TBD | TBD | TBD |
| GRU | Black | 305 | TBD | TBD | TBD |
| GRU | Asian | 158 | TBD | TBD | TBD |
| Transformer | Overall | 2000 | TBD | TBD | TBD |
| Transformer | White | 1537 | TBD | TBD | TBD |
| Transformer | Black | 305 | TBD | TBD | TBD |
| Transformer | Asian | 158 | TBD | TBD | TBD |

*TBD values will be filled after training completes*

---

## 📝 For the Final Report

### Sections Needed:

1. **Introduction** (~0.5 page)
   - Glaucoma detection importance
   - Clinical notes as data source
   - Fairness concerns in healthcare AI

2. **Related Work** (~0.5 page)
   - Text classification with LSTM/GRU
   - Transformers for medical text
   - Fairness in healthcare ML

3. **Method** (~1 page)
   - Data preprocessing pipeline
   - Model architectures (include figure!)
   - Training procedure

4. **Experiments** (~1.5 pages)
   - Dataset description
   - Hyperparameters
   - Evaluation methodology
   - Results tables and figures
   - Fairness analysis

5. **Conclusions** (~0.5 page)
   - Best performing model
   - Fairness findings
   - Limitations and future work

### Figures to Include:
1. EDA visualizations (class distribution, demographics)
2. Model architecture diagram
3. Training curves (loss/AUC over epochs)
4. ROC curves (overall and by race)
5. Fairness comparison bar charts

---

## 🎯 Next Steps

1. **Run Training**: Execute `train_all_models.py` to get results
2. **Run Fairness Eval**: Execute `5_fairness_evaluation.py`
3. **Write Report**: Use results to populate 4-page report
4. **Create GitHub Repo**: 
   - Initialize repository
   - Upload all code
   - Add README
   - Get repository link
5. **Write Reflection**:
   - Biggest challenge
   - What was learned
   - Self-evaluation (A/B/C/D)

---

## 💡 Key Insights

1. **Data Quality**: Clinical notes are noisy but information-rich
2. **Class Imbalance by Race**: Black patients have higher glaucoma prevalence
3. **Model Selection**: Transformer may outperform RNNs despite fewer parameters
4. **Fairness**: Critical to evaluate performance across demographic groups

---

## ⚠️ Important Notes

- Training on CPU will take 30-60 minutes per model
- Training on GPU will take 5-15 minutes per model
- Ensure sufficient disk space (~500MB for all outputs)
- All random seeds should be set for reproducibility

---

## 📧 Contact

For questions about this project:
- Email: min.shi@louisiana.edu (Professor)
- Office: 350

---

**Status**: Ready for training and evaluation
**Last Updated**: November 20, 2025
