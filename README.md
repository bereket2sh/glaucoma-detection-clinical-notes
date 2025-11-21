# Glaucoma Detection from Clinical Notes

A machine learning project for detecting glaucoma from clinical text notes with fairness evaluation across demographic groups.

## 📁 Project Structure

```
.
├── data/
│   ├── raw/              # Original clinical_notes.csv
│   └── processed/        # Preprocessed datasets (.pt, .pkl)
├── src/
│   ├── models.py         # Neural network architectures (LSTM, GRU, Transformer, CNN)
│   ├── data_preprocessing.py  # Text cleaning and tokenization
│   ├── train.py          # Training pipeline
│   └── evaluate_fairness.py   # Fairness evaluation
├── outputs/
│   ├── figures/          # Visualizations (EDA, ROC curves)
│   ├── models/           # Trained model checkpoints (.pt)
│   └── logs/             # Training logs
├── scripts/              # Utility scripts for monitoring and testing
├── docs/                 # Documentation (templates, guides)
├── requirements.txt      # Python dependencies
├── run_pipeline.sh       # Complete pipeline execution script
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Easy way: Run everything at once
./run_pipeline.sh

# Or run steps individually:
cd src
python data_preprocessing.py  # Step 1: Preprocess data
python train.py              # Step 2: Train all models
python evaluate_fairness.py  # Step 3: Evaluate fairness
cd ..
```

### 3. View Results

- **Models**: `outputs/models/best_*.pt`
- **Visualizations**: `outputs/figures/*.png`
- **Metrics**: `outputs/training_summary.csv` and `outputs/model_comparison_table.csv`

## 🎯 Models Implemented

1. **LSTM** - Bidirectional 2-layer (3.7M parameters)
2. **GRU** - Bidirectional 2-layer (3.1M parameters)
3. **Transformer** - 3 encoder layers, 8 heads (1.9M parameters)
4. **CNN-1D** - Multi-filter convolutional (1.5M parameters)

## 📊 Dataset

- **Source**: FairCLIP Dataset
- **Size**: 10,000 clinical notes
- **Split**: 7,000 train / 1,000 validation / 2,000 test
- **Task**: Binary classification (glaucoma detection)
- **Demographics**: Age, Gender, Race (Asian, Black, White)

## 📈 Evaluation Metrics

- **Overall**: AUC, Sensitivity, Specificity, Accuracy
- **Stratified by race**: Asian, Black, White subgroups
- **Fairness analysis**: ROC curves and comparison tables

## 📚 Documentation

See the `docs/` folder for detailed documentation.

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for complete list

## 📝 License

Academic project for CSCE 566 Data Mining course.
