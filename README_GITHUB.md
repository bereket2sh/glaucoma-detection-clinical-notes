# Glaucoma Detection from Clinical Notes

Deep learning models for detecting glaucoma disease from clinical text notes with fairness evaluation across racial groups.

**Course**: CSCE566 - Data Mining, Fall 2025  
**Dataset**: FairCLIP  
**Task**: Binary classification with fairness analysis

---

## 🎯 Project Overview

This project implements and compares multiple deep learning architectures (LSTM, GRU, Transformer) for glaucoma detection from clinical notes, with explicit evaluation of model fairness across Asian, Black, and White patient populations.

### Key Features
- ✅ 4 deep learning models (LSTM, GRU, Transformer, CNN)
- ✅ Comprehensive fairness evaluation by race
- ✅ Complete pipeline from raw text to predictions
- ✅ Detailed performance metrics (AUC, Sensitivity, Specificity)

---

## 📊 Dataset

- **Source**: FairCLIP dataset
- **Size**: 10,000 clinical notes
- **Split**: 7,000 train / 1,000 validation / 2,000 test
- **Labels**: Binary (glaucoma: yes/no)
- **Demographics**: Age, gender, race, ethnicity, language

**Important**: Dataset shows different glaucoma prevalence by race:
- Black patients: 64.9%
- White patients: 47.9%
- Asian patients: 48.7%

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/glaucoma-detection-clinical-notes.git
cd glaucoma-detection-clinical-notes
```

### 2. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
# Data exploration
python 1_data_exploration.py

# Preprocessing
python 2_data_preprocessing.py

# Train all models (~20-30 min)
python train_all_models.py

# Fairness evaluation
python 5_fairness_evaluation.py
```

---

## 📁 Project Structure

```
├── models.py                      # Model architectures (LSTM, GRU, Transformer, CNN)
├── train_all_models.py           # Training pipeline
├── 5_fairness_evaluation.py      # Fairness analysis
├── 2_data_preprocessing.py       # Text preprocessing
├── 1_data_exploration.py         # EDA and visualization
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── results/
    ├── best_*_model.pt          # Trained models
    ├── *_results.json           # Metrics
    ├── model_comparison_table.csv
    └── *.png                     # Visualizations
```

---

## 🤖 Models Implemented

| Model | Parameters | Description |
|-------|-----------|-------------|
| **LSTM** | 3.7M | Bidirectional, 2-layer LSTM |
| **GRU** | 3.1M | Bidirectional, 2-layer GRU |
| **Transformer** | 1.9M | 3 encoder layers, 8 attention heads |
| **CNN** | 1.5M | 1D CNN with multiple filter sizes |

---

## 📈 Results

### Overall Performance (Test Set)

| Model | AUC | Sensitivity | Specificity | Accuracy |
|-------|-----|-------------|-------------|----------|
| LSTM | TBD | TBD | TBD | TBD |
| GRU | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD |

### Fairness Analysis

Performance metrics stratified by racial group (White, Black, Asian) are available in `model_comparison_table.csv`.

---

## 🔧 Technical Details

### Preprocessing
- Text cleaning and normalization
- Medical abbreviation expansion (OU→both eyes, IOP→intraocular pressure)
- Vocabulary: ~10,000 tokens (min frequency=2)
- Max sequence length: 512 tokens

### Training
- Batch size: 32
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 10
- Loss: Binary Cross-Entropy with Logits
- Early stopping with learning rate scheduling

### Evaluation Metrics
- **Overall**: AUC, Sensitivity, Specificity, Accuracy
- **Fairness**: Same metrics stratified by race (Asian, Black, White)

---

## 📊 Visualizations

Generated visualizations include:
- EDA plots (class distribution, demographics)
- ROC curves by model and race
- Fairness comparison charts
- Training history plots

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- nltk, tqdm

See `requirements.txt` for complete list.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{glaucoma-detection-2025,
  author = {Your Name},
  title = {Glaucoma Detection from Clinical Notes with Fairness Evaluation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/glaucoma-detection-clinical-notes}
}
```

---

## 📄 License

This project is for academic purposes (CSCE566 Final Project).

---

## 👤 Author

**Your Name**  
University of Louisiana at Lafayette  
CSCE566 - Data Mining, Fall 2025

---

## 🙏 Acknowledgments

- FairCLIP dataset for providing fairness-annotated clinical data
- Course instructor: Dr. Min Shi

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact: your.email@louisiana.edu
