# Glaucoma Detection from Clinical Notes
## CSCE566 - Data Mining Final Project

### Project Overview
This project implements deep learning models for detecting glaucoma disease from clinical notes text data. The project focuses on fairness evaluation across different racial groups (White, Black, Asian).

### Dataset
- **Source**: FairCLIP Dataset
- **Size**: 10,000 clinical notes
- **Split**: 7,000 training / 1,000 validation / 2,000 test
- **Task**: Binary classification (glaucoma: yes/no)
- **Features**: Clinical notes, demographics (age, gender, race, ethnicity)

### Models Implemented
1. **LSTM** (Long Short-Term Memory) - Bidirectional, 2 layers
2. **GRU** (Gated Recurrent Unit) - Bidirectional, 2 layers  
3. **Transformer** - Custom transformer encoder with positional encoding

### Evaluation Metrics
- **Overall Metrics**: AUC, Sensitivity, Specificity, Accuracy
- **Fairness Metrics**: AUC, Sensitivity, Specificity by racial group (Asian, Black, White)

### Project Structure
```
├── clinical_notes.csv                  # Raw dataset
├── 1_data_exploration.py              # EDA and visualization
├── 2_data_preprocessing.py            # Text cleaning and tokenization
├── models.py                          # Model architectures
├── train_all_models.py                # Training pipeline
├── 5_fairness_evaluation.py           # Fairness analysis
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Exploration
```bash
python 1_data_exploration.py
```
Generates:
- `eda_visualizations.png` - Distribution plots
- `glaucoma_rate_by_race.png` - Fairness analysis

#### 2. Data Preprocessing
```bash
python 2_data_preprocessing.py
```
Generates:
- `vocab.pkl` - Vocabulary dictionary
- `train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt` - PyTorch datasets
- `*_processed.csv` - Cleaned data

#### 3. Train All Models
```bash
python train_all_models.py
```
Trains LSTM, GRU, and Transformer models. Generates:
- `best_*_model.pt` - Saved model checkpoints
- `*_results.json` - Training metrics and history
- `*_predictions.npy` - Test set predictions
- `training_summary.csv` - Comparison table

#### 4. Fairness Evaluation
```bash
python 5_fairness_evaluation.py
```
Generates:
- `*_fairness_results.json` - Race-stratified metrics
- `model_comparison_table.csv` - Comprehensive comparison
- `fairness_comparison.png` - Bar charts by race
- `roc_curves_by_race.png` - ROC curves

### Key Results

**Dataset Statistics:**
- 10,000 clinical notes (5,048 positive, 4,952 negative)
- Average note length: 147 words
- Racial distribution: White (76.9%), Black (14.9%), Asian (8.2%)
- **Important**: Black patients show 64.9% glaucoma rate vs White (47.9%) and Asian (48.7%)

**Model Performance** (on test set):
| Model | Parameters | AUC | Sensitivity | Specificity |
|-------|----------- |-----|-------------|-------------|
| LSTM  | 3.7M       |  TBD|         TBD |         TBD |
| GRU   | 3.1M       | TBD |         TBD |         TBD |
| Transformer | 1.9M | TBD |         TBD |         TBD |

### Hyperparameters
- Embedding dimension: 128
- Hidden dimension: 256 (LSTM/GRU)
- Batch size: 32
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 10
- Max sequence length: 512 tokens
- Dropout: 0.3

### Technical Details

**Text Preprocessing:**
- Lowercase normalization
- Medical abbreviation expansion (OU → both eyes, IOP → intraocular pressure)
- Special character removal
- Vocabulary size: ~10,000 tokens (min frequency=2)

**Model Architectures:**
- LSTM: Bidirectional, 2-layer, embedding → LSTM → FC layers
- GRU: Similar to LSTM but with GRU cells
- Transformer: 3 encoder layers, 8 attention heads, positional encoding

### Fairness Considerations
This project explicitly evaluates model fairness across racial groups to ensure equitable performance. Key findings:
- Models evaluated on Asian, Black, and White populations separately
- Metrics reported for each subgroup to identify potential bias
- Higher glaucoma prevalence in Black patients requires careful interpretation

### Future Work
- Fine-tune clinical BERT models (BioBERT, ClinicalBERT)
- Ensemble methods for improved performance
- Address class imbalance in racial subgroups
- Incorporate additional clinical features (age, medical history)

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- nltk, tqdm

### Author
Bereket
CSCE566 - Data Mining
University of Louisiana at Lafayette
Fall 2025

### License
Academic use only - CSCE566 Final Project
