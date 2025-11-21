"""
Data Preprocessing and Feature Engineering
CSCE566 - Data Mining - Glaucoma Detection
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA PREPROCESSING FOR GLAUCOMA DETECTION")
print("="*80)

# Download NLTK data
print("\n[1] Downloading NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK resources ready")
except:
    print("⚠ NLTK download skipped")

# Load dataset
print("\n[2] Loading dataset...")
df = pd.read_csv('../data/raw/clinical_notes.csv')
print(f"✓ Loaded {len(df):,} records")

# Text cleaning function
def clean_text(text):
    """Clean clinical notes text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Replace common medical abbreviations with full forms
    text = text.replace(' ou ', ' both eyes ')
    text = text.replace(' od ', ' right eye ')
    text = text.replace(' os ', ' left eye ')
    text = text.replace(' iop ', ' intraocular pressure ')
    text = text.replace(' hvf ', ' visual field ')
    text = text.replace(' oct ', ' optical coherence tomography ')
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("\n[3] Cleaning text data...")
df['cleaned_note'] = df['note'].apply(clean_text)
df['cleaned_summary'] = df['gpt4_summary'].apply(clean_text)

print("  Sample cleaned text:")
print(f"  Original length: {len(df['note'].iloc[0])}")
print(f"  Cleaned length: {len(df['cleaned_note'].iloc[0])}")
print(f"  First 200 chars: {df['cleaned_note'].iloc[0][:200]}...")

# Convert labels
print("\n[4] Encoding labels...")
df['label'] = (df['glaucoma'] == 'yes').astype(int)
print(f"  Positive class (glaucoma=yes): {df['label'].sum()} samples")
print(f"  Negative class (glaucoma=no): {(1-df['label']).sum()} samples")

# Encode race for fairness evaluation
print("\n[5] Encoding demographic features...")
race_encoder = LabelEncoder()
df['race_encoded'] = race_encoder.fit_transform(df['race'])
race_mapping = dict(zip(race_encoder.classes_, race_encoder.transform(race_encoder.classes_)))
print(f"  Race encoding: {race_mapping}")

# Split data using the provided 'use' column
print("\n[6] Splitting dataset...")
train_df = df[df['use'] == 'training'].copy()
val_df = df[df['use'] == 'validation'].copy()
test_df = df[df['use'] == 'test'].copy()

print(f"  Training set: {len(train_df)} samples")
print(f"  Validation set: {len(val_df)} samples")
print(f"  Test set: {len(test_df)} samples")

# Check class balance in splits
for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    pos_rate = subset['label'].mean() * 100
    print(f"  {name} - Positive rate: {pos_rate:.2f}%")

# Build vocabulary
print("\n[7] Building vocabulary...")
all_text = ' '.join(train_df['cleaned_note'].values)
words = all_text.split()
word_counts = Counter(words)

# Filter vocabulary (min frequency = 2, max vocab size = 20000)
MIN_FREQ = 2
MAX_VOCAB = 20000
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(MAX_VOCAB) if count >= MIN_FREQ]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(f"  Total unique words: {len(word_counts):,}")
print(f"  Vocabulary size: {len(vocab):,}")
print(f"  Most common words: {word_counts.most_common(10)}")

# Text to sequence function
def text_to_sequence(text, word2idx, max_len=512):
    """Convert text to sequence of indices"""
    words = text.split()[:max_len]
    sequence = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    return sequence

# Pad sequences
def pad_sequence(sequence, max_len=512):
    """Pad sequence to max_len"""
    if len(sequence) >= max_len:
        return sequence[:max_len]
    else:
        return sequence + [word2idx['<PAD>']] * (max_len - len(sequence))

print("\n[8] Converting text to sequences...")
MAX_LEN = 512

for name, subset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    sequences = subset['cleaned_note'].apply(lambda x: text_to_sequence(x, word2idx, MAX_LEN))
    sequences_padded = sequences.apply(lambda x: pad_sequence(x, MAX_LEN))
    subset['sequence'] = sequences_padded
    subset['seq_length'] = sequences.apply(len)
    
    print(f"  {name.capitalize()}: Avg sequence length = {subset['seq_length'].mean():.1f}")

# Create PyTorch Dataset
class GlaucomaDataset(Dataset):
    """Custom Dataset for Glaucoma Detection"""
    
    def __init__(self, dataframe):
        self.texts = np.array(dataframe['sequence'].tolist())
        self.labels = dataframe['label'].values
        self.races = dataframe['race'].values
        self.ages = dataframe['age'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text': torch.LongTensor(self.texts[idx]),
            'label': torch.LongTensor([self.labels[idx]]),
            'race': self.races[idx],
            'age': self.ages[idx]
        }

# Create datasets
print("\n[9] Creating PyTorch datasets...")
train_dataset = GlaucomaDataset(train_df)
val_dataset = GlaucomaDataset(val_df)
test_dataset = GlaucomaDataset(test_df)

print(f"  Train dataset: {len(train_dataset)} samples")
print(f"  Val dataset: {len(val_dataset)} samples")
print(f"  Test dataset: {len(test_dataset)} samples")

# Save preprocessed data
print("\n[10] Saving preprocessed data...")

# Save vocabulary
with open('../data/processed/vocab.pkl', 'wb') as f:
    pickle.dump({'word2idx': word2idx, 'idx2word': idx2word, 'vocab': vocab}, f)
print("  ✓ Vocabulary saved to 'data/processed/vocab.pkl'")

# Save race encoder
with open('../data/processed/race_encoder.pkl', 'wb') as f:
    pickle.dump(race_encoder, f)
print("  ✓ Race encoder saved to 'data/processed/race_encoder.pkl'")

# Save processed dataframes
train_df.to_csv('../data/processed/train_processed.csv', index=False)
val_df.to_csv('../data/processed/val_processed.csv', index=False)
test_df.to_csv('../data/processed/test_processed.csv', index=False)
print("  ✓ Processed CSV files saved")

# Save PyTorch datasets
torch.save(train_dataset, '../data/processed/train_dataset.pt')
torch.save(val_dataset, '../data/processed/val_dataset.pt')
torch.save(test_dataset, '../data/processed/test_dataset.pt')
print("  ✓ PyTorch datasets saved")

# Statistics summary
print("\n[11] Preprocessing Statistics Summary:")
print(f"  Vocabulary size: {len(vocab):,}")
print(f"  Max sequence length: {MAX_LEN}")
print(f"  Training samples: {len(train_df):,}")
print(f"  Validation samples: {len(val_df):,}")
print(f"  Test samples: {len(test_df):,}")

# Racial distribution in test set (important for fairness)
print("\n[12] Test Set Racial Distribution:")
test_race_counts = test_df['race'].value_counts()
for race in ['white', 'black', 'asian']:
    if race in test_race_counts.index:
        count = test_race_counts[race]
        pct = count / len(test_df) * 100
        positive = len(test_df[(test_df['race'] == race) & (test_df['label'] == 1)])
        pos_rate = positive / count * 100 if count > 0 else 0
        print(f"  {race.capitalize()}: {count} samples ({pct:.1f}%), {positive} positive ({pos_rate:.1f}%)")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  - vocab.pkl (vocabulary)")
print("  - race_encoder.pkl (demographic encoder)")
print("  - train_processed.csv, val_processed.csv, test_processed.csv")
print("  - train_dataset.pt, val_dataset.pt, test_dataset.pt")
print("="*80)
