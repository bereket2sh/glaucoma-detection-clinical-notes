"""
Complete Training Pipeline - All Models
CSCE566 - Data Mining
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
import time
from models import get_model, count_parameters

# Define GlaucomaDataset class (needed for loading saved datasets)
class GlaucomaDataset(Dataset):
    """Custom Dataset for Glaucoma Detection"""
    
    def __init__(self, dataframe):
        self.texts = np.array(dataframe['sequence'].tolist()) if hasattr(dataframe, 'tolist') else dataframe.texts
        self.labels = dataframe['label'].values if hasattr(dataframe, 'values') else dataframe.labels
        self.races = dataframe['race'].values if hasattr(dataframe, 'values') else dataframe.races
        self.ages = dataframe['age'].values if hasattr(dataframe, 'values') else dataframe.ages
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text': torch.LongTensor(self.texts[idx]),
            'label': torch.LongTensor([self.labels[idx]]),
            'race': self.races[idx],
            'age': self.ages[idx]
        }

print("="*80)
print("TRAINING ALL MODELS FOR GLAUCOMA DETECTION")
print("="*80)

# Configuration
CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,  # Reduced for faster training
    'models': ['lstm', 'gru', 'transformer']
}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
print("\nLoading data...")
with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
vocab_size = len(vocab_data['vocab'])

train_dataset = torch.load('train_dataset.pt', weights_only=False)
val_dataset = torch.load('val_dataset.pt', weights_only=False)
test_dataset = torch.load('test_dataset.pt', weights_only=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"  Vocab size: {vocab_size:,}")

def train_and_evaluate(model_name):
    """Train and evaluate a single model"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Create model
    model = get_model(model_name, vocab_size, device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_auc = 0
    history = []
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch in pbar:
            texts = batch['text'].to(device)
            labels = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(probs.flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text'].to(device)
                labels = batch['label'].float().to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        scheduler.step(val_auc)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc
        })
        
        print(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f} | Val: Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, f'best_{model_name}_model.pt')
            print(f"  ✓ Best model saved (AUC: {val_auc:.4f})")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load(f'best_{model_name}_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            texts = batch['text'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(texts)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs.flatten())
            test_labels.extend(labels.cpu().numpy().flatten())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    pred_binary = (test_preds >= 0.5).astype(int)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, pred_binary)
    test_prec = precision_score(test_labels, pred_binary, zero_division=0)
    test_recall = recall_score(test_labels, pred_binary, zero_division=0)
    
    # Specificity
    tn = ((test_labels == 0) & (pred_binary == 0)).sum()
    fp = ((test_labels == 0) & (pred_binary == 1)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    training_time = time.time() - start_time
    
    results = {
        'model_name': model_name,
        'params': count_parameters(model),
        'training_time': training_time,
        'best_val_auc': best_val_auc,
        'test_metrics': {
            'auc': float(test_auc),
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'sensitivity': float(test_recall),
            'specificity': float(specificity)
        },
        'history': history
    }
    
    # Save
    with open(f'{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(f'{model_name}_predictions.npy', test_preds)
    np.save(f'{model_name}_labels.npy', test_labels)
    
    print(f"\nResults:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Sensitivity: {test_recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return results

# Train all models
all_results = {}
for model_name in CONFIG['models']:
    try:
        results = train_and_evaluate(model_name)
        all_results[model_name] = results
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

summary_data = []
for model_name, results in all_results.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Parameters: {results['params']:,}")
    print(f"  Training Time: {results['training_time']:.2f}s")
    print(f"  Test AUC: {results['test_metrics']['auc']:.4f}")
    print(f"  Sensitivity: {results['test_metrics']['sensitivity']:.4f}")
    print(f"  Specificity: {results['test_metrics']['specificity']:.4f}")
    
    summary_data.append({
        'Model': model_name.upper(),
        'Parameters': f"{results['params']:,}",
        'Time (s)': f"{results['training_time']:.1f}",
        'AUC': f"{results['test_metrics']['auc']:.4f}",
        'Sensitivity': f"{results['test_metrics']['sensitivity']:.4f}",
        'Specificity': f"{results['test_metrics']['specificity']:.4f}"
    })

# Save summary
import pandas as pd
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('training_summary.csv', index=False)

print("\n" + "="*80)
print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*80)
