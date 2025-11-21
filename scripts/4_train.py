"""
Training Script for Glaucoma Detection Models
CSCE566 - Data Mining
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import json
import time
from models import get_model, count_parameters

print("="*80)
print("GLAUCOMA DETECTION - MODEL TRAINING")
print("="*80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[1] Device: {device}")

# Load vocabulary
print("\n[2] Loading vocabulary...")
with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
vocab_size = len(vocab_data['vocab'])
print(f"  Vocabulary size: {vocab_size:,}")

# Load datasets
print("\n[3] Loading datasets...")
train_dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('val_dataset.pt')
test_dataset = torch.load('test_dataset.pt')
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts = batch['text'].to(device)
        labels = batch['label'].float().to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            texts = batch['text'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    # Calculate metrics with threshold 0.5
    pred_labels = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, all_preds, all_labels

# Train model function
def train_model(model_name, num_epochs=15, batch_size=32, learning_rate=0.001):
    """Train a model"""
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    # Create model
    print(f"\n[1] Creating {model_name} model...")
    model = get_model(model_name, vocab_size, device)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Create dataloaders
    print(f"\n[2] Creating dataloaders (batch_size={batch_size})...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                      patience=3, verbose=True)
    
    # Training loop
    print(f"\n[3] Training for {num_epochs} epochs...")
    best_val_auc = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_auc = val_metrics['auc']
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, f'best_{model_name}_model.pt')
            print(f"  ✓ Best model saved (AUC: {val_auc:.4f})")
    
    training_time = time.time() - start_time
    print(f"\n[4] Training completed in {training_time:.2f} seconds")
    print(f"  Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    
    # Load best model
    print(f"\n[5] Loading best model for evaluation...")
    checkpoint = torch.load(f'best_{model_name}_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print(f"\n[6] Evaluating on test set...")
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    # Calculate specificity
    tn = ((test_labels == 0) & (test_preds < 0.5)).sum()
    fp = ((test_labels == 0) & (test_preds >= 0.5)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Specificity: {specificity:.4f}")
    
    # Save results
    results = {
        'model_name': model_name,
        'num_parameters': count_parameters(model),
        'training_time': training_time,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_metrics': {
            'auc': float(test_metrics['auc']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'sensitivity': float(test_metrics['recall']),
            'specificity': float(specificity),
            'f1': float(test_metrics['f1'])
        },
        'history': history
    }
    
    # Save results and predictions
    with open(f'{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(f'{model_name}_predictions.npy', test_preds)
    np.save(f'{model_name}_labels.npy', test_labels)
    
    print(f"\n[7] Results saved:")
    print(f"  - {model_name}_results.json")
    print(f"  - {model_name}_predictions.npy")
    print(f"  - {model_name}_labels.npy")
    print(f"  - best_{model_name}_model.pt")
    
    return results

# Main training
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    
    print(f"\n[4] Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    
    # Train models
    models_to_train = ['lstm', 'gru', 'transformer']
    all_results = {}
    
    for model_name in models_to_train:
        results = train_model(model_name, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
        all_results[model_name] = results
        print(f"\n{'='*80}\n")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Parameters: {results['num_parameters']:,}")
        print(f"  Training Time: {results['training_time']:.2f}s")
        print(f"  Test AUC: {results['test_metrics']['auc']:.4f}")
        print(f"  Test Sensitivity: {results['test_metrics']['sensitivity']:.4f}")
        print(f"  Test Specificity: {results['test_metrics']['specificity']:.4f}")
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
