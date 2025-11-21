"""
Quick training test with LSTM model
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
print("QUICK TRAINING TEST - LSTM MODEL")
print("="*80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load vocabulary
print("\nLoading vocabulary...")
with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
vocab_size = len(vocab_data['vocab'])
print(f"Vocabulary size: {vocab_size:,}")

# Load datasets
print("\nLoading datasets...")
train_dataset = torch.load('train_dataset.pt')
val_dataset = torch.load('val_dataset.pt')

# Create model
print("\nCreating LSTM model...")
model = get_model('lstm', vocab_size, device)
print(f"Parameters: {count_parameters(model):,}")

# Create dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training for 3 epochs (quick test)
NUM_EPOCHS = 3
print(f"\nTraining for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
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
    
    train_loss = total_loss / len(train_loader)
    train_auc = roc_auc_score(all_labels, all_preds)
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
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
    
    val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

print("\n" + "="*80)
print("QUICK TEST COMPLETED SUCCESSFULLY!")
print("="*80)
