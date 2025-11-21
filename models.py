"""
Model Architectures for Glaucoma Detection
CSCE566 - Data Mining
Implements: LSTM, GRU, and 1D CNN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """LSTM-based text classifier for glaucoma detection"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout1(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim*2)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # hidden shape: (batch_size, hidden_dim*2)
        out = self.dropout2(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


class GRUClassifier(nn.Module):
    """GRU-based text classifier for glaucoma detection"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(GRUClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout1(embedded)
        
        # GRU
        gru_out, hidden = self.gru(embedded)
        # gru_out shape: (batch_size, seq_len, hidden_dim*2)
        
        # Use the last hidden state
        if self.gru.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # hidden shape: (batch_size, hidden_dim*2)
        out = self.dropout2(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


class CNN1DClassifier(nn.Module):
    """1D CNN-based text classifier for glaucoma detection"""
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=128, 
                 filter_sizes=[3, 4, 5], dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # Multiple convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout1(embedded)
        
        # Transpose for Conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolution and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_len - filter_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate outputs from different filter sizes
        out = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


class TransformerClassifier(nn.Module):
    """Transformer-based text classifier for glaucoma detection"""
    
    def __init__(self, vocab_size, embedding_dim=128, nhead=8, 
                 num_encoder_layers=3, dim_feedforward=512, dropout=0.3, max_len=512):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        embedded = self.pos_encoder(embedded)
        
        # Create padding mask
        padding_mask = (x == 0)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)  # (batch_size, embedding_dim)
        
        out = self.dropout(pooled)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def get_model(model_name, vocab_size, device='cuda'):
    """Factory function to create models"""
    
    if model_name.lower() == 'lstm':
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
    elif model_name.lower() == 'gru':
        model = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
    elif model_name.lower() == 'cnn':
        model = CNN1DClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=128,
            filter_sizes=[3, 4, 5],
            dropout=0.3
        )
    elif model_name.lower() == 'transformer':
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=512,
            dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*80)
    print("MODEL ARCHITECTURE TEST")
    print("="*80)
    
    vocab_size = 10000
    batch_size = 16
    seq_len = 512
    
    # Test input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    models = ['lstm', 'gru', 'cnn', 'transformer']
    
    for model_name in models:
        print(f"\n{model_name.upper()} Model:")
        model = get_model(model_name, vocab_size, device='cpu')
        print(f"  Parameters: {count_parameters(model):,}")
        
        with torch.no_grad():
            output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Model test passed")
    
    print("\n" + "="*80)
    print("ALL MODELS TESTED SUCCESSFULLY!")
    print("="*80)
