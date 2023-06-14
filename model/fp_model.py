import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #[N, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[N, D]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #[N, D]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:[B, N, D]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FingerPrintEncoder(nn.Module):
    def __init__(self, word_dim, out_dim, num_head=8, num_layer=1):
        super(FingerPrintEncoder, self).__init__()
        self.embedding = nn.Embedding(2048, word_dim)
        encoder_layer = nn.TransformerEncoderLayer(word_dim, nhead=num_head, dim_feedforward=out_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layer)
        self.pe = PositionalEncoding(word_dim)
        self.linear = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.embedding(x.int())
        x = self.pe(x)
        output = self.transformer(x)
        output = self.linear(output.sum(2))
    
        return output

# ========= Transformer for SMILES =========
class PE_Smiles(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=510):
        super(PE_Smiles, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #[N, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[N, D]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #[N, D]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:[B, N, D]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, word_dim, out_dim, num_head=8, num_layer=1):
        super(SmilesEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim)
        encoder_layer = nn.TransformerEncoderLayer(word_dim, nhead=num_head, dim_feedforward=out_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layer)
        self.pe = PE_Smiles(word_dim)
        self.linear = nn.Linear(510, out_dim)

    def forward(self, x, mask):
        x = self.embedding(x.int())
        x = self.pe(x)
        x = x * mask.unsqueeze(-1)
        output = self.transformer(x)
        output = self.linear(output.sum(2))
    
        return output
