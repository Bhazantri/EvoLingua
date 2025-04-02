
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .config import EvoLinguaConfig

class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction (MTP) module."""
    def __init__(self, config: EvoLinguaConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mtp_depth = config.mtp_depth
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads, dropout=config.dropout)
            for _ in range(config.mtp_depth)
        ])
        self.projection = nn.ModuleList([
            nn.Linear(config.embed_dim * 2, config.embed_dim, bias=False)
            for _ in range(config.mtp_depth)
        ])
        self.norm = nn.RMSNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        predictions = []
        h = x
        for k in range(self.mtp_depth):
            future_tokens = input_ids[:, k:k + seq_len]
            token_emb = self.embedding(future_tokens)
            combined = torch.cat([self.norm(h), self.norm(token_emb)], dim=-1)
            h_prime = self.projection[k](combined)
            h = self.transformer_blocks[k](h_prime)
            pred = self.output_head(h)
            predictions.append(pred)
        return predictions
