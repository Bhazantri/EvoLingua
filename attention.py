# evolingua/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .config import EvoLinguaConfig

def apply_rotary_pos_emb(x: torch.Tensor, seq_len: int, dim: int) -> torch.Tensor:
    """Apply Rotary Positional Embedding (RoPE)."""
    positions = torch.arange(seq_len, device=x.device).unsqueeze(-1)
    freqs = torch.pow(10000, -torch.arange(0, dim, 2, device=x.device) / dim)
    angles = positions * freqs
    sin_angles = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    cos_angles = torch.cos(angles).unsqueeze(0).unsqueeze(2)
    x_reshaped = x.view(*x.shape[:-1], -1, 2)
    x_rotated = torch.cat([x_reshaped[..., 0] * cos_angles - x_reshaped[..., 1] * sin_angles,
                           x_reshaped[..., 0] * sin_angles + x_reshaped[..., 1] * cos_angles], dim=-1)
    return x_rotated.view_as(x)

class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (MLA) module."""
    def __init__(self, config: EvoLinguaConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.kv_compress_dim = config.kv_compress_dim

        self.W_dkv = nn.Linear(self.embed_dim, self.kv_compress_dim, bias=False)
        self.W_uk = nn.Linear(self.kv_compress_dim, self.embed_dim, bias=False)
        self.W_uv = nn.Linear(self.kv_compress_dim, self.embed_dim, bias=False)
        self.W_kr = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_dq = nn.Linear(self.embed_dim, self.kv_compress_dim, bias=False)
        self.W_uq = nn.Linear(self.kv_compress_dim, self.embed_dim, bias=False)
        self.W_qr = nn.Linear(self.kv_compress_dim, self.embed_dim, bias=False)
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        kv_compressed = self.W_dkv(x)
        keys = self.W_uk(kv_compressed)
        values = self.W_uv(kv_compressed)
        keys_rope = apply_rotary_pos_emb(self.W_kr(x), seq_len, self.head_dim)

        q_compressed = self.W_dq(x)
        queries = self.W_uq(q_compressed)
        queries_rope = apply_rotary_pos_emb(self.W_qr(q_compressed), seq_len, self.head_dim)

        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys_rope.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.W_o(attn_output)
