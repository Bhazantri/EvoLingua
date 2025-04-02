
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from .config import EvoLinguaConfig
from .attention import MultiHeadLatentAttention
from .moe import DeepSeekMoE
from .mtp import MultiTokenPrediction

try:
    from transformer_engine.pytorch import fp8_autocast
except ImportError:
    from contextlib import nullcontext as fp8_autocast

class EvoLingua(nn.Module):
    """EvoLingua: A scalable MoE language model."""
    def __init__(self, config: EvoLinguaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.embed_dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": MultiHeadLatentAttention(config),
                "moe": DeepSeekMoE(config),
                "norm1": nn.LayerNorm(config.embed_dim),
                "norm2": nn.LayerNorm(config.embed_dim)
            }) for _ in range(config.num_layers)
        ])
        self.mtp = MultiTokenPrediction(config)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids) + self.positional_embedding[:, :seq_len, :]

        with fp8_autocast():
            for layer in self.layers:
                attn_output = layer["attention"](x, attention_mask)
                x = layer["norm1"](x + attn_output)
                moe_output = layer["moe"](x)
                x = layer["norm2"](x + moe_output)

        x = self.final_norm(x)
        main_output = self.output_head(x)
        mtp_outputs = self.mtp(x, input_ids)
        return main_output, mtp_outputs
