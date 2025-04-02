
import torch
import- torch.nn as nn
import torch.nn.functional as F
from .config import EvoLinguaConfig

class DeepSeekMoE(nn.Module):
    """Mixture-of-Experts (MoE) with auxiliary-loss-free load balancing."""
    def __init__(self, config: EvoLinguaConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.num_shared_experts = 2

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim)
            ) for _ in range(self.num_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim)
            ) for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_experts))
        self.dropout = nn.Dropout(config.dropout)
        self.bias_update_speed = config.bias_update_speed

    def update_load_balancing(self, expert_load: torch.Tensor):
        """Adjust bias terms for load balancing."""
        target_load = expert_load.mean()
        overload = expert_load > target_load
        underload = expert_load < target_load
        with torch.no_grad():
            self.bias[overload] -= self.bias_update_speed
            self.bias[underload] += self.bias_update_speed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        shared_output = sum(expert(x) for expert in self.shared_experts)

        gate_scores = self.gate(x) + self.bias
        gate_probs = torch.sigmoid(gate_scores)
        topk_scores, topk_indices = gate_probs.topk(self.experts_per_token, dim=-1)
        gate_values = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        routed_output = torch.zeros_like(x)
        for i in range(self.experts_per_token):
            expert_idx = topk_indices[:, :, i]
            expert_output = torch.stack([self.routed_experts[idx](x[b]) for b, idx in enumerate(expert_idx)])
            routed_output += expert_output * gate_values[:, :, i].unsqueeze(-1)

        expert_load = gate_probs.sum(dim=(0, 1))
        self.update_load_balancing(expert_load)

        return x + shared_output + self.dropout(routed_output)
