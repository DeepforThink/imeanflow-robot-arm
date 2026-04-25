import math

import torch
from torch import Tensor, nn


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Create sinusoidal embeddings for scalar timesteps."""
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Small MLP on top of sinusoidal timestep features."""

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.net = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.net(sinusoidal_embedding(t, self.freq_dim))

