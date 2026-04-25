from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.time_embedding import TimestepEmbedder


class Mlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float):
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    """Manual attention implementation compatible with torch.func.jvp on CPU."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim**-0.5
        weights = torch.matmul(q * scale, k.transpose(-2, -1))
        weights = torch.softmax(weights, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, mlp_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class IMeanFlowTransformer(nn.Module):
    """Transformer backbone with shared trunk and u/v action heads.

    The model is conditional on robot observations and the interval length h=t-r.
    It intentionally does not implement classifier-free guidance. In robot action
    generation, observation conditioning is usually strong enough, and a fake CFG
    interface without unconditional training is misleading.
    """

    def __init__(self, config: IMeanFlowConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.action_in = nn.Linear(config.action_dim, hidden_dim)
        self.obs_in = nn.Linear(config.obs_dim, hidden_dim)
        self.h_in = TimestepEmbedder(hidden_dim)

        self.action_pos = nn.Parameter(torch.randn(1, config.horizon, hidden_dim) * 0.02)
        self.obs_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.time_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        shared_layers = max(1, config.num_layers - 2)
        head_layers = max(1, config.num_layers - shared_layers)

        self.shared = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(shared_layers)
            ]
        )
        self.u_head = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(head_layers)
            ]
        )
        self.v_head = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(head_layers)
            ]
        )

        self.u_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, config.action_dim))
        self.v_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, config.action_dim))

        nn.init.zeros_(self.u_out[-1].weight)
        nn.init.zeros_(self.u_out[-1].bias)
        nn.init.zeros_(self.v_out[-1].weight)
        nn.init.zeros_(self.v_out[-1].bias)

    def forward(self, noisy_actions: Tensor, h: Tensor, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Predict average velocity u and auxiliary instantaneous velocity v.

        Args:
            noisy_actions: Tensor with shape [B, H, A].
            h: Interval length t-r, shape [B].
            obs: Observation vector, shape [B, obs_dim].
        """
        batch_size = noisy_actions.shape[0]

        action_tokens = self.action_in(noisy_actions) + self.action_pos
        obs_tokens = self.obs_token.expand(batch_size, -1, -1) + self.obs_in(obs).unsqueeze(1)
        time_tokens = self.time_token.expand(batch_size, -1, -1) + self.h_in(h).unsqueeze(1)

        seq = torch.cat([obs_tokens, time_tokens, action_tokens], dim=1)
        for layer in self.shared:
            seq = layer(seq)

        u_seq = seq
        v_seq = seq
        for layer in self.u_head:
            u_seq = layer(u_seq)
        for layer in self.v_head:
            v_seq = layer(v_seq)

        action_start = 2
        u = self.u_out(u_seq[:, action_start:])
        v = self.v_out(v_seq[:, action_start:])
        return u, v
