from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticArmDataset(Dataset):
    """Small synthetic dataset for smoke-testing action generation.

    Observation layout:
        [current_joint_positions, target_position]

    Action chunk:
        Smooth future joint commands that move from current joints toward a
        deterministic target-dependent joint configuration.
    """

    def __init__(
        self,
        num_samples: int = 4096,
        obs_dim: int = 9,
        action_dim: int = 6,
        horizon: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        if obs_dim < action_dim + 3:
            raise ValueError("obs_dim must be at least action_dim + 3")

        generator = torch.Generator().manual_seed(seed)
        current = torch.randn(num_samples, action_dim, generator=generator) * 0.5
        target_xyz = torch.empty(num_samples, 3).uniform_(-1.0, 1.0, generator=generator)

        target_joints = torch.zeros(num_samples, action_dim)
        target_joints[:, 0] = torch.atan2(target_xyz[:, 1], target_xyz[:, 0])
        target_joints[:, 1] = 0.8 * target_xyz[:, 2]
        target_joints[:, 2] = torch.linalg.norm(target_xyz[:, :2], dim=-1) - 0.5
        for j in range(3, action_dim):
            target_joints[:, j] = torch.sin((j + 1) * target_xyz[:, j % 3])

        phase = torch.linspace(0.0, 1.0, horizon)
        smooth = 0.5 - 0.5 * torch.cos(math.pi * phase)
        actions = current[:, None, :] + smooth[None, :, None] * (target_joints - current)[:, None, :]

        actions += 0.015 * torch.randn(actions.shape, generator=generator)

        obs = torch.zeros(num_samples, obs_dim)
        obs[:, :action_dim] = current
        obs[:, action_dim : action_dim + 3] = target_xyz
        if obs_dim > action_dim + 3:
            obs[:, action_dim + 3 :] = torch.randn(
                num_samples, obs_dim - action_dim - 3, generator=generator
            ) * 0.05

        self.obs = obs.float()
        self.actions = actions.float()
        self.action_is_pad = torch.zeros(num_samples, horizon, dtype=torch.bool)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.obs[index], self.actions[index], self.action_is_pad[index]

