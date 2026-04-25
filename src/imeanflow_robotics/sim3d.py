from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Arm3DState:
    joints: Tensor
    ee_xyz: Tensor
    target_xyz: Tensor


class SimpleArm3D:
    """3-DoF yaw-shoulder-elbow arm used by the MuJoCo demo."""

    def __init__(
        self,
        upper_arm: float = 0.55,
        forearm: float = 0.45,
        base_height: float = 0.18,
    ):
        self.upper_arm = float(upper_arm)
        self.forearm = float(forearm)
        self.base_height = float(base_height)

    @property
    def obs_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def max_reach(self) -> float:
        return self.upper_arm + self.forearm

    def observation(self, joints: Tensor, target_xyz: Tensor) -> Tensor:
        return torch.cat([joints, target_xyz.to(device=joints.device, dtype=joints.dtype)], dim=-1)

    def forward_kinematics(self, joints: Tensor) -> Tensor:
        """Return base, elbow, and end-effector points with shape [..., 3, 3]."""
        yaw = joints[..., 0]
        shoulder = joints[..., 1]
        elbow = joints[..., 2]

        radial_1 = self.upper_arm * torch.cos(shoulder)
        z_1 = self.base_height + self.upper_arm * torch.sin(shoulder)
        radial_2 = radial_1 + self.forearm * torch.cos(shoulder + elbow)
        z_2 = z_1 + self.forearm * torch.sin(shoulder + elbow)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        base = torch.zeros(*joints.shape[:-1], 3, device=joints.device, dtype=joints.dtype)
        base[..., 2] = self.base_height
        elbow_xyz = torch.stack([radial_1 * cos_yaw, radial_1 * sin_yaw, z_1], dim=-1)
        ee_xyz = torch.stack([radial_2 * cos_yaw, radial_2 * sin_yaw, z_2], dim=-1)
        return torch.stack([base, elbow_xyz, ee_xyz], dim=-2)

    def inverse_kinematics(self, target_xyz: Tensor) -> Tensor:
        """Analytic IK for a yaw plus 2-link vertical arm."""
        target_xyz = target_xyz.to(dtype=torch.float32)
        x = target_xyz[..., 0]
        y = target_xyz[..., 1]
        z = target_xyz[..., 2] - self.base_height
        yaw = torch.atan2(y, x)
        radius = torch.linalg.norm(target_xyz[..., :2], dim=-1)

        dist = torch.sqrt(radius.square() + z.square()).clamp(0.08, self.max_reach - 0.02)
        scale = dist / torch.sqrt(radius.square() + z.square()).clamp_min(1e-6)
        radius = radius * scale
        z = z * scale

        l1 = torch.as_tensor(self.upper_arm, device=target_xyz.device, dtype=target_xyz.dtype)
        l2 = torch.as_tensor(self.forearm, device=target_xyz.device, dtype=target_xyz.dtype)
        cos_elbow = ((dist.square() - l1.square() - l2.square()) / (2.0 * l1 * l2)).clamp(
            -0.98, 0.98
        )
        elbow = torch.acos(cos_elbow)
        shoulder = torch.atan2(z, radius) - torch.atan2(
            l2 * torch.sin(elbow), l1 + l2 * torch.cos(elbow)
        )
        return torch.stack([yaw, shoulder, elbow], dim=-1)

    def sample_target_xyz(self, num_samples: int, generator: torch.Generator) -> Tensor:
        yaw = (torch.rand(num_samples, generator=generator) - 0.5) * 1.7 * math.pi
        radius = 0.25 + 0.65 * torch.rand(num_samples, generator=generator)
        z = 0.08 + 0.65 * torch.rand(num_samples, generator=generator)
        return torch.stack([radius * torch.cos(yaw), radius * torch.sin(yaw), z], dim=-1)


class Reach3DDataset(Dataset):
    """Generated 3D reaching demonstrations for iMeanFlow training."""

    def __init__(
        self,
        num_samples: int = 8192,
        horizon: int = 16,
        seed: int = 13,
        noise_std: float = 0.006,
        arm: SimpleArm3D | None = None,
    ):
        super().__init__()
        self.arm = arm or SimpleArm3D()
        generator = torch.Generator().manual_seed(seed)

        current = torch.empty(num_samples, self.arm.action_dim)
        current[:, 0] = (torch.rand(num_samples, generator=generator) - 0.5) * 2.0 * math.pi
        current[:, 1] = (torch.rand(num_samples, generator=generator) - 0.5) * 1.8
        current[:, 2] = torch.rand(num_samples, generator=generator) * 1.8
        target_xyz = self.arm.sample_target_xyz(num_samples, generator)
        target_joints = self.arm.inverse_kinematics(target_xyz)

        phase = torch.linspace(0.0, 1.0, horizon)
        smooth = 0.5 - 0.5 * torch.cos(math.pi * phase)
        actions = current[:, None, :] + smooth[None, :, None] * (target_joints - current)[:, None, :]
        actions += noise_std * torch.randn(actions.shape, generator=generator)
        actions[:, 0] = current
        actions[:, -1] = target_joints

        self.obs = self.arm.observation(current, target_xyz).float()
        self.actions = actions.float()
        self.action_is_pad = torch.zeros(num_samples, horizon, dtype=torch.bool)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.obs[index], self.actions[index], self.action_is_pad[index]
