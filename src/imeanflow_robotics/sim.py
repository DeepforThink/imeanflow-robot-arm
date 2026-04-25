from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from imeanflow_robotics.policy import IMeanFlowPolicy


@dataclass(frozen=True)
class RolloutResult:
    """Container returned by the planar-arm simulation rollout."""

    joints: Tensor
    points: Tensor
    target_xy: Tensor
    final_distance: float


class PlanarArm2D:
    """Simple 3-DoF planar arm used for a lightweight action-generation demo."""

    def __init__(self, link_lengths: tuple[float, float, float] = (0.65, 0.45, 0.30)):
        self.link_lengths = torch.tensor(link_lengths, dtype=torch.float32)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def obs_dim(self) -> int:
        return 5

    @property
    def max_reach(self) -> float:
        return float(self.link_lengths.sum())

    def forward_kinematics(self, joints: Tensor) -> Tensor:
        """Return link endpoints with shape [..., 4, 2], including the base."""
        if joints.shape[-1] != self.action_dim:
            raise ValueError(f"expected last joint dimension {self.action_dim}, got {joints.shape[-1]}")

        lengths = self.link_lengths.to(device=joints.device, dtype=joints.dtype)
        angles = torch.cumsum(joints, dim=-1)
        deltas = torch.stack([lengths * torch.cos(angles), lengths * torch.sin(angles)], dim=-1)
        endpoints = torch.cumsum(deltas, dim=-2)
        base = torch.zeros(*joints.shape[:-1], 1, 2, device=joints.device, dtype=joints.dtype)
        return torch.cat([base, endpoints], dim=-2)

    def inverse_kinematics(self, target_xy: Tensor) -> Tensor:
        """Analytic elbow-up target joints for the planar reaching task.

        The second and third links are treated as a straight wrist segment during
        target construction. This makes each target position map to one stable
        joint target, which keeps the demo focused on action generation rather
        than multimodal inverse kinematics.
        """
        target_xy = target_xy.to(dtype=torch.float32)
        l1 = self.link_lengths[0].to(target_xy.device)
        l23 = self.link_lengths[1:].sum().to(target_xy.device)

        x = target_xy[..., 0]
        y = target_xy[..., 1]
        radius = torch.linalg.norm(target_xy, dim=-1).clamp(0.12, self.max_reach - 0.03)
        scale = radius / torch.linalg.norm(target_xy, dim=-1).clamp_min(1e-6)
        x = x * scale
        y = y * scale

        cos_q2 = ((radius**2 - l1**2 - l23**2) / (2.0 * l1 * l23)).clamp(-0.98, 0.98)
        q2 = torch.acos(cos_q2)
        q1 = torch.atan2(y, x) - torch.atan2(l23 * torch.sin(q2), l1 + l23 * torch.cos(q2))
        q3 = torch.zeros_like(q1)
        return torch.stack([q1, q2, q3], dim=-1)

    def observation(self, joints: Tensor, target_xy: Tensor) -> Tensor:
        return torch.cat([joints, target_xy.to(device=joints.device, dtype=joints.dtype)], dim=-1)


class PlanarReachDataset(Dataset):
    """Reaching demonstrations for the planar-arm simulation.

    Observation layout:
        [current_joint_0, current_joint_1, current_joint_2, target_x, target_y]

    Action chunk:
        Absolute joint commands that smoothly move from current joints to the
        deterministic inverse-kinematics target for the requested end-effector
        position.
    """

    def __init__(
        self,
        num_samples: int = 4096,
        horizon: int = 16,
        seed: int = 7,
        noise_std: float = 0.01,
        arm: PlanarArm2D | None = None,
    ):
        super().__init__()
        self.arm = arm or PlanarArm2D()
        generator = torch.Generator().manual_seed(seed)

        current = (torch.rand(num_samples, self.arm.action_dim, generator=generator) - 0.5) * 2.4
        target_xy = self._sample_reachable_targets(num_samples, generator)
        target_joints = self.arm.inverse_kinematics(target_xy)

        phase = torch.linspace(0.0, 1.0, horizon)
        smooth = 0.5 - 0.5 * torch.cos(math.pi * phase)
        actions = current[:, None, :] + smooth[None, :, None] * (target_joints - current)[:, None, :]
        actions += noise_std * torch.randn(actions.shape, generator=generator)
        actions[:, 0] = current
        actions[:, -1] = target_joints

        self.obs = self.arm.observation(current, target_xy).float()
        self.actions = actions.float()
        self.action_is_pad = torch.zeros(num_samples, horizon, dtype=torch.bool)
        self.target_xy = target_xy.float()
        self.target_joints = target_joints.float()

    def _sample_reachable_targets(self, num_samples: int, generator: torch.Generator) -> Tensor:
        theta = (torch.rand(num_samples, generator=generator) - 0.5) * 2.0 * math.pi
        radius = 0.25 + (self.arm.max_reach - 0.35) * torch.rand(num_samples, generator=generator)
        return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.obs[index], self.actions[index], self.action_is_pad[index]


@torch.no_grad()
def rollout_planar_policy(
    policy: IMeanFlowPolicy,
    arm: PlanarArm2D,
    initial_joints: Tensor,
    target_xy: Tensor,
    control_cycles: int = 12,
    execute_steps: int = 4,
    num_inference_steps: int | None = None,
    num_candidates: int = 1,
    tracking_gain: float = 0.70,
    max_joint_step: float = 0.25,
) -> RolloutResult:
    """Run receding-horizon control with generated action chunks."""
    device = policy.device
    joints = initial_joints.to(device=device, dtype=torch.float32).clone()
    target_xy = target_xy.to(device=device, dtype=torch.float32)
    joint_history = [joints.detach().cpu()]

    for _ in range(control_cycles):
        obs = arm.observation(joints, target_xy).unsqueeze(0).to(device)
        if num_candidates > 1:
            obs_batch = obs.expand(num_candidates, -1)
            candidates = policy.sample_action_chunk(obs_batch, num_steps=num_inference_steps)
            candidate_ee = arm.forward_kinematics(candidates[:, -1])[:, -1]
            distances = torch.linalg.norm(candidate_ee - target_xy, dim=-1)
            chunk = candidates[distances.argmin()]
        else:
            chunk = policy.sample_action_chunk(obs, num_steps=num_inference_steps)[0]
        for command in chunk[:execute_steps]:
            delta = tracking_gain * (command - joints)
            delta = torch.clamp(delta, min=-max_joint_step, max=max_joint_step)
            joints = joints + delta
            joints = torch.clamp(joints, -math.pi, math.pi)
            joint_history.append(joints.detach().cpu())

    joint_tensor = torch.stack(joint_history)
    points = arm.forward_kinematics(joint_tensor)
    target_cpu = target_xy.detach().cpu()
    final_distance = torch.linalg.norm(points[-1, -1] - target_cpu).item()
    return RolloutResult(
        joints=joint_tensor,
        points=points,
        target_xy=target_cpu,
        final_distance=final_distance,
    )
