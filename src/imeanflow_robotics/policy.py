from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.model import IMeanFlowTransformer


class IMeanFlowPolicy(nn.Module):
    """Conditional iMeanFlow policy for robotic arm action chunks."""

    def __init__(self, config: IMeanFlowConfig):
        super().__init__()
        self.config = config
        self.model = IMeanFlowTransformer(config)
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset(self) -> None:
        self._action_queue.clear()

    def sample_timesteps(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Sample t>=r, with some r=t samples for local flow matching stabilization."""
        rnd_t = torch.randn(batch_size, device=device)
        rnd_r = torch.randn(batch_size, device=device)
        t = torch.sigmoid(rnd_t * self.config.p_std + self.config.p_mean)
        r = torch.sigmoid(rnd_r * self.config.p_std + self.config.p_mean)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        if self.config.ratio < 1.0:
            local_mask = torch.rand(batch_size, device=device) < (1.0 - self.config.ratio)
            r = torch.where(local_mask, t, r)
        return t, r

    def compute_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the no-CFG iMeanFlow training objective.

        The target velocity is the straight-line flow velocity from action to noise.
        The u-head learns the compound iMeanFlow velocity:

            V = u + (t-r) * stopgrad(JVP(u))

        The v-head is an auxiliary instantaneous velocity predictor and provides
        the tangent direction used by the JVP.
        """
        batch_size = actions.shape[0]
        device = actions.device

        t, r = self.sample_timesteps(batch_size, device)
        noise = torch.randn_like(actions)
        z = (1.0 - t[:, None, None]) * actions + t[:, None, None] * noise
        v_target = noise - actions

        with torch.no_grad():
            h0 = torch.zeros_like(t)
            _, v_tangent = self.model(z, h0, obs)

        def u_func(z_arg: Tensor, t_arg: Tensor, r_arg: Tensor) -> Tensor:
            h_arg = t_arg - r_arg
            u_arg, _ = self.model(z_arg, h_arg, obs)
            return u_arg

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)
        u_pred, dudt = torch.func.jvp(
            u_func,
            (z, t, r),
            (v_tangent.detach(), dtdt, drdt),
        )
        v_compound = u_pred + (t - r)[:, None, None] * dudt.detach()

        _, v_pred = self.model(z, t - r, obs)
        loss_u_sq = (v_compound - v_target).pow(2)
        loss_v_sq = (v_pred - v_target).pow(2)

        if action_is_pad is not None:
            valid = ~action_is_pad
            loss_u_sq = loss_u_sq * valid.unsqueeze(-1)
            loss_v_sq = loss_v_sq * valid.unsqueeze(-1)

        loss_u_per_sample = loss_u_sq.sum(dim=(1, 2))
        loss_v_per_sample = loss_v_sq.sum(dim=(1, 2))

        if self.config.norm_p > 0:
            loss_u = (
                loss_u_per_sample
                / (loss_u_per_sample.detach() + self.config.norm_eps).pow(self.config.norm_p)
            ).mean()
            loss_v = (
                loss_v_per_sample
                / (loss_v_per_sample.detach() + self.config.norm_eps).pow(self.config.norm_p)
            ).mean()
        else:
            loss_u = loss_u_per_sample.mean()
            loss_v = loss_v_per_sample.mean()

        loss = loss_u + self.config.v_loss_weight * loss_v
        metrics = {
            "loss": loss.detach(),
            "loss_u": loss_u.detach(),
            "loss_v": loss_v.detach(),
            "mean_h": (t - r).mean().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Generate a full action chunk using few-step Euler integration."""
        self.eval()
        batch_size = obs.shape[0]
        num_steps = num_steps or self.config.num_inference_steps
        if noise is None:
            z = torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_dim,
                device=obs.device,
                dtype=obs.dtype,
            )
        else:
            z = noise

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=obs.device, dtype=obs.dtype)
        for i in range(num_steps):
            h = (t_steps[i] - t_steps[i + 1]).expand(batch_size)
            u, _ = self.model(z, h, obs)
            z = z - h[:, None, None] * u
        return z

    @torch.no_grad()
    def select_action(self, obs: Tensor) -> Tensor:
        """Return one executable action, caching generated chunks internally."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if not self._action_queue:
            chunk = self.sample_action_chunk(obs)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

