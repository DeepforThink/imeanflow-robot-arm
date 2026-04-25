from __future__ import annotations

import argparse
from pathlib import Path

import torch

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.data import SyntheticArmDataset
from imeanflow_robotics.policy import IMeanFlowPolicy


def load_policy(path: Path, device: str) -> IMeanFlowPolicy:
    checkpoint = torch.load(path, map_location=device)
    config = IMeanFlowConfig(**checkpoint["config"])
    policy = IMeanFlowPolicy(config).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained iMeanFlow checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/imeanflow_synthetic.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-steps", type=int, default=None)
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, args.device)
    dataset = SyntheticArmDataset(
        num_samples=256,
        obs_dim=policy.config.obs_dim,
        action_dim=policy.config.action_dim,
        horizon=policy.config.horizon,
        seed=123,
    )

    obs = dataset.obs.to(args.device)
    target = dataset.actions.to(args.device)
    with torch.no_grad():
        pred = policy.sample_action_chunk(obs, num_steps=args.num_steps)
        mse = torch.nn.functional.mse_loss(pred, target).item()
        first_action_mse = torch.nn.functional.mse_loss(pred[:, 0], target[:, 0]).item()

    print(f"full chunk MSE: {mse:.6f}")
    print(f"first action MSE: {first_action_mse:.6f}")
    print("sample predicted action[0, :3]:")
    print(pred[0, :3].cpu())


if __name__ == "__main__":
    main()

