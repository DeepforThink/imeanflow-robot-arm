from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.data import SyntheticArmDataset
from imeanflow_robotics.policy import IMeanFlowPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train iMeanFlow on a synthetic arm dataset.")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/imeanflow_synthetic.pt"))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    config = IMeanFlowConfig(
        obs_dim=9,
        action_dim=6,
        horizon=16,
        n_action_steps=8,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inference_steps=args.num_inference_steps,
    )
    dataset = SyntheticArmDataset(obs_dim=config.obs_dim, action_dim=config.action_dim, horizon=config.horizon)
    train_set, val_set = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    policy = IMeanFlowPolicy(config).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    iterator = iter(train_loader)
    progress = tqdm(range(1, args.steps + 1), desc="training")
    for step in progress:
        try:
            obs, actions, action_is_pad = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            obs, actions, action_is_pad = next(iterator)

        obs = obs.to(args.device)
        actions = actions.to(args.device)
        action_is_pad = action_is_pad.to(args.device)

        policy.train()
        loss, metrics = policy.compute_loss(obs, actions, action_is_pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        if step % 20 == 0:
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.4f}",
                loss_u=f"{metrics['loss_u'].item():.4f}",
                loss_v=f"{metrics['loss_v'].item():.4f}",
            )

    policy.eval()
    val_mse = 0.0
    total = 0
    with torch.no_grad():
        for obs, actions, _ in val_loader:
            obs = obs.to(args.device)
            actions = actions.to(args.device)
            pred = policy.sample_action_chunk(obs)
            val_mse += torch.nn.functional.mse_loss(pred, actions, reduction="sum").item()
            total += actions.numel()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "model": policy.state_dict(), "val_mse": val_mse / total}, args.output)
    print(f"saved checkpoint to {args.output}")
    print(f"validation action MSE: {val_mse / total:.6f}")


if __name__ == "__main__":
    main()

