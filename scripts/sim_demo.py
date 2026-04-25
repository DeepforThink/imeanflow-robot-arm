from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.policy import IMeanFlowPolicy
from imeanflow_robotics.sim import PlanarArm2D, PlanarReachDataset, RolloutResult, rollout_planar_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and render a planar-arm iMeanFlow demo.")
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/planar_arm_demo.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--rollout-seed", type=int, default=7)
    parser.add_argument("--control-cycles", type=int, default=12)
    parser.add_argument("--tracking-gain", type=float, default=0.70)
    parser.add_argument("--max-joint-step", type=float, default=0.25)
    return parser.parse_args()


def build_policy(args: argparse.Namespace) -> IMeanFlowPolicy:
    config = IMeanFlowConfig(
        obs_dim=5,
        action_dim=3,
        horizon=16,
        n_action_steps=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inference_steps=args.num_inference_steps,
        norm_p=0.0,
        v_loss_weight=0.5,
    )
    return IMeanFlowPolicy(config).to(args.device)


def train_or_load_policy(args: argparse.Namespace, arm: PlanarArm2D) -> IMeanFlowPolicy:
    policy = build_policy(args)
    if args.checkpoint.exists() and not args.force_train:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        policy.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint from {args.checkpoint}")
        return policy

    dataset = PlanarReachDataset(num_samples=4096, horizon=policy.config.horizon, arm=arm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    progress = tqdm(range(1, args.train_steps + 1), desc="training planar demo")
    for step in progress:
        try:
            obs, actions, action_is_pad = next(iterator)
        except StopIteration:
            iterator = iter(loader)
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

        if step % 25 == 0 or step == args.train_steps:
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.4f}",
                u=f"{metrics['loss_u'].item():.4f}",
                v=f"{metrics['loss_v'].item():.4f}",
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": policy.config.__dict__, "model": policy.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")
    return policy


def render_static(result: RolloutResult, path: Path, arm: PlanarArm2D) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.set_title("iMeanFlow planar arm reaching demo")
    ax.set_xlim(-arm.max_reach - 0.15, arm.max_reach + 0.15)
    ax.set_ylim(-arm.max_reach - 0.15, arm.max_reach + 0.15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ee = result.points[:, -1]
    ax.plot(ee[:, 0], ee[:, 1], color="#2563eb", linewidth=2, label="end-effector path")
    ax.scatter([result.target_xy[0]], [result.target_xy[1]], marker="*", s=190, color="#dc2626", label="target")
    ax.plot(result.points[0, :, 0], result.points[0, :, 1], "o--", color="#64748b", label="start")
    ax.plot(result.points[-1, :, 0], result.points[-1, :, 1], "o-", color="#0f172a", linewidth=3, label="final")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        f"final distance: {result.final_distance:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_gif(result: RolloutResult, path: Path, arm: PlanarArm2D) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_xlim(-arm.max_reach - 0.15, arm.max_reach + 0.15)
    ax.set_ylim(-arm.max_reach - 0.15, arm.max_reach + 0.15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.scatter([result.target_xy[0]], [result.target_xy[1]], marker="*", s=170, color="#dc2626")
    (arm_line,) = ax.plot([], [], "o-", color="#0f172a", linewidth=3, markersize=6)
    (trace_line,) = ax.plot([], [], color="#2563eb", linewidth=2, alpha=0.85)
    status = ax.text(0.02, 0.94, "", transform=ax.transAxes, fontsize=10)

    def update(frame: int):
        points = result.points[frame]
        ee_trace = result.points[: frame + 1, -1]
        arm_line.set_data(points[:, 0], points[:, 1])
        trace_line.set_data(ee_trace[:, 0], ee_trace[:, 1])
        dist = torch.linalg.norm(points[-1] - result.target_xy).item()
        status.set_text(f"step {frame:02d} | dist {dist:.3f}")
        return arm_line, trace_line, status

    animation = FuncAnimation(fig, update, frames=len(result.points), interval=90, blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(11)

    arm = PlanarArm2D()
    policy = train_or_load_policy(args, arm)
    policy.eval()

    initial_joints = torch.tensor([-1.25, 0.55, -0.35])
    target_xy = torch.tensor([1.10, 0.50])
    torch.manual_seed(args.rollout_seed)
    result = rollout_planar_policy(
        policy=policy,
        arm=arm,
        initial_joints=initial_joints,
        target_xy=target_xy,
        control_cycles=args.control_cycles,
        execute_steps=policy.config.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        num_candidates=args.num_candidates,
        tracking_gain=args.tracking_gain,
        max_joint_step=args.max_joint_step,
    )

    png_path = args.output_dir / "planar_arm_demo.png"
    gif_path = args.output_dir / "planar_arm_demo.gif"
    render_static(result, png_path, arm)
    render_gif(result, gif_path, arm)
    print(f"saved {png_path}")
    print(f"saved {gif_path}")
    print(f"final end-effector distance: {result.final_distance:.4f}")


if __name__ == "__main__":
    main()
