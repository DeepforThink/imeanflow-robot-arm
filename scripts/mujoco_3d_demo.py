from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MuJoCo is required: pip install -e '.[mujoco]'") from exc

from imeanflow_robotics.config import IMeanFlowConfig
from imeanflow_robotics.policy import IMeanFlowPolicy
from imeanflow_robotics.sim3d import Reach3DDataset, SimpleArm3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3D MuJoCo iMeanFlow reaching demo.")
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/mujoco_3d_demo.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--control-cycles", type=int, default=10)
    parser.add_argument("--execute-steps", type=int, default=12)
    parser.add_argument("--frameskip", type=int, default=16)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--mount-height", type=float, default=0.18)
    parser.add_argument("--floor-height", type=float, default=-0.25)
    parser.add_argument("--target-x", type=float, default=0.62)
    parser.add_argument("--target-y", type=float, default=0.18)
    parser.add_argument("--target-z", type=float, default=0.66)
    parser.add_argument("--initial-yaw", type=float, default=-0.35)
    parser.add_argument("--initial-shoulder", type=float, default=-0.30)
    parser.add_argument("--initial-elbow", type=float, default=1.15)
    parser.add_argument("--tracking-gain", type=float, default=1.0)
    parser.add_argument("--max-joint-step", type=float, default=10.0)
    return parser.parse_args()


def build_policy(args: argparse.Namespace) -> IMeanFlowPolicy:
    config = IMeanFlowConfig(
        obs_dim=6,
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


def train_or_load(args: argparse.Namespace, arm: SimpleArm3D) -> IMeanFlowPolicy:
    policy = build_policy(args)
    if args.checkpoint.exists() and not args.force_train:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        policy.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint from {args.checkpoint}")
        return policy

    dataset = Reach3DDataset(num_samples=8192, horizon=policy.config.horizon, arm=arm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    progress = tqdm(range(1, args.train_steps + 1), desc="training mujoco 3d policy")
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

        if step % 50 == 0 or step == args.train_steps:
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.3f}",
                u=f"{metrics['loss_u'].item():.3f}",
                v=f"{metrics['loss_v'].item():.3f}",
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": policy.config.__dict__, "model": policy.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")
    return policy


def make_mujoco_xml(target_xyz: np.ndarray, mount_height: float, floor_height: float) -> str:
    tx, ty, tz = target_xyz
    pedestal_z = 0.5 * (mount_height + floor_height)
    pedestal_half_height = 0.5 * (mount_height - floor_height)
    return textwrap.dedent(
        f"""
        <mujoco model="imeanflow_3d_arm">
          <compiler angle="radian" inertiafromgeom="true"/>
          <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
          <visual>
            <global offwidth="640" offheight="480"/>
            <quality shadowsize="2048"/>
          </visual>
          <worldbody>
            <light pos="0 -2 3" dir="0 1 -1"/>
            <camera name="demo" pos="1.8 -2.4 1.6" xyaxes="0.82 0.57 0 -0.30 0.43 0.85"/>
            <geom type="plane" pos="0 0 {floor_height:.4f}" size="2 2 0.02" rgba="0.92 0.92 0.88 1"/>
            <geom type="cylinder" pos="0 0 {pedestal_z:.4f}" size="0.075 {pedestal_half_height:.4f}" rgba="0.35 0.35 0.38 1"/>
            <site name="target" pos="{tx:.4f} {ty:.4f} {tz:.4f}" type="sphere" size="0.045" rgba="1 0.1 0.1 1"/>
            <body name="base" pos="0 0 {mount_height:.4f}">
              <joint name="yaw" type="hinge" axis="0 0 1" damping="1.2" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.08 0.08" rgba="0.2 0.2 0.25 1"/>
              <body name="upper" pos="0 0 0">
                <joint name="shoulder" type="hinge" axis="0 -1 0" damping="1.4" range="-1.57 1.57"/>
                <geom type="capsule" fromto="0 0 0 0.55 0 0" size="0.035" rgba="0.2 0.4 0.9 1"/>
                <body name="forearm" pos="0.55 0 0">
                  <joint name="elbow" type="hinge" axis="0 -1 0" damping="1.0" range="-0.05 2.60"/>
                  <geom type="capsule" fromto="0 0 0 0.45 0 0" size="0.03" rgba="0.1 0.7 0.5 1"/>
                  <site name="ee" pos="0.45 0 0" type="sphere" size="0.035" rgba="0.05 0.05 0.08 1"/>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="yaw_pos" joint="yaw" kp="45" kv="7" ctrlrange="-3.14 3.14"/>
            <position name="shoulder_pos" joint="shoulder" kp="55" kv="8" ctrlrange="-1.57 1.57"/>
            <position name="elbow_pos" joint="elbow" kp="45" kv="7" ctrlrange="-0.05 2.60"/>
          </actuator>
        </mujoco>
        """
    )


@torch.no_grad()
def run_mujoco_rollout(
    policy: IMeanFlowPolicy,
    arm: SimpleArm3D,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[np.ndarray]]:
    torch.manual_seed(args.seed)
    policy_target_xyz = torch.tensor([args.target_x, args.target_y, args.target_z], dtype=torch.float32)
    mount_offset = args.mount_height - arm.base_height
    mujoco_target_xyz = policy_target_xyz + torch.tensor([0.0, 0.0, mount_offset], dtype=torch.float32)
    initial_q = torch.tensor(
        [args.initial_yaw, args.initial_shoulder, args.initial_elbow], dtype=torch.float32
    )

    model = mujoco.MjModel.from_xml_string(
        make_mujoco_xml(mujoco_target_xyz.numpy(), args.mount_height, args.floor_height)
    )
    data = mujoco.MjData(model)
    data.qpos[:] = initial_q.numpy()
    data.ctrl[:] = initial_q.numpy()
    mujoco.mj_forward(model, data)

    ee_id = model.site("ee").id
    camera_id = model.camera("demo").id
    renderer = mujoco.Renderer(model, height=480, width=640)

    q_history = []
    ee_history = []
    frames: list[np.ndarray] = []

    policy.eval()
    for _ in range(args.control_cycles):
        q = torch.tensor(data.qpos[:3].copy(), dtype=torch.float32, device=policy.device)
        obs = arm.observation(q, policy_target_xyz.to(policy.device)).unsqueeze(0)
        if args.num_candidates > 1:
            obs_batch = obs.expand(args.num_candidates, -1)
            candidates = policy.sample_action_chunk(obs_batch, num_steps=args.num_inference_steps)
            candidate_ee = arm.forward_kinematics(candidates[:, -1])[:, -1]
            distances = torch.linalg.norm(candidate_ee - policy_target_xyz.to(policy.device), dim=-1)
            chunk = candidates[distances.argmin()].cpu().numpy()
        else:
            chunk = policy.sample_action_chunk(obs, num_steps=args.num_inference_steps)[0].cpu().numpy()
        for command in chunk[1 : args.execute_steps + 1]:
            command = np.clip(command, [-np.pi, -1.35, 0.02], [np.pi, 1.35, 2.55])
            delta = np.clip(
                args.tracking_gain * (command - data.ctrl[:]),
                -args.max_joint_step,
                args.max_joint_step,
            )
            data.ctrl[:] = data.ctrl[:] + delta
            for _ in range(args.frameskip):
                mujoco.mj_step(model, data)
            q_history.append(data.qpos[:3].copy())
            ee_history.append(data.site_xpos[ee_id].copy())
            renderer.update_scene(data, camera=camera_id)
            frames.append(renderer.render().copy())

    renderer.close()
    ee = np.asarray(ee_history)
    final_distance = float(np.linalg.norm(ee[-1] - mujoco_target_xyz.numpy()))
    return np.asarray(q_history), ee, mujoco_target_xyz.numpy(), final_distance, frames


def save_plot(ee: np.ndarray, target_xyz: np.ndarray, final_distance: float, path: Path) -> None:
    fig = plt.figure(figsize=(6, 5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#2563eb", linewidth=2, label="end-effector")
    ax.scatter([target_xyz[0]], [target_xyz[1]], [target_xyz[2]], color="#dc2626", s=90, label="target")
    ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], color="#64748b", s=45, label="start")
    ax.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], color="#0f172a", s=55, label="final")
    ax.set_title(f"MuJoCo 3D reach | final distance {final_distance:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.legend(loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    arm = SimpleArm3D()
    policy = train_or_load(args, arm)

    _, ee, target_xyz, final_distance, frames = run_mujoco_rollout(policy, arm, args)
    png_path = args.output_dir / "mujoco_3d_demo.png"
    gif_path = args.output_dir / "mujoco_3d_demo.gif"
    save_plot(ee, target_xyz, final_distance, png_path)
    save_gif(frames, gif_path)
    print(f"saved {png_path}")
    print(f"saved {gif_path}")
    print(f"final MuJoCo end-effector distance: {final_distance:.4f}")


if __name__ == "__main__":
    main()
