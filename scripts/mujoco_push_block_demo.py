from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MuJoCo is required: pip install -e '.[mujoco]'") from exc

from imeanflow_robotics.sim3d import SimpleArm3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one scripted MuJoCo push-block episode.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--data-path", type=Path, default=Path("data/push_block_demo/episode_000.npz"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--mount-height", type=float, default=0.34)
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument("--block-x", type=float, default=0.42)
    parser.add_argument("--block-y", type=float, default=-0.12)
    parser.add_argument("--target-x", type=float, default=0.60)
    parser.add_argument("--target-y", type=float, default=0.08)
    parser.add_argument("--frameskip", type=int, default=12)
    parser.add_argument("--save-data", action="store_true")
    return parser.parse_args()


def make_xml(mount_height: float, table_height: float, block_xy: np.ndarray, target_xy: np.ndarray) -> str:
    bx, by = block_xy
    tx, ty = target_xy
    return textwrap.dedent(
        f"""
        <mujoco model="imeanflow_push_block">
          <compiler angle="radian" inertiafromgeom="true"/>
          <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
          <visual>
            <global offwidth="640" offheight="480"/>
            <quality shadowsize="2048"/>
          </visual>
          <default>
            <geom solref="0.008 1" solimp="0.95 0.99 0.001" friction="1.2 0.05 0.01"/>
          </default>
          <worldbody>
            <light pos="0 -2 3" dir="0 1 -1"/>
            <camera name="demo" pos="1.35 -1.75 1.15" xyaxes="0.79 0.61 0 -0.33 0.43 0.84"/>
            <geom name="table" type="box" pos="0.50 0 {table_height - 0.035:.4f}" size="0.65 0.45 0.035" rgba="0.88 0.86 0.80 1"/>
            <geom name="base_stand" type="cylinder" pos="0 0 {0.5 * mount_height:.4f}" size="0.075 {0.5 * mount_height:.4f}" rgba="0.35 0.35 0.38 1"/>
            <site name="target" pos="{tx:.4f} {ty:.4f} {table_height + 0.012:.4f}" type="cylinder" size="0.075 0.006" rgba="1 0.1 0.1 0.6"/>
            <body name="block" pos="{bx:.4f} {by:.4f} {table_height + 0.045:.4f}">
              <freejoint/>
              <geom type="box" size="0.045 0.045 0.045" mass="0.25" rgba="0.95 0.55 0.15 1"/>
            </body>
            <body name="base" pos="0 0 {mount_height:.4f}">
              <joint name="yaw" type="hinge" axis="0 0 1" damping="1.2" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.08 0.08" rgba="0.2 0.2 0.25 1" contype="0" conaffinity="0"/>
              <body name="upper" pos="0 0 0">
                <joint name="shoulder" type="hinge" axis="0 -1 0" damping="1.4" range="-1.57 1.40"/>
                <geom type="capsule" fromto="0 0 0 0.55 0 0" size="0.03" rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>
                <body name="forearm" pos="0.55 0 0">
                  <joint name="elbow" type="hinge" axis="0 -1 0" damping="1.0" range="0.02 2.60"/>
                  <geom type="capsule" fromto="0 0 0 0.45 0 0" size="0.026" rgba="0.1 0.7 0.5 1" contype="0" conaffinity="0"/>
                  <body name="tool" pos="0.45 0 0">
                    <geom name="pusher" type="sphere" size="0.04" mass="0.03" rgba="0.05 0.05 0.08 1"/>
                    <site name="ee" pos="0 0 0" type="sphere" size="0.02" rgba="0.05 0.05 0.08 1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="yaw_pos" joint="yaw" kp="55" kv="8" ctrlrange="-3.14 3.14"/>
            <position name="shoulder_pos" joint="shoulder" kp="65" kv="9" ctrlrange="-1.57 1.40"/>
            <position name="elbow_pos" joint="elbow" kp="55" kv="8" ctrlrange="0.02 2.60"/>
          </actuator>
        </mujoco>
        """
    )


def scripted_waypoints(block_xy: np.ndarray, target_xy: np.ndarray, z: float) -> np.ndarray:
    direction = target_xy - block_xy
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    side = np.array([-direction[1], direction[0]])
    return np.asarray(
        [
            np.r_[block_xy - 0.22 * direction + 0.04 * side, z + 0.14],
            np.r_[block_xy - 0.18 * direction + 0.02 * side, z],
            np.r_[block_xy - 0.07 * direction, z],
            np.r_[block_xy + 0.10 * direction, z],
            np.r_[target_xy - 0.06 * direction, z],
            np.r_[target_xy - 0.06 * direction, z + 0.12],
        ],
        dtype=np.float32,
    )


def render_plot(ee: np.ndarray, block_xy: np.ndarray, target_xy: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.set_title("MuJoCo push-block scripted episode")
    ax.set_aspect("equal")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(-0.35, 0.35)
    ax.grid(True, alpha=0.25)
    ax.plot(ee[:, 0], ee[:, 1], color="#2563eb", linewidth=2, label="end-effector path")
    ax.scatter([block_xy[0]], [block_xy[1]], color="#f97316", s=75, label="final block")
    ax.scatter([target_xy[0]], [target_xy[1]], color="#dc2626", s=120, marker="*", label="target")
    ax.legend(loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0)


def run_episode(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    block_xy = np.array([args.block_x, args.block_y], dtype=np.float32)
    target_xy = np.array([args.target_x, args.target_y], dtype=np.float32)
    block_xy += rng.normal(0.0, 0.01, size=2).astype(np.float32)

    model = mujoco.MjModel.from_xml_string(
        make_xml(args.mount_height, args.table_height, block_xy, target_xy)
    )
    data = mujoco.MjData(model)
    arm = SimpleArm3D(base_height=args.mount_height)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = model.camera("demo").id
    ee_id = model.site("ee").id
    block_body_id = model.body("block").id
    robot_joint_ids = [model.joint(name).id for name in ("yaw", "shoulder", "elbow")]
    robot_qpos_addr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in robot_joint_ids])
    robot_qvel_addr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in robot_joint_ids])

    home_q = arm.inverse_kinematics(
        np_to_torch(np.array([0.45, -0.22, args.table_height + 0.20], dtype=np.float32))
    ).numpy()
    data.qpos[robot_qpos_addr] = home_q
    data.ctrl[:] = home_q
    mujoco.mj_forward(model, data)

    waypoints = scripted_waypoints(block_xy, target_xy, args.table_height + 0.105)
    obs_history = []
    action_history = []
    ee_history = []
    block_history = []
    frames: list[np.ndarray] = []

    current_ctrl = data.ctrl[:].copy()
    for waypoint in waypoints:
        q_target = arm.inverse_kinematics(np_to_torch(waypoint)).numpy()
        q_target = np.clip(q_target, [-np.pi, -1.45, 0.04], [np.pi, 1.25, 2.50])
        for alpha in np.linspace(0.0, 1.0, 28, endpoint=True):
            data.ctrl[:] = (1.0 - alpha) * current_ctrl + alpha * q_target
            for _ in range(args.frameskip):
                mujoco.mj_step(model, data)
            ee_xyz = data.site_xpos[ee_id].copy()
            current_block_xy = data.xpos[block_body_id][:2].copy()
            obs = np.concatenate(
                [
                    data.qpos[robot_qpos_addr].copy(),
                    data.qvel[robot_qvel_addr].copy(),
                    ee_xyz,
                    current_block_xy,
                    target_xy,
                ]
            )
            obs_history.append(obs)
            action_history.append(waypoint.copy())
            ee_history.append(ee_xyz)
            block_history.append(current_block_xy)
            renderer.update_scene(data, camera=camera_id)
            frames.append(renderer.render().copy())
        current_ctrl = q_target

    renderer.close()
    return (
        np.asarray(obs_history, dtype=np.float32),
        np.asarray(action_history, dtype=np.float32),
        np.asarray(ee_history, dtype=np.float32),
        np.asarray(block_history, dtype=np.float32),
        frames,
    )


def np_to_torch(array: np.ndarray):
    import torch

    return torch.tensor(array, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    obs, actions, ee, block_xy, frames = run_episode(args)
    target_xy = obs[-1, -2:]
    final_error = float(np.linalg.norm(block_xy[-1] - target_xy))
    success = final_error < 0.08

    png_path = args.output_dir / "push_block_scripted.png"
    gif_path = args.output_dir / "push_block_scripted.gif"
    render_plot(ee, block_xy[-1], target_xy, png_path)
    save_gif(frames, gif_path)

    if args.save_data:
        args.data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.data_path,
            observation=obs,
            action=actions,
            ee_xyz=ee,
            block_xy=block_xy,
            target_xy=target_xy,
            success=np.asarray(success),
            final_error=np.asarray(final_error, dtype=np.float32),
        )
        print(f"saved episode data to {args.data_path}")

    print(f"saved {png_path}")
    print(f"saved {gif_path}")
    print(f"final block error: {final_error:.4f}")
    print(f"success: {success}")


if __name__ == "__main__":
    main()
