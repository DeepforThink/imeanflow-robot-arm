from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MuJoCo viewer is required: pip install -e '.[mujoco]'") from exc

from imeanflow_robotics.sim3d import SimpleArm3D
from scripts.mujoco_push_block_demo import make_xml, np_to_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive MuJoCo push-block viewer.")
    parser.add_argument("--mount-height", type=float, default=0.34)
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument("--block-x", type=float, default=0.42)
    parser.add_argument("--block-y", type=float, default=-0.12)
    parser.add_argument("--target-x", type=float, default=0.60)
    parser.add_argument("--target-y", type=float, default=0.08)
    parser.add_argument("--step-size", type=float, default=0.025)
    return parser.parse_args()


def robot_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [model.joint(name).id for name in ("yaw", "shoulder", "elbow")]
    qpos_addr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    qvel_addr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    return qpos_addr, qvel_addr


def set_arm_target(data: mujoco.MjData, arm: SimpleArm3D, target_xyz: np.ndarray) -> None:
    q_target = arm.inverse_kinematics(np_to_torch(target_xyz.astype(np.float32))).numpy()
    data.ctrl[:] = np.clip(q_target, [-np.pi, -1.45, 0.04], [np.pi, 1.25, 2.50])


def print_help() -> None:
    print("")
    print("Interactive Push-Block Viewer")
    print("  W/S: move end-effector +Y / -Y")
    print("  A/D: move end-effector -X / +X")
    print("  Q/E: move end-effector +Z / -Z")
    print("  R: reset end-effector target")
    print("  P: print robot, block, and target state")
    print("  Esc or close the viewer window to exit")
    print("")


def main() -> None:
    args = parse_args()
    block_xy = np.array([args.block_x, args.block_y], dtype=np.float32)
    target_xy = np.array([args.target_x, args.target_y], dtype=np.float32)
    model = mujoco.MjModel.from_xml_string(
        make_xml(args.mount_height, args.table_height, block_xy, target_xy)
    )
    data = mujoco.MjData(model)
    arm = SimpleArm3D(base_height=args.mount_height)
    robot_qpos_addr, _ = robot_addresses(model)
    block_body_id = model.body("block").id
    ee_id = model.site("ee").id

    home_xyz = np.array([0.38, -0.26, args.table_height + 0.17], dtype=np.float32)
    ee_target = home_xyz.copy()
    data.qpos[robot_qpos_addr] = arm.inverse_kinematics(np_to_torch(home_xyz)).numpy()
    set_arm_target(data, arm, ee_target)
    mujoco.mj_forward(model, data)

    print_help()

    def key_callback(keycode: int) -> None:
        nonlocal ee_target
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "w":
            ee_target[1] += args.step_size
        elif key == "s":
            ee_target[1] -= args.step_size
        elif key == "a":
            ee_target[0] -= args.step_size
        elif key == "d":
            ee_target[0] += args.step_size
        elif key == "q":
            ee_target[2] += args.step_size
        elif key == "e":
            ee_target[2] -= args.step_size
        elif key == "r":
            ee_target = home_xyz.copy()
        elif key == "p":
            print(f"qpos={data.qpos[robot_qpos_addr]}")
            print(f"ee={data.site_xpos[ee_id]}")
            print(f"block_xy={data.xpos[block_body_id][:2]}, target_xy={target_xy}")
            print(f"ee_target={ee_target}")
        ee_target = np.clip(
            ee_target,
            [0.15, -0.38, args.table_height + 0.06],
            [0.88, 0.38, args.table_height + 0.35],
        )

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            set_arm_target(data, arm, ee_target)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
