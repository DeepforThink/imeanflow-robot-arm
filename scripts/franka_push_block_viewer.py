from __future__ import annotations

import argparse
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
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

try:
    from robot_descriptions import panda_mj_description
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Franka Panda assets are required: pip install -e '.[mujoco]'"
    ) from exc


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")


@dataclass
class RobotAddresses:
    qpos: np.ndarray
    dof: np.ndarray
    ctrl: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a Franka Panda MuJoCo push-block viewer."
    )
    parser.add_argument("--block-x", type=float, default=0.50)
    parser.add_argument("--block-y", type=float, default=-0.10)
    parser.add_argument("--target-x", type=float, default=0.58)
    parser.add_argument("--target-y", type=float, default=0.12)
    parser.add_argument("--table-top-z", type=float, default=0.0)
    parser.add_argument("--step-size", type=float, default=0.025)
    parser.add_argument("--headless-steps", type=int, default=0)
    return parser.parse_args()


def build_scene_xml(
    panda_xml_path: Path,
    block_xy: np.ndarray,
    target_xy: np.ndarray,
    table_top_z: float,
) -> str:
    """Append a tabletop push-block scene to the official MuJoCo Menagerie Panda XML."""
    base_xml = panda_xml_path.read_text(encoding="utf-8")
    base_xml = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", base_xml, flags=re.DOTALL)
    block_half = 0.035
    table_half_z = 0.025
    scene = f"""
    <camera name="demo" pos="1.05 -1.05 0.72" xyaxes="0.707 0.707 0 -0.35 0.35 0.869"/>
    <light name="task_light" pos="0.4 -0.4 1.2" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="box" pos="0.55 0 {table_top_z - table_half_z:.4f}"
          size="0.42 0.36 {table_half_z:.4f}" rgba="0.68 0.68 0.62 1"
          friction="1.0 0.01 0.0001"/>
    <geom name="target_disk" type="cylinder"
          pos="{target_xy[0]:.4f} {target_xy[1]:.4f} {table_top_z + 0.002:.4f}"
          size="0.055 0.002" rgba="0.9 0.12 0.08 0.85" contype="0" conaffinity="0"/>
    <body name="block" pos="{block_xy[0]:.4f} {block_xy[1]:.4f} {table_top_z + block_half:.4f}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{block_half:.4f} {block_half:.4f} {block_half:.4f}"
            mass="0.08" rgba="0.95 0.55 0.08 1" friction="1.2 0.02 0.0001"/>
    </body>
    """
    return base_xml.replace("<worldbody>", "<worldbody>\n" + textwrap.dedent(scene), 1)


def write_temp_scene_xml(scene_xml: str) -> Path:
    # Mesh paths in panda.xml are relative to the XML file, so keep the generated
    # scene beside the official Menagerie XML.
    panda_dir = Path(panda_mj_description.MJCF_PATH).resolve().parent
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_imeanflow_panda_push.xml",
        dir=panda_dir,
        encoding="utf-8",
        delete=False,
    )
    with temp:
        temp.write(scene_xml)
    return Path(temp.name)


def robot_addresses(model: mujoco.MjModel) -> RobotAddresses:
    qpos = np.asarray([int(model.joint(name).qposadr[0]) for name in ARM_JOINTS])
    dof = np.asarray([int(model.joint(name).dofadr[0]) for name in ARM_JOINTS])
    ctrl = np.asarray([model.actuator(name=f"actuator{i}").id for i in range(1, 8)])
    return RobotAddresses(qpos=qpos, dof=dof, ctrl=ctrl)


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: RobotAddresses,
    home_q: np.ndarray,
    block_xy: np.ndarray,
    table_top_z: float,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[addresses.qpos] = home_q
    for name in FINGER_JOINTS:
        data.qpos[int(model.joint(name).qposadr[0])] = 0.04
    block_qpos = int(model.joint("block_free").qposadr[0])
    data.qpos[block_qpos : block_qpos + 7] = [
        block_xy[0],
        block_xy[1],
        table_top_z + 0.035,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.ctrl[addresses.ctrl] = home_q
    data.ctrl[model.actuator("actuator8").id] = 255.0
    mujoco.mj_forward(model, data)


def damped_ik_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_data: mujoco.MjData,
    addresses: RobotAddresses,
    body_id: int,
    target_xyz: np.ndarray,
    joint_target: np.ndarray,
    home_q: np.ndarray,
) -> tuple[float, np.ndarray]:
    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = 0.0
    ik_data.qpos[addresses.qpos] = joint_target
    for name in FINGER_JOINTS:
        ik_data.qpos[int(model.joint(name).qposadr[0])] = 0.04
    mujoco.mj_forward(model, ik_data)
    current = ik_data.xpos[body_id].copy()
    error = target_xyz - current
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, ik_data, jacp, jacr, body_id)
    jac = jacp[:, addresses.dof]
    damping = 0.01
    step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), 0.55 * error)
    posture = 0.002 * (home_q - joint_target)
    q_next = joint_target + np.clip(step, -0.035, 0.035) + posture
    q_low = model.jnt_range[:7, 0]
    q_high = model.jnt_range[:7, 1]
    q_next = np.clip(q_next, q_low, q_high)
    data.ctrl[addresses.ctrl] = q_next
    data.ctrl[model.actuator("actuator8").id] = 255.0
    return float(np.linalg.norm(error)), q_next


def print_help() -> None:
    print("")
    print("Franka Panda Push-Block Viewer")
    print("  W/S: move wrist target +Y / -Y")
    print("  A/D: move wrist target -X / +X")
    print("  Q/E: move wrist target +Z / -Z")
    print("  R: reset robot and block")
    print("  P: print wrist, block, target, and IK error")
    print("  Esc or close the viewer window to exit")
    print("")


def main() -> None:
    args = parse_args()
    block_xy = np.array([args.block_x, args.block_y], dtype=np.float64)
    target_xy = np.array([args.target_x, args.target_y], dtype=np.float64)
    scene_xml = build_scene_xml(
        Path(panda_mj_description.MJCF_PATH),
        block_xy=block_xy,
        target_xy=target_xy,
        table_top_z=args.table_top_z,
    )
    scene_path = write_temp_scene_xml(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    scene_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    addresses = robot_addresses(model)
    hand_id = model.body("hand").id
    block_id = model.body("block").id
    home_q = np.array(
        [-0.53, 0.53, -0.08, -2.31, 0.01, 2.88, 0.79],
        dtype=np.float64,
    )
    home_xyz = np.array([0.42, -0.30, args.table_top_z + 0.12], dtype=np.float64)
    wrist_target = home_xyz.copy()
    joint_target = home_q.copy()

    reset_state(model, data, addresses, home_q, block_xy, args.table_top_z)

    if args.headless_steps > 0:
        for _ in range(args.headless_steps):
            ik_error, joint_target = damped_ik_control(
                model, data, ik_data, addresses, hand_id, wrist_target, joint_target, home_q
            )
            mujoco.mj_step(model, data)
        print(f"panda_xml={panda_mj_description.MJCF_PATH}")
        print(f"wrist={np.round(data.xpos[hand_id], 4)}")
        print(f"block_xy={np.round(data.xpos[block_id][:2], 4)}")
        print(f"target_xy={np.round(target_xy, 4)}")
        print(f"ik_error={ik_error:.4f}")
        return

    print_help()

    def key_callback(keycode: int) -> None:
        nonlocal wrist_target, joint_target
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "w":
            wrist_target[1] += args.step_size
        elif key == "s":
            wrist_target[1] -= args.step_size
        elif key == "a":
            wrist_target[0] -= args.step_size
        elif key == "d":
            wrist_target[0] += args.step_size
        elif key == "q":
            wrist_target[2] += args.step_size
        elif key == "e":
            wrist_target[2] -= args.step_size
        elif key == "r":
            wrist_target = home_xyz.copy()
            joint_target = home_q.copy()
            reset_state(model, data, addresses, home_q, block_xy, args.table_top_z)
        elif key == "p":
            ik_error = float(np.linalg.norm(wrist_target - data.xpos[hand_id]))
            print(f"wrist={np.round(data.xpos[hand_id], 4)}")
            print(f"block_xy={np.round(data.xpos[block_id][:2], 4)}, target_xy={target_xy}")
            print(f"wrist_target={np.round(wrist_target, 4)}, ik_error={ik_error:.4f}")
        wrist_target = np.clip(
            wrist_target,
            [0.28, -0.36, args.table_top_z + 0.09],
            [0.72, 0.36, args.table_top_z + 0.22],
        )

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            _, joint_target = damped_ik_control(
                model, data, ik_data, addresses, hand_id, wrist_target, joint_target, home_q
            )
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
