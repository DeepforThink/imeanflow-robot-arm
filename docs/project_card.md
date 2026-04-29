# Project Card

## Project name

iMeanFlow Robotics

## Scope

This repository implements a compact action-generation policy for robotic arms.
It is meant for research demonstration, interview discussion, and future
extension to real imitation-learning datasets.

## Claims

The repository demonstrates:

- a no-CFG conditional iMeanFlow training objective,
- u/v dual-head action velocity prediction,
- JVP-based compound velocity training,
- few-step Euler action chunk generation,
- a synthetic robotic arm dataset for reproducible smoke tests.
- a lightweight planar-arm simulation demo with PNG/GIF rollout artifacts.
- an optional 3D MuJoCo reaching demo with a yaw/shoulder/elbow arm.
- an early MuJoCo push-block environment with scripted demonstration capture.
- a Franka Panda MuJoCo viewer built from the official MuJoCo Menagerie assets.

It does not claim:

- official iMeanFlow reproduction,
- state-of-the-art robot performance,
- validation on a real robot arm.
- physical-engine benchmark performance.
- real-world robot validation.

## Suggested evaluation

When connected to real demonstration data, evaluate:

- task success rate,
- action smoothness,
- inference latency,
- performance as a function of NFE,
- comparison with standard Flow Matching.

## Included demo

Run:

```bash
python scripts/sim_demo.py --train-steps 300
```

This trains a small 3-DoF planar reaching policy, rolls it out with
receding-horizon action chunks, and saves `assets/planar_arm_demo.png` and
`assets/planar_arm_demo.gif`.

For MuJoCo:

```bash
pip install -e ".[mujoco]"
python scripts/mujoco_3d_demo.py --train-steps 1200
```

This runs the generated joint targets through MuJoCo position actuators and
saves `assets/mujoco_3d_demo.png` and `assets/mujoco_3d_demo.gif`.

For push-block data collection:

```bash
python scripts/mujoco_push_block_demo.py --save-data
python scripts/mujoco_push_block_viewer.py
```

The scripted demo saves one `.npz` episode and visual artifacts. The viewer is
intended for manually inspecting contact behavior before scaling up data
collection.

For a real robot-style push-block viewer:

```bash
python scripts/franka_push_block_viewer.py
```

This loads the Franka Panda model through `robot_descriptions`, adds a tabletop
block-pushing scene, and controls the wrist target with damped least-squares IK.
It should be used as the project baseline for future Panda/FR3 experiments
instead of the hand-written 3-DoF prototype.
