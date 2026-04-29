# iMeanFlow Robotics

Improved Mean Flow policy for robotic arm action chunk generation.

This repository implements a **no-CFG conditional iMeanFlow** model for generating
future robot arm actions from observation vectors. It is designed as a compact,
readable research project that connects three ideas:

- flow matching for action generation,
- Mean Flow / iMeanFlow for few-step generation,
- action chunking for robotic imitation learning.

The implementation is intentionally independent of any large robotics framework,
so the method can be inspected, tested, and extended easily.

## Current Baseline Result

The companion FlowPolicy baseline replaces Diffusion Policy's denoising
objective with a Flow Matching velocity objective while keeping the original
low-dimensional Push-T data, rollout, and evaluation pipeline. A successful
evaluation rollout is shown below:

```text
Flow Matching lowdim Push-T test mean score: 0.818 over 50 test seeds
```

![Flow Matching Push-T rollout](assets/flow_matching_pusht.gif)

## Why This Project

Modern robot policies such as diffusion-style policies and pi0 generate an action
chunk by transforming Gaussian noise into a smooth action trajectory. Standard
Flow Matching learns an instantaneous velocity field and usually needs multiple
Euler steps at inference time.

iMeanFlow instead trains a model to support larger time jumps. The goal is:

```text
generate an action chunk with 1-2 model evaluations
instead of a longer multi-step sampler
```

This can be useful for real-time robot control where inference latency matters.

## Method Overview

For a demonstrated action chunk `x` and Gaussian noise `e`:

```text
z_t = (1 - t) * x + t * e
v_target = e - x
```

The model predicts:

```text
u(z_t, h, obs)      # interval-average velocity
v_hat(z_t, h, obs)  # auxiliary instantaneous velocity
h = t - r
```

The iMeanFlow training target is built with a Jacobian-vector product:

```text
v_tangent = v_hat(z_t, h=0, obs)
dudt = JVP(u, direction=(v_tangent, dt=1, dr=0))
V = u + (t - r) * stopgrad(dudt)
loss = ||V - v_target||^2 + ||v_hat - v_target||^2
```

At inference time:

```text
z ~ N(0, I)
for t -> r:
    z = z - (t - r) * u(z, t-r, obs)
```

The default sampler uses two Euler steps.

## Why No CFG

Classifier-free guidance is important in many image-generation systems, but it
requires an unconditional branch and condition dropout. Robot action generation is
already strongly conditioned by robot state, task, and visual observations. A fake
`omega` parameter without unconditional training would be misleading, so this
project uses direct conditional generation only.

## Repository Structure

```text
src/imeanflow_robotics/
  config.py          # model and training configuration
  model.py           # observation-conditioned Transformer with u/v heads
  policy.py          # iMeanFlow loss and few-step action sampling
  data.py            # synthetic robotic arm dataset
  sim.py             # lightweight 2D planar-arm simulation
  sim3d.py           # generated 3D reaching demonstrations
  train.py           # training entry point
  evaluate.py        # checkpoint evaluation

scripts/
  train_synthetic.py
  rollout_demo.py
  sim_demo.py
  mujoco_3d_demo.py
  franka_push_block_viewer.py

docs/
  method.md
  interview_guide_zh.md

tests/
  test_policy.py
```

## Quick Start

Install:

```bash
git clone https://github.com/DeepforThink/imeanflow-robot-arm.git
cd imeanflow-robot-arm
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Train on the synthetic arm dataset:

```bash
python scripts/train_synthetic.py --steps 800
```

Evaluate:

```bash
python -m imeanflow_robotics.evaluate --checkpoint checkpoints/imeanflow_synthetic.pt
```

Run a minimal action queue demo:

```bash
python scripts/rollout_demo.py
```

Run the planar-arm simulation demo:

```bash
python scripts/sim_demo.py --train-steps 300
```

The demo trains a small no-CFG iMeanFlow policy on generated reaching
demonstrations, rolls it out with receding-horizon action chunks, and writes:

```text
assets/planar_arm_demo.png
assets/planar_arm_demo.gif
```

![Planar arm simulation](assets/planar_arm_demo.gif)

The demo is a kinematic simulation, not a physics benchmark. Its purpose is to
show the policy generating action chunks that move a simple robot arm toward a
target. The default rollout uses 4 model evaluations per chunk and a fixed seed
for reproducible visualization. A small joint-step limiter is applied during
rollout, matching the kind of low-level command smoothing used in real robot
control loops.

Run the 3D MuJoCo simulation demo:

```bash
pip install -e ".[mujoco]"
python scripts/mujoco_3d_demo.py --train-steps 1200
```

This trains a small 3-DoF yaw/shoulder/elbow reaching policy, then runs it in
MuJoCo with position actuators. The arm is mounted on a small pedestal so the
visualization looks like a table-mounted robot rather than a linkage lying on
the floor. The script writes:

```text
assets/mujoco_3d_demo.png
assets/mujoco_3d_demo.gif
```

![MuJoCo 3D simulation](assets/mujoco_3d_demo.gif)

Run the early MuJoCo push-block environment:

```bash
python scripts/mujoco_push_block_demo.py --save-data
python scripts/mujoco_push_block_viewer.py
```

The first command runs a scripted pushing episode, saves
`assets/push_block_scripted.png`, `assets/push_block_scripted.gif`, and optionally
`data/push_block_demo/episode_000.npz`. The second command opens an interactive
MuJoCo viewer. Keyboard controls in the terminal move the end-effector target so
the pushing setup can be inspected and tuned manually.

Run the Franka Panda push-block viewer:

```bash
python scripts/franka_push_block_viewer.py
```

This viewer loads the official MuJoCo Menagerie Franka Panda model through
`robot_descriptions`, then adds a tabletop block-pushing scene. It is the better
starting point for real robot-style experiments than the small hand-written
3-DoF arm above: the Panda has the full 7-DoF kinematic chain, gripper geometry,
joint limits, and collision meshes. Keyboard controls move a Cartesian wrist
target with damped least-squares IK:

```text
W/S: move wrist target +Y / -Y
A/D: move wrist target -X / +X
Q/E: move wrist target +Z / -Z
R: reset robot and block
P: print current state
```

## Example Code

```python
import torch
from imeanflow_robotics import IMeanFlowConfig, IMeanFlowPolicy

config = IMeanFlowConfig(obs_dim=9, action_dim=6, horizon=16, n_action_steps=8)
policy = IMeanFlowPolicy(config)

obs = torch.randn(4, config.obs_dim)
actions = torch.randn(4, config.horizon, config.action_dim)
loss, metrics = policy.compute_loss(obs, actions)

chunk = policy.sample_action_chunk(obs)
single_action = policy.select_action(obs[0])
```

## What This Repository Is Honest About

This is a compact research/engineering implementation, not an official iMeanFlow
release and not a claim of state-of-the-art robot performance. The included
synthetic dataset exists to verify the training loop and sampling API. For real
robot use, replace `SyntheticArmDataset` with teleoperation or demonstration data
from your robot.

Recommended next experiments:

- compare 2-step iMeanFlow with 10-step Flow Matching,
- measure success rate and action smoothness on a real or simulated arm,
- add image/state encoders for visual imitation learning,
- benchmark control latency under different NFE values.

## References

- Lipman et al., Flow Matching for Generative Modeling.
- Geng et al., Mean Flows for One-step Generative Modeling.
- Geng et al., Improved Mean Flows: On the Challenges of Fastforward Generative Models.
- pi0 / OpenPI style action flow matching for robot policies.

## License

MIT License.
