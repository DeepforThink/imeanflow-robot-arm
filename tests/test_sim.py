import torch

from imeanflow_robotics import IMeanFlowConfig, IMeanFlowPolicy
from imeanflow_robotics.sim import PlanarArm2D, PlanarReachDataset, rollout_planar_policy


def test_planar_reach_dataset_shapes():
    dataset = PlanarReachDataset(num_samples=8, horizon=6)
    obs, actions, action_is_pad = dataset[0]

    assert obs.shape == (5,)
    assert actions.shape == (6, 3)
    assert action_is_pad.shape == (6,)
    assert not action_is_pad.any()


def test_planar_rollout_shapes():
    arm = PlanarArm2D()
    config = IMeanFlowConfig(
        obs_dim=arm.obs_dim,
        action_dim=arm.action_dim,
        horizon=6,
        n_action_steps=2,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )
    policy = IMeanFlowPolicy(config)
    result = rollout_planar_policy(
        policy=policy,
        arm=arm,
        initial_joints=torch.zeros(3),
        target_xy=torch.tensor([0.6, 0.4]),
        control_cycles=2,
        execute_steps=2,
    )

    assert result.joints.shape == (5, 3)
    assert result.points.shape == (5, 4, 2)
    assert result.target_xy.shape == (2,)
    assert result.final_distance >= 0.0
