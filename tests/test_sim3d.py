import torch

from imeanflow_robotics.sim3d import Reach3DDataset, SimpleArm3D


def test_simple_arm_3d_shapes():
    arm = SimpleArm3D()
    joints = torch.zeros(2, 3)
    points = arm.forward_kinematics(joints)
    obs = arm.observation(joints, torch.zeros(2, 3))

    assert points.shape == (2, 3, 3)
    assert obs.shape == (2, 6)


def test_reach_3d_dataset_shapes():
    dataset = Reach3DDataset(num_samples=8, horizon=6)
    obs, actions, action_is_pad = dataset[0]

    assert obs.shape == (6,)
    assert actions.shape == (6, 3)
    assert action_is_pad.shape == (6,)
    assert not action_is_pad.any()
