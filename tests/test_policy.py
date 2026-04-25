import torch

from imeanflow_robotics import IMeanFlowConfig, IMeanFlowPolicy


def test_loss_and_sampling_shapes():
    config = IMeanFlowConfig(
        obs_dim=9,
        action_dim=6,
        horizon=8,
        n_action_steps=4,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )
    policy = IMeanFlowPolicy(config)

    obs = torch.randn(3, config.obs_dim)
    actions = torch.randn(3, config.horizon, config.action_dim)
    action_is_pad = torch.zeros(3, config.horizon, dtype=torch.bool)

    loss, metrics = policy.compute_loss(obs, actions, action_is_pad)
    assert loss.ndim == 0
    assert metrics["loss_u"].ndim == 0
    loss.backward()

    chunk = policy.sample_action_chunk(obs)
    assert chunk.shape == (3, config.horizon, config.action_dim)


def test_action_queue_returns_single_action():
    config = IMeanFlowConfig(
        obs_dim=9,
        action_dim=6,
        horizon=8,
        n_action_steps=4,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )
    policy = IMeanFlowPolicy(config)
    obs = torch.randn(config.obs_dim)
    action = policy.select_action(obs)
    assert action.shape == (1, config.action_dim)

