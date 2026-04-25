import torch

from imeanflow_robotics import IMeanFlowConfig, IMeanFlowPolicy


def main() -> None:
    config = IMeanFlowConfig(
        obs_dim=9,
        action_dim=6,
        horizon=16,
        n_action_steps=8,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
    )
    policy = IMeanFlowPolicy(config)
    obs = torch.randn(config.obs_dim)

    print("Untrained rollout demo. Train first for meaningful actions.")
    for step in range(10):
        action = policy.select_action(obs)
        print(f"step={step:02d} action_shape={tuple(action.shape)} action={action[0].numpy()}")


if __name__ == "__main__":
    main()

