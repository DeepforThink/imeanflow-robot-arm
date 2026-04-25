from dataclasses import dataclass


@dataclass
class IMeanFlowConfig:
    """Configuration for conditional iMeanFlow action generation."""

    obs_dim: int = 16
    action_dim: int = 6
    horizon: int = 16
    n_action_steps: int = 8

    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Logit-normal sampling for t and r.
    p_mean: float = -0.4
    p_std: float = 1.0
    ratio: float = 0.5

    # Loss balancing.
    norm_p: float = 1.0
    norm_eps: float = 0.01
    v_loss_weight: float = 1.0

    # Inference.
    num_inference_steps: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must be <= horizon")
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")

