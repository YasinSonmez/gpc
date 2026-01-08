"""Configuration management for GPC experiments."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TrainingConfig:
    """Configuration for training a GPC policy."""
    
    # Environment settings
    task_name: str = "cart_pole"
    episode_length: int = 200
    
    # Controller settings
    controller_type: str = "predictive_sampling"  # or "evosax", "mppi", "cem"
    num_samples: int = 8
    noise_level: float = 0.1
    num_policy_samples: int = 2
    num_latent_samples: int = 0  # Number of latent space samples (0=disabled)
    latent_noise_level: float = 1.0  # Noise level for latent sampling
    temperature: float = 0.1  # For MPPI controller
    num_randomizations: int = 1  # Domain randomizations
    # CEM-specific parameters
    num_elites: int = 2  # For CEM controller
    sigma_start: float = 1.0  # For CEM controller
    sigma_min: float = 0.01  # For CEM controller
    # Spline interpolation
    spline_type: str = "zero"  # "zero", "linear", or "cubic"
    num_knots: int = 4  # Number of control knots
    plan_horizon: float = 0.5  # Planning horizon in seconds
    
    # Network architecture
    architecture: str = "mlp"  # or "cnn"
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64])
    feature_dims: list[int] = field(default_factory=lambda: [32, 32])  # CNN only
    timestep_embedding_dim: int = 8  # CNN only
    
    # Training hyperparameters
    num_iters: int = 10
    num_envs: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    exploration_noise_level: float = 0.0
    normalize_observations: bool = True
    
    # Cost-Conditioned Flow Matching (CFG) settings
    use_cfg: bool = False  # Enable cost-conditioned flow matching with CFG
    cfg_drop_prob: float = 0.1  # Probability of dropping cost condition during training
    cfg_guidance_scale: float = 2.0  # Guidance scale w for inference (w > 1 = stronger guidance)
    use_replay_buffer: bool = False  # Retain all historical data (True) or clear each iter (False)
    replay_buffer_size: int = 500_000  # Maximum replay buffer size
    
    # Evaluation and logging
    checkpoint_every: int = 10
    num_eval_episodes: int = 3
    video_fps: int = 30
    video_quality: int = 8  # CRF value: lower = better (0-51)
    video_resolution: tuple[int, int] = (640, 480)  # (width, height)
    record_training_videos: bool = True
    num_training_videos: int = 2
    record_eval_videos: bool = True
    
    # Output settings
    experiment_name: Optional[str] = None
    log_verbosity: int = 1  # 0=minimal, 1=normal, 2=verbose
    strategy: str = "policy"  # "policy" or "best" for simulation advancement
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.video_quality < 0 or self.video_quality > 51:
            raise ValueError("video_quality must be between 0 and 51")
        if self.log_verbosity not in [0, 1, 2]:
            raise ValueError("log_verbosity must be 0, 1, or 2")


@dataclass
class EvaluationConfig:
    """Configuration for evaluating a trained policy."""
    
    num_episodes: int = 5
    video_fps: int = 30
    video_quality: int = 5
    inference_timestep: float = 0.1
    warm_start_level: float = 1.0
    record_videos: bool = True
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_config(config_path: Optional[str | Path] = None) -> TrainingConfig:
    """Load training configuration from file or return defaults."""
    if config_path is None:
        return TrainingConfig()
    return TrainingConfig.from_yaml(config_path)
