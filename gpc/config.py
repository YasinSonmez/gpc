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
    action_repeat: int = 1  # Number of simulation steps per control step
    terminate_when_unhealthy: bool = False  # Early termination if agent falls (e.g. Ant)
    
    # Controller settings
    controller_type: str = "predictive_sampling"  # or "evosax", "mppi", "cem"
    num_samples: int = 8
    iterations: int = 1  # Number of optimization iterations per control step
    noise_level: float = 0.1
    noise_level_start: Optional[float] = None
    noise_level_end: Optional[float] = None
    num_policy_samples: int = 2
    num_latent_samples: int = 0  # Number of latent space samples (0=disabled)
    latent_noise_level: float = 1.0  # Noise level for latent sampling
    temperature: float = 0.1  # For MPPI controller
    num_randomizations: int = 1  # Domain randomizations
    # Evosax-specific settings
    evosax_strategy: str = "CMA_ES"  # evosax strategy name (e.g. CMA_ES, Sep_CMA_ES, OpenES)
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
    seed: int = 0
    
    # Cost-Conditioned Flow Matching (CFG) settings
    use_cost_conditioning: bool = False  # Enable cost-conditioned flow matching
    use_cfg: bool = False  # Enable Classifier-Free Guidance
    cfg_drop_prob: float = 0.1  # Probability of dropping cost condition during training
    cfg_guidance_scale: float = 2.0  # Guidance scale w for inference (w > 1 = stronger guidance)
    cost_weight_temperature: float = 1.0  # Temperature for cost weighting
    
    use_replay_buffer: bool = False  # Retain all historical data (True) or clear each iter (False)
    replay_buffer_size: int = 500_000  # Maximum replay buffer size
    buffer_filtering_fraction: float = 0.0  # Fraction of expert return to keep (0 = disable)
    buffer_filtering_mode: str = "fraction"  # "fraction" (top X%), "mean" (better than buffer mean), "expert" (better than expert fraction)
    initial_replay_buffer_path: Optional[str] = None
    warmstart_policy_buffer: bool = False
    warmstart_value_buffer: bool = False
    
    # Evaluation and logging
    checkpoint_every: int = 10
    num_eval_episodes: int = 3
    video_fps: int = 30
    video_quality: int = 8  # CRF value: lower = better (0-51)
    video_resolution: tuple[int, int] = (640, 480)  # (width, height)
    record_training_videos: bool = True
    num_training_videos: int = 2
    record_eval_videos: bool = True
    proposal_overlay: bool = False  # Overlay proposal stats on videos
    proposal_video_ghost_count: int = 3  # Ghost bodies per frame (0=disabled)
    proposal_video_ghost_steps: int = 4  # Future steps per ghost body
    proposal_video_num_display: int = 0  # Max proposals to draw (0=all)
    proposal_video_trace_points: int = 10  # Horizon subsample points for trajectory lines (0=all)
    proposal_video_palette: str = "tableau10"  # Color palette
    render_camera: str = "floating"
    
    # WandB settings
    use_wandb: bool = False
    wandb_project: str = "gpc"
    
    # Output settings
    experiment_name: Optional[str] = None
    log_verbosity: int = 1  # 0=minimal, 1=normal, 2=verbose
    strategy: str = "policy"  # "policy" or "best" for simulation advancement
    save_spc_data: bool = False  # Custom flag for saving training data as .npz
    
    # Chunked SPC settings
    chunked_spc: bool = False
    chunk_size: int = 4
    chunk_temperature: float = 0.1

    # Value Function Integration
    use_value_function: bool = False
    value_hidden_layers: list[int] = field(default_factory=lambda: [256, 256, 256])
    value_learning_rate: float = 1e-3
    iql_tau: float = 0.1
    discount_factor: float = 0.99
    polyak_tau: float = 0.005
    value_alpha_start: float = 0.0
    value_alpha_end: float = 0.1
    value_alpha_start_iter: int = 0
    value_alpha_start_iter: int = 0
    value_alpha_ramp_type: str = "linear"
    use_task_terminal_cost: bool = True  # Add task heuristic terminal cost to value function

    # ── Anti-collapse diversity options ──────────────────────────────────────
    # Solution 1 (Stratified SPC) + Solution 4 (DAgger Schedule) — unified
    exploration_floor: float = 0.0      # Fraction of SPC's N samples replaced with Uniform[-1,1] noise.
                                        #   e.g. 0.2 → 20% of num_samples are wide-random seeds every step.
    policy_rampup_iters: int = 0        # Ramp policy contribution 0→full over this many iters.
                                        #   0 (default) = always use full num_policy_samples.
    # Solution 6 (Dropout Regularisation against mode memorisation)
    # Dropout is active during training only (disabled at inference to stay compatible
    # with jax.lax.while_loop).  It prevents the MLP from memorising a single mode.
    inference_dropout: bool = False     # Enable dropout in DenoisingMLP (training-time regularisation).
    inference_dropout_rate: float = 0.1 # Dropout probability when inference_dropout=True (DenoisingMLP only).
    value_buffer_window: int = 0  # Number of recent iterations to keep in buffer (0 = keep all)
    value_train_epochs: int = 4
    
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
        if self.chunked_spc:
            if self.chunk_size < 1:
                raise ValueError("chunk_size must be at least 1")
            if self.chunk_temperature <= 0:
                raise ValueError("chunk_temperature must be positive")


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
