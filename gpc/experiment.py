"""Experiment management utilities for GPC."""
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import jax
import numpy as np
from flax import nnx

from gpc.config import EvaluationConfig, TrainingConfig
from gpc.envs import TrainingEnv
from gpc.policy import Policy


class ExperimentManager:
    """Manages experiment directories, configs, and outputs."""
    
    def __init__(
        self,
        config: TrainingConfig,
        base_dir: str | Path = "experiments",
    ):
        """Initialize experiment manager.
        
        Args:
            config: Training configuration.
            base_dir: Base directory for all experiments.
        """
        self.config = config
        self.base_dir = Path(base_dir)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.experiment_name:
            exp_name = f"{config.experiment_name}_{timestamp}"
        else:
            exp_name = timestamp
        self.exp_dir = self.base_dir / config.task_name / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.log_dir = self.exp_dir / "logs"
        self.video_dir = self.exp_dir / "videos"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.eval_dir = self.exp_dir / "evaluation"
        
        self.log_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.eval_dir.mkdir(exist_ok=True)
        
        # Initialize training log file
        self.training_log_path = self.exp_dir / "training_log.md"
        self._init_training_log()
        
        # Save configuration
        self.config.to_yaml(self.exp_dir / "config.yaml")
        
        print(f"\n{'='*80}")
        print(f"Experiment Directory: {self.exp_dir.absolute()}")
        print(f"{'='*80}\n")
    
    def _init_training_log(self) -> None:
        """Initialize training log markdown file."""
        with open(self.training_log_path, "w") as f:
            f.write(f"# Training Log: {self.config.task_name}\n\n")
            f.write(f"**Experiment**: {self.config.experiment_name or 'default'}\n\n")
            f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
    
    def log_iteration_stats(
        self,
        iteration: int,
        total_iters: int,
        policy_cost: float,
        spc_cost: float,
        policy_best_frac: float,
        training_loss: float,
        sim_time: float,
        fit_time: float,
        render_time: float,
        num_episodes: int,
        num_data_points: int,
        obs_mean: float,
        obs_std: float,
    ) -> None:
        """Log iteration statistics to markdown file.
        
        Args:
            iteration: Current iteration (1-indexed for display).
            total_iters: Total number of iterations.
            policy_cost: Mean cost from policy samples.
            spc_cost: Mean cost from SPC samples.
            policy_best_frac: Fraction where policy beats SPC.
            training_loss: Training loss for this iteration.
            sim_time: Time spent simulating episodes.
            fit_time: Time spent fitting policy.
            render_time: Time spent rendering videos.
            num_episodes: Number of episodes collected.
            num_data_points: Total data points collected.
            obs_mean: Mean of observations.
            obs_std: Standard deviation of observations.
        """
        total_time = sim_time + fit_time + render_time
        
        with open(self.training_log_path, "a") as f:
            f.write(f"## Iteration {iteration}/{total_iters}\n\n")
            
            # Performance metrics
            f.write("### Performance\n\n")
            f.write(f"- **Policy Cost**: {policy_cost:.4f}\n")
            f.write(f"- **SPC Cost**: {spc_cost:.4f}\n")
            f.write(f"- **Policy Best**: {policy_best_frac*100:.2f}%\n")
            f.write(f"- **Training Loss**: {training_loss:.4f}\n\n")
            
            # Timing breakdown
            f.write("### Timing Breakdown\n\n")
            f.write(f"- Simulation: {sim_time:.3f}s\n")
            f.write(f"- Policy Fit: {fit_time:.3f}s\n")
            f.write(f"- Video Render: {render_time:.3f}s\n")
            f.write(f"- **Total**: {total_time:.3f}s\n\n")
            
            # Data statistics
            f.write("### Data Statistics\n\n")
            f.write(f"- Episodes: {num_episodes}\n")
            f.write(f"- Data Points: {num_data_points}\n")
            f.write(f"- Obs Mean: {obs_mean:.4f}\n")
            f.write(f"- Obs Std: {obs_std:.4f}\n\n")
            
            f.write("---\n\n")
    
    def finalize_training_log(self, total_time: str) -> None:
        """Add final summary to training log.
        
        Args:
            total_time: Total training time as formatted string.
        """
        with open(self.training_log_path, "a") as f:
            f.write(f"## Training Complete\n\n")
            f.write(f"**Total Time**: {total_time}\n\n")
            f.write(f"**Finished**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def get_policy_path(self, iteration: Optional[int] = None) -> Path:
        """Get path for saving/loading policy.
        
        Args:
            iteration: Iteration number for checkpoint, or None for final.
        """
        if iteration is None:
            return self.exp_dir / "policy.pkl"
        return self.checkpoint_dir / f"policy_iter_{iteration:04d}.pkl"
    
    def get_video_path(
        self,
        iteration: Optional[int] = None,
        episode: Optional[int] = None,
        prefix: str = "train",
    ) -> Path:
        """Get path for saving video.
        
        Args:
            iteration: Training iteration (for training videos).
            episode: Episode number (for evaluation videos).
            prefix: Prefix for video filename (train/eval).
        """
        if iteration is not None:
            return self.video_dir / f"{prefix}_iter_{iteration:04d}.mp4"
        elif episode is not None:
            return self.eval_dir / f"episode_{episode:03d}.mp4"
        return self.video_dir / f"{prefix}.mp4"
    
    def save_video(
        self,
        frames: np.ndarray,
        path: Path,
        fps: int = 30,
        quality: int = 5,
        resolution: Optional[tuple[int, int]] = None,
    ) -> None:
        """Save video with high quality settings.
        
        Args:
            frames: Array of frames (T, C, H, W) or (T, H, W, C).
            path: Output path.
            fps: Frames per second.
            quality: CRF quality (0-51, lower=better).
            resolution: Target (width, height). If None, uses original.
        """
        # Ensure frames are in (T, H, W, C) format
        if frames.shape[1] == 3:  # (T, C, H, W)
            frames = frames.transpose(0, 2, 3, 1)
        
        # Convert to uint8 if needed
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        # Resize if resolution specified
        if resolution is not None:
            from PIL import Image
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize(resolution, Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frames = np.stack(resized_frames)
        
        # Save with high quality
        imageio.mimsave(
            path,
            frames,
            fps=fps,
            codec="libx264",
            quality=quality,
            pixelformat="yuv420p",
            output_params=["-crf", str(quality)],
        )
    
    def log_iteration_summary(
        self,
        iteration: int,
        total_iters: int,
        metrics: dict,
        verbosity: int = 1,
    ) -> None:
        """Print iteration summary with appropriate verbosity.
        
        Args:
            iteration: Current iteration (0-indexed).
            total_iters: Total number of iterations.
            metrics: Dictionary of metrics to log.
            verbosity: 0=minimal, 1=normal, 2=verbose.
        """
        if verbosity == 0:
            return
        
        # Always show progress
        print(f"\n{'─'*80}")
        print(f"Iteration {iteration + 1}/{total_iters}")
        print(f"{'─'*80}")
        
        if verbosity >= 1:
            # Standard metrics
            print(f"  Policy Cost:      {metrics.get('policy_cost', 0):.4f}")
            print(f"  SPC Cost:         {metrics.get('spc_cost', 0):.4f}")
            print(f"  Policy Best:      {metrics.get('policy_best_frac', 0)*100:.2f}%")
            print(f"  Training Loss:    {metrics.get('loss', 0):.4f}")
        
        if verbosity >= 2:
            # Detailed timing
            print(f"\n  Timing Breakdown:")
            print(f"    Simulation:     {metrics.get('sim_time', 0):.3f}s")
            print(f"    Policy Fit:     {metrics.get('fit_time', 0):.3f}s")
            if "render_time" in metrics:
                print(f"    Video Render:   {metrics.get('render_time', 0):.3f}s")
            print(f"    Total:          {metrics.get('total_time', 0):.3f}s")
            
            # Data statistics
            if "data_size" in metrics:
                print(f"\n  Data Statistics:")
                print(f"    Episodes:       {metrics.get('num_episodes', 0)}")
                print(f"    Data Points:    {metrics.get('data_size', 0)}")
                print(f"    Obs Mean:       {metrics.get('obs_mean', 0):.4f}")
                print(f"    Obs Std:        {metrics.get('obs_std', 1):.4f}")
    
    @staticmethod
    def find_latest_experiment(
        task_name: str,
        base_dir: str | Path = "experiments",
    ) -> Optional[Path]:
        """Find the most recent experiment directory for a task.
        
        Args:
            task_name: Name of the task (e.g., "cart_pole").
            base_dir: Base experiments directory.
            
        Returns:
            Path to latest experiment with trained policy, or None.
        """
        exp_base = Path(base_dir) / task_name
        if not exp_base.exists():
            return None
        
        # Find directories with policy.pkl
        exp_dirs = [
            d for d in exp_base.iterdir()
            if d.is_dir() and (d / "policy.pkl").exists()
        ]
        
        if not exp_dirs:
            return None
        
        return max(exp_dirs, key=lambda p: p.stat().st_mtime)
    
    def create_summary(self) -> None:
        """Create a summary markdown file for the experiment."""
        summary_path = self.exp_dir / "SUMMARY.md"
        
        with open(summary_path, "w") as f:
            f.write(f"# Experiment Summary: {self.config.task_name}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Task**: {self.config.task_name}\n")
            f.write(f"- **Architecture**: {self.config.architecture}\n")
            f.write(f"- **Hidden Layers**: {self.config.hidden_layers}\n")
            f.write(f"- **Training Iterations**: {self.config.num_iters}\n")
            f.write(f"- **Environments**: {self.config.num_envs}\n")
            f.write(f"- **Learning Rate**: {self.config.learning_rate}\n\n")
            
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.exp_dir.name}/\n")
            f.write("├── config.yaml           # Training configuration\n")
            f.write("├── policy.pkl            # Final trained policy\n")
            f.write("├── SUMMARY.md            # This file\n")
            f.write("├── logs/                 # TensorBoard logs\n")
            f.write("├── videos/               # Training videos\n")
            f.write("├── checkpoints/          # Policy checkpoints\n")
            f.write("└── evaluation/           # Evaluation results\n")
            f.write("```\n\n")
            
            f.write("## Usage\n\n")
            f.write("### View Training Logs\n")
            f.write(f"```bash\n")
            f.write(f"tensorboard --logdir {self.log_dir}\n")
            f.write("```\n\n")
            
            f.write("### Load Policy\n")
            f.write("```python\n")
            f.write("from gpc.policy import Policy\n")
            f.write(f"policy = Policy.load('{self.exp_dir / 'policy.pkl'}')\n")
            f.write("```\n")
