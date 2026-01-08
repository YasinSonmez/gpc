"""Train and evaluate GPC policies using configuration files.

This is a general-purpose script that works for any task defined in gpc.envs.

Usage:
    python run_experiment.py train --config configs/cart_pole.yaml
    python run_experiment.py eval --exp-dir experiments/cart_pole/20260106_100943
    python run_experiment.py eval --task cart_pole  # Uses latest experiment
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Set before any MuJoCo imports

import argparse
import sys
from pathlib import Path

# Import gpc first for CUDA setup
import gpc  # noqa: F401

from flax import nnx
from hydrax.algs import CEM, Evosax, PredictiveSampling

from gpc.architectures import DenoisingCNN, DenoisingMLP
from gpc.cost_conditioned import CostConditionedMLP
from gpc.augmented import PolicyAugmentedController
from gpc.config import EvaluationConfig, TrainingConfig
from gpc.envs import (
    CartPoleEnv,
    CraneEnv,
    DoubleCartPoleEnv,
    HumanoidEnv,
    ParticleEnv,
    PendulumEnv,
    PushTEnv,
    WalkerEnv,
)
from gpc.experiment import ExperimentManager
from gpc.policy import Policy
from gpc.testing import test_and_record, test_interactive
from gpc.train_config import train_with_config
from gpc.sweep import expand_config_sweep, print_sweep_summary


# Task name to environment class mapping
TASK_ENVS = {
    "cart_pole": CartPoleEnv,
    "double_cart_pole": DoubleCartPoleEnv,
    "pendulum": PendulumEnv,
    "particle": ParticleEnv,
    "walker": WalkerEnv,
    "crane": CraneEnv,
    "humanoid": HumanoidEnv,
    "pusht": PushTEnv,
}


def create_environment(config: TrainingConfig):
    """Create environment from config."""
    if config.task_name not in TASK_ENVS:
        raise ValueError(
            f"Unknown task: {config.task_name}. "
            f"Available: {list(TASK_ENVS.keys())}"
        )
    
    env_class = TASK_ENVS[config.task_name]
    return env_class(episode_length=config.episode_length)


def create_controller(env, config: TrainingConfig):
    """Create controller from config."""
    # Common parameters for all controllers
    common_params = {
        "plan_horizon": config.plan_horizon,
        "spline_type": config.spline_type,
        "num_knots": config.num_knots,
    }
    
    if config.controller_type == "predictive_sampling":
        base_ctrl = PredictiveSampling(
            env.task,
            num_samples=config.num_samples,
            noise_level=config.noise_level,
            **common_params,
        )
    elif config.controller_type == "cem":
        base_ctrl = CEM(
            env.task,
            num_samples=config.num_samples,
            num_elites=getattr(config, 'num_elites', config.num_samples // 4),
            sigma_start=getattr(config, 'sigma_start', 1.0),
            sigma_min=getattr(config, 'sigma_min', 0.01),
            **common_params,
        )
    elif config.controller_type == "evosax":
        import evosax
        base_ctrl = Evosax(
            env.task,
            num_samples=config.num_samples,
            strategy=evosax.OpenES,
            **common_params,
        )
    elif config.controller_type == "mppi":
        from hydrax.algs import MPPI
        base_ctrl = MPPI(
            env.task,
            num_samples=config.num_samples,
            noise_level=config.noise_level,
            temperature=config.temperature,
            num_randomizations=config.num_randomizations,
            **common_params,
        )
    else:
        raise ValueError(f"Unknown controller type: {config.controller_type}")
    
    return PolicyAugmentedController(base_ctrl, config.num_policy_samples)


def create_network(env, config: TrainingConfig):
    """Create neural network from config."""
    # Network horizon is the number of knots (new API) not planning_horizon
    horizon = config.num_knots
    
    # Use cost-conditioned architecture if CFG is enabled
    if config.use_cfg:
        if config.architecture == "mlp":
            return CostConditionedMLP(
                action_size=env.task.model.nu,
                observation_size=env.observation_size,
                horizon=horizon,
                hidden_layers=config.hidden_layers,
                rngs=nnx.Rngs(0),
            )
        else:
            raise ValueError(
                f"CFG only supports 'mlp' architecture, got '{config.architecture}'"
            )
    
    if config.architecture == "mlp":
        return DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=horizon,
            hidden_layers=config.hidden_layers,
            rngs=nnx.Rngs(0),
        )
    elif config.architecture == "cnn":
        return DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=horizon,
            feature_dims=config.feature_dims,
            timestep_embedding_dim=config.timestep_embedding_dim,
            rngs=nnx.Rngs(0),
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def train_command(args):
    """Run training."""
    # Expand config if it contains hyperparameter sweeps
    configs = expand_config_sweep(args.config)
    
    # Print sweep summary if multiple configs
    if len(configs) > 1:
        print_sweep_summary(configs)
        
        # Ask for confirmation
        response = input(f"Run all {len(configs)} experiments? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return 0
    
    # Run each configuration
    for config_idx, (config, suffix) in enumerate(configs, 1):
        if len(configs) > 1:
            print(f"\n{'='*80}")
            print(f"Running configuration {config_idx}/{len(configs)}: {suffix}")
            print(f"{'='*80}\n")
        
        # Override config with command-line args if provided
        if args.num_iters is not None:
            config.num_iters = args.num_iters
        if args.num_envs is not None:
            config.num_envs = args.num_envs
        if args.exp_name is not None:
            config.experiment_name = args.exp_name
        
        # Append suffix to experiment name
        if suffix:
            if config.experiment_name:
                config.experiment_name = f"{config.experiment_name}_{suffix}"
            else:
                config.experiment_name = f"{config.task_name}_{suffix}"
        
        # Create experiment manager
        exp_manager = ExperimentManager(config, base_dir=args.output_dir)
        
        # Create environment, controller, and network
        env = create_environment(config)
        ctrl = create_controller(env, config)
        net = create_network(env, config)
        
        # Train
        policy = train_with_config(env, ctrl, net, config, exp_manager)
        
        print(f"\n✓ Training complete!")
        print(f"  Policy: {exp_manager.get_policy_path()}")
        print(f"  Logs: {exp_manager.log_dir}")
        print(f"  Videos: {exp_manager.video_dir}")
        
        if len(configs) > 1:
            print(f"\n  Progress: {config_idx}/{len(configs)} configurations completed")
    
    if len(configs) > 1:
        print(f"\n{'='*80}")
        print(f"✓ All {len(configs)} experiments completed!")
        print(f"{'='*80}")
    else:
        print(f"\nView logs with: tensorboard --logdir {exp_manager.log_dir}")


def eval_command(args):
    """Run evaluation."""
    # Find experiment directory
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    elif args.task:
        exp_dir = ExperimentManager.find_latest_experiment(
            args.task, base_dir=args.output_dir
        )
        if exp_dir is None:
            print(f"Error: No experiments found for task '{args.task}'")
            print(f"Run training first: python run_experiment.py train --config configs/{args.task}.yaml")
            return 1
        print(f"Using latest experiment: {exp_dir}")
    else:
        print("Error: Must provide either --exp-dir or --task")
        return 1
    
    # Load configuration and policy
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: No config found at {config_path}")
        return 1
    
    config = TrainingConfig.from_yaml(config_path)
    policy_path = exp_dir / "policy.pkl"
    
    if not policy_path.exists():
        print(f"Error: No policy found at {policy_path}")
        return 1
    
    print(f"Loading policy from {policy_path}")
    policy = Policy.load(policy_path)
    
    # Create environment
    env = create_environment(config)
    
    # Load evaluation config
    eval_config = EvaluationConfig(
        num_episodes=args.num_episodes,
        video_fps=config.video_fps,
        video_quality=config.video_quality,
    )
    
    # Check for display
    import os
    if args.interactive and os.environ.get("DISPLAY"):
        try:
            print("Launching interactive viewer...")
            test_interactive(env, policy)
        except Exception as e:
            print(f"Interactive viewer failed: {e}")
            print("Falling back to video recording...")
            video_dir = exp_dir / "evaluation_interactive"
            test_and_record(
                env, policy,
                num_episodes=eval_config.num_episodes,
                output_dir=video_dir,
                video_fps=eval_config.video_fps,
            )
    else:
        print("Recording evaluation videos...")
        video_dir = exp_dir / "evaluation_videos"
        test_and_record(
            env, policy,
            num_episodes=eval_config.num_episodes,
            output_dir=video_dir,
            video_fps=eval_config.video_fps,
        )
        print(f"\nVideos saved to: {video_dir.absolute()}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate GPC policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new policy")
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    train_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="experiments",
        help="Base output directory (default: experiments)",
    )
    train_parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Override number of training iterations",
    )
    train_parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel environments",
    )
    train_parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Custom experiment name (default: timestamp)",
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory to evaluate",
    )
    eval_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (uses latest experiment)",
    )
    eval_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="experiments",
        help="Base output directory (default: experiments)",
    )
    eval_parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    eval_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Try interactive viewer (requires display)",
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        return train_command(args)
    elif args.command == "eval":
        return eval_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
