import argparse
from datetime import datetime
from pathlib import Path

# Import gpc first to apply CUDA fixes
import gpc  # noqa: F401

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingMLP
from gpc.envs import CartPoleEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_and_record, test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Balance an inverted pendulum on a cart"
    )
    subparsers = parser.add_subparsers(
        dest="task", help="What to do (choose one)"
    )
    subparsers.add_parser("train", help="Train (and save) a generative policy")
    test_parser = subparsers.add_parser("test", help="Test a generative policy")
    test_parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory to load policy from (default: latest)",
    )
    subparsers.add_parser(
        "sample", help="Bootstrap sampling-based MPC with a generative policy"
    )
    args = parser.parse_args()

    # Set up the environment
    env = CartPoleEnv(episode_length=200)
    
    # Determine experiment directory
    if args.task == "train":
        # Create new experiment directory for training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path("experiments/cart_pole") / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        # For test/sample, use specified exp_dir or find latest with policy
        if hasattr(args, "exp_dir") and args.exp_dir:
            exp_dir = Path(args.exp_dir)
        else:
            # Find the latest experiment directory with a policy.pkl file
            exp_base = Path("experiments/cart_pole")
            if not exp_base.exists():
                print(f"Error: No experiments found in {exp_base}")
                print("Please run training first with: python examples/cart_pole.py train")
                exit(1)
            
            # Get all directories with policy.pkl, sorted by modification time
            exp_dirs_with_policy = [
                d for d in exp_base.iterdir() 
                if d.is_dir() and (d / "policy.pkl").exists()
            ]
            
            if not exp_dirs_with_policy:
                print(f"Error: No trained policies found in {exp_base}")
                print("Please run training first with: python examples/cart_pole.py train")
                exit(1)
            
            exp_dir = max(exp_dirs_with_policy, key=lambda p: p.stat().st_mtime)
            print(f"Using latest experiment: {exp_dir}")
    
    save_file = exp_dir / "policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
        net = DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            hidden_layers=[64, 64],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=2,
            log_dir=str(exp_dir / "logs"),
            num_iters=10,
            num_envs=128,
            num_epochs=100,
            num_videos=0,  # Disable video rendering during training
        )
        policy.save(save_file)
        print(f"\nExperiment saved to: {exp_dir.absolute()}")
        print(f"  - Policy: {save_file}")
        print(f"  - Logs: {exp_dir / 'logs'}")

    elif args.task == "test":
        # Load the policy from a file and test it
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        
        # Check if display is available for interactive viewer
        import os
        if os.environ.get("DISPLAY"):
            # Try interactive viewer
            try:
                test_interactive(env, policy)
            except Exception as e:
                print(f"\nInteractive viewer failed ({e})")
                print("Recording evaluation videos instead...\n")
                video_dir = exp_dir / "evaluation_videos"
                test_and_record(
                    env, policy, num_episodes=3, output_dir=video_dir, video_fps=30
                )
                print(f"\nEvaluation complete! Videos saved to: {video_dir.absolute()}")
        else:
            # No display available, record videos
            print("No display available, recording evaluation videos...\n")
            video_dir = exp_dir / "evaluation_videos"
            test_and_record(
                env, policy, num_episodes=3, output_dir=video_dir, video_fps=30
            )
            print(f"\nEvaluation complete! Videos saved to: {video_dir.absolute()}")

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            observation_fn=env.get_obs,
            num_policy_samples=2,
            task=env.task,
            num_samples=1,
            noise_level=0.1,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50)

    else:
        parser.print_help()
