# Configuration Files

This directory contains YAML configuration files for different GPC experiments.

## Quick Start

Train a policy:
```bash
export MUJOCO_GL=egl  # For headless rendering
python run_experiment.py train --config configs/cart_pole.yaml
```

Evaluate a trained policy:
```bash
python run_experiment.py eval --task cart_pole
```

## Available Configs

- `cart_pole.yaml` - Cart-pole swing-up task (simple, fast)
- `particle.yaml` - Particle navigation (very simple, for testing)
- `walker.yaml` - Walker locomotion (more complex, slower)

## Config Parameters

### Environment
- `task_name`: Name of the task (must match environment in gpc.envs)
- `episode_length`: Number of simulation steps per episode

### Controller
- `controller_type`: "predictive_sampling" or "evosax"
- `num_samples`: Number of control samples from base controller
- `noise_level`: Noise level for sampling
- `num_policy_samples`: Number of samples from learned policy

### Architecture
- `architecture`: "mlp" or "cnn"
- `hidden_layers`: List of hidden layer sizes

### Training
- `num_iters`: Number of training iterations
- `num_envs`: Number of parallel environments
- `learning_rate`: Adam learning rate
- `batch_size`: Batch size for policy training
- `num_epochs`: Epochs per training iteration
- `exploration_noise_level`: Action noise during data collection
- `normalize_observations`: Whether to normalize observations

### Checkpointing & Evaluation
- `checkpoint_every`: Save checkpoint every N iterations
- `num_eval_episodes`: Number of evaluation episodes
- `video_fps`: Video frame rate
- `video_quality`: CRF quality (0-51, lower=better)

### Logging & Visualization
- `record_training_videos`: Record videos during training
- `num_training_videos`: Number of training videos per iteration
- `record_eval_videos`: Record evaluation videos
- `log_verbosity`: 0=minimal, 1=normal, 2=verbose

- `experiment_name`: Custom name (default: timestamp)

## Creating Custom Configs

1. Copy an existing config file
2. Modify parameters for your task
3. Run: `python run_experiment.py train --config configs/your_config.yaml`

## Command-Line Overrides

You can override config parameters from command line:
```bash
python run_experiment.py train --config configs/cart_pole.yaml \
    --num-iters 50 \
    --num-envs 256 \
    --exp-name my_experiment
```
