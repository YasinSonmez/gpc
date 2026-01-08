"""Enhanced training utilities with config support and video recording."""
import time
from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tensorboardX import SummaryWriter

from gpc.augmented import PolicyAugmentedController
from gpc.config import TrainingConfig
from gpc.envs import SimulatorState, TrainingEnv
from gpc.experiment import ExperimentManager
from gpc.policy import Policy
from gpc.training import fit_policy, simulate_episode


def train_with_config(
    env: TrainingEnv,
    ctrl: PolicyAugmentedController,
    net: nnx.Module,
    config: TrainingConfig,
    exp_manager: ExperimentManager,
) -> Policy:
    """Train a GPC policy using configuration-based settings.
    
    Args:
        env: Training environment.
        ctrl: Policy-augmented controller.
        net: Neural network architecture.
        config: Training configuration.
        exp_manager: Experiment manager for organizing outputs.
        
    Returns:
        Trained policy.
    """
    rng = jax.random.key(0)
    train_start = datetime.now()
    
    # Validate environment
    assert jnp.all(jnp.isfinite(env.task.u_min)), "Task must have finite action bounds"
    assert jnp.all(jnp.isfinite(env.task.u_max)), "Task must have finite action bounds"
    
    # Print training configuration
    if config.log_verbosity >= 1:
        _print_training_header(env, ctrl, net, config)
    
    # Set up policy and optimizer
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)
    optimizer = nnx.Optimizer(net, optax.adamw(config.learning_rate))
    
    # Set up TensorBoard
    tb_writer = SummaryWriter(str(exp_manager.log_dir))
    
    # Set up replay buffer for CFG training
    replay_buffer: Optional["ReplayBuffer"] = None
    if config.use_cfg and config.use_replay_buffer:
        from gpc.cost_conditioned import ReplayBuffer
        replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
        if config.log_verbosity >= 1:
            print(f"  Using replay buffer (max size: {config.replay_buffer_size:,})")
    
    # JIT-compiled functions
    @nnx.jit
    def jit_simulate(
        policy: Policy, rng: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState]:
        """Simulate episodes in parallel."""
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(
            simulate_episode,
            in_axes=(None, None, None, None, 0, None, None, None),
        )(env, ctrl, policy, config.exploration_noise_level, rngs, config.strategy,
          config.num_latent_samples, config.latent_noise_level)
    
    @nnx.jit
    def jit_fit(
        policy: Policy,
        optimizer: nnx.Optimizer,
        y: jax.Array,
        U: jax.Array,
        U_guess: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """Fit policy to data."""
        return fit_policy(
            y, U, U_guess, policy.model, optimizer,
            config.batch_size, config.num_epochs, rng
        )
    
    # Training loop
    for iteration in range(config.num_iters):
        iter_start = time.time()
        
        # === Data Generation Phase ===
        if config.log_verbosity >= 2:
            print(f"\n  [1/3] Generating training data ({config.num_envs} parallel episodes)...")
        
        sim_start = time.time()
        rng, sim_rng = jax.random.split(rng)
        y, U, U_guess, J_spc, J_policy, traj = jit_simulate(policy, sim_rng)
        y.block_until_ready()
        sim_time = time.time() - sim_start
        
        # Save raw data for replay buffer (before any filtering)
        y_raw, U_raw, U_guess_raw = y, U, U_guess
        
        # Compute statistics from ALL episodes
        J_spc_mean = float(jnp.mean(J_spc))
        J_policy_mean = float(jnp.mean(J_policy))
        frac_policy_best = float(jnp.mean(J_policy < J_spc))
        
        # === CFG: Add to replay buffer (before flattening) ===
        if config.use_cfg and replay_buffer is not None:
            # Compute episode costs for replay buffer tagging
            episode_costs_raw = jnp.minimum(J_spc, J_policy)
            if episode_costs_raw.ndim == 2:
                episode_costs_for_buffer = jnp.mean(episode_costs_raw, axis=1)
            else:
                episode_costs_for_buffer = episode_costs_raw
            
            # Add raw (unfiltered, unflattened) data to buffer
            replay_buffer.add(y_raw, U_raw, U_guess_raw, episode_costs_for_buffer)
            
            if config.log_verbosity >= 2:
                print(f"  Replay buffer size: {replay_buffer.size:,}, best cost: {replay_buffer.best_cost:.2f}")
        
        # Flatten batch data
        y = y.reshape(-1, y.shape[-1])
        U = U.reshape(-1, U.shape[-2], U.shape[-1])
        U_guess = U_guess.reshape(-1, U_guess.shape[-2], U_guess.shape[-1])
        
        # === Video Recording Phase ===
        render_time = 0.0
        if config.record_training_videos and iteration % max(1, config.num_iters // 10) == 0:
            if config.log_verbosity >= 2:
                print(f"  [2/3] Rendering training videos...")
            
            render_start = time.time()
            for vid_idx in range(min(config.num_training_videos, config.num_envs)):
                states = jax.tree.map(lambda x: x[vid_idx], traj)
                frames = env.render(states, fps=config.video_fps)
                video_path = exp_manager.get_video_path(iteration=iteration, prefix=f"train_ep{vid_idx}")
                exp_manager.save_video(
                    frames, video_path, fps=config.video_fps, 
                    quality=config.video_quality, resolution=config.video_resolution
                )
            render_time = time.time() - render_start
            
            if config.log_verbosity >= 2:
                print(f"      Saved {config.num_training_videos} videos ({render_time:.2f}s)")
        
        # === Policy Fitting Phase ===
        if config.log_verbosity >= 2:
            print(f"  [3/3] Fitting policy to data...")
        
        policy.model.train()
        fit_start = time.time()
        rng, fit_rng = jax.random.split(rng)
        
        cfg_info = None
        if config.use_cfg and replay_buffer is not None:
            # CFG training with replay buffer
            from gpc.cost_conditioned import fit_policy_cfg
            loss, cfg_info = fit_policy_cfg(
                replay_buffer, policy.model, optimizer,
                config.batch_size, config.num_epochs, fit_rng,
                cfg_drop_prob=config.cfg_drop_prob,
                verbose=(config.log_verbosity >= 2),
            )
        else:
            # Standard training
            loss = jit_fit(policy, optimizer, y, U, U_guess, fit_rng)
        
        loss.block_until_ready()
        fit_time = time.time() - fit_start
        
        # === Checkpointing ===
        if (iteration + 1) % config.checkpoint_every == 0:
            ckpt_path = exp_manager.get_policy_path(iteration + 1)
            policy.save(ckpt_path)
            if config.log_verbosity >= 1:
                print(f"\n  💾 Checkpoint saved: {ckpt_path.name}")
        
        # === Logging ===
        iter_time = time.time() - iter_start
        
        metrics = {
            "policy_cost": J_policy_mean,
            "spc_cost": J_spc_mean,
            "policy_best_frac": frac_policy_best,
            "loss": float(loss),
            "sim_time": sim_time,
            "fit_time": fit_time,
            "render_time": render_time,
            "total_time": iter_time,
            "num_episodes": config.num_envs,
            "data_size": y.shape[0] if not (config.use_cfg and replay_buffer) else replay_buffer.size,
            "obs_mean": float(jnp.mean(y)),
            "obs_std": float(jnp.std(y)),
        }
        
        exp_manager.log_iteration_summary(
            iteration, config.num_iters, metrics, config.log_verbosity
        )
        
        # Log to markdown file
        exp_manager.log_iteration_stats(
            iteration=iteration + 1,
            total_iters=config.num_iters,
            policy_cost=J_policy_mean,
            spc_cost=J_spc_mean,
            policy_best_frac=frac_policy_best,
            training_loss=float(loss),
            sim_time=sim_time,
            fit_time=fit_time,
            render_time=render_time,
            num_episodes=config.num_envs,
            num_data_points=y.shape[0],
            obs_mean=float(jnp.mean(y)),
            obs_std=float(jnp.std(y)),
        )
        
        # TensorBoard logging
        tb_writer.add_scalar("cost/policy", J_policy_mean, iteration)
        tb_writer.add_scalar("cost/spc", J_spc_mean, iteration)
        tb_writer.add_scalar("policy/best_fraction", frac_policy_best, iteration)
        tb_writer.add_scalar("policy/loss", float(loss), iteration)
        tb_writer.add_scalar("time/simulation", sim_time, iteration)
        tb_writer.add_scalar("time/fitting", fit_time, iteration)
        tb_writer.add_scalar("time/total", iter_time, iteration)
        tb_writer.add_scalar("data/obs_mean", float(jnp.mean(y)), iteration)
        tb_writer.add_scalar("data/obs_std", float(jnp.std(y)), iteration)
        if render_time > 0:
            tb_writer.add_scalar("time/rendering", render_time, iteration)
        tb_writer.flush()
    
    # === Final Evaluation ===
    if config.log_verbosity >= 1:
        print(f"\n{'='*80}")
        print("Running final evaluation...")
        print(f"{'='*80}")
    
    policy.model.eval()
    eval_costs = []
    
    for ep in range(config.num_eval_episodes):
        if config.log_verbosity >= 1:
            print(f"  Evaluation episode {ep + 1}/{config.num_eval_episodes}...", end=" ")
        
        rng, eval_rng = jax.random.split(rng)
        _, _, _, _, J_eval, states = simulate_episode(
            env, ctrl, policy, 0.0, eval_rng, config.strategy
        )
        eval_cost = float(jnp.mean(J_eval))
        eval_costs.append(eval_cost)
        
        if config.log_verbosity >= 1:
            print(f"cost={eval_cost:.4f}")
        
        # Record evaluation video
        if config.record_eval_videos:
            frames = env.render(states, fps=config.video_fps)
            video_path = exp_manager.get_video_path(episode=ep + 1)
            exp_manager.save_video(
                frames, video_path, fps=config.video_fps, 
                quality=config.video_quality, resolution=config.video_resolution
            )
    
    if config.log_verbosity >= 1:
        print(f"\n  Evaluation cost: {np.mean(eval_costs):.4f} ± {np.std(eval_costs):.4f}")
        print(f"  Videos saved to: {exp_manager.eval_dir}")
    
    # Save final policy
    final_path = exp_manager.get_policy_path()
    policy.save(final_path)
    
    # Create experiment summary
    exp_manager.create_summary()
    
    # Training complete
    total_time = datetime.now() - train_start
    total_time_str = str(total_time).split('.')[0]  # Remove microseconds
    
    # Finalize training log
    exp_manager.finalize_training_log(total_time_str)
    
    if config.log_verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"Training Complete! Total time: {total_time_str}")
        print(f"Experiment saved to: {exp_manager.exp_dir.absolute()}")
        print(f"{'='*80}\n")
    
    tb_writer.close()
    return policy


def _print_training_header(
    env: TrainingEnv,
    ctrl: PolicyAugmentedController,
    net: nnx.Module,
    config: TrainingConfig,
) -> None:
    """Print detailed training configuration header."""
    episode_seconds = env.episode_length * env.task.model.opt.timestep
    # Use controller's plan_horizon instead of task.planning_horizon
    horizon_seconds = ctrl.base_ctrl.plan_horizon
    num_knots = ctrl.base_ctrl.num_knots
    num_samples = config.num_policy_samples + ctrl.base_ctrl.num_samples
    
    params = nnx.state(net, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration: {config.task_name}")
    print(f"{'='*80}\n")
    
    print("Environment:")
    print(f"  Episode Length:      {episode_seconds:.1f}s ({env.episode_length} steps)")
    print(f"  Planning Horizon:    {horizon_seconds:.1f}s ({num_knots} knots)")
    print(f"  Observation Dim:     {env.observation_size}")
    print(f"  Action Dim:          {env.task.model.nu}")
    
    print(f"\nController:")
    print(f"  Type:                {ctrl.base_ctrl.__class__.__name__}")
    print(f"  Base Samples:        {ctrl.base_ctrl.num_samples}")
    print(f"  Policy Samples:      {config.num_policy_samples}")
    print(f"  Total per Step:      {num_samples * ctrl.num_randomizations * config.num_envs}")
    print(f"    = {num_samples} × {ctrl.num_randomizations} × {config.num_envs}")
    
    print(f"\nPolicy Architecture:")
    print(f"  Type:                {net.__class__.__name__}")
    print(f"  Parameters:          {total_params:,}")
    print(f"  Hidden Layers:       {config.hidden_layers}")
    
    print(f"\nTraining:")
    print(f"  Iterations:          {config.num_iters}")
    print(f"  Parallel Envs:       {config.num_envs}")
    print(f"  Learning Rate:       {config.learning_rate}")
    print(f"  Batch Size:          {config.batch_size}")
    print(f"  Epochs per Iter:     {config.num_epochs}")
    
    # Print CFG settings if enabled
    if config.use_cfg:
        print(f"\nCost-Conditioned Flow Matching (CFG):")
        print(f"  Enabled:             Yes")
        print(f"  Drop Probability:    {config.cfg_drop_prob:.1%}")
        print(f"  Guidance Scale:      {config.cfg_guidance_scale}")
        print(f"  Replay Buffer:       {'Yes' if config.use_replay_buffer else 'No'}")
        if config.use_replay_buffer:
            print(f"  Buffer Size:         {config.replay_buffer_size:,}")
    
    print(f"\nVisualization:")
    print(f"  Training Videos:     {'Yes' if config.record_training_videos else 'No'}")
    print(f"  Eval Videos:         {'Yes' if config.record_eval_videos else 'No'}")
    print(f"  Video FPS:           {config.video_fps}")
    print(f"  Video Quality:       {config.video_quality} (CRF)")
    
    print(f"\n{'='*80}\n")
