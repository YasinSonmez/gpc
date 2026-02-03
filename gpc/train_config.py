"""Enhanced training utilities with config support and video recording."""
import json
import pickle
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from tensorboardX import SummaryWriter

from gpc.augmented import PolicyAugmentedController
from gpc.config import TrainingConfig
from gpc.envs import SimulatorState, TrainingEnv
from gpc.experiment import ExperimentManager
from gpc.policy import Policy
from gpc.training import fit_policy, simulate_episode
from gpc.value_function import ValueFunctionTrainer as ValueTrainer
from gpc.replay_buffer import ReplayBuffer as TrajectoryReplayBuffer


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
    rng = jax.random.key(config.seed)
    np.random.seed(config.seed)  # For replay buffer sampling
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
    optimizer = nnx.Optimizer(net, optax.adamw(config.learning_rate), wrt=nnx.Param)
    
    tb_writer = SummaryWriter(str(exp_manager.log_dir))
    
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=exp_manager.exp_dir.name,
            config={
                "task_name": config.task_name,
                "num_iters": config.num_iters,
                "num_envs": config.num_envs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "num_policy_samples": config.num_policy_samples,
                "controller_type": config.controller_type,
            },
            dir=str(exp_manager.exp_dir),
        )
    
    replay_buffer: Optional[TrajectoryReplayBuffer] = None
    if config.use_replay_buffer:
        replay_buffer = TrajectoryReplayBuffer(
            capacity=config.replay_buffer_size,
            observation_size=env.observation_size,
            action_size=env.task.model.nu,
            horizon=config.num_knots,
        )
        if config.log_verbosity >= 1:
            print(f"  Using replay buffer (max size: {config.replay_buffer_size:,})")
            
    # Set up Value Function training
    value_trainer = None
    value_replay_buffer = None
    value_fn = None
    if config.use_value_function:
        value_trainer = ValueTrainer(
            observation_size=env.observation_size,
            hidden_layers=config.value_hidden_layers,
            learning_rate=config.value_learning_rate,
            polyak_tau=config.polyak_tau,
            rngs=nnx.Rngs(config.seed + 1)
        )
        plan_horizon_steps = int(round(config.plan_horizon / env.task.model.opt.timestep))

        # Calculate capacity for a true sliding window of size N iterations
        if config.value_buffer_window > 0:
            # Transitions per episode = config.episode_length
            # Total capacity = (envs * length) * window_size
            value_buffer_capacity = config.num_envs * config.episode_length * config.value_buffer_window
            if config.log_verbosity >= 1:
                print(f"  Value buffer window enabled: {config.value_buffer_window} iterations (capacity: {value_buffer_capacity})")
        else:
            value_buffer_capacity = config.replay_buffer_size

        value_replay_buffer = TrajectoryReplayBuffer(
            capacity=value_buffer_capacity,
            observation_size=env.observation_size,
            action_size=env.task.model.nu,
            horizon=config.num_knots,
        )

        if config.log_verbosity >= 1:
            print(f"  Initialized Value Trainer (expectile: {config.iql_tau})")

        # Define value function for the controller
        v_graphdef, _ = nnx.split(value_trainer.model)
        def value_fn(params, data):
            if params is None:
                return jnp.array(0.0)
            # Merge state into a temporary model to compute value
            v_model = nnx.merge(v_graphdef, params)
            return v_model.get_value(env.get_obs(data), aggregate="max")
    
    # Load initial replay buffer if provided
    expert_average_cost = 0.0
    if config.initial_replay_buffer_path:
        if os.path.exists(config.initial_replay_buffer_path):
            if config.log_verbosity >= 1:
                print(f"  Loading initial replay buffer from {config.initial_replay_buffer_path}...")
            try:
                # Load using the unified loader
                expert_buffer = TrajectoryReplayBuffer.load(
                    config.initial_replay_buffer_path, 
                    capacity=config.replay_buffer_size
                )
                
                # 1. Load into Value Replay Buffer
                if config.warmstart_value_buffer and value_replay_buffer is not None:
                    # Sync data from expert_buffer
                    value_replay_buffer.add(
                        expert_buffer.obs[:expert_buffer.size],
                        expert_buffer.actions[:expert_buffer.size],
                        expert_buffer.costs[:expert_buffer.size],
                        expert_buffer.next_obs[:expert_buffer.size],
                        expert_buffer.dones[:expert_buffer.size],
                        discount=config.discount_factor
                    )
                    if config.log_verbosity >= 1:
                        print(f"    Loaded {expert_buffer.size} transitions into Value buffer.")
                
                # 2. Load into Policy Replay Buffer
                if config.warmstart_policy_buffer and replay_buffer is not None:
                     # For the policy buffer, we use the sequences as well
                     replay_buffer.add(
                         expert_buffer.obs[:expert_buffer.size],
                         expert_buffer.actions[:expert_buffer.size],
                         expert_buffer.costs[:expert_buffer.size],
                         expert_buffer.next_obs[:expert_buffer.size],
                         expert_buffer.dones[:expert_buffer.size],
                         u_sequences=expert_buffer.u_sequences[:expert_buffer.size],
                         u_prev_sequences=expert_buffer.u_prev_sequences[:expert_buffer.size],
                         discount=config.discount_factor
                     )
                     if config.log_verbosity >= 1:
                        print(f"    Loaded {expert_buffer.size} transitions into Policy buffer.")
                        
                     # Calculate expert average return (cost) for filtering
                     # Note: Costs in buffer are per-step. Return is sum over episode.
                     expert_average_cost = float(np.mean(expert_buffer.costs[:expert_buffer.size])) * config.episode_length
                     if config.log_verbosity >= 1:
                        print(f"    Expert average episode cost: {expert_average_cost:.2f}")
                        
                if config.log_verbosity >= 1:
                    targets = []
                    if config.warmstart_value_buffer and value_replay_buffer is not None: targets.append("Value")
                    if config.warmstart_policy_buffer and replay_buffer is not None: targets.append("Policy")
                    if targets:
                        print(f"  Warm-started: {' & '.join(targets)} buffers from expert data ({expert_buffer.size} transitions).")
            except Exception as e:
                print(f"  ERROR: Failed to load initial replay buffer: {e}")
        else:
            print(f"  WARNING: Initial replay buffer path {config.initial_replay_buffer_path} does not exist.")
    
    # Update controller with value function if enabled
    if value_trainer is not None:
        # Inject value_fn into the controller
        ctrl.value_fn = value_fn
    
    # JIT-compiled functions
    @nnx.jit
    def jit_simulate(
        policy: Policy, rng: jax.Array, value_params=None, value_alpha=0.0
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState]:
        """Simulate episodes in parallel."""
        rngs = jax.random.split(rng, config.num_envs)
        
        target_cost = 0.0
        if replay_buffer is not None:
            target_cost = replay_buffer.target_cost_normalized
            
        return jax.vmap(
            simulate_episode,
            in_axes=(None, None, None, None, 0, None, None, None, None, None),
        )(
            env, 
            ctrl, 
            policy, 
            config.exploration_noise_level, 
            rngs, 
            config.strategy,
            target_cost,
            config.cfg_guidance_scale if config.use_cfg else 1.0,
            value_params,
            value_alpha,
        )
    
    @nnx.jit(static_argnums=(6,))
    def jit_fit(
        policy: Policy,
        optimizer: nnx.Optimizer,
        observations: jax.Array,
        actions: jax.Array,
        previous_actions: jax.Array,
        rng: jax.Array,
        num_epochs: int,
        costs: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Fit the policy network to the data (unconditional model)."""
        # Flatten across timesteps and initial conditions
        y = observations.reshape(-1, observations.shape[-1])
        U = actions.reshape(-1, config.num_knots, env.task.model.nu)
        U_guess = previous_actions.reshape(
            -1, config.num_knots, env.task.model.nu
        )
        
        flat_costs = None
        if costs is not None:
             # Flatten costs: (num_episodes, num_timesteps) -> (flat_size,)
             # Note: Input costs are typically (num_episodes, num_timesteps)
             # Reshape to (-1,) to match y and U
             flat_costs = costs.reshape(-1)

        # Rescale the actions from [u_min, u_max] to [-1, 1]
        mean = (env.task.u_max + env.task.u_min) / 2
        scale = (env.task.u_max - env.task.u_min) / 2
        U = (U - mean) / scale
        U_guess = (U_guess - mean) / scale

        # Normalize the observations, updating the running statistics stored
        # in the policy
        y = policy.normalizer(y, use_running_average=not config.normalize_observations)

        # Do the regression
        return fit_policy(
            y,
            U,
            U_guess,
            policy.model,
            optimizer,
            config.batch_size,
            num_epochs,
            rng,
            costs=flat_costs,
            cost_weight_temperature=config.cost_weight_temperature,
        )
    
    training_history = []
    
    # Training loop
    for iteration in range(config.num_iters):
        iter_start = time.time()
        
        # === Data Generation Phase ===
        skip_simulation = (iteration == 0) and (replay_buffer is not None) and (replay_buffer.size > 0)
        
        # Calculate current value_alpha with delayed start and optional exponential ramping
        if iteration < config.value_alpha_start_iter:
            # Before start iteration, keep at 0
            value_alpha = 0.0
        elif config.num_iters <= config.value_alpha_start_iter:
            # If start_iter >= num_iters, just use end value
            value_alpha = config.value_alpha_end
        else:
            # Calculate progress from start_iter to num_iters
            denom = config.num_iters - config.value_alpha_start_iter - 1
            if denom > 0:
                progress = (iteration - config.value_alpha_start_iter) / denom
            else:
                progress = 1.0
            progress = jnp.clip(progress, 0.0, 1.0)
            
            if config.value_alpha_ramp_type == "exponential":
                # Exponential ramp: slow start, fast finish
                progress = progress ** 3
            # else: linear (default)
            
            value_alpha = config.value_alpha_start + (config.value_alpha_end - config.value_alpha_start) * progress

        if not skip_simulation:
            if config.log_verbosity >= 2:
                print(f"\n  [1/3] Generating training data ({config.num_envs} parallel episodes)...")
        
            sim_start = time.time()
            rng, sim_rng = jax.random.split(rng)
            
            v_params = nnx.state(value_trainer.model) if value_trainer else None
            
            y_all = jit_simulate(policy, sim_rng, v_params, value_alpha)
            y, U, U_guess, J_spc, J_policy, J_inst, actions_taken, next_y, dones, states = y_all
            y.block_until_ready()
            sim_time = time.time() - sim_start
            
            # Save raw data for replay buffer (before any filtering)
            y_raw, U_raw, U_guess_raw = y, U, U_guess
            
            # Compute statistics from ALL episodes
            J_spc_mean = float(jnp.mean(J_spc))
            J_policy_mean = float(jnp.mean(J_policy))
            frac_policy_best = float(jnp.mean(J_policy < J_spc))
            
            # Total episode cost
            episode_costs = jnp.sum(J_inst, axis=1)
            J_episode_mean = float(jnp.mean(episode_costs))
            J_episode_std = float(jnp.std(episode_costs))
            
            # episode_costs_for_buffer: (num_envs,) - True episode cost from simulator
            episode_costs_for_buffer = episode_costs
            if replay_buffer is not None:
                # Apply return-based filtering if configured
                mask = jnp.ones(config.num_envs, dtype=bool)
                if config.buffer_filtering_fraction > 0 and expert_average_cost != 0:
                    # Threshold is defined relative to expert mean return (negative cost)
                    # Return >= fraction * expert_mean_return  =>  Cost <= fraction * expert_mean_cost
                    threshold_cost = config.buffer_filtering_fraction * expert_average_cost
                    mask = episode_costs_for_buffer <= threshold_cost
                    
                    num_added = int(jnp.sum(mask))
                    if config.log_verbosity >= 1:
                        print(f"    Buffer filtering: {num_added}/{config.num_envs} episodes met return threshold ({threshold_cost:.2f})")
                
                if jnp.any(mask):
                    replay_buffer.add(
                        y_raw[mask], actions_taken[mask], episode_costs_for_buffer[mask], next_y[mask], dones[mask],
                        u_sequences=U_raw[mask], u_prev_sequences=U_guess_raw[mask]
                    )
    
            # === Value Transition Extraction ===
            if value_trainer is not None:
                if config.log_verbosity >= 2:
                    print(f"  Extracting value transitions...")
                
                # Flatten across environments and time
                obs_flat = y.reshape(-1, y.shape[-1])
                act_flat = actions_taken.reshape(-1, actions_taken.shape[-1])
                cost_flat = J_inst.reshape(-1)
                next_obs_flat = next_y.reshape(-1, next_y.shape[-1])
                dones_flat = dones.reshape(-1)
                
                value_replay_buffer.add(
                    obs_flat, act_flat, cost_flat, next_obs_flat, dones_flat,
                    discount=config.discount_factor
                )
    
            # Flatten batch data for fitting (if needed)
            y = y.reshape(-1, y.shape[-1])
            U = U.reshape(-1, U.shape[-2], U.shape[-1])
            U_guess = U_guess.reshape(-1, U_guess.shape[-2], U_guess.shape[-1])
            
            render_time = 0.0
            training_videos = []
            if config.record_training_videos and iteration % max(1, config.num_iters // 10) == 0:
                if config.log_verbosity >= 2:
                    print(f"  [2/3] Rendering training videos...")
                
                render_start = time.time()
                for vid_idx in range(min(config.num_training_videos, config.num_envs)):
                    traj_ep = jax.tree.map(lambda x: x[vid_idx], states)
                    frames = env.render(traj_ep, fps=config.video_fps)
                    
                    if frames.dtype != np.uint8:
                        frames = (frames * 255).astype(np.uint8)
                    frames = np.ascontiguousarray(frames)
                    training_videos.append(frames)
                    
                    frames_for_save = frames.transpose(0, 2, 3, 1) if frames.shape[1] == 3 else frames
                    video_path = exp_manager.get_video_path(iteration=iteration, prefix=f"train_ep{vid_idx}")
                    exp_manager.save_video(
                        frames_for_save, video_path, fps=config.video_fps, 
                        quality=config.video_quality, resolution=config.video_resolution
                    )
                render_time = time.time() - render_start
                
                if config.log_verbosity >= 2:
                    print(f"      Saved {config.num_training_videos} videos ({render_time:.2f}s)")
        else:
            if config.log_verbosity >= 1:
                print(f"\n  [1/3] Skipping simulation (Warm-start Optimization)...")
            
            sim_time = 0.0
            render_time = 0.0
            
            # Dummy stats for logging
            J_spc_mean = 0.0
            J_policy_mean = 0.0
            frac_policy_best = 0.0
            J_episode_mean = 0.0
            J_episode_std = 0.0
            
            # Dummy y/U/etc for fitting if buffer is missing
            y = jnp.zeros((1, env.observation_size))
            U = jnp.zeros((1, config.num_knots, env.task.model.nu))
            U_guess = jnp.zeros((1, config.num_knots, env.task.model.nu))
            episode_costs_for_buffer = jnp.zeros((1,))
            
            training_videos = []

        
        # === Policy Fitting Phase ===
        if config.log_verbosity >= 2:
            print(f"  [3/3] Fitting policy to data...")
        
        policy.model.train()
        fit_start = time.time()
        
        # Debug: Check normalizer stats before fitting
        if policy.normalizer is not None and config.log_verbosity >= 2:
            print(f"    [Debug] Normalizer - mean: {jnp.mean(policy.normalizer.mean.value):.4f}, var: {jnp.mean(policy.normalizer.var.value):.4f}")
            
        # Epoch multiplier for warm-start
        current_num_epochs = config.num_epochs
        if iteration == 0 and skip_simulation:
            current_num_epochs *= 20
            if config.log_verbosity >= 1:
                print(f"    Increasing fitting epochs for warm-start: {config.num_epochs} -> {current_num_epochs}")

        rng, fit_rng = jax.random.split(rng)
        
        cfg_info = None
        if config.use_cost_conditioning:
            from gpc.cost_conditioned import fit_policy_cost_conditioned, fit_policy_cfg, ReplayBuffer
            
            if config.use_cfg:
                if replay_buffer is not None:
                    loss, cfg_info = fit_policy_cfg(
                        replay_buffer,
                        policy.model,
                        optimizer,
                        config.batch_size,
                        current_num_epochs,
                        fit_rng,
                        normalizer=policy.normalizer,
                        update_normalizer=config.normalize_observations,
                        cfg_drop_prob=config.cfg_drop_prob,
                        cost_weight_temperature=config.cost_weight_temperature,
                        verbose=(config.log_verbosity >= 2),
                    )
                else:
                    temp_buffer = TrajectoryReplayBuffer(
                        capacity=y_raw.shape[0] * y_raw.shape[1],
                        observation_size=env.observation_size,
                        action_size=env.task.model.nu,
                        horizon=config.num_knots,
                    )
                    temp_buffer.add(
                        y_raw, actions_taken, episode_costs_for_buffer, next_y, dones,
                        u_sequences=U_raw, u_prev_sequences=U_guess_raw
                    )
                    loss, cfg_info = fit_policy_cfg(
                        temp_buffer, policy.model, optimizer,
                        config.batch_size, current_num_epochs, fit_rng,
                        normalizer=policy.normalizer,
                        update_normalizer=config.normalize_observations,
                        cfg_drop_prob=config.cfg_drop_prob,
                        verbose=(config.log_verbosity >= 2),
                    )
            else:
                if replay_buffer is not None:
                    batch_obs, batch_act, batch_old_act, batch_costs = replay_buffer.get_all()
                else:
                    batch_obs = y
                    batch_act = U
                    batch_old_act = U_guess
                    costs_per_timestep = jnp.broadcast_to(
                        episode_costs_for_buffer[:, None], 
                        (config.num_envs, env.episode_length)
                    )
                    batch_costs_flat = costs_per_timestep.flatten()
                    cost_min = jnp.min(batch_costs_flat)
                    cost_std = jnp.std(batch_costs_flat) + 1e-8
                    batch_costs = jnp.clip((batch_costs_flat - cost_min) / (cost_std * 3 + 1e-8), 0.0, 1.0)
                
                loss = fit_policy_cost_conditioned(
                    batch_obs, batch_act, batch_old_act, batch_costs,
                    policy.model, optimizer,
                    config.batch_size, current_num_epochs, fit_rng,
                    normalizer=policy.normalizer,
                    update_normalizer=config.normalize_observations,
                )
        else:
            if replay_buffer is not None:
                # Get all data including costs
                batch_obs, batch_act, batch_old_act, batch_costs = replay_buffer.get_all()
                loss = jit_fit(policy, optimizer, batch_obs, batch_act, batch_old_act, fit_rng, current_num_epochs, costs=batch_costs)
            else:
                # Standard online training
                # Calculate costs for current batch
                # episode_costs is (num_episodes,) -> broadcast to (num_episodes, num_timesteps)?
                # Wait, simulate_episode returns J_spc/J_policy but NOT per-timestep costs in the output tuple (it returns scalars mean J)
                # Correction: The output 'y' and 'U' are (num_episodes, num_timesteps, ...).
                # We need per-timestep costs.
                # However, the `y` and `U` collected in `simulate_episode` are typically full trajectories?
                
                # In this online mode, we have:
                # y: (num_envs, time, obs)
                # U: (num_envs, time, horizon, action)
                
                # We also have `costs_per_timestep` from earlier (lines 173-176 in original code I saw?)
                # Let's check if `costs_per_timestep` is available.
                costs_per_timestep = jnp.broadcast_to(
                    episode_costs[:, None], 
                    (config.num_envs, y.shape[1])
                )
                
                # Re-calculating normalized costs using the same logic as replay buffer would be ideal.
                # Since we don't have a buffer, we use batch statistics.
                # Use standard normalization (min-max scaled by std)
                batch_costs_flat = costs_per_timestep.flatten()
                cost_min = jnp.min(batch_costs_flat)
                cost_std = jnp.std(batch_costs_flat) + 1e-8
                batch_costs_normalized = jnp.clip((costs_per_timestep - cost_min) / (cost_std + 1e-8), 0.0, 1.0)
                
                loss = jit_fit(policy, optimizer, y, U, U_guess, fit_rng, current_num_epochs, costs=batch_costs_normalized)
        
        loss.block_until_ready()
        fit_time = time.time() - fit_start
        
        # === Value Network Training Phase ===
        value_loss = 0.0
        v_mae, v_corr, v_mean, v_true_mean = 0.0, 0.0, 0.0, 0.0
        # Start training N iterations before it's first used, to allow for warm-up
        value_train_start_iter = max(0, config.value_alpha_start_iter - config.value_buffer_window)
        
        if value_trainer is not None and iteration >= value_train_start_iter and value_replay_buffer.size > config.batch_size:
            current_value_epochs = config.value_train_epochs
            if iteration == 0 and skip_simulation:
                current_value_epochs *= 20
                if config.log_verbosity >= 1:
                    print(f"    Increasing value fitting epochs for warm-start: {config.value_train_epochs} -> {current_value_epochs}")
            
            if config.log_verbosity >= 2:
                print(f"  Value Network training ({current_value_epochs} epochs)...")
            
            rng, v_fit_rng = jax.random.split(rng)
            n_step = plan_horizon_steps
            
            # Vectorized sampling: get a single block of data for the GPU
            # This is significantly faster than sampling per batch in Python
            value_data = value_replay_buffer.sample_n_step_all(n_step, config.discount_factor)
            
            # Convert all to jnp arrays once
            value_data = {k: jnp.array(v) for k, v in value_data.items()}
            
            # Use JAX-optimized fit method (scanned on GPU)
            value_loss = value_trainer.fit(
                value_data, 
                config.batch_size, 
                current_value_epochs, 
                v_fit_rng, 
                config.discount_factor, 
                config.iql_tau, 
                n_step
            )
            value_loss = float(value_loss)
            
            # Diagnostics: Check value function accuracy (every iteration)
            if value_replay_buffer.size > 1000:
                indices = np.random.randint(0, value_replay_buffer.size, size=min(500, value_replay_buffer.size))
                sample_obs = value_replay_buffer.obs[indices]
                true_returns = value_replay_buffer.returns_to_go[indices]
                
                v_pred = value_trainer.model.get_value(jnp.array(sample_obs), aggregate="max")
                errors = v_pred - true_returns
                v_mae = float(jnp.mean(jnp.abs(errors)))
                v_corr = float(jnp.corrcoef(v_pred, true_returns)[0, 1])
                v_mean = float(jnp.mean(v_pred))
                v_true_mean = float(np.mean(true_returns))
                
                if config.log_verbosity >= 2:
                    print(f"      Value loss: {value_loss:.4f}")
                    print(f"      V diagnostics: MAE={v_mae:.1f}, Corr={v_corr:.3f}, "
                          f"V_mean={v_mean:.1f}, True_mean={v_true_mean:.1f}")
            elif config.log_verbosity >= 2:
                print(f"      Value loss: {value_loss:.4f}")
        
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
            "episode_cost": J_episode_mean,
            "episode_cost_std": J_episode_std,
            "policy_best_frac": frac_policy_best,
            "loss": float(loss),
            "sim_time": sim_time,
            "fit_time": fit_time,
            "render_time": render_time,
            "total_time": iter_time,
            "num_episodes": config.num_envs,
            "data_size": replay_buffer.size if replay_buffer is not None else y.shape[0],
            "obs_mean": float(jnp.mean(y)),
            "obs_std": float(jnp.std(y)),
        }
        
        if value_trainer is not None:
            metrics["value_loss"] = value_loss
            metrics["value_alpha"] = float(value_alpha)
            metrics["value_buffer_size"] = value_replay_buffer.size
            if value_replay_buffer.size > 1000:
                metrics["value_mae"] = v_mae
                metrics["value_corr"] = v_corr
                metrics["value_v_mean"] = v_mean
                metrics["v_true_mean"] = v_true_mean
        training_history.append(metrics)
        
        # Save training history incrementally
        with open(exp_manager.exp_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        exp_manager.log_iteration_summary(
            iteration, config.num_iters, metrics, config.log_verbosity
        )
        
        # Log to markdown file
        exp_manager.log_iteration_stats(
            iteration=iteration + 1,
            total_iters=config.num_iters,
            policy_cost=J_policy_mean,
            spc_cost=J_spc_mean,
            episode_cost=J_episode_mean,
            episode_cost_std=J_episode_std,
            policy_best_frac=frac_policy_best,
            training_loss=float(loss),
            sim_time=sim_time,
            fit_time=fit_time,
            render_time=render_time,
            num_episodes=config.num_envs,
            num_data_points=metrics["data_size"],
            obs_mean=float(jnp.mean(y)),
            obs_std=float(jnp.std(y)),
        )
        
        tb_writer.add_scalar("cost/policy", J_policy_mean, iteration)
        tb_writer.add_scalar("cost/spc", J_spc_mean, iteration)
        tb_writer.add_scalar("cost/episode", J_episode_mean, iteration)
        tb_writer.add_scalar("cost/episode_std", J_episode_std, iteration)
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
        
        wandb_log = {
            "loss": float(loss),
            "policy_best_fraction": frac_policy_best,
            "cost/policy": J_policy_mean,
            "cost/spc": J_spc_mean,
            "cost/episode": J_episode_mean,
            "cost/episode_std": J_episode_std,
            "time/simulation": sim_time,
            "time/fitting": fit_time,
            "time/total": iter_time,
            "value_alpha": float(value_alpha),
        }
        if value_trainer is not None:
            wandb_log["value_loss"] = value_loss
            wandb_log["value_buffer_size"] = value_replay_buffer.size
            if value_replay_buffer.size > 1000:
                wandb_log["value/mae"] = v_mae
                wandb_log["value/correlation"] = v_corr
                wandb_log["value/v_mean"] = v_mean
                wandb_log["value/true_mean"] = v_true_mean
        if render_time > 0:
            wandb_log["time/rendering"] = render_time
        if training_videos and config.use_wandb:
            wandb_log["training_videos"] = [wandb.Video(vid, fps=config.video_fps, format="mp4") for vid in training_videos]
        
        if value_trainer is not None:
            wandb_log["value_loss"] = value_loss
            wandb_log["value_alpha"] = value_alpha
            tb_writer.add_scalar("value/loss", value_loss, iteration)
            tb_writer.add_scalar("value/alpha", value_alpha, iteration)
            
        if config.use_wandb:
            wandb.log(wandb_log, step=iteration)
    
    # === Final Evaluation ===
    if config.log_verbosity >= 1:
        print(f"\n{'='*80}")
        print("Running final evaluation...")
        print(f"{'='*80}")
    
    policy.model.eval()
    eval_costs = []
    eval_spc_costs = []
    eval_videos = []
    
    for ep in range(config.num_eval_episodes):
        if config.log_verbosity >= 1:
            print(f"  Evaluation episode {ep + 1}/{config.num_eval_episodes}...", end=" ")
        
        rng, eval_rng = jax.random.split(rng)
        _, _, _, J_spc_eval, J_eval, _, _, _, _, states = simulate_episode(
            env, ctrl, policy, 0.0, eval_rng, config.strategy
        )
        eval_cost = float(jnp.mean(J_eval))
        eval_spc_cost = float(jnp.mean(J_spc_eval))
        eval_costs.append(eval_cost)
        eval_spc_costs.append(eval_spc_cost)
        
        if config.log_verbosity >= 1:
            print(f"cost={eval_cost:.4f}")
        
        if config.record_eval_videos:
            frames = env.render(states, fps=config.video_fps)
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8)
            frames = np.ascontiguousarray(frames)
            eval_videos.append(frames)
            
            frames_for_save = frames.transpose(0, 2, 3, 1) if frames.shape[1] == 3 else frames
            video_path = exp_manager.get_video_path(episode=ep + 1)
            exp_manager.save_video(
                frames_for_save, video_path, fps=config.video_fps, 
                quality=config.video_quality, resolution=config.video_resolution
            )
    
    if eval_costs:
        eval_mean = np.mean(eval_costs)
        eval_std = np.std(eval_costs)
        eval_spc_mean = np.mean(eval_spc_costs)
        
        if config.log_verbosity >= 1:
            print(f"\n  Evaluation cost: {eval_mean:.4f} ± {eval_std:.4f}")
            print(f"  Videos saved to: {exp_manager.eval_dir}")
        
        wandb.log({
            "eval/cost_mean": eval_mean,
            "eval/cost_std": eval_std,
            "eval/cost_policy": eval_mean,
            "eval/cost_spc": eval_spc_mean,
            "eval/cost_min": np.min(eval_costs),
            "eval/cost_max": np.max(eval_costs),
        })
        
        if eval_videos:
            wandb.log({
                "eval_videos": [wandb.Video(vid, fps=config.video_fps, format="mp4") for vid in eval_videos]
            })
    else:
        if config.log_verbosity >= 1:
            print("\n  Evaluation skipped (num_eval_episodes=0)")
    
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
    
    # Save training history
    with open(exp_manager.exp_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    tb_writer.close()
    wandb.finish()
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
    
    if config.use_cost_conditioning:
        print(f"\nCost Conditioning:")
        print(f"  Enabled:             Yes")
        if config.use_cfg:
            print(f"  CFG:                 Yes")
            print(f"  Drop Probability:    {config.cfg_drop_prob:.1%}")
            print(f"  Guidance Scale:      {config.cfg_guidance_scale}")
        else:
            print(f"  CFG:                 No")
        print(f"  Replay Buffer:       {'Yes' if config.use_replay_buffer else 'No'}")
        if config.use_replay_buffer:
            print(f"  Buffer Size:         {config.replay_buffer_size:,}")
    
    print(f"\nVisualization:")
    print(f"  Training Videos:     {'Yes' if config.record_training_videos else 'No'}")
    print(f"  Eval Videos:         {'Yes' if config.record_eval_videos else 'No'}")
    print(f"  Video FPS:           {config.video_fps}")
    print(f"  Video Quality:       {config.video_quality} (CRF)")
    
    print(f"\n{'='*80}\n")
