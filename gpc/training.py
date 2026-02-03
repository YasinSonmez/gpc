import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from hydrax.alg_base import SamplingBasedController
from tensorboardX import SummaryWriter

from gpc.augmented import PACParams, PolicyAugmentedController
from gpc.envs import SimulatorState, TrainingEnv
from gpc.policy import Policy
from gpc.replay_buffer import ReplayBuffer
from gpc.value_function import ValueFunctionTrainer

Params = Any


def simulate_episode(
    env: TrainingEnv,
    ctrl: PolicyAugmentedController,
    policy: Policy,
    exploration_noise_level: float,
    rng: jax.Array,
    strategy: str = "policy",
    target_cost: float | None = None,
    cfg_scale: float = 1.0,
    value_params: Optional[Any] = None,
    value_alpha: float = 0.0,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState]:
    """Starting from a random initial state, run SPC and record training data.

    Args:
        env: The training environment.
        ctrl: The sampling-based controller (augmented with a learned policy).
        policy: The generative policy network.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action.
        rng: The random number generator key.
        strategy: The strategy for advancing the simulation. "policy" uses the
                  first policy sample, while "best" agregates all samples.
        target_cost: Target normalized cost for CFG.
        cfg_scale: Guidance scale (w) for CFG.

    Returns:
        y: The observations at each time step.
        U: The optimal actions at each time step.
        U_guess: The initial guess for the optimal actions at each time step.
        J_spc: cost of the best action sequence found by SPC at each time step.
        J_policy: cost of the best action sequence found by the policy.
        states: Vmapped simulator states at each time step.
    """
    rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

    x = env.init_state(env_rng)
    psi = ctrl.init_params()
    psi = psi.replace(
        base_params=psi.base_params.replace(rng=ctrl_rng),
        target_cost=target_cost,
        cfg_scale=cfg_scale,
        value_params=value_params,
        value_alpha=value_alpha,
    )

    def _scan_fn(
        carry: Tuple[SimulatorState, jax.Array, PACParams], t: int
    ) -> Tuple:
        x, U, psi = carry

        y = env._get_observation(x)
        rng, policy_rng, explore_rng = jax.random.split(psi.base_params.rng, 3)
        warm_start_level = 0.0
        
        policy_rngs = jax.random.split(policy_rng, ctrl.num_policy_samples)
        Us = jax.vmap(policy.apply, in_axes=(0, None, 0, None, None, None))(
            U, y, policy_rngs, warm_start_level, target_cost, cfg_scale
        )

        psi = psi.replace(
            policy_samples=Us, base_params=psi.base_params.replace(rng=rng)
        )

        U_guess = psi.mean
        psi, rollouts = ctrl.optimize(x.data, psi)
        U_star_knots = psi.mean

        costs = jnp.sum(rollouts.costs, axis=1)
        policy_best_idx = jnp.argmin(costs[: ctrl.num_policy_samples])
        spc_best_idx = (
            jnp.argmin(costs[ctrl.num_policy_samples :])
            + ctrl.num_policy_samples
        )
        spc_best = costs[spc_best_idx]
        policy_best = costs[policy_best_idx]

        if strategy == "policy":
            u = Us[0, 0]
        elif strategy == "best":
            u = ctrl.get_action_sequence(psi)[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        exploration_noise = exploration_noise_level * jax.random.normal(
            explore_rng, u.shape
        )
        u_taken = u + exploration_noise
        x = env.step(x, u_taken)

        # Calculate instantaneous cost (true cost at current step)
        J_inst = env.task.running_cost(x.data, u)

        return (x, Us, psi), (y, U_star_knots, U_guess, spc_best, policy_best, J_inst, u_taken, x)

    rng, u_rng = jax.random.split(rng)
    U = jax.random.normal(
        u_rng,
        (ctrl.num_policy_samples, ctrl.num_knots, env.task.model.nu),
    )
    _, (y, U, U_guess, J_spc, J_policy, J_inst, actions_taken, states) = jax.lax.scan(
        _scan_fn, (x, U, psi), jnp.arange(env.episode_length)
    )

    # Produce next observations for replay buffer
    next_y = jax.vmap(env._get_observation)(states)
    
    # Calculate dones (last step is done)
    # states.t is (episode_length,)
    dones = jnp.zeros(env.episode_length)
    dones = dones.at[-1].set(1.0)
    
    return y, U, U_guess, J_spc, J_policy, J_inst, actions_taken, next_y, dones, states


def fit_policy(
    observations: jax.Array,
    action_sequences: jax.Array,
    old_action_sequences: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
    sigma_min: float = 1e-2,
    costs: Optional[jax.Array] = None,
    cost_weight_temperature: float = 1.0,
) -> jax.Array:
    """Fit a flow matching model to the data.

    This model generates samples U ~ π(U|y) from the policy by flowing from
    U ~ N(0, I) to the target action sequence U*.

    Args:
        observations: The (normalized) observations y.
        action_sequences: The corresponding target action sequences U.
        old_action_sequences: The previous action sequences U_guess.
        model: The policy network, outputs the flow matching vector field.
        optimizer: The optimizer (e.g. Adam).
        batch_size: The batch size.
        num_epochs: The number of epochs.
        rng: The random number generator key.
        sigma_min: Target distribution width for flow matching, see
                   https://arxiv.org/pdf/2210.02747, eq (20-23).
        costs: Optional normalized costs for weighting the loss.
        cost_weight_temperature: Temperature for cost weighting.

    Returns:
        The loss from the last epoch.

    Note that model and optimizer are updated in-place by flax.nnx.
    """
    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        act: jax.Array,
        old_act: jax.Array,
        noise: jax.Array,
        t: jax.Array,
        step_costs: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute the flow-matching loss."""
        alpha = 1.0 - sigma_min
        noised_action = t[..., None] * act + (1 - alpha * t[..., None]) * noise
        target = act - alpha * noise
        pred = model(noised_action, obs, t)

        # Weigh the loss by how close the noise is to the old action sequence.
        v1 = (old_act - act).reshape(act.shape[0], -1)
        v2 = (noise - act).reshape(noise.shape[0], -1)
        dot = jnp.sum(v1 * v2, axis=-1)
        norm1 = jnp.linalg.norm(v1, axis=-1)
        norm2 = jnp.linalg.norm(v2, axis=-1)
        cosine_similarity = dot / (norm1 * norm2 + 1e-8)
        flow_weight = jax.lax.stop_gradient(jnp.exp(2 * (cosine_similarity - 1)))
        
        # Calculate per-sample MSE
        # pred: (batch, horizon, dim)
        sq_error = jnp.mean(jnp.square(pred - target), axis=(-2, -1))

        # Apply cost weighting if costs are provided and temperature > 0
        if step_costs is not None and cost_weight_temperature > 0:
             cost_weight = jnp.exp(-step_costs / cost_weight_temperature)
             cost_weight = jax.lax.stop_gradient(cost_weight)
             # flow_weight is scalar, cost_weight is (batch,)
             # sq_error is (batch,)
             return jnp.mean(flow_weight * cost_weight * sq_error)
        else:
             return jnp.mean(flow_weight * sq_error)

    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Perform a gradient descent step on a batch of data."""
        # Get a random batch of data
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(
            batch_rng, (batch_size,), 0, num_data_points
        )
        batch_obs = observations[batch_idx]
        batch_act = action_sequences[batch_idx]
        batch_old_act = old_action_sequences[batch_idx]
        
        batch_costs = None
        if costs is not None:
            batch_costs = costs[batch_idx]

        # Sample noise and time steps for the flow matching targets
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # Compute the loss and its gradient
        loss, grad = nnx.value_and_grad(_loss_fn)(
            model, batch_obs, batch_act, batch_old_act, noise, t, batch_costs
        )
        
        # Update the optimizer and model parameters in-place via flax.nnx
        optimizer.update(model, grad)

        return rng, loss

    # for i in range(num_batches * num_epochs): take a training step
    @nnx.scan
    def _scan_fn(carry: Tuple, i: int) -> Tuple:
        model, optimizer, rng = carry
        rng, loss = _train_step(model, optimizer, rng)
        return (model, optimizer, rng), loss

    _, losses = _scan_fn(
        (model, optimizer, rng), jnp.arange(num_batches * num_epochs)
    )

    return losses[-1]


def train(  # noqa: PLR0915 this is a long function, don't limit to 50 lines
    env: TrainingEnv,
    ctrl: SamplingBasedController,
    net: nnx.Module,
    num_policy_samples: int,
    log_dir: Union[Path, str],
    num_iters: int,
    num_envs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 10,
    checkpoint_every: int = 10,
    exploration_noise_level: float = 0.0,
    normalize_observations: bool = True,
    num_videos: int = 2,
    video_fps: int = 10,
    strategy: str = "policy",
) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The sampling-based predictive control method to use.
        net: The flow matching network architecture.
        num_policy_samples: The number of samples to draw from the policy.
        log_dir: The directory to log TensorBoard data to.
        num_iters: The number of training iterations.
        num_envs: The number of parallel environments to simulate.
        learning_rate: The learning rate for the policy network.
        batch_size: The batch size for training the policy network.
        num_epochs: The number of epochs to train the policy network.
        checkpoint_every: Number of iterations between policy checkpoint saves.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action during episode simulation.
        normalize_observations: Flag for observation normalization.
        num_videos: Number of videos to render for visualization.
        video_fps: Frames per second for rendered videos.
        strategy: The strategy for choosing a control action to advance the
                  simulation during the data collection phase. "policy" uses the
                  first policy sample, while "best" agregates all samples.

    """
    rng = jax.random.key(0)

    # Check that the task has finite input bounds
    assert jnp.all(jnp.isfinite(env.task.u_min))
    assert jnp.all(jnp.isfinite(env.task.u_max))

    # Check that the sampling-based predictive controller is compatible. In
    # particular, we need access to the mean of the sampling distribution.
    _spc_params = ctrl.init_params()
    assert hasattr(
        _spc_params, "mean"
    ), f"Controller '{type(ctrl).__name__}' is not compatible with GPC."

    # Print some information about the training setup
    episode_seconds = env.episode_length * env.task.model.opt.timestep
    horizon_seconds = env.task.planning_horizon * env.task.dt
    num_samples = num_policy_samples + ctrl.num_samples
    print("Training with:")
    print(
        f"  episode length: {episode_seconds} seconds"
        f" ({env.episode_length} simulation steps)"
    )
    (
        print(
            f"  planning horizon: {horizon_seconds} seconds"
            f" ({env.task.planning_horizon} knots)"
        ),
    )
    print(
        "  Parallel rollouts per simulation step:"
        f" {num_samples * ctrl.num_randomizations * num_envs}"
        f" (= {num_samples} x {ctrl.num_randomizations} x {num_envs})"
    )
    print("")

    # Print some info about the policy architecture
    params = nnx.state(net, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print(f"Policy: {type(net).__name__} with {total_params} parameters")
    print("")

    # Set up the sampling-based controller and policy network
    # Provide value_fn to the controller if enabled
    value_fn = None
    if use_value_function:
        # Wrap observation function for value network
        # Since the value function in PAC.optimize receives mjx.Data
        # but the ValueNetwork receives observations.
        value_fn = lambda params, data: value_trainer.model.get_value(env.get_obs(data), aggregate="max")

    ctrl = PolicyAugmentedController(ctrl, num_policy_samples, value_fn=value_fn)
    assert env.task == ctrl.task

    # Set up the policy
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adamw(learning_rate), wrt=nnx.Param)

    # Set up Value Function Trainer and Replay Buffer
    value_trainer = None
    if use_value_function:
        value_trainer = ValueFunctionTrainer(
            observation_size=env.observation_size,
            hidden_layers=value_hidden_layers,
            learning_rate=value_learning_rate,
            polyak_tau=polyak_tau,
            rngs=nnx.Rngs(1),
        )

    replay_buffer = None
    if use_replay_buffer or use_value_function:
        replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            observation_size=env.observation_size,
            action_size=env.task.model.nu,
            horizon=env.task.planning_horizon,
        )

    # Set up the TensorBoard logger
    log_dir = Path(log_dir) / time.strftime("%Y%m%d_%H%M%S")
    print("Logging to", log_dir)
    tb_writer = SummaryWriter(log_dir)

    # Set up some helper functions
    @nnx.jit
    def jit_simulate(
        policy: Policy, 
        rng: jax.Array,
        value_params: Optional[Any] = None,
        value_alpha: float = 0.0,
    ) -> Tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState
    ]:
        """Simulate episodes in parallel.

        Args:
            policy: The policy network.
            rng: The random number generator key.

        Returns:
            The observations at each time step.
            The best action sequence at each time step.
            Average cost of SPC's best action sequence.
            Average cost of the policy's best action sequence.
            Fraction of times the policy generated the best action sequence.
            First four simulation trajectories for visualization.
        """
        rngs = jax.random.split(rng, num_envs)

        y, U, U_guess, J_spc, J_policy, J_inst, actions_taken, next_y, dones, states = jax.vmap(
            simulate_episode, in_axes=(None, None, None, None, 0, None, None, None, None, None)
        )(env, ctrl, policy, exploration_noise_level, rngs, strategy, None, 1.0, value_params, value_alpha)

        # Get the first few simulated trajectories
        selected_states = jax.tree.map(lambda x: x[:num_videos], states)

        frac = jnp.mean(J_policy < J_spc)
        return (
            y,
            U,
            U_guess,
            jnp.mean(J_spc),
            jnp.mean(J_policy),
            frac,
            J_inst,
            actions_taken,
            next_y,
            dones,
            selected_states,
        )

    @nnx.jit
    def jit_fit(
        policy: Policy,
        optimizer: nnx.Optimizer,
        observations: jax.Array,
        actions: jax.Array,
        previous_actions: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """Fit the policy network to the data.

        Args:
            policy: The policy network (updated in place).
            optimizer: The optimizer (updated in place).
            observations: The observations.
            actions: The best action sequences.
            previous_actions: The initial/guessed action sequences.
            rng: The random number generator key.

        Returns:
            The loss from the last epoch.
        """
        # Flatten across timesteps and initial conditions
        y = observations.reshape(-1, observations.shape[-1])
        U = actions.reshape(-1, env.task.planning_horizon, env.task.model.nu)
        U_guess = previous_actions.reshape(
            -1, env.task.planning_horizon, env.task.model.nu
        )

        # Rescale the actions from [u_min, u_max] to [-1, 1]
        mean = (env.task.u_max + env.task.u_min) / 2
        scale = (env.task.u_max - env.task.u_min) / 2
        U = (U - mean) / scale
        U_guess = (U_guess - mean) / scale

        # Normalize the observations, updating the running statistics stored
        # in the policy
        y = policy.normalizer(y, use_running_average=not normalize_observations)

        # Do the regression
        return fit_policy(
            y,
            U,
            U_guess,
            policy.model,
            optimizer,
            batch_size,
            num_epochs,
            rng,
            costs=None,
            cost_weight_temperature=1.0,
        )

    train_start = datetime.now()
    for i in range(num_iters):
        # Calculate current value_alpha
        value_alpha = value_alpha_start
        if num_iters > 1:
            value_alpha += (value_alpha_end - value_alpha_start) * (i / (num_iters - 1))

        # Simulate and record the best action sequences. Some of the action
        # samples are generated via SPC and others are generated by the policy.
        policy.model.eval()
        sim_start = time.time()
        rng, episode_rng = jax.random.split(rng)
        
        # Get value params state if using value function
        v_params = None
        if use_value_function:
            v_params = nnx.state(value_trainer.model)

        y, U, U_guess, J_spc, J_policy, frac, J_inst, actions_taken, next_y, dones, traj = jit_simulate(
            policy, episode_rng, v_params, value_alpha
        )
        y.block_until_ready()
        sim_time = time.time() - sim_start

        # Store data in replay buffer
        if replay_buffer is not None:
            # y: (num_envs, ep_len, obs_dim)
            # U: (num_envs, ep_len, horiz, nu) -> we want the first action u: (num_envs, ep_len, nu)
            # J_inst: (num_envs, ep_len)
            u_first = U[:, :, 0, :]
            
            # Flatten across envs and episodes
            rb_y = y.reshape(-1, y.shape[-1])
            rb_u = u_first.reshape(-1, u_first.shape[-1])
            rb_j = J_inst.reshape(-1)
            rb_next_y = next_y.reshape(-1, next_y.shape[-1])
            rb_dones = dones.reshape(-1)
            
            replay_buffer.add(rb_y, rb_u, rb_j, rb_next_y, rb_dones)

        # Render the first few trajectories for visualization
        # N.B. this uses CPU mujoco's rendering utils, so we need to do it
        # sequentially and outside a jit-compiled function
        if num_videos > 0:
            render_start = time.time()
            video_frames = []
            for j in range(num_videos):
                states = jax.tree.map(lambda x: x[j], traj)  # noqa: B023
                video_frames.append(env.render(states, video_fps))
            video_frames = np.stack(video_frames)
            render_time = time.time() - render_start

        # Fit the policy network U = NNet(y) to the data
        policy.model.train()
        fit_start = time.time()
        rng, fit_rng = jax.random.split(rng)
        loss = jit_fit(policy, optimizer, y, U, U_guess, fit_rng)
        loss.block_until_ready()
        fit_time = time.time() - fit_start

        # TODO: run some evaluation tests

        # Save a policy checkpoint
        if i % checkpoint_every == 0 and i > 0:
            ckpt_path = log_dir / f"policy_ckpt_{i}.pkl"
            policy.save(ckpt_path)
            print(f"Saved policy checkpoint to {ckpt_path}")

        # Print a performance summary
        time_elapsed = datetime.now() - train_start
        print(
            f"  {i+1}/{num_iters} |"
            f" policy cost {J_policy:.4f} |"
            f" spc cost {J_spc:.4f} |"
            f" {100 * frac:.2f}% policy is best |"
            f" loss {loss:.4f} |"
            f" {time_elapsed} elapsed"
        )

        # Tensorboard logging
        tb_writer.add_scalar("sim/policy_cost", J_policy, i)
        tb_writer.add_scalar("sim/spc_cost", J_spc, i)
        tb_writer.add_scalar("sim/time", sim_time, i)
        tb_writer.add_scalar("sim/policy_best_frac", frac, i)
        tb_writer.add_scalar("sim/value_alpha", value_alpha, i)
        
        # Fit the value function if enabled
        if use_value_function:
            v_fit_start = time.time()
            v_losses = []
            # Plan horizon n for TD(n)
            n_step = env.task.planning_horizon
            
            # Fit for several epochs
            num_batches = max(1, replay_buffer.size // batch_size)
            for _ in range(num_batches * value_train_epochs):
                batch = replay_buffer.sample_n_step(batch_size, n_step, discount_factor)
                # Convert to jax arrays
                jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                v_loss = value_trainer.train_step(
                    jax_batch, discount_factor, iql_tau, n_step
                )
                v_losses.append(v_loss)
                value_trainer.update_targets()
                
            v_fit_time = time.time() - v_fit_start
            print(f"  value loss {jnp.mean(jnp.array(v_losses)):.4f} | value fit time {v_fit_time:.2f}s")
            tb_writer.add_scalar("fit/value_loss", jnp.mean(jnp.array(v_losses)), i)
            tb_writer.add_scalar("fit/value_time", v_fit_time, i)

        tb_writer.add_scalar("fit/loss", loss, i)
        tb_writer.add_scalar("fit/time", fit_time, i)
        if num_videos > 0:
            tb_writer.add_scalar("render/time", render_time, i)
            tb_writer.add_video(
                "render/trajectories", video_frames, i, fps=video_fps
            )
        tb_writer.flush()

    return policy
