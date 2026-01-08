"""Cost-Conditioned Flow Matching with Classifier-Free Guidance (CFG).

This module implements return/cost-conditioned flow matching to improve sample
efficiency by learning from all data while guiding toward low-cost behaviors.
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from gpc.architectures import MLP, PositionalEmbedding


class CostEmbedding(nnx.Module):
    """Embeds a scalar cost into a higher-dimensional space."""

    def __init__(self, embed_dim: int, rngs: nnx.Rngs):
        """Initialize cost embedding.
        
        Args:
            embed_dim: Output embedding dimension.
            rngs: Random number generators.
        """
        self.linear1 = nnx.Linear(1, embed_dim, rngs=rngs)
        self.linear2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, cost: jax.Array) -> jax.Array:
        """Embed cost scalar to embedding vector.
        
        Args:
            cost: Normalized cost value, shape (...,) or (..., 1).
            
        Returns:
            Embedding of shape (..., embed_dim).
        """
        if cost.ndim == 0 or (cost.ndim >= 1 and cost.shape[-1] != 1):
            cost = cost[..., None]
        x = self.linear1(cost)
        x = nnx.swish(x)
        x = self.linear2(x)
        return x


class CostConditionedMLP(nnx.Module):
    """Flow matching MLP conditioned on observation, time, and cost.
    
    Computes U* = NNet(U, y, t, c), where:
    - U: noisy action sequence
    - y: observation  
    - t: diffusion time
    - c: normalized episode cost (or null token for unconditional)
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        hidden_layers: Sequence[int],
        rngs: nnx.Rngs,
        cost_embed_dim: int = 16,
    ):
        """Initialize the network.

        Args:
            action_size: Dimension of actions.
            observation_size: Dimension of observations.
            horizon: Number of steps in action sequence.
            hidden_layers: Sizes of hidden layers.
            rngs: Random number generators.
            cost_embed_dim: Dimension of cost embedding.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon
        self.hidden_layers = list(hidden_layers)
        self.cost_embed_dim = cost_embed_dim

        # Cost embedding network
        self.cost_embed = CostEmbedding(cost_embed_dim, rngs)
        
        # Learnable null embedding for unconditional (CFG dropout)
        self.null_embed = nnx.Param(jax.random.normal(rngs.params(), (cost_embed_dim,)) * 0.02)

        # Main MLP: input = [U_flat, y, t, cost_embed]
        input_size = horizon * action_size + observation_size + 1 + cost_embed_dim
        output_size = horizon * action_size
        self.mlp = MLP(
            [input_size] + self.hidden_layers + [output_size], rngs=rngs
        )

    def __call__(
        self,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
        cost: Optional[jax.Array] = None,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Forward pass.
        
        Args:
            u: Noisy action sequence, shape (..., horizon, action_size).
            y: Observation, shape (..., observation_size).
            t: Diffusion time, shape (..., 1).
            cost: Normalized cost, shape (...,) or (..., 1). If None, uses null.
            use_running_average: Unused, for API compatibility.
            
        Returns:
            Predicted velocity field, shape (..., horizon, action_size).
        """
        batches = u.shape[:-2]
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        
        # Get cost embedding
        if cost is None:
            # Unconditional: use null embedding
            cost_emb = jnp.broadcast_to(self.null_embed.value, batches + (self.cost_embed_dim,))
        else:
            cost_emb = self.cost_embed(cost)
        
        x = jnp.concatenate([u_flat, y, t, cost_emb], axis=-1)
        x = self.mlp(x)
        return x.reshape(batches + (self.horizon, self.action_size))

    def forward_conditional(
        self,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
        cost: jax.Array,
    ) -> jax.Array:
        """Forward pass with cost conditioning."""
        return self(u, y, t, cost=cost)

    def forward_unconditional(
        self,
        u: jax.Array,
        y: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Forward pass without cost conditioning (null token)."""
        return self(u, y, t, cost=None)


class ReplayBuffer:
    """Simple replay buffer that accumulates data across iterations.
    
    Stores (observation, action, old_action, cost) tuples with normalized costs.
    """

    def __init__(self, max_size: int = 1_000_000):
        """Initialize buffer.
        
        Args:
            max_size: Maximum number of data points to store.
        """
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.old_actions = []
        self.costs = []  # Per-datapoint normalized cost
        self._size = 0
        
        # Running statistics for cost normalization
        self._cost_mean = 0.0
        self._cost_std = 1.0
        self._cost_min = float('inf')
        self._num_episodes = 0

    def add(
        self,
        observations: jax.Array,
        actions: jax.Array,
        old_actions: jax.Array,
        episode_costs: jax.Array,
    ) -> None:
        """Add new data to the buffer.
        
        Args:
            observations: Shape [num_episodes, episode_length, obs_dim].
            actions: Shape [num_episodes, episode_length, horizon, action_dim].
            old_actions: Shape [num_episodes, episode_length, horizon, action_dim].
            episode_costs: Per-episode costs, shape [num_episodes] or [num_episodes, episode_length].
        """
        # Convert to numpy for storage
        obs_np = jnp.asarray(observations)
        act_np = jnp.asarray(actions)
        old_act_np = jnp.asarray(old_actions)
        costs_np = jnp.asarray(episode_costs)
        
        num_episodes = obs_np.shape[0]
        episode_length = obs_np.shape[1]
        
        # Reduce episode costs to scalar per episode if needed
        if costs_np.ndim == 2:
            costs_np = jnp.mean(costs_np, axis=1)
        
        # Update cost statistics
        self._num_episodes += num_episodes
        all_costs = costs_np
        self._cost_mean = float(jnp.mean(all_costs))
        self._cost_std = float(jnp.std(all_costs) + 1e-8)
        self._cost_min = float(jnp.min(all_costs))
        
        # Broadcast episode cost to all timesteps
        # costs_per_timestep shape: [num_episodes, episode_length]
        costs_per_timestep = jnp.broadcast_to(
            costs_np[:, None], (num_episodes, episode_length)
        )
        
        # Flatten and append
        self.observations.append(obs_np.reshape(-1, obs_np.shape[-1]))
        self.actions.append(act_np.reshape(-1, act_np.shape[-2], act_np.shape[-1]))
        self.old_actions.append(old_act_np.reshape(-1, old_act_np.shape[-2], old_act_np.shape[-1]))
        self.costs.append(costs_per_timestep.flatten())
        
        self._size += num_episodes * episode_length
        
        # Trim if over capacity
        if self._size > self.max_size:
            self._trim()

    def _trim(self) -> None:
        """Trim buffer to max_size using FIFO."""
        # Concatenate all data
        all_obs = jnp.concatenate(self.observations, axis=0)
        all_act = jnp.concatenate(self.actions, axis=0)
        all_old_act = jnp.concatenate(self.old_actions, axis=0)
        all_costs = jnp.concatenate(self.costs, axis=0)
        
        # Keep only the most recent data
        keep_from = all_obs.shape[0] - self.max_size
        self.observations = [all_obs[keep_from:]]
        self.actions = [all_act[keep_from:]]
        self.old_actions = [all_old_act[keep_from:]]
        self.costs = [all_costs[keep_from:]]
        self._size = self.max_size

    def sample(
        self, batch_size: int, rng: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample a batch of data.
        
        Args:
            batch_size: Number of samples.
            rng: Random key.
            
        Returns:
            (observations, actions, old_actions, normalized_costs)
        """
        # Concatenate all stored data
        all_obs = jnp.concatenate(self.observations, axis=0)
        all_act = jnp.concatenate(self.actions, axis=0)
        all_old_act = jnp.concatenate(self.old_actions, axis=0)
        all_costs = jnp.concatenate(self.costs, axis=0)
        
        # Sample indices
        indices = jax.random.randint(rng, (batch_size,), 0, all_obs.shape[0])
        
        # Normalize costs to [0, 1] range (0 = best, 1 = worst)
        normalized_costs = (all_costs - self._cost_min) / (self._cost_std * 3 + 1e-8)
        normalized_costs = jnp.clip(normalized_costs, 0.0, 1.0)
        
        return (
            all_obs[indices],
            all_act[indices],
            all_old_act[indices],
            normalized_costs[indices],
        )

    def get_all(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get all data in the buffer.
        
        Returns:
            (observations, actions, old_actions, normalized_costs)
        """
        all_obs = jnp.concatenate(self.observations, axis=0)
        all_act = jnp.concatenate(self.actions, axis=0)
        all_old_act = jnp.concatenate(self.old_actions, axis=0)
        all_costs = jnp.concatenate(self.costs, axis=0)
        
        # Normalize costs
        normalized_costs = (all_costs - self._cost_min) / (self._cost_std * 3 + 1e-8)
        normalized_costs = jnp.clip(normalized_costs, 0.0, 1.0)
        
        return all_obs, all_act, all_old_act, normalized_costs

    @property
    def size(self) -> int:
        """Current number of data points."""
        return self._size

    @property
    def best_cost(self) -> float:
        """Best (lowest) cost seen so far."""
        return self._cost_min
    
    @property
    def target_cost_normalized(self) -> float:
        """Target cost for CFG inference (normalized, 0 = best)."""
        return 0.0  # Always target the best


def fit_policy_cfg(
    replay_buffer: ReplayBuffer,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
    cfg_drop_prob: float = 0.1,
    sigma_min: float = 1e-2,
    verbose: bool = False,
) -> Tuple[jax.Array, dict]:
    """Fit a cost-conditioned flow matching model with CFG.
    
    Args:
        replay_buffer: Buffer containing all historical data.
        model: CostConditionedMLP model.
        optimizer: Optimizer.
        batch_size: Batch size.
        num_epochs: Number of epochs.
        rng: Random key.
        cfg_drop_prob: Probability of dropping cost condition (for CFG).
        sigma_min: Flow matching sigma_min parameter.
        verbose: Print CFG validation info.
        
    Returns:
        Tuple of (final_loss, info_dict with CFG diagnostics).
    """
    num_data_points = replay_buffer.size
    num_batches = max(1, num_data_points // batch_size)
    
    # Validate model is cost-conditioned
    assert hasattr(model, 'null_embed'), "Model must be CostConditionedMLP for CFG training"
    assert hasattr(model, 'cost_embed'), "Model must have cost_embed for CFG training"

    def _loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        act: jax.Array,
        old_act: jax.Array,
        costs: jax.Array,
        noise: jax.Array,
        t: jax.Array,
        drop_mask: jax.Array,
    ) -> jax.Array:
        """Compute CFG flow-matching loss."""
        alpha = 1.0 - sigma_min
        noised_action = t[..., None] * act + (1 - alpha * t[..., None]) * noise
        target = act - alpha * noise
        
        # Vectorized: compute both conditional and unconditional, then select
        pred_cond = model(noised_action, obs, t, cost=costs)
        pred_uncond = model(noised_action, obs, t, cost=None)
        
        # Select based on drop_mask (True = unconditional)
        pred = jnp.where(drop_mask[..., None, None], pred_uncond, pred_cond)

        # Cosine similarity weighting (same as original)
        v1 = (old_act - act).reshape(act.shape[0], -1)
        v2 = (noise - act).reshape(noise.shape[0], -1)
        dot = jnp.sum(v1 * v2, axis=-1)
        norm1 = jnp.linalg.norm(v1, axis=-1)
        norm2 = jnp.linalg.norm(v2, axis=-1)
        cosine_similarity = dot / (norm1 * norm2 + 1e-8)
        weight = jax.lax.stop_gradient(jnp.exp(2 * (cosine_similarity - 1)))

        # MSE loss
        sq_error = jnp.mean(jnp.square(pred - target), axis=(-2, -1))
        return jnp.mean(weight * sq_error)

    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Perform a gradient descent step."""
        rng, batch_rng, noise_rng, t_rng, drop_rng = jax.random.split(rng, 5)
        
        # Sample batch from replay buffer
        batch_obs, batch_act, batch_old_act, batch_costs = replay_buffer.sample(
            batch_size, batch_rng
        )

        # Sample noise and time
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        
        # CFG: drop cost with probability cfg_drop_prob
        drop_mask = jax.random.uniform(drop_rng, (batch_size,)) < cfg_drop_prob
        num_dropped = jnp.sum(drop_mask)

        # Compute loss and gradient
        loss, grad = nnx.value_and_grad(_loss_fn)(
            model, batch_obs, batch_act, batch_old_act, batch_costs, noise, t, drop_mask
        )
        optimizer.update(grad)

        return rng, loss, num_dropped

    # Training loop
    @nnx.scan
    def _scan_fn(carry: Tuple, i: int) -> Tuple:
        model, optimizer, rng = carry
        rng, loss, num_dropped = _train_step(model, optimizer, rng)
        return (model, optimizer, rng), (loss, num_dropped)

    _, (losses, drops) = _scan_fn(
        (model, optimizer, rng), jnp.arange(num_batches * num_epochs)
    )
    
    # Compute CFG diagnostics
    total_steps = num_batches * num_epochs
    total_dropped = float(jnp.sum(drops))
    actual_drop_rate = total_dropped / (total_steps * batch_size)
    
    info = {
        "cfg_drop_prob_config": cfg_drop_prob,
        "cfg_actual_drop_rate": actual_drop_rate,
        "total_train_steps": total_steps,
        "replay_buffer_size": num_data_points,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
    }
    
    if verbose:
        print(f"    CFG Training: drop_rate={actual_drop_rate:.2%} (target={cfg_drop_prob:.2%})")
        print(f"    Loss: {float(losses[0]):.4f} -> {float(losses[-1]):.4f}")

    return losses[-1], info
