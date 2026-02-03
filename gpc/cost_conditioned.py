"""Cost-Conditioned Flow Matching with Classifier-Free Guidance (CFG).

This module implements return/cost-conditioned flow matching to improve sample
efficiency by learning from all data while guiding toward low-cost behaviors.
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from gpc.architectures import MLP, PositionalEmbedding
from gpc.replay_buffer import ReplayBuffer


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
        
        # Fixed null embedding for unconditional (CFG dropout)
        # Using -1.0 as sentinel: clearly distinct from cost_embed outputs
        # (cost=0 is our target, so zeros would be ambiguous)
        # nnx.Variable (not Param) = stored but not trained
        self.null_embed = nnx.Variable(jnp.full((cost_embed_dim,), -1.0))

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
        drop_cost_mask: Optional[jax.Array] = None,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Forward pass.
        
        Args:
            u: Noisy action sequence, shape (..., horizon, action_size).
            y: Observation, shape (..., observation_size).
            t: Diffusion time, shape (..., 1).
            cost: Normalized cost, shape (...,) or (..., 1). If None, uses null.
            drop_cost_mask: Boolean mask, True means drop cost (use null). 
                           Shape (...,) or (..., 1).
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
            
            # Apply dropout mask if provided
            if drop_cost_mask is not None:
                # Ensure mask handles broadcasting
                if drop_cost_mask.ndim < cost_emb.ndim:
                     # Add feature dimension to mask: (batch,) -> (batch, 1) to match (batch, dim)
                    drop_cost_mask = drop_cost_mask[..., None]
                
                null_emb = jnp.broadcast_to(self.null_embed.value, cost_emb.shape)
                cost_emb = jnp.where(drop_cost_mask, null_emb, cost_emb)

        # Broadcast cost_emb to match batch dimensions if necessary
        # u_flat has shape (batches..., dim)
        # cost_emb has shape (batches_from_cost..., dim)
        if len(batches) > 0 and cost_emb.ndim == 1:
             # Broadcast (dim,) -> (batches..., dim)
             cost_emb = jnp.broadcast_to(cost_emb, batches + (self.cost_embed_dim,))
        elif cost_emb.shape[:-1] != batches:
             # Try to broadcast to match batches
             try:
                 cost_emb = jnp.broadcast_to(cost_emb, batches + (self.cost_embed_dim,))
             except ValueError:
                 # If shapes are incompatible, let concatenate raise the error
                 pass
        
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




def fit_policy_cfg(
    replay_buffer: ReplayBuffer,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
    normalizer: Optional[nnx.BatchNorm] = None,
    update_normalizer: bool = True,
    cfg_drop_prob: float = 0.1,
    cost_weight_temperature: float = 1.0,
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
        normalizer: Optional normalizer (BatchNorm) to apply to observations.
        update_normalizer: Whether to update normalizer stats during training.
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
        # Normalize observations if normalizer provided
        if normalizer is not None:
            # use_running_average=False means update running statistics (if training)
            # use_running_average=True means use stored statistics (if eval/expert)
            obs = normalizer(obs, use_running_average=not update_normalizer)

        # Debugging: show normalized range
        # Note: We can't easily use 'i' here because it's in train_step not loss_fn. 
        # But we can just use a debug callback if normalizer stats look weird.
        # Let's check the normalizer stats in train_config instead.
        
        alpha = 1.0 - sigma_min
        noised_action = t[..., None] * act + (1 - alpha * t[..., None]) * noise
        target = act - alpha * noise
        
        # Compute prediction with random cost dropping
        # drop_mask: True means use null embedding (unconditional)
        pred = model(noised_action, obs, t, cost=costs, drop_cost_mask=drop_mask)

        # Cosine similarity weighting (same as original)
        v1 = (old_act - act).reshape(act.shape[0], -1)
        v2 = (noise - act).reshape(noise.shape[0], -1)
        dot = jnp.sum(v1 * v2, axis=-1)
        norm1 = jnp.linalg.norm(v1, axis=-1)
        norm2 = jnp.linalg.norm(v2, axis=-1)
        cosine_similarity = dot / (norm1 * norm2 + 1e-8)
        flow_weight = jax.lax.stop_gradient(jnp.exp(2 * (cosine_similarity - 1)))
        
        # Cost-based weighting: prioritize low-cost samples
        # costs are normalized to [0, 1] (roughly)
        # If temperature <= 0, disable cost weighting (pure CFG)
        if cost_weight_temperature > 0:
            cost_weight = jnp.exp(-costs / cost_weight_temperature)
        else:
            cost_weight = jnp.ones_like(costs)
        cost_weight = jax.lax.stop_gradient(cost_weight)
        
        # Combine weights
        # Note: drop_mask doesn't affect weighting; we want to learn good unconditional flows too
        total_weight = flow_weight * cost_weight

        # MSE loss
        sq_error = jnp.mean(jnp.square(pred - target), axis=(-2, -1))
        return jnp.mean(total_weight * sq_error)

    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Perform a gradient descent step."""
        rng, batch_rng, noise_rng, t_rng, drop_rng = jax.random.split(rng, 5)
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        batch_obs = batch['obs']
        batch_act = batch['actions']
        batch_old_act = batch['old_actions']
        batch_costs = batch['costs']

        # Debug: Print stats intermittently
        def _print_stats(obs, act, costs):
             print(f"      [Step {i}] Obs range: [{jnp.min(obs):.2f}, {jnp.max(obs):.2f}] | Act range: [{jnp.min(act):.2f}, {jnp.max(act):.2f}] | Costs range: [{jnp.min(costs):.2f}, {jnp.max(costs):.2f}]")
             return None
        
        # Only print occasionally to avoid slowing down training too much
        # But for warm-start we want more visibility.
        if verbose:
             jax.lax.cond(
                 i % 100 == 0,
                 lambda _: jax.debug.callback(_print_stats, batch_obs, batch_act, batch_costs),
                 lambda _: None,
                 operand=None
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
        
        # If normalizer is provided and we want to update it, we need to pass its gradient too?
        # NO: nnx.BatchNorm typically updates its state variables (running_mean) via side effects.
        # But for that to work in a JITted grad pass, we must include its state in the diff.
        # However, for simply imitating an expert (warmstart), we might prefer to just update it.
        optimizer.update(model, grad)

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


def fit_policy_cost_conditioned(
    observations: jax.Array,
    action_sequences: jax.Array,
    old_action_sequences: jax.Array,
    costs: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
    normalizer: Optional[nnx.BatchNorm] = None,
    update_normalizer: bool = True,
    sigma_min: float = 1e-2,
) -> jax.Array:
    """Fit a cost-conditioned flow matching model (without CFG).
    
    Args:
        observations: Observations y.
        action_sequences: Target action sequences U.
        old_action_sequences: Previous action sequences U_guess.
        costs: Normalized costs for each sample.
        model: CostConditionedMLP model.
        optimizer: Optimizer.
        batch_size: Batch size.
        num_epochs: Number of epochs.
        rng: Random key.
        sigma_min: Flow matching sigma_min parameter.
        
    Returns:
        Final loss.
    """
    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)
    
    assert hasattr(model, 'cost_embed'), "Model must be CostConditionedMLP"
    
    def _loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        act: jax.Array,
        old_act: jax.Array,
        costs: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        # Normalize observations
        if normalizer is not None:
             obs = normalizer(obs, use_running_average=not update_normalizer)

        alpha = 1.0 - sigma_min
        noised_action = t[..., None] * act + (1 - alpha * t[..., None]) * noise
        target = act - alpha * noise
        pred = model(noised_action, obs, t, cost=costs)
        # Per-sample cosine-similarity weighting (match CFG implementation)
        v1 = (old_act - act).reshape(act.shape[0], -1)
        v2 = (noise - act).reshape(noise.shape[0], -1)
        dot = jnp.sum(v1 * v2, axis=-1)
        norm1 = jnp.linalg.norm(v1, axis=-1)
        norm2 = jnp.linalg.norm(v2, axis=-1)
        cosine_similarity = dot / (norm1 * norm2 + 1e-8)
        weight = jax.lax.stop_gradient(jnp.exp(2 * (cosine_similarity - 1)))

        # Per-sample squared error, then weight and average
        sq_error = jnp.mean(jnp.square(pred - target), axis=(-2, -1))
        return jnp.mean(weight * sq_error)
    
    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(
            batch_rng, (batch_size,), 0, num_data_points
        )
        batch_obs = observations[batch_idx]
        batch_act = action_sequences[batch_idx]
        batch_old_act = old_action_sequences[batch_idx]
        batch_costs = costs[batch_idx]
        
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        
        loss, grad = nnx.value_and_grad(_loss_fn)(
            model, batch_obs, batch_act, batch_old_act, batch_costs, noise, t
        )
        optimizer.update(model, grad)
        
        return rng, loss
    
    @nnx.scan
    def _scan_fn(carry: Tuple, i: int) -> Tuple:
        model, optimizer, rng = carry
        rng, loss = _train_step(model, optimizer, rng)
        return (model, optimizer, rng), loss
    
    _, losses = _scan_fn(
        (model, optimizer, rng), jnp.arange(num_batches * num_epochs)
    )
    
    return losses[-1]
