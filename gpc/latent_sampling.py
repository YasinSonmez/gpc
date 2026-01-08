"""Latent space sampling for flow matching policies.

This module provides utilities for sampling-based optimization in the latent
noise space of flow matching policies. The key insight is that we can:
1. Sample noise z ~ N(0, I) 
2. Decode to actions U via the flow model
3. Optimize the noise distribution based on costs
4. Use the optimal noise to generate actions

This can be more efficient than direct action space optimization because
the flow model has learned meaningful structure.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from gpc.policy import Policy


def sample_from_latent_distribution(
    policy: Policy,
    observation: jax.Array,
    num_samples: int,
    noise_level: float,
    rng: jax.Array,
    mean_latent: jax.Array | None = None,
    warm_start_level: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    """Sample action sequences by sampling and decoding through flow model.
    
    Args:
        policy: Flow matching policy for decoding.
        observation: Current observation for conditioning.
        num_samples: Number of samples to generate.
        noise_level: Standard deviation for latent sampling.
        rng: Random key.
        mean_latent: Mean of latent distribution (if None, uses standard normal).
        warm_start_level: Warm start level for policy decode (0-1).
        
    Returns:
        action_samples: Action knots (num_samples, num_knots, nu).
        latent_samples: The latent noise used (num_samples, num_knots, nu).
    """
    # Get shape from model
    horizon = policy.model.horizon
    action_size = policy.model.action_size
    
    # Determine latent shape from policy
    # We'll sample standard normal and let policy do the integration
    if mean_latent is None or mean_latent.size == 0:
        # Start from standard normal
        rngs = jax.random.split(rng, num_samples)
        latent_samples = jax.vmap(
            lambda key: jax.random.normal(key, (horizon, action_size))
        )(rngs) * noise_level
    else:
        # Sample around provided mean
        rngs = jax.random.split(rng, num_samples)
        noise = jax.vmap(
            lambda key: jax.random.normal(key, mean_latent.shape)
        )(rngs) * noise_level
        latent_samples = mean_latent + noise
    
    # Decode latent samples through flow matching policy
    # Policy.apply expects: (U_init, observation, rng, warm_start_level)
    # For latent sampling, U_init is the latent noise
    action_samples = jax.vmap(
        lambda z, key: policy.apply(z, observation, key, warm_start_level)
    )(latent_samples, rngs)
    
    return action_samples, latent_samples


def update_latent_distribution(
    latent_samples: jax.Array,
    costs: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """Update latent distribution mean using weighted average based on costs.
    
    Args:
        latent_samples: Latent samples (num_samples, num_knots, nu).
        costs: Cost for each sample (num_samples,).
        temperature: Temperature for softmax weighting (lower = more peaked).
        
    Returns:
        Updated mean latent (num_knots, nu).
    """
    # Compute weights (lower cost = higher weight)
    min_cost = jnp.min(costs)
    weights = jnp.exp(-(costs - min_cost) / temperature)
    weights = weights / jnp.sum(weights)
    
    # Compute weighted mean in latent space
    mean_latent = jnp.sum(latent_samples * weights[:, None, None], axis=0)
    
    return mean_latent


def hybrid_sample_actions(
    policy: Policy,
    observation: jax.Array,
    num_policy_samples: int,
    num_latent_samples: int,
    noise_level: float,
    rng: jax.Array,
    mean_latent: jax.Array | None = None,
    warm_start_level: float = 0.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Generate samples using both direct policy and latent space sampling.
    
    This creates a hybrid approach:
    - num_policy_samples from direct policy.apply() calls
    - num_latent_samples from latent space optimization
    
    Args:
        policy: Flow matching policy.
        observation: Current observation.
        num_policy_samples: Number of direct policy samples.
        num_latent_samples: Number of latent space samples.
        noise_level: Noise level for latent sampling.
        rng: Random key.
        mean_latent: Mean of latent distribution for focused sampling.
        warm_start_level: Warm start level.
        
    Returns:
        all_samples: Combined action samples.
        policy_samples: Direct policy samples.
        latent_info: (latent_samples, decoded_actions) for tracking.
    """
    rng, policy_rng, latent_rng = jax.random.split(rng, 3)
    
    # Get shape from model
    horizon = policy.model.horizon
    action_size = policy.model.action_size
    
    # Direct policy samples (original approach)
    policy_rngs = jax.random.split(policy_rng, num_policy_samples)
    U_init = jnp.zeros((horizon, action_size))  # Start from zeros
    policy_samples = jax.vmap(
        lambda key: policy.apply(U_init, observation, key, warm_start_level)
    )(policy_rngs)
    
    # Latent space samples
    if num_latent_samples > 0:
        latent_actions, latent_samples = sample_from_latent_distribution(
            policy, observation, num_latent_samples, noise_level,
            latent_rng, mean_latent, warm_start_level
        )
        all_samples = jnp.concatenate([policy_samples, latent_actions], axis=0)
        latent_info = (latent_samples, latent_actions)
    else:
        all_samples = policy_samples
        latent_info = None
    
    return all_samples, policy_samples, latent_info

