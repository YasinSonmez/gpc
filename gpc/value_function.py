import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Optional, Any

from gpc.architectures import MLP

class ValueNetwork(nnx.Module):
    """A dual Value Network for IQL-based cost minimization."""

    def __init__(
        self,
        observation_size: int,
        hidden_layers: list[int],
        rngs: nnx.Rngs,
    ):
        layer_sizes = [observation_size] + hidden_layers + [1]
        self.v1 = MLP(layer_sizes, rngs=rngs)
        self.v2 = MLP(layer_sizes, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Return the value of the observation from both networks."""
        v1 = self.v1(obs).squeeze(-1)
        v2 = self.v2(obs).squeeze(-1)
        return v1, v2

    def get_value(self, obs: jax.Array, aggregate: str = "max") -> jax.Array:
        """Return a single value estimate using max (pessimistic for costs)."""
        v1, v2 = self(obs)
        if aggregate == "max":
            return jnp.maximum(v1, v2)
        elif aggregate == "min":
            return jnp.minimum(v1, v2)
        else:
            return (v1 + v2) / 2.0

def iql_expectile_loss(
    diff: jax.Array,
    tau: float,
) -> jax.Array:
    """Compute the expectile loss for IQL.
    
    For costs: tau = 0.1 pulls V towards the lower tail of Q.
    """
    weight = jnp.where(diff < 0, 1.0 - tau, tau)
    return jnp.mean(weight * jnp.square(diff))

def fit_value_function(
    model: ValueNetwork,
    target_model: ValueNetwork,
    optimizer: nnx.Optimizer,
    batch: dict,
    discount: float,
    tau: float,
    n_step: int,
) -> jax.Array:
    """Perform one training step for the Value function.
    
    Args:
        batch: Dict containing 'obs', 'target_q' (accumulated cost), 
               'last_obs', 'last_done'.
    """
    
    def loss_fn(model: ValueNetwork):
        # Current predictions
        v1, v2 = model(batch["obs"])
        
        # Target calculation (n-step TD)
        # Q_target = Sum(gamma^k * c_k) + gamma^n * V_target(s_n)
        v_next = target_model.get_value(batch["last_obs"], aggregate="max")
        # Mask out value if last state was terminal
        v_next = v_next * (1.0 - batch["last_done"])
        
        q_target = batch["target_q"] + (discount ** n_step) * v_next
        q_target = jax.lax.stop_gradient(q_target)
        
        loss1 = iql_expectile_loss(q_target - v1, tau)
        loss2 = iql_expectile_loss(q_target - v2, tau)
        
        return loss1 + loss2

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss_fn(model)

class ValueFunctionTrainer(nnx.Module):
    """Helper to manage value network training and target updates."""
    
    def __init__(
        self,
        observation_size: int,
        hidden_layers: list[int],
        learning_rate: float,
        polyak_tau: float,
        rngs: nnx.Rngs,
    ):
        self.model = ValueNetwork(observation_size, hidden_layers, rngs)
        self.target_model = ValueNetwork(observation_size, hidden_layers, rngs)
        
        # Initialize target model with same weights
        self.target_model.v1.l0.kernel = self.model.v1.l0.kernel
        # ... actually easier to just copy state
        nnx.update(self.target_model, nnx.state(self.model))
        
        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate), wrt=nnx.Param)
        self.polyak_tau = polyak_tau
        
    def update_targets(self):
        """Update target networks via Polyak averaging."""
        state = nnx.state(self.model)
        target_state = nnx.state(self.target_model)
        
        new_target_state = jax.tree_util.tree_map(
            lambda m, t: (1 - self.polyak_tau) * t + self.polyak_tau * m,
            state,
            target_state
        )
        nnx.update(self.target_model, new_target_state)

    @nnx.jit
    def train_step(self, batch, discount, tau, n_step):
        return fit_value_function(
            self.model, self.target_model, self.optimizer, 
            batch, discount, tau, n_step
        )

    def fit(self, data, batch_size, num_epochs, rng, discount, tau, n_step):
        """Fit the value function to a large batch of data using nnx.scan for speed."""
        num_data_points = data['obs'].shape[0]
        num_batches = max(1, num_data_points // batch_size)
        total_steps = num_batches * num_epochs

        def _train_step(model, target_model, optimizer, rng):
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, num_data_points
            )
            
            batch = {k: v[batch_idx] for k, v in data.items()}
            
            loss = fit_value_function(
                model, target_model, optimizer, 
                batch, discount, tau, n_step
            )
            
            # Polyak update targets
            state = nnx.state(model)
            target_state = nnx.state(target_model)
            new_target_state = jax.tree_util.tree_map(
                lambda m, t: (1 - self.polyak_tau) * t + self.polyak_tau * m,
                state,
                target_state
            )
            nnx.update(target_model, new_target_state)
            
            return rng, loss

        @nnx.scan
        def _scan_fn(carry, i):
            model, target_model, optimizer, rng = carry
            rng, loss = _train_step(model, target_model, optimizer, rng)
            return (model, target_model, optimizer, rng), loss

        # Initial state for scan
        carry_in = (self.model, self.target_model, self.optimizer, rng)
        (self.model, self.target_model, self.optimizer, _), losses = _scan_fn(
            carry_in, jnp.arange(total_steps)
        )
        
        return losses[-1]

# Alias for compatibility with other parts of the codebase
ValueTrainer = ValueFunctionTrainer
