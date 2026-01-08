from pathlib import Path
from typing import Tuple, Union

import cloudpickle
import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass


@dataclass
class Policy:
    """A pickle-able Generative Predictive Control policy.

    Generates action sequences using flow matching, conditioned on the latest
    observation, e.g., samples U = [u_0, u_1, ...] ~ p(U | y).

    Attributes:
        model: The flow matching network that generates the action sequence.
        normalizer: Observation normalization module.
        u_min: The minimum action values.
        u_max: The maximum action values.
        dt: The integration step size for flow matching.
    """

    model: nnx.Module
    normalizer: nnx.BatchNorm
    u_min: jax.Array
    u_max: jax.Array
    dt: float = 0.1

    def save(self, path: Union[Path, str]) -> None:
        """Save the policy to a file.

        Args:
            path: The path to save the policy to.
        """
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "Policy":
        """Load a policy from a file.

        Args:
            path: The path to load the policy from.

        Returns:
            The loaded policy instance
        """
        with open(path, "rb") as f:
            policy = cloudpickle.load(f)
        return policy

    def apply(
        self,
        prev: jax.Array,
        y: jax.Array,
        rng: jax.Array,
        warm_start_level: float = 0.0,
        target_cost: float | None = None,
        cfg_scale: float = 1.0,
    ) -> jax.Array:
        """Generate an action sequence conditioned on the observation.

        Args:
            prev: The previous action sequence.
            y: The current observation.
            rng: The random number generator key.
            warm_start_level: The degree of warm-starting to use, in [0, 1].
            target_cost: Target normalized cost for CFG (0=best). If None, no CFG.
            cfg_scale: Guidance scale for CFG (w > 1 increases guidance strength).

        A warm-start level of 0.0 means the action sequence is generated from
        scratch, with the seed for flow matching drawn from a random normal
        distribution. A warm-start level of 1.0 means the seed is the previous
        action sequence. Values in between interpolate between these two, with
        larger values giving smoother but less exploratory action sequences.

        Returns:
            The updated action sequence
        """
        # Normalize the observation, but don't update the stored statistics
        y = self.normalizer(y, use_running_average=True)

        # Set the initial sample
        warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        noise = jax.random.normal(rng, prev.shape)
        U = warm_start_level * prev + (1 - warm_start_level) * noise

        # Check if model supports cost conditioning (has cost parameter)
        use_cfg = target_cost is not None and cfg_scale != 1.0

        def _step(args: Tuple[jax.Array, float]) -> Tuple[jax.Array, float]:
            """Flow the sample U along the learned vector field."""
            U, t = args
            
            if use_cfg and hasattr(self.model, 'forward_conditional'):
                # CFG: v_final = v_uncond + w * (v_cond - v_uncond)
                v_uncond = self.model(U, y, t, cost=None)
                v_cond = self.model(U, y, t, cost=jnp.array(target_cost))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(U, y, t, use_running_average=True)
            
            U += self.dt * v
            U = jax.numpy.clip(U, -1, 1)
            return U, t + self.dt

        # While t < 1, U += dt * model(U, y, t)
        U, t = jax.lax.while_loop(
            lambda args: jnp.all(args[1] < 1.0),
            _step,
            (U, jnp.zeros(1)),
        )

        # Rescale actions from [-1, 1] to [u_min, u_max]
        mean = (self.u_max + self.u_min) / 2
        scale = (self.u_max - self.u_min) / 2
        U = U * scale + mean

        return U
