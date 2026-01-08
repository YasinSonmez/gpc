from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory


@dataclass
class PACParams:
    """Parameters for the policy-augmented controller.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        base_params: The parameters for the base controller.
        policy_samples: Control knots sampled from the policy.
        rng: Random number generator key for domain randomization.
    """

    tk: jax.Array
    mean: jax.Array
    base_params: Any
    policy_samples: jax.Array
    rng: jax.Array


class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        # Pass through all parameters from base controller
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
            plan_horizon=base_ctrl.plan_horizon,
            spline_type=base_ctrl.spline_type,
            num_knots=base_ctrl.num_knots,
            iterations=base_ctrl.iterations,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        base_rng, our_rng = jax.random.split(base_params.rng)
        base_params = base_params.replace(rng=base_rng)
        # Policy samples are now knots, not full control sequences
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.num_knots,
                self.task.model.nu,
            )
        )
        return PACParams(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params,
            policy_samples=policy_samples,
            rng=our_rng,
        )

    def sample_knots(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control knots from the base controller and the policy."""
        # Samples from the base controller
        base_samples, base_params = self.base_ctrl.sample_knots(
            params.base_params
        )

        # Include samples from the policy. Assumes that these have already been
        # generated and stored in params.policy_samples.
        samples = jnp.append(base_samples, params.policy_samples, axis=0)

        return samples, params.replace(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params
        )

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update the policy parameters according to the base controller."""
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        return params.replace(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params
        )

    def get_action(self, params: PACParams, t: float) -> jax.Array:
        """Get the action from the base controller at a given time."""
        return self.base_ctrl.get_action(params.base_params, t)

    def get_action_sequence(self, params: PACParams) -> jax.Array:
        """Get the full control trajectory by interpolating knots."""
        # Get the full interpolated control trajectory from base controller
        # Use the interp_func to convert knots to control trajectory
        tq = jnp.arange(self.ctrl_steps) * self.dt  # Query times
        tk = params.tk  # Knot times
        knots = params.mean  # Control knots
        
        # Interpolate each action dimension
        controls = self.interp_func(tq, tk, knots.T).T
        
        return controls
