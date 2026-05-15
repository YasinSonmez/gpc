from typing import Any, Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.algs.cem import CEM, CEMParams
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mujoco import mjx

from gpc.policy import Policy


class UniformRandomShooting(SamplingBasedController):
    """Random-shooting MPC with knots sampled uniformly from action bounds."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.num_samples = int(num_samples)

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> SamplingParams:
        return super().init_params(initial_knots, seed)

    def sample_knots(
        self, params: SamplingParams
    ) -> Tuple[jax.Array, SamplingParams]:
        rng, sample_rng = jax.random.split(params.rng)
        knots = jax.random.uniform(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        return knots, params.replace(rng=rng)

    def update_params(
        self, params: SamplingParams, rollouts: Trajectory
    ) -> SamplingParams:
        costs = jnp.sum(rollouts.costs, axis=1)
        best_idx = jnp.argmin(costs)
        return params.replace(mean=rollouts.knots[best_idx])


class CEMNoWarmStart(CEM):
    """CEM variant that disables temporal warm-start across MPC steps.

    Standard Hydrax CEM advances previous optimal knots to initialize the next
    control step. This variant instead resets mean/cov every control step.
    """

    def optimize(
        self,
        state: mjx.Data,
        params: CEMParams,
    ) -> Tuple[CEMParams, Trajectory]:
        """Perform one CEM optimization step without knot warm-start."""
        new_tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        params = params.replace(
            tk=new_tk,
            mean=jnp.zeros_like(params.mean),
            cov=jnp.full_like(params.cov, self.sigma_start),
        )

        def _optimize_scan_body(scan_params: CEMParams, _: jax.Array):
            knots, scan_params = self.sample_knots(scan_params)
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

            rng, dr_rng = jax.random.split(scan_params.rng)
            rollouts = self.rollout_with_randomizations(
                state,
                new_tk,
                knots,
                dr_rng,
            )
            scan_params = scan_params.replace(rng=rng)
            scan_params = self.update_params(scan_params, rollouts)
            return scan_params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
        )
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final


class BootstrappedPredictiveSampling(PredictiveSampling):
    """Perform predictive sampling, but add samples from a generative policy."""

    def __init__(
        self,
        policy: Policy,
        observation_fn: Callable[[mjx.Data], jax.Array],
        num_policy_samples: int,
        warm_start_level: float = 0.0,
        inference_timestep: float = 0.1,
        **kwargs,
    ):
        """Initialize the controller.

        Args:
            policy: The generative policy to sample from.
            observation_fn: A function that produces an observation vector.
            num_policy_samples: The number of samples to take from the policy.
            warm_start_level: The warm start level in [0, 1] to use for the
                policy samples. 0.0 generates samples from scratch, while 1.0
                seed all samples from the previous action sequence.
            inference_timestep: The timestep dt for flow matching inference.
            **kwargs: Constructor arguments for PredictiveSampling.
        """
        self.observation_fn = observation_fn
        self.policy = policy.replace(dt=inference_timestep)
        self.policy.model.eval()  # Don't update batch statistics
        self.warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        self.num_policy_samples = num_policy_samples

        super().__init__(**kwargs)

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        In addition to sampling random control sequences, also sample control
        sequences from the generative policy.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        rng, policy_rng, dr_rng = jax.random.split(params.rng, 3)

        # Sample random control sequences
        controls, params = self.sample_controls(params)
        controls = jnp.clip(controls, self.task.u_min, self.task.u_max)

        # Update sensor readings and get an observation
        state = mjx.forward(self.task.model, state)
        y = self.observation_fn(state)

        # Sample from the generative policy, which is conditioned on the latest
        # observation.
        target_cost = getattr(params, 'target_cost', None)
        cfg_scale = getattr(params, 'cfg_scale', 1.0)
        
        policy_rngs = jax.random.split(policy_rng, self.num_policy_samples)
        policy_controls = jax.vmap(
            self.policy.apply, in_axes=(None, None, 0, None, None, None)
        )(
            params.mean,
            y,
            policy_rngs,
            self.warm_start_level,
            target_cost,
            cfg_scale,
        )

        # Combine the random and policy samples
        controls = jnp.concatenate([controls, policy_controls], axis=0)

        # Roll out the control sequences, applying domain randomizations and
        # combining costs using self.risk_strategy.
        rollouts = self.rollout_with_randomizations(state, controls, dr_rng)

        # Update the policy parameters based on the combined costs
        params = params.replace(rng=rng)
        params = self.update_params(params, rollouts)
        return params, rollouts
