"""Chunked SPC with Boltzmann particle resampling."""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from hydrax.alg_base import Trajectory
from mujoco import mjx

from gpc.augmented import PACParams, PolicyAugmentedController


def boltzmann_resample_indices(
    rng: jax.Array, costs: jax.Array, temperature: float
) -> jax.Array:
    """Sample particle indices with probability proportional to exp(-J / T).

    Args:
        rng: JAX random key.
        costs: One scalar cost per particle, shape ``(num_particles,)``.
        temperature: Positive Boltzmann temperature. Smaller values concentrate
            mass on lower-cost particles; larger values approach uniform
            resampling.

    Returns:
        Resampled particle indices, shape ``(num_particles,)``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    centered = costs - jnp.min(costs)
    logits = -centered / temperature
    return jax.random.categorical(rng, logits, shape=costs.shape)


class ChunkedPAC(PolicyAugmentedController):
    """Policy-augmented SPC with chunked Boltzmann particle extension.

    The controller keeps the public ``PolicyAugmentedController`` interface, but
    replaces each full-horizon rollout batch by a particle system:

    1. Sample one proposal segment for every particle and roll it out.
    2. At each later chunk, resample parent endpoints with Boltzmann weights
       based on cumulative intermediate cost.
    3. Sample one fresh extension segment for each resampled endpoint.
    4. Resample after extension so the next chunk starts from the posterior
       particle cloud over low-cost prefixes.

    A complete optimization iteration still evaluates ``S`` particles over the
    full horizon, where ``S = num_policy_samples + base_ctrl.num_samples``.
    Thus, for a fixed horizon and iteration count, the simulator-step budget is
    matched to ordinary SPC; the difference is where selection pressure is
    applied.
    """

    def __init__(
        self,
        base_ctrl: Any,
        num_policy_samples: int,
        chunk_size: int,
        temperature: float,
        exploration_floor: float = 0.0,
    ) -> None:
        """Initialize the chunked controller.

        Args:
            base_ctrl: Hydrax sampling controller with ``sample_knots`` and
                ``update_params``.
            num_policy_samples: Number of learned-policy proposal knots.
            chunk_size: Desired chunk size in knots. The implementation converts
                this to an equal number of time chunks over the control horizon.
            temperature: Positive Boltzmann resampling temperature.
            exploration_floor: Fraction of base samples replaced by wide random
                proposals, forwarded to ``PolicyAugmentedController``.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        super().__init__(
            base_ctrl,
            num_policy_samples,
            exploration_floor=exploration_floor,
        )
        self.chunk_size = int(chunk_size)
        self.temperature = float(temperature)
        self.num_particles = (
            self.num_policy_samples + self.base_ctrl.num_samples
        )

        self.num_chunks = max(
            1,
            (self.num_knots + self.chunk_size - 1) // self.chunk_size,
        )
        self._control_slices = _make_slices(self.ctrl_steps, self.num_chunks)
        self._knot_slices = _make_slices(self.num_knots, self.num_chunks)

    def optimize(
        self, state: mjx.Data, params: PACParams
    ) -> Tuple[PACParams, Trajectory]:
        """Run chunked Boltzmann SPC and update controller parameters."""
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(
            new_tk, params.tk, params.mean[None, ...]
        )[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(
            scan_params: PACParams, _: jax.Array
        ) -> Tuple[PACParams, Trajectory]:
            rng, rollout_rng = jax.random.split(scan_params.rng)
            scan_params = scan_params.replace(rng=rng)

            rollouts, scan_params = self._chunked_rollout(
                state, new_tk, scan_params, rollout_rng
            )
            scan_params = self.update_params(scan_params, rollouts)
            return scan_params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
        )
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final

    def _chunked_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        params: PACParams,
        rng: jax.Array,
    ) -> Tuple[Trajectory, PACParams]:
        """Generate one full-horizon particle rollout batch."""
        rng, domain_rng = jax.random.split(rng)
        particle_states = self._initial_particle_states(
            state, self.num_particles, domain_rng
        )

        controls = jnp.zeros(
            (self.num_particles, self.ctrl_steps, self.task.model.nu)
        )
        knots = jnp.zeros(
            (self.num_particles, self.num_knots, self.task.model.nu)
        )
        domain_costs = jnp.zeros(
            (self.num_randomizations, self.num_particles, self.ctrl_steps + 1)
        )
        trace_shape = self.task.get_trace_sites(state).shape
        trace_sites = jnp.zeros(
            (self.num_particles, self.ctrl_steps + 1, *trace_shape)
        )
        cumulative_domain_costs = jnp.zeros(
            (self.num_randomizations, self.num_particles)
        )

        for chunk_idx, ((c0, c1), (k0, k1)) in enumerate(
            zip(self._control_slices, self._knot_slices, strict=True)
        ):
            if chunk_idx > 0:
                rng, parent_rng = jax.random.split(rng)
                parent_idx = self._resample_indices(
                    parent_rng, cumulative_domain_costs
                )
                particle_states = _take_particles(particle_states, parent_idx)
                controls = controls[parent_idx]
                knots = knots[parent_idx]
                domain_costs = domain_costs[:, parent_idx]
                trace_sites = trace_sites[parent_idx]
                cumulative_domain_costs = cumulative_domain_costs[:, parent_idx]

            sampled_knots, params = self.sample_knots(params)
            sampled_knots = jnp.clip(
                sampled_knots, self.task.u_min, self.task.u_max
            )
            tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
            sampled_controls = self.interp_func(tq, tk, sampled_knots)
            segment_controls = sampled_controls[:, c0:c1]

            particle_states, segment_costs, segment_traces = self._eval_segment(
                particle_states, segment_controls
            )
            cumulative_domain_costs = cumulative_domain_costs + jnp.sum(
                segment_costs, axis=-1
            )

            controls = controls.at[:, c0:c1].set(segment_controls)
            knots = knots.at[:, k0:k1].set(sampled_knots[:, k0:k1])
            domain_costs = domain_costs.at[:, :, c0:c1].set(segment_costs)
            trace_sites = trace_sites.at[:, c0:c1].set(segment_traces[0])

            rng, child_rng = jax.random.split(rng)
            child_idx = self._resample_indices(
                child_rng, cumulative_domain_costs
            )
            particle_states = _take_particles(particle_states, child_idx)
            controls = controls[child_idx]
            knots = knots[child_idx]
            domain_costs = domain_costs[:, child_idx]
            trace_sites = trace_sites[child_idx]
            cumulative_domain_costs = cumulative_domain_costs[:, child_idx]

        terminal_costs = jax.vmap(jax.vmap(self.task.terminal_cost))(
            particle_states
        )
        terminal_traces = jax.vmap(self.task.get_trace_sites)(
            particle_states[0]
        )
        domain_costs = domain_costs.at[:, :, -1].set(terminal_costs)
        trace_sites = trace_sites.at[:, -1].set(terminal_traces)

        costs = self.risk_strategy.combine_costs(domain_costs)
        return (
            Trajectory(
                controls=controls,
                knots=knots,
                costs=costs,
                trace_sites=trace_sites,
            ),
            params,
        )

    def _initial_particle_states(
        self, state: mjx.Data, num_particles: int, rng: jax.Array
    ) -> mjx.Data:
        """Replicate initial states over domains and particles."""
        domain_states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                domain_states, subrngs
            )
            domain_states = domain_states.tree_replace(randomizations)

        return jax.vmap(
            lambda domain_state: jax.vmap(lambda _: domain_state)(
                jnp.arange(num_particles)
            )
        )(domain_states)

    def _eval_segment(
        self, particle_states: mjx.Data, controls: jax.Array
    ) -> Tuple[mjx.Data, jax.Array, jax.Array]:
        """Roll out one chunk from each particle endpoint."""

        def _eval_domain(model: mjx.Model, states: mjx.Data):
            return jax.vmap(
                lambda state, control: self._eval_one_particle_segment(
                    model, state, control
                )
            )(states, controls)

        if self.num_randomizations > 1:
            return jax.vmap(
                _eval_domain, in_axes=(self.randomized_axes, 0)
            )(self.model, particle_states)

        final_states, costs, traces = _eval_domain(
            self.model, particle_states[0]
        )
        return (
            jax.tree.map(lambda x: x[None], final_states),
            costs[None],
            traces[None],
        )

    def _eval_one_particle_segment(
        self, model: mjx.Model, state: mjx.Data, controls: jax.Array
    ) -> Tuple[mjx.Data, jax.Array, jax.Array]:
        """Roll out one particle over one contiguous control chunk."""

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[jax.Array, jax.Array]]:
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (cost, sites)

        final_state, (costs, traces) = jax.lax.scan(_scan_fn, state, controls)
        return final_state, costs, traces

    def _resample_indices(
        self, rng: jax.Array, cumulative_domain_costs: jax.Array
    ) -> jax.Array:
        """Resample particles using risk-aggregated cumulative cost."""
        cumulative_costs = self.risk_strategy.combine_costs(
            cumulative_domain_costs[:, :, None]
        )[:, 0]
        return boltzmann_resample_indices(
            rng, cumulative_costs, self.temperature
        )


def _make_slices(length: int, num_chunks: int) -> tuple[tuple[int, int], ...]:
    """Create non-empty contiguous slices covering ``range(length)``."""
    boundaries = _round_linspace(0, length, num_chunks + 1)
    slices = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop > start:
            slices.append((start, stop))
    return tuple(slices)


def _round_linspace(start: int, stop: int, num: int) -> tuple[int, ...]:
    """Integer linspace with monotone rounded boundaries."""
    if num == 1:
        return (start,)
    step = (stop - start) / (num - 1)
    return tuple(round(start + step * idx) for idx in range(num))


def _take_particles(pytree: Any, indices: jax.Array) -> Any:
    """Index the particle axis of a domain-by-particle pytree."""
    return jax.tree.map(lambda leaf: leaf[:, indices], pytree)
