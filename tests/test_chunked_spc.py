import jax
import jax.numpy as jnp
import pytest
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.chunked_spc import ChunkedPAC, boltzmann_resample_indices
from gpc.config import TrainingConfig


def test_boltzmann_low_temperature_selects_best() -> None:
    """Near-zero temperature concentrates on the lowest-cost particle."""
    costs = jnp.array([0.0, 10.0, 20.0])
    indices = boltzmann_resample_indices(
        jax.random.key(0), costs, temperature=1e-6
    )
    assert jnp.all(indices == 0)


def test_boltzmann_rejects_nonpositive_temperature() -> None:
    """Boltzmann resampling is undefined for nonpositive temperature."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        boltzmann_resample_indices(
            jax.random.key(0), jnp.array([0.0, 1.0]), temperature=0.0
        )


def test_chunked_predictive_sampling_shapes_and_update() -> None:
    """Chunked SPC returns a normal Hydrax trajectory batch."""
    task = Particle()
    base_ctrl = PredictiveSampling(
        task,
        num_samples=6,
        noise_level=0.2,
        plan_horizon=0.2,
        num_knots=4,
        iterations=1,
    )
    ctrl = ChunkedPAC(
        base_ctrl,
        num_policy_samples=0,
        chunk_size=2,
        temperature=1.0,
    )

    state = mjx.make_data(task.model)
    state = state.replace(
        mocap_pos=state.mocap_pos.at[0, 0:2].set(jnp.array([0.01, 0.01]))
    )
    params = ctrl.init_params()

    params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert rollouts.controls.shape == (
        base_ctrl.num_samples,
        ctrl.ctrl_steps,
        task.model.nu,
    )
    assert rollouts.knots.shape == (
        base_ctrl.num_samples,
        ctrl.num_knots,
        task.model.nu,
    )
    assert rollouts.costs.shape == (base_ctrl.num_samples, ctrl.ctrl_steps + 1)
    assert rollouts.trace_sites.shape[0] == base_ctrl.num_samples
    assert jnp.all(jnp.isfinite(rollouts.costs))

    total_costs = jnp.sum(rollouts.costs, axis=1)
    best_idx = jnp.argmin(total_costs)
    assert jnp.allclose(params.mean, rollouts.knots[best_idx])


def test_chunked_config_validation() -> None:
    """Invalid chunked-SPC configs fail before controller construction."""
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        TrainingConfig(chunked_spc=True, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_temperature must be positive"):
        TrainingConfig(chunked_spc=True, chunk_temperature=0.0)
