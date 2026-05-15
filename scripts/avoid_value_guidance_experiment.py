"""Short-horizon avoid ablation with HJ terminal value and grad(V) guidance.

This script keeps only three cases:
1) `spc_short`: short-horizon SPC baseline.
2) `spc_hjV`: SPC with HJ terminal value.
3) `spc_hjV_grad`: SPC with HJ terminal value + adjoint-free grad(V) shift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hj_reachability as hj
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib import patches

from gpc.augmented import PACParams, PolicyAugmentedController
from gpc.config import TrainingConfig
from gpc.hj_solver import HJPolicyModel, HJTablePolicy, obstacle_signed_distance
from gpc.training import simulate_episode
from run_experiment import create_controller, create_environment


@dataclass
class HJValueArtifacts:
    grid: hj.Grid
    values: jax.Array
    grad_values: jax.Array
    policy: HJTablePolicy
    metadata: dict[str, Any]


class HJGradientGuidedController(PolicyAugmentedController):
    """PAC wrapper with an adjoint-free HJ value-gradient proposal shift."""

    def __init__(
        self,
        *args,
        hj_grid: hj.Grid,
        hj_grad_values: jax.Array,
        grad_step: float,
        grad_clip_norm: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hj_grid = hj_grid
        self.hj_grad_values = hj_grad_values
        self.grad_step = float(grad_step)
        self.grad_clip_norm = float(grad_clip_norm)

    def update_params(self, params: PACParams, rollouts) -> PACParams:
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        if self.grad_step <= 0.0:
            return params.replace(
                tk=base_params.tk,
                mean=base_params.mean,
                base_params=base_params,
                prev_elites=params.prev_elites,
            )

        # Rollout trace sites provide pointmass position; estimate terminal velocity
        # by finite-difference over the final two states.
        pos = rollouts.trace_sites[:, :, 0, :2]
        pos_h = pos[:, -1]
        vel_h = (pos[:, -1] - pos[:, -2]) / self.task.dt
        x_h = jnp.concatenate([pos_h, vel_h], axis=-1)
        x_bar = jnp.mean(x_h, axis=0)
        x_bar = jnp.clip(x_bar, self.hj_grid.domain.lo, self.hj_grid.domain.hi)

        p = self.hj_grid.interpolate(self.hj_grad_values, x_bar)
        p = jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        directional_value_change = (x_h - x_bar) @ p

        sample_mean = params.base_params.mean
        delta_u = rollouts.knots - sample_mean
        cov = getattr(params.base_params, "cov", None)
        if cov is None:
            inv_var = 1.0
        else:
            inv_var = 1.0 / (jnp.square(cov) + 1e-4)

        grad_u = jnp.mean(delta_u * directional_value_change[:, None, None], axis=0) * inv_var
        grad_u = jnp.nan_to_num(grad_u, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = jnp.linalg.norm(grad_u)
        grad_u = grad_u * jnp.minimum(1.0, self.grad_clip_norm / (grad_norm + 1e-6))

        guided_mean = jnp.clip(
            base_params.mean - self.grad_step * grad_u,
            self.task.u_min,
            self.task.u_max,
        )
        base_params = base_params.replace(mean=guided_mean)
        return params.replace(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params,
            prev_elites=params.prev_elites,
        )


def load_hj_artifacts(hj_dir: Path, config: TrainingConfig, env) -> HJValueArtifacts:
    with open(hj_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    coords = np.load(hj_dir / "grid_coordinates.npz")
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=jnp.asarray(coords["domain_lo"], dtype=jnp.float32),
            hi=jnp.asarray(coords["domain_hi"], dtype=jnp.float32),
        ),
        (
            len(coords["axis_0"]),
            len(coords["axis_1"]),
            len(coords["axis_2"]),
            len(coords["axis_3"]),
        ),
    )
    values = jnp.asarray(np.load(hj_dir / "value_function.npy"), dtype=jnp.float32)
    solver_settings = hj.SolverSettings.with_accuracy(config.hj_solver_accuracy)
    grad_values = grid.grad_values(values, upwind_scheme=solver_settings.upwind_scheme)

    if (hj_dir / "policy.npy").exists():
        policy_table = jnp.asarray(np.load(hj_dir / "policy.npy"), dtype=jnp.float32)
    else:
        policy_table = jnp.zeros((*grid.shape, env.task.model.nu), dtype=jnp.float32)
    policy = HJTablePolicy(
        model=HJPolicyModel(horizon=config.num_knots),
        u_min=env.task.u_min,
        u_max=env.task.u_max,
        grid=grid,
        policy_table=policy_table,
        goal_pos=jnp.asarray(metadata["goal_pos"], dtype=jnp.float32),
    )
    return HJValueArtifacts(
        grid=grid,
        values=values,
        grad_values=grad_values,
        policy=policy,
        metadata=metadata,
    )


def perturb_hj_values(
    artifacts: HJValueArtifacts,
    config: TrainingConfig,
    noise_sigma: float,
    noise_seed: int,
    noise_mean: float,
    smooth_passes: int,
    noise_distribution: str,
) -> HJValueArtifacts:
    """Apply configured noise model to V and recompute grad(V)."""
    if noise_mean < 0.0:
        raise ValueError("noise_mean must be nonnegative")
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma must be nonnegative")
    if noise_distribution not in {
        "softplus_gaussian",
        "lognormal",
        "clipped_gaussian",
        "blend_correlated",
    }:
        raise ValueError(
            "noise_distribution must be 'softplus_gaussian', 'lognormal', 'clipped_gaussian', or 'blend_correlated'"
        )

    values_np = np.asarray(artifacts.values, dtype=np.float32)
    rng = np.random.default_rng(int(noise_seed))
    eps = np.finfo(np.float32).tiny

    if noise_distribution == "blend_correlated":
        if noise_sigma > 1.0:
            raise ValueError("for blend_correlated, noise_sigma must be in [0, 1]")
        noisy_values = blend_to_correlated_noise(
            values=values_np,
            noise_scale=float(noise_sigma),
            noise_seed=int(noise_seed),
            smooth_passes=int(smooth_passes),
        )
        delta = noisy_values - values_np
        noise_stat_field = delta
        noise_model = "Vhat = (1 - lambda) V + lambda N, lambda in [0,1], N smoothed Gaussian"
    else:
        if noise_sigma == 0.0:
            z = np.full(values_np.shape, float(noise_mean), dtype=np.float32)
        else:
            z0 = rng.standard_normal(values_np.shape, dtype=np.float32)
            z0 = smooth_noise_field(z0, smooth_passes=smooth_passes)
            z0 = (z0 - np.mean(z0)) / (np.std(z0) + 1e-8)
            if noise_distribution == "softplus_gaussian":
                z = positive_softplus_gaussian_field(
                    z0,
                    mean=float(noise_mean),
                    small_noise_std=float(noise_sigma),
                )
            elif noise_distribution == "lognormal":
                z = positive_correlated_field_from_normal(
                    z0,
                    mean=float(noise_mean),
                    std=float(noise_sigma),
                )
            else:
                z = float(noise_mean) + float(noise_sigma) * z0
                z = np.maximum(z, eps).astype(np.float32)

        noisy_values = values_np + z * values_np
        delta = noisy_values - values_np
        noise_stat_field = z
        noise_model = "Vhat = V + z(x) V, z(x) > 0"

    noisy_values_jax = jnp.asarray(noisy_values, dtype=jnp.float32)

    solver_settings = hj.SolverSettings.with_accuracy(config.hj_solver_accuracy)
    noisy_grad_values = artifacts.grid.grad_values(
        noisy_values_jax, upwind_scheme=solver_settings.upwind_scheme
    )
    return HJValueArtifacts(
        grid=artifacts.grid,
        values=noisy_values_jax,
        grad_values=noisy_grad_values,
        policy=artifacts.policy,
        metadata={
            **artifacts.metadata,
            "value_noise_model": noise_model,
            "value_noise_distribution": noise_distribution,
            "value_noise_mean": float(noise_mean),
            "value_noise_sigma": float(noise_sigma),
            "value_noise_smooth_passes": int(smooth_passes),
            "value_noise_stat_name": "delta" if noise_distribution == "blend_correlated" else "z",
            "value_noise_min": float(np.min(noise_stat_field)),
            "value_noise_max": float(np.max(noise_stat_field)),
            "value_noise_empirical_mean": float(np.mean(noise_stat_field)),
            "value_noise_empirical_std": float(np.std(noise_stat_field)),
            "value_noise_q01": float(np.percentile(noise_stat_field, 1.0)),
            "value_noise_q50": float(np.percentile(noise_stat_field, 50.0)),
            "value_noise_q99": float(np.percentile(noise_stat_field, 99.0)),
            "value_delta_abs_mean": float(np.mean(np.abs(delta))),
            "value_delta_abs_max": float(np.max(np.abs(delta))),
            "value_delta_rel_l2": float(
                np.linalg.norm(delta.reshape(-1)) / (np.linalg.norm(values_np.reshape(-1)) + 1e-12)
            ),
            "value_correlation": float(np.corrcoef(values_np.reshape(-1), noisy_values.reshape(-1))[0, 1]),
        },
    )


def smooth_noise_field(noise: np.ndarray, smooth_passes: int) -> np.ndarray:
    """Create a spatially correlated grid field using nearest-neighbor smoothing."""
    out = np.asarray(noise, dtype=np.float32)
    for _ in range(max(0, int(smooth_passes))):
        padded = np.pad(out, [(1, 1)] * out.ndim, mode="edge")
        center = tuple(slice(1, -1) for _ in range(out.ndim))
        acc = padded[center].copy()
        count = 1.0
        for axis in range(out.ndim):
            lower = list(center)
            upper = list(center)
            lower[axis] = slice(0, -2)
            upper[axis] = slice(2, None)
            acc = acc + padded[tuple(lower)] + padded[tuple(upper)]
            count += 2.0
        out = acc / count
    return out.astype(np.float32)


def blend_to_correlated_noise(
    values: np.ndarray,
    noise_scale: float,
    noise_seed: int,
    smooth_passes: int = 4,
) -> np.ndarray:
    """Interpolate from data to spatially correlated noise.

    Formula:
        V_hat(lambda) = (1 - lambda) * V + lambda * N,
    where lambda in [0, 1], and N is a smoothed Gaussian field rescaled to
    match mean/std of V.

    This creates a clean path from original data (lambda=0) to complete
    correlated noise (lambda=1) without heavy-tailed spikes.
    """
    if not (0.0 <= noise_scale <= 1.0):
        raise ValueError("noise_scale must be in [0, 1]")

    values_np = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(int(noise_seed))

    z = rng.standard_normal(values_np.shape, dtype=np.float32)
    z = smooth_noise_field(z, smooth_passes=smooth_passes)
    z = (z - np.mean(z)) / (np.std(z) + 1e-8)

    v_mean = float(np.mean(values_np))
    v_std = float(np.std(values_np))
    if v_std < 1e-8:
        noise_field = np.full_like(values_np, v_mean)
    else:
        noise_field = v_mean + v_std * z

    lam = float(noise_scale)
    v_hat = (1.0 - lam) * values_np + lam * noise_field
    return v_hat.astype(np.float32)


def _obstacle_mask_xy(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    variant: str,
    obstacle_pos: np.ndarray,
) -> np.ndarray:
    """Return boolean mask where True indicates obstacle interior."""
    x_vec = np.asarray(x_vec, dtype=np.float32)
    y_vec = np.asarray(y_vec, dtype=np.float32)
    X, Y = np.meshgrid(x_vec, y_vec, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    obs = jnp.asarray(obstacle_pos, dtype=jnp.float32)
    sdf = jax.vmap(lambda p: obstacle_signed_distance(jnp.asarray(p), obs, variant))(
        jnp.asarray(pts, dtype=jnp.float32)
    )
    inside = np.asarray(sdf < 0.0).reshape(Y.shape)
    return inside


def _hj_style_limits(masked: np.ndarray) -> tuple[float | None, float | None]:
    vals = masked[np.isfinite(masked)]
    if vals.size == 0:
        return None, None
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if vmax <= vmin:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    return vmin, vmax


def plot_blend_noise_sweep_hj_style(
    values: np.ndarray,
    noise_scales: list[float],
    noise_seed: int,
    smooth_passes: int,
    out_path: Path,
    variant: str,
    obstacle_pos: np.ndarray,
    x_vec: np.ndarray | None = None,
    y_vec: np.ndarray | None = None,
    ix_vx: int | None = None,
    ix_vy: int | None = None,
) -> None:
    """Plot a data->noise sweep using the HJ-style obstacle-masked scale.

    All panels share one color range computed from the base free-space values
    (2-98 percentiles), matching the matrix-plot style used elsewhere.
    """
    if len(noise_scales) == 0:
        raise ValueError("noise_scales must be non-empty")

    values_np = np.asarray(values, dtype=np.float32)
    if values_np.ndim not in (2, 4):
        raise ValueError("values must be 2D or 4D")

    if values_np.ndim == 4:
        if ix_vx is None:
            ix_vx = values_np.shape[2] // 2
        if ix_vy is None:
            ix_vy = values_np.shape[3] // 2
        base_slice = np.asarray(values_np[:, :, ix_vx, ix_vy]).T
    else:
        base_slice = np.asarray(values_np).T

    if x_vec is None:
        x_vec = np.arange(base_slice.shape[1], dtype=np.float32)
    if y_vec is None:
        y_vec = np.arange(base_slice.shape[0], dtype=np.float32)
    x_vec = np.asarray(x_vec, dtype=np.float32)
    y_vec = np.asarray(y_vec, dtype=np.float32)

    mask = _obstacle_mask_xy(
        x_vec=x_vec,
        y_vec=y_vec,
        variant=str(variant),
        obstacle_pos=np.asarray(obstacle_pos, dtype=np.float32),
    )

    base_masked = np.array(base_slice, copy=True)
    base_masked[mask] = np.nan
    vmin, vmax = _hj_style_limits(base_masked)
    if vmin is None or vmax is None:
        raise ValueError("could not compute HJ-style limits from free-space cells")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.96))
    extent = [x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]]

    fig, axes = plt.subplots(
        1,
        len(noise_scales),
        figsize=(4.2 * len(noise_scales), 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(noise_scales) == 1:
        axes = [axes]

    im_ref = None
    for ax, scale in zip(axes, noise_scales):
        v_hat = blend_to_correlated_noise(
            values=values_np,
            noise_scale=float(scale),
            noise_seed=int(noise_seed),
            smooth_passes=int(smooth_passes),
        )
        if values_np.ndim == 4:
            sl = np.asarray(v_hat[:, :, ix_vx, ix_vy]).T
        else:
            sl = np.asarray(v_hat).T
        sl = np.array(sl, copy=True)
        sl[mask] = np.nan

        im = ax.imshow(
            sl,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        if im_ref is None:
            im_ref = im
        ax.set_title(f"noise_scale={float(scale):.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    fig.suptitle(
        "Blend from data to correlated noise (HJ-style shared free-space scale)",
        y=1.02,
    )
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("V_hat")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def positive_correlated_field_from_normal(
    field: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Map a normalized field to positive values with empirical mean/std.

    We use a normalized lognormal family ``exp(a * field)``. For each ``a``,
    the field is rescaled to the requested empirical mean, and ``a`` is found by
    bisection so the empirical standard deviation matches ``std``. This keeps
    every grid value strictly positive without the zero-mass artifact produced
    by clipping a Gaussian field.
    """
    if mean <= 0.0:
        raise ValueError("positive lognormal noise requires noise_mean > 0")
    if std == 0.0:
        return np.full(field.shape, mean, dtype=np.float32)

    x = np.asarray(field, dtype=np.float64)

    def _make(scale: float) -> np.ndarray:
        y = np.exp(np.clip(scale * x, -60.0, 60.0))
        y *= mean / (np.mean(y) + 1e-300)
        return y

    lo = 0.0
    hi = 1.0
    while np.std(_make(hi)) < std and hi < 64.0:
        hi *= 2.0

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if np.std(_make(mid)) < std:
            lo = mid
        else:
            hi = mid

    return _make(hi).astype(np.float32)


def positive_softplus_gaussian_field(
    field: np.ndarray,
    mean: float,
    small_noise_std: float,
) -> np.ndarray:
    """Positive Gaussian-like field matching N(mean, std) to first order.

    Let ``a = softplus^{-1}(mean)``. We set

        z = softplus(a + (std / sigmoid(a)) * field).

    Since d softplus(a) / da = sigmoid(a), small ``std`` gives
    ``z approx mean + std * field`` while preserving strict positivity for all
    scales. Unlike clipping, this introduces no atom at zero; unlike a
    fixed-mean high-variance lognormal, large scales do not make most cells
    collapse to nearly zero.
    """
    if mean <= 0.0:
        raise ValueError("softplus Gaussian noise requires noise_mean > 0")
    if small_noise_std == 0.0:
        return np.full(field.shape, mean, dtype=np.float32)

    x = np.asarray(field, dtype=np.float64)
    a = np.log(np.expm1(mean))
    slope = 1.0 / (1.0 + np.exp(-a))
    latent = a + (small_noise_std / (slope + 1e-12)) * x
    z = np.logaddexp(0.0, latent)
    return np.maximum(z, np.finfo(np.float32).tiny).astype(np.float32)


def make_hj_terminal_value(env, artifacts: HJValueArtifacts):
    def terminal_value(_params, data):
        pos = data.site_xpos[env.task.pointmass_id, 0:2]
        state = jnp.concatenate([pos, data.qvel[:2]])
        state = jnp.clip(state, artifacts.grid.domain.lo, artifacts.grid.domain.hi)
        value = artifacts.grid.interpolate(artifacts.values, state)
        return jnp.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e6)

    return terminal_value


def configure_base(
    config_path: Path,
    episodes: int,
    batch_size: int,
    seed: int,
    controller_type: str | None,
    plan_horizon: float,
    num_knots: int,
    num_samples: int | None,
    iterations: int | None,
) -> TrainingConfig:
    config = TrainingConfig.from_yaml(config_path)
    config.task_name = "avoid"
    config.method = "gpc"
    config.seed = int(seed)
    if controller_type is not None:
        config.controller_type = str(controller_type)
    config.plan_horizon = float(plan_horizon)
    config.num_knots = int(num_knots)
    if num_samples is not None:
        config.num_samples = int(num_samples)
    if iterations is not None:
        config.iterations = int(iterations)
    config.hj_eval_episodes = int(episodes)
    config.num_envs = int(batch_size)
    config.num_policy_samples = 0
    config.strategy = "best"
    config.use_wandb = False
    config.record_training_videos = False
    config.record_eval_videos = False
    config.proposal_overlay = False
    config.proposal_video_trace_points = 1
    return config


def build_controller(
    config: TrainingConfig,
    env,
    artifacts: HJValueArtifacts,
    case_name: str,
    value_alpha: float,
    grad_step: float,
    grad_clip_norm: float,
):
    base = create_controller(env, config)
    if case_name == "spc_short":
        ctrl = base
        ctrl.use_task_terminal_cost = True
        return ctrl
    if case_name == "spc_hjV":
        ctrl = base
    elif case_name == "spc_hjV_grad":
        ctrl = HJGradientGuidedController(
            base.base_ctrl,
            config.num_policy_samples,
            exploration_floor=config.exploration_floor,
            hj_grid=artifacts.grid,
            hj_grad_values=artifacts.grad_values,
            grad_step=grad_step,
            grad_clip_norm=grad_clip_norm,
        )
    else:
        raise ValueError(f"Unknown case: {case_name}")

    ctrl.value_fn = make_hj_terminal_value(env, artifacts)
    ctrl.use_task_terminal_cost = False
    ctrl.value_alpha = jnp.asarray(value_alpha, dtype=jnp.float32)
    return ctrl


def trajectory_metrics(env, qpos: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    pointmass_body_id = env.task.mj_model.body("pointmass").id
    offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)
    pos = qpos[..., :2] + offset[None, None, :]
    goal = np.asarray(metadata["goal_pos"], dtype=np.float32)
    obstacle = np.asarray(metadata["obstacle_pos"], dtype=np.float32)
    variant = str(metadata.get("task_variant", "u_trap"))

    final_dist = np.linalg.norm(pos[:, -1] - goal[None, :], axis=-1)
    flat_pos = jnp.asarray(pos.reshape(-1, 2), dtype=jnp.float32)
    sdf = jax.vmap(lambda p: obstacle_signed_distance(p, jnp.asarray(obstacle), variant))(flat_pos)
    sdf = np.asarray(sdf).reshape(pos.shape[:2])
    min_sdf = np.min(sdf, axis=1)
    collision = min_sdf < 0.0
    success = (final_dist < 0.05) & (~collision)
    return {
        "final_distance_mean": float(np.mean(final_dist)),
        "final_distance_std": float(np.std(final_dist)),
        "min_sdf_mean": float(np.mean(min_sdf)),
        "collision_rate": float(np.mean(collision)),
        "success_rate": float(np.mean(success)),
    }


def evaluate_case(
    config: TrainingConfig,
    env,
    artifacts: HJValueArtifacts,
    case_name: str,
    value_alpha: float,
    grad_step: float,
    grad_clip_norm: float,
) -> tuple[dict[str, Any], np.ndarray]:
    ctrl = build_controller(
        config=config,
        env=env,
        artifacts=artifacts,
        case_name=case_name,
        value_alpha=value_alpha,
        grad_step=grad_step,
        grad_clip_norm=grad_clip_norm,
    )
    rng = jax.random.key(config.seed)
    num_total = int(config.hj_eval_episodes)
    num_batch = int(config.num_envs)
    num_batches = int(np.ceil(num_total / num_batch))

    @nnx.jit
    def jit_simulate_batch(rng_in: jax.Array):
        rngs = jax.random.split(rng_in, config.num_envs)
        return jax.vmap(
            simulate_episode,
            in_axes=(None, None, None, None, 0, None, None, None, None, None, None, None),
        )(
            env,
            ctrl,
            artifacts.policy,
            0.0,
            rngs,
            config.strategy,
            None,
            1.0,
            None,
            value_alpha,
            jnp.array(1.0),
            config.proposal_video_trace_points,
        )

    episode_costs: list[np.ndarray] = []
    proposal_costs: list[np.ndarray] = []
    qpos_chunks: list[np.ndarray] = []
    t0 = time.time()
    for _ in range(num_batches):
        rng, sim_rng = jax.random.split(rng)
        out = jit_simulate_batch(sim_rng)
        _, _, _, j_spc, _, j_inst, _, _, _, qpos, _, _, _ = out
        episode_costs.append(np.asarray(jnp.sum(j_inst, axis=1)))
        proposal_costs.append(np.asarray(j_spc))
        qpos_chunks.append(np.asarray(qpos))
    wall_seconds = time.time() - t0

    ep = np.concatenate(episode_costs, axis=0)[:num_total]
    prop = np.concatenate(proposal_costs, axis=0)[:num_total]
    qpos = np.concatenate(qpos_chunks, axis=0)[:num_total]
    metrics = trajectory_metrics(env, qpos, artifacts.metadata)
    result = {
        "case_name": case_name,
        "episode_costs": ep.tolist(),
        "proposal_costs": prop.tolist(),
        "episode_cost_mean": float(np.mean(ep)),
        "episode_cost_std": float(np.std(ep)),
        "episode_cost_median": float(np.median(ep)),
        "episode_cost_min": float(np.min(ep)),
        "episode_cost_max": float(np.max(ep)),
        "proposal_cost_mean": float(np.mean(prop)),
        "proposal_cost_std": float(np.std(prop)),
        "wall_seconds": float(wall_seconds),
        **metrics,
    }
    return result, qpos


def paired_delta_stats(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, float]:
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(delta[idx])))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def add_obstacle_patch(ax, obstacle: np.ndarray, variant: str) -> None:
    ox, oy = float(obstacle[0]), float(obstacle[1])
    if variant == "sphere":
        ax.add_patch(
            patches.Circle((ox, oy), radius=0.1, facecolor="black", alpha=0.18, edgecolor="black")
        )
        return
    if variant == "vertical_block":
        ax.add_patch(
            patches.Rectangle((ox - 0.02, oy - 0.12), 0.04, 0.24, facecolor="black", alpha=0.18, edgecolor="black")
        )
        return
    specs = [
        ((ox + 0.04 - 0.02, oy - 0.10), 0.04, 0.20),
        ((ox - 0.01 - 0.05, oy + 0.10 - 0.02), 0.10, 0.04),
        ((ox - 0.01 - 0.05, oy - 0.10 - 0.02), 0.10, 0.04),
    ]
    for xy, w, h in specs:
        ax.add_patch(
            patches.Rectangle(xy, w, h, facecolor="black", alpha=0.18, edgecolor="black")
        )


def save_plots(
    out_dir: Path,
    results: list[dict[str, Any]],
    qpos_by_case: dict[str, np.ndarray],
    env,
    metadata: dict[str, Any],
) -> None:
    labels = [r["case_name"] for r in results]
    costs = [np.asarray(r["episode_costs"]) for r in results]
    variant = str(metadata.get("task_variant", "u_trap"))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.boxplot(costs, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Episode cost")
    ax.set_title(f"Avoid {variant}: short horizon + HJ guidance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_cost_boxplot.png", dpi=180)
    plt.close(fig)

    pointmass_body_id = env.task.mj_model.body("pointmass").id
    offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)
    goal = np.asarray(metadata["goal_pos"], dtype=np.float32)
    obstacle = np.asarray(metadata["obstacle_pos"], dtype=np.float32)

    fig, axes = plt.subplots(1, len(results), figsize=(4.8 * len(results), 4.5), sharex=True, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        qpos = qpos_by_case[result["case_name"]]
        pos = qpos[: min(16, len(qpos)), :, :2] + offset[None, None, :]
        for traj in pos:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.55, linewidth=1.2)
        add_obstacle_patch(ax, obstacle, variant)
        ax.plot(goal[0], goal[1], marker="*", color="gold", markeredgecolor="k", markersize=12)
        ax.set_title(result["case_name"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=180)
    plt.close(fig)


def save_noise_matrix_plots(
    out_dir: Path,
    case_names: list[str],
    noise_levels: list[float],
    results_by_noise: dict[float, dict[str, dict[str, Any]]],
    qpos_by_noise_case: dict[tuple[float, str], np.ndarray],
    env,
    metadata: dict[str, Any],
) -> None:
    variant = str(metadata.get("task_variant", "u_trap"))
    means = np.array(
        [
            [results_by_noise[n][c]["episode_cost_mean"] for c in case_names]
            for n in noise_levels
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(case_names)), labels=case_names)
    ax.set_yticks(np.arange(len(noise_levels)), labels=[f"{n:.3f}" for n in noise_levels])
    ax.set_xlabel("Case")
    ax.set_ylabel("Value-noise scale")
    ax.set_title(f"Avoid {variant}: mean episode cost by noise and method")
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean episode cost")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_episode_cost_matrix.png", dpi=200)
    plt.close(fig)

    successes = np.array(
        [
            [results_by_noise[n][c]["success_rate"] for c in case_names]
            for n in noise_levels
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    im = ax.imshow(successes, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(case_names)), labels=case_names)
    ax.set_yticks(np.arange(len(noise_levels)), labels=[f"{n:.4f}" for n in noise_levels])
    ax.set_xlabel("Case")
    ax.set_ylabel("Value-noise sigma")
    ax.set_title(f"Avoid {variant}: success rate by noise and method")
    for i in range(successes.shape[0]):
        for j in range(successes.shape[1]):
            ax.text(j, i, f"{successes[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Success rate")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_success_rate_matrix.png", dpi=200)
    plt.close(fig)

    pointmass_body_id = env.task.mj_model.body("pointmass").id
    offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)
    goal = np.asarray(metadata["goal_pos"], dtype=np.float32)
    obstacle = np.asarray(metadata["obstacle_pos"], dtype=np.float32)

    fig, axes = plt.subplots(
        len(noise_levels),
        len(case_names),
        figsize=(4.2 * len(case_names), 3.2 * len(noise_levels)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if len(noise_levels) == 1:
            axes = axes[None, :]
        else:
            axes = axes[:, None]

    for i, noise in enumerate(noise_levels):
        for j, case_name in enumerate(case_names):
            ax = axes[i, j]
            qpos = qpos_by_noise_case[(noise, case_name)]
            pos = qpos[: min(8, len(qpos)), :, :2] + offset[None, None, :]
            for traj in pos:
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.55, linewidth=1.0, color="#1f77b4")
            add_obstacle_patch(ax, obstacle, variant)
            ax.plot(goal[0], goal[1], marker="*", color="gold", markeredgecolor="k", markersize=10)
            if i == 0:
                ax.set_title(case_name)
            if j == 0:
                ax.set_ylabel(f"noise={noise:.3f}\ny")
            else:
                ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            m = results_by_noise[noise][case_name]["episode_cost_mean"]
            ax.text(
                0.02,
                0.98,
                f"mean={m:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if i == len(noise_levels) - 1:
                ax.set_xlabel("x")

    fig.suptitle(f"Avoid {variant}: trajectory matrix (rows=noise, cols=method)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_trajectory_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    config: TrainingConfig,
    metadata: dict[str, Any],
    hj_dir: Path,
    results: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> None:
    variant = str(metadata.get("task_variant", config.task_variant))
    grid_size = int(metadata.get("grid_size", -1))
    lines = [
        f"# Avoid {variant} HJ Value Guidance (Exp1 + Exp4)",
        "",
        "## Setup",
        "",
        f"- Config source: `{config.task_name}/{config.task_variant}` with short-horizon override.",
        f"- HJ artifact: `{hj_dir}`.",
        f"- HJ grid size (per dim): `{grid_size}`.",
        f"- Episodes: `{config.hj_eval_episodes}`.",
        f"- Parallel env batch: `{config.num_envs}`.",
        f"- Controller: `{config.controller_type}`, samples `{config.num_samples}`, iterations `{config.iterations}`, elites `{config.num_elites}`.",
        f"- Short horizon: `plan_horizon={config.plan_horizon}`, `num_knots={config.num_knots}`.",
        "",
        "## Results",
        "",
        "| case | mean episode cost | std | median | final dist | collision rate | success rate | wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| {name} | {mean:.4f} | {std:.4f} | {median:.4f} | {fd:.4f} | {col:.3f} | {succ:.3f} | {wall:.1f} |".format(
                name=r["case_name"],
                mean=r["episode_cost_mean"],
                std=r["episode_cost_std"],
                median=r["episode_cost_median"],
                fd=r["final_distance_mean"],
                col=r["collision_rate"],
                succ=r["success_rate"],
                wall=r["wall_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| comparison | mean delta | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['mean']:.4f} | [{c['ci95_low']:.4f}, {c['ci95_high']:.4f}] |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `spc_short` is the requested short-horizon baseline.",
            "- `spc_hjV` uses terminal shaping `alpha * V_HJ(x_H)`.",
            "- `spc_hjV_grad` adds an adjoint-free proposal shift driven by `grad V_HJ`.",
            "- Episode cost is always measured by the same rollout/cost pipeline.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_noise_report(
    out_dir: Path,
    config: TrainingConfig,
    metadata: dict[str, Any],
    noise_levels: list[float],
    case_names: list[str],
    results_by_noise: dict[float, dict[str, dict[str, Any]]],
    noise_stats_by_noise: dict[float, dict[str, float]],
    noise_mean: float,
    noise_seed: int,
    noise_seed_mode: str,
    noise_seed_by_noise: dict[float, int],
    smooth_passes: int,
    noise_distribution: str,
) -> None:
    variant = str(metadata.get("task_variant", config.task_variant))
    if noise_distribution == "blend_correlated":
        noise_model_desc = (
            "Vhat(x) = (1-lambda) V(x) + lambda N(x), with lambda=noise sigma in [0,1] "
            "and N a spatially correlated Gaussian field matched to mean/std(V)."
        )
    else:
        noise_model_desc = "Vhat(x) = V(x) + z(x)V(x), with positive spatially correlated z(x)."
    lines = [
        f"# Avoid {variant} V-Noise Sensitivity (Exp1 + Exp4)",
        "",
        "## Setup",
        "",
        f"- Short horizon: `plan_horizon={config.plan_horizon}`, `num_knots={config.num_knots}`.",
        f"- Controller: `{config.controller_type}`, samples `{config.num_samples}`, iterations `{config.iterations}`.",
        f"- Episodes per case: `{config.hj_eval_episodes}`.",
        f"- Value perturbation: `{noise_model_desc}`",
        f"- Noise distribution: `{noise_distribution}`.",
        f"- Noise field base mean parameter: `{noise_mean}`.",
        f"- Noise sigma parameter sweep: `{noise_levels}`.",
        f"- Noise seed base: `{noise_seed}`.",
        f"- Noise seed mode: `{noise_seed_mode}`.",
        "- The table below reports empirical perturbation statistics actually applied to the HJ grid.",
        f"- Correlation operator: `{smooth_passes}` nearest-neighbor smoothing passes on the 4D HJ grid.",
        "- Success: final distance `< 0.05` and no obstacle collision under the existing signed-distance check.",
        "",
        "## Mean Episode Cost Matrix",
        "",
    ]
    header = "| sigma | " + " | ".join(case_names) + " |"
    lines.append(header)
    lines.append("|---:" + "|---:" * len(case_names) + "|")
    for noise in noise_levels:
        row = results_by_noise[noise]
        vals = " | ".join(f"{row[c]['episode_cost_mean']:.4f}" for c in case_names)
        lines.append(f"| {noise:.4f} | {vals} |")
    lines.extend(
        [
            "",
            "## Success Rate Matrix",
            "",
        ]
    )
    lines.append(header)
    lines.append("|---:" + "|---:" * len(case_names) + "|")
    for noise in noise_levels:
        row = results_by_noise[noise]
        vals = " | ".join(f"{row[c]['success_rate']:.3f}" for c in case_names)
        lines.append(f"| {noise:.4f} | {vals} |")
    lines.extend(
        [
            "",
            "## Noise Field Checks",
            "",
            "| sigma | empirical mean | empirical std | min z | q50 z | q99 z | max z | rel L2 delta | corr(V,Vhat) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        [
            "",
            "## Noise Seed Mapping",
            "",
            "| sigma | noise seed |",
            "|---:|---:|",
        ]
    )
    for noise in noise_levels:
        lines.append(f"| {noise:.4f} | {int(noise_seed_by_noise[noise])} |")
    lines.extend(
        [
            "",
        ]
    )
    for noise in noise_levels:
        stats = noise_stats_by_noise[noise]
        lines.append(
            "| {sigma:.4f} | {mean:.6f} | {std:.6f} | {min_z:.6g} | {q50:.6f} | {q99:.6f} | {max_z:.6f} | {rel_l2:.6f} | {corr:.6f} |".format(
                sigma=noise,
                mean=stats["empirical_mean"],
                std=stats["empirical_std"],
                min_z=stats["min"],
                q50=stats["q50"],
                q99=stats["q99"],
                max_z=stats["max"],
                rel_l2=stats["value_delta_rel_l2"],
                corr=stats["value_correlation"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The perturbation is multiplicative and strictly positive, so it preserves the sign of the HJ value while changing the relative terminal-value scale across the grid.",
            "- Spatial smoothing makes nearby HJ states receive similar perturbations; this avoids measuring only high-frequency interpolation noise.",
            "- `spc_short` is independent of `Vhat`; any variation in value-guided cases is attributable to the terminal value and gradient table perturbation under matched episode seeds.",
            "",
            "## Figures",
            "",
            "![Mean episode cost](noise_episode_cost_matrix.png)",
            "",
            "![Success rate](noise_success_rate_matrix.png)",
            "",
            "![Trajectory matrix](noise_trajectory_matrix.png)",
        ]
    )
    (out_dir / "noise_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_noise_levels(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("noise-level list must be non-empty")
    values = [max(0.0, v) for v in values]
    return values


def resolve_noise_seed(
    base_seed: int,
    run_seed: int,
    sigma_index: int,
    mode: str,
) -> int:
    """Return deterministic noise RNG seed for one (run seed, sigma index)."""
    if mode == "fixed":
        return int(base_seed)
    if mode == "per_sigma":
        return int(base_seed + 7919 * (sigma_index + 1))
    if mode == "per_seed_sigma":
        return int(base_seed + 1000003 * run_seed + 7919 * (sigma_index + 1))
    raise ValueError(f"unknown noise seed mode: {mode}")


def parse_case_names(text: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    allowed = {"spc_short", "spc_hjV", "spc_hjV_grad"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown case(s): {unknown}; allowed: {sorted(allowed)}")
    if not values:
        raise ValueError("case list must be non-empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/avoid_u_trap.yaml"))
    parser.add_argument(
        "--hj-dir",
        type=Path,
        default=Path("experiments/avoid/hj_avoid_binary_u_trap_20260502_131354/hj"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/avoid_value_guidance_u_trap"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller-type",
        type=str,
        default=None,
        choices=["predictive_sampling", "uniform", "cem", "cem_no_warm_start", "evosax", "mppi"],
    )
    parser.add_argument("--plan-horizon", type=float, default=0.5)
    parser.add_argument("--num-knots", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--cases", type=str, default="spc_short,spc_hjV,spc_hjV_grad")
    parser.add_argument("--value-alpha", type=float, default=1.0)
    parser.add_argument("--grad-step", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--noise-analysis", action="store_true")
    parser.add_argument("--noise-levels", type=str, default="0.001,0.002,0.003,0.004")
    parser.add_argument("--noise-mean", type=float, default=0.1)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument(
        "--noise-seed-mode",
        type=str,
        default="per_seed_sigma",
        choices=["fixed", "per_sigma", "per_seed_sigma"],
        help=(
            "How to derive HJ-noise RNG seeds across the sigma sweep: "
            "fixed (same field every sigma), per_sigma (different per sigma), "
            "per_seed_sigma (different per run seed and sigma)."
        ),
    )
    parser.add_argument("--noise-smooth-passes", type=int, default=4)
    parser.add_argument(
        "--noise-distribution",
        type=str,
        default="softplus_gaussian",
        choices=["softplus_gaussian", "lognormal", "clipped_gaussian", "blend_correlated"],
    )
    args = parser.parse_args()

    out_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configure_base(
        args.config,
        args.episodes,
        args.batch_size,
        args.seed,
        args.controller_type,
        args.plan_horizon,
        args.num_knots,
        args.num_samples,
        args.iterations,
    )
    env = create_environment(config)
    artifacts = load_hj_artifacts(args.hj_dir, config, env)

    case_names = parse_case_names(args.cases)
    if not args.noise_analysis:
        results: list[dict[str, Any]] = []
        qpos_by_case: dict[str, np.ndarray] = {}
        for case_name in case_names:
            print(f"Running {case_name} ...", flush=True)
            result, qpos = evaluate_case(
                config=config,
                env=env,
                artifacts=artifacts,
                case_name=case_name,
                value_alpha=float(args.value_alpha),
                grad_step=float(args.grad_step),
                grad_clip_norm=float(args.grad_clip_norm),
            )
            results.append(result)
            qpos_by_case[case_name] = qpos
            np.savez_compressed(out_dir / f"{case_name}_qpos.npz", qpos=qpos)
            print(
                f"  mean={result['episode_cost_mean']:.4f}, std={result['episode_cost_std']:.4f}, "
                f"success={result['success_rate']:.3f}, collision={result['collision_rate']:.3f}",
                flush=True,
            )

        costs = {r["case_name"]: np.asarray(r["episode_costs"]) for r in results}
        comparison_pairs = [
            ("spc_hjV", "spc_short"),
            ("spc_hjV_grad", "spc_short"),
            ("spc_hjV_grad", "spc_hjV"),
        ]
        comparisons = {
            f"{a} - {b}": paired_delta_stats(costs[a], costs[b])
            for a, b in comparison_pairs
            if a in costs and b in costs
        }

        payload = {
            "mode": "single",
            "config": config.__dict__,
            "hj_dir": str(args.hj_dir),
            "cases": case_names,
            "results": results,
            "comparisons": comparisons,
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        save_plots(out_dir, results, qpos_by_case, env, artifacts.metadata)
        write_report(out_dir, config, artifacts.metadata, args.hj_dir, results, comparisons)
    else:
        noise_levels = parse_noise_levels(args.noise_levels)
        results_by_noise: dict[float, dict[str, dict[str, Any]]] = {}
        qpos_by_noise_case: dict[tuple[float, str], np.ndarray] = {}
        noise_stats_by_noise: dict[float, dict[str, float]] = {}
        noise_seed_by_noise: dict[float, int] = {}

        short_result = None
        short_qpos = None
        if "spc_short" in case_names:
            print("Running spc_short baseline once ...", flush=True)
            short_result, short_qpos = evaluate_case(
                config=config,
                env=env,
                artifacts=artifacts,
                case_name="spc_short",
                value_alpha=float(args.value_alpha),
                grad_step=float(args.grad_step),
                grad_clip_norm=float(args.grad_clip_norm),
            )
        for noise_idx, noise in enumerate(noise_levels):
            row: dict[str, dict[str, Any]] = {}
            results_by_noise[noise] = row
            if short_result is not None and short_qpos is not None:
                row["spc_short"] = short_result
                qpos_by_noise_case[(noise, "spc_short")] = short_qpos

            sigma_noise_seed = resolve_noise_seed(
                base_seed=int(args.noise_seed),
                run_seed=int(args.seed),
                sigma_index=int(noise_idx),
                mode=str(args.noise_seed_mode),
            )
            noise_seed_by_noise[noise] = int(sigma_noise_seed)

            noisy_artifacts = perturb_hj_values(
                artifacts=artifacts,
                config=config,
                noise_sigma=float(noise),
                noise_seed=int(sigma_noise_seed),
                noise_mean=float(args.noise_mean),
                smooth_passes=int(args.noise_smooth_passes),
                noise_distribution=str(args.noise_distribution),
            )
            noise_stats_by_noise[noise] = {
                "empirical_mean": float(noisy_artifacts.metadata["value_noise_empirical_mean"]),
                "empirical_std": float(noisy_artifacts.metadata["value_noise_empirical_std"]),
                "min": float(noisy_artifacts.metadata["value_noise_min"]),
                "max": float(noisy_artifacts.metadata["value_noise_max"]),
                "q01": float(noisy_artifacts.metadata["value_noise_q01"]),
                "q50": float(noisy_artifacts.metadata["value_noise_q50"]),
                "q99": float(noisy_artifacts.metadata["value_noise_q99"]),
                "value_delta_abs_mean": float(noisy_artifacts.metadata["value_delta_abs_mean"]),
                "value_delta_abs_max": float(noisy_artifacts.metadata["value_delta_abs_max"]),
                "value_delta_rel_l2": float(noisy_artifacts.metadata["value_delta_rel_l2"]),
                "value_correlation": float(noisy_artifacts.metadata["value_correlation"]),
            }

            for case_name in [c for c in case_names if c != "spc_short"]:
                print(f"Running noise={noise:.4f}, case={case_name} ...", flush=True)
                result, qpos = evaluate_case(
                    config=config,
                    env=env,
                    artifacts=noisy_artifacts,
                    case_name=case_name,
                    value_alpha=float(args.value_alpha),
                    grad_step=float(args.grad_step),
                    grad_clip_norm=float(args.grad_clip_norm),
                )
                row[case_name] = result
                qpos_by_noise_case[(noise, case_name)] = qpos
                np.savez_compressed(
                    out_dir / f"noise_{noise:.4f}_{case_name}_qpos.npz",
                    qpos=qpos,
                )
                print(
                    f"  mean={result['episode_cost_mean']:.4f}, std={result['episode_cost_std']:.4f}, "
                    f"success={result['success_rate']:.3f}, collision={result['collision_rate']:.3f}",
                    flush=True,
                )

        payload = {
            "mode": "noise_analysis",
            "config": config.__dict__,
            "hj_dir": str(args.hj_dir),
            "cases": case_names,
            "noise_levels": noise_levels,
            "noise_seed": int(args.noise_seed),
            "noise_seed_mode": str(args.noise_seed_mode),
            "noise_seed_by_noise": {
                f"{noise:.6f}": int(seed)
                for noise, seed in noise_seed_by_noise.items()
            },
            "noise_stats_by_noise": {
                f"{noise:.6f}": stats
                for noise, stats in noise_stats_by_noise.items()
            },
            "results_by_noise": {
                f"{noise:.6f}": {k: v for k, v in row.items()}
                for noise, row in results_by_noise.items()
            },
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        save_noise_matrix_plots(
            out_dir=out_dir,
            case_names=case_names,
            noise_levels=noise_levels,
            results_by_noise=results_by_noise,
            qpos_by_noise_case=qpos_by_noise_case,
            env=env,
            metadata=artifacts.metadata,
        )
        write_noise_report(
            out_dir=out_dir,
            config=config,
            metadata=artifacts.metadata,
            noise_levels=noise_levels,
            case_names=case_names,
            results_by_noise=results_by_noise,
            noise_stats_by_noise=noise_stats_by_noise,
            noise_mean=float(args.noise_mean),
            noise_seed=int(args.noise_seed),
            noise_seed_mode=str(args.noise_seed_mode),
            noise_seed_by_noise=noise_seed_by_noise,
            smooth_passes=int(args.noise_smooth_passes),
            noise_distribution=str(args.noise_distribution),
        )

    print(f"Saved results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
