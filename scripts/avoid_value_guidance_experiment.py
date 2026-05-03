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
    noise_scale: float,
    noise_seed: int,
) -> HJValueArtifacts:
    """Add Gaussian noise to V and recompute grad(V)."""
    if noise_scale <= 0.0:
        return artifacts

    values_np = np.asarray(artifacts.values, dtype=np.float32)
    value_std = float(np.std(values_np)) + 1e-6
    rng = np.random.default_rng(int(noise_seed))
    noise = rng.standard_normal(values_np.shape, dtype=np.float32)
    noisy_values = values_np + float(noise_scale) * value_std * noise
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
        metadata={**artifacts.metadata, "value_noise_scale": float(noise_scale)},
    )


def make_hj_terminal_value(env, artifacts: HJValueArtifacts):
    def terminal_value(_params, data):
        pos = data.site_xpos[env.task.pointmass_id, 0:2]
        state = jnp.concatenate([pos, data.qvel[:2]])
        state = jnp.clip(state, artifacts.grid.domain.lo, artifacts.grid.domain.hi)
        value = artifacts.grid.interpolate(artifacts.values, state)
        return jnp.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e6)

    return terminal_value


def configure_base(config_path: Path, episodes: int, batch_size: int, seed: int) -> TrainingConfig:
    config = TrainingConfig.from_yaml(config_path)
    config.task_name = "avoid"
    config.method = "gpc"
    config.seed = int(seed)
    config.plan_horizon = 0.5
    config.num_knots = 4
    config.hj_eval_episodes = int(episodes)
    config.num_envs = int(batch_size)
    config.num_policy_samples = 0
    config.strategy = "best"
    config.use_wandb = False
    config.record_training_videos = False
    config.record_eval_videos = False
    config.proposal_overlay = False
    config.proposal_video_trace_points = 0
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
    if axes.ndim == 1:
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
) -> None:
    variant = str(metadata.get("task_variant", config.task_variant))
    lines = [
        f"# Avoid {variant} V-Noise Sensitivity (Exp1 + Exp4)",
        "",
        "## Setup",
        "",
        f"- Short horizon: `plan_horizon={config.plan_horizon}`, `num_knots={config.num_knots}`.",
        f"- Episodes per case: `{config.hj_eval_episodes}`.",
        "- Rows are increasing noise injected into the HJ value table.",
        "- Columns are `spc_short`, `spc_hjV`, `spc_hjV_grad`.",
        "",
        "## Mean Episode Cost Matrix",
        "",
        "| noise scale | spc_short | spc_hjV | spc_hjV_grad | hjV-short | grad-short | grad-hjV |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for noise in noise_levels:
        row = results_by_noise[noise]
        short = row["spc_short"]["episode_cost_mean"]
        hjv = row["spc_hjV"]["episode_cost_mean"]
        grad = row["spc_hjV_grad"]["episode_cost_mean"]
        lines.append(
            f"| {noise:.4f} | {short:.4f} | {hjv:.4f} | {grad:.4f} | {hjv - short:.4f} | {grad - short:.4f} | {grad - hjv:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `noise_episode_cost_matrix.png`: mean-cost heatmap (rows=noise, cols=method).",
            "- `noise_trajectory_matrix.png`: 5x3 trajectory panel (rows=noise, cols=method).",
        ]
    )
    (out_dir / "noise_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_noise_levels(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("noise-level list must be non-empty")
    values = [max(0.0, v) for v in values]
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
    parser.add_argument("--value-alpha", type=float, default=1.0)
    parser.add_argument("--grad-step", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--noise-analysis", action="store_true")
    parser.add_argument("--noise-levels", type=str, default="0.0,0.05,0.1,0.2,0.35")
    parser.add_argument("--noise-seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configure_base(args.config, args.episodes, args.batch_size, args.seed)
    env = create_environment(config)
    artifacts = load_hj_artifacts(args.hj_dir, config, env)

    case_names = ["spc_short", "spc_hjV", "spc_hjV_grad"]
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
        comparisons = {
            "spc_hjV - spc_short": paired_delta_stats(costs["spc_hjV"], costs["spc_short"]),
            "spc_hjV_grad - spc_short": paired_delta_stats(costs["spc_hjV_grad"], costs["spc_short"]),
            "spc_hjV_grad - spc_hjV": paired_delta_stats(costs["spc_hjV_grad"], costs["spc_hjV"]),
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
        for noise in noise_levels:
            row: dict[str, dict[str, Any]] = {"spc_short": short_result}
            results_by_noise[noise] = row
            qpos_by_noise_case[(noise, "spc_short")] = short_qpos

            noisy_artifacts = perturb_hj_values(
                artifacts=artifacts,
                config=config,
                noise_scale=float(noise),
                noise_seed=int(args.noise_seed + int(round(1000 * noise))),
            )

            for case_name in ["spc_hjV", "spc_hjV_grad"]:
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
        )

    print(f"Saved results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
