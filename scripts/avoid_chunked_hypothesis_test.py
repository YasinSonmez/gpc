#!/usr/bin/env python3
"""Rigorous hypothesis tests for chunked SPC on avoid_u_trap.

Tests three hypotheses from the underperformance investigation:

H2: Double-resampling is harmful.
H3: ESS-gated low-variance resampling improves diversity/performance.
H1: Boundary lookahead score improves chunk ranking in prefix-deceptive U-trap.

The script runs a fixed ablation set, computes paired deltas vs baseline,
bootstrap confidence intervals, and writes figures/report artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpc.config import TrainingConfig
from gpc.hj_solver import obstacle_signed_distance
from gpc.policy import Policy
from gpc.training import simulate_episode
from run_experiment import create_controller, create_environment, create_network


@dataclass(frozen=True)
class Variant:
    name: str
    chunked_spc: bool
    chunk_size: int
    tau: float
    resample_pre: bool = True
    resample_post: bool = True
    resample_post_last: bool = True
    resample_scheme: str = "multinomial"
    ess_threshold: float = 0.0
    lookahead_alpha: float = 0.0


@dataclass
class VariantResult:
    name: str
    aggregate: dict[str, Any]
    per_seed: list[dict[str, Any]]
    episode_costs: list[float]
    episode_success: list[float]


def parse_int_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def bootstrap_ci_mean(
    data: np.ndarray,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI for sample mean."""
    data = np.asarray(data, dtype=np.float64)
    mean = float(np.mean(data))
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty((n_boot,), dtype=np.float64)
    n = data.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(np.mean(data[idx]))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return mean, lo, hi


def build_base_config(args) -> TrainingConfig:
    cfg = TrainingConfig.from_yaml(args.config)
    cfg.task_name = "avoid"
    cfg.task_variant = "u_trap"
    cfg.method = "gpc"
    cfg.seed = int(args.seed)
    cfg.controller_type = str(args.controller_type)
    cfg.plan_horizon = float(args.horizon)
    cfg.num_knots = int(args.num_knots)
    cfg.num_samples = int(args.num_samples)
    cfg.iterations = int(args.spc_iterations)
    cfg.num_policy_samples = 0
    cfg.strategy = "best"
    cfg.use_wandb = False
    cfg.num_envs = int(args.eval_batch_size)
    cfg.record_training_videos = False
    cfg.record_eval_videos = False
    cfg.proposal_overlay = False
    cfg.proposal_video_trace_points = 1
    # Keep controller-specific defaults stable.
    cfg.num_elites = int(args.num_elites)
    cfg.sigma_start = float(args.sigma_start)
    cfg.sigma_min = float(args.sigma_min)
    cfg.noise_level = float(args.noise_level)
    return cfg


def build_eval_policy(env, cfg: TrainingConfig) -> Policy:
    net = create_network(env, cfg)
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    return Policy(net, normalizer, env.task.u_min, env.task.u_max)


def make_eval_runner(env, ctrl, policy: Policy, batch_size: int):
    @nnx.jit
    def run_batch(rng_in: jax.Array):
        rngs = jax.random.split(rng_in, batch_size)
        return jax.vmap(
            simulate_episode,
            in_axes=(None, None, None, None, 0, None, None, None, None, None, None, None),
        )(
            env,
            ctrl,
            policy,
            0.0,
            rngs,
            "best",
            None,
            1.0,
            None,
            0.0,
            jnp.array(1.0),
            1,
        )

    return run_batch


def trajectory_success_flags(env, qpos: np.ndarray, variant: str = "u_trap") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointmass_body_id = env.task.mj_model.body("pointmass").id
    obstacle_body_id = env.task.mj_model.body("obstacle").id
    offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)
    obstacle_pos = np.asarray(env.task.mj_model.body_pos[obstacle_body_id, :2], dtype=np.float32)
    goal_pos = np.array([0.25, 0.0], dtype=np.float32)

    pos = qpos[..., :2] + offset[None, None, :]
    final_dist = np.linalg.norm(pos[:, -1] - goal_pos[None, :], axis=-1)
    flat_pos = jnp.asarray(pos.reshape(-1, 2), dtype=jnp.float32)
    sdf = jax.vmap(lambda p: obstacle_signed_distance(p, jnp.asarray(obstacle_pos), variant))(flat_pos)
    sdf = np.asarray(sdf).reshape(pos.shape[:2])
    min_sdf = np.min(sdf, axis=1)
    collision = min_sdf < 0.0
    success = (final_dist < 0.05) & (~collision)
    return success.astype(np.float32), collision.astype(np.float32), final_dist.astype(np.float32)


def evaluate_variant(
    env,
    base_cfg: TrainingConfig,
    policy: Policy,
    variant: Variant,
    seeds: list[int],
    episodes_per_seed: int,
    eval_batch_size: int,
) -> VariantResult:
    cfg = copy.copy(base_cfg)
    cfg.chunked_spc = bool(variant.chunked_spc)
    cfg.chunk_size = int(variant.chunk_size)
    cfg.chunk_temperature = float(variant.tau)
    cfg.chunk_resample_pre = bool(variant.resample_pre)
    cfg.chunk_resample_post = bool(variant.resample_post)
    cfg.chunk_resample_post_last = bool(variant.resample_post_last)
    cfg.chunk_resample_scheme = str(variant.resample_scheme)
    cfg.chunk_ess_threshold = float(variant.ess_threshold)
    cfg.chunk_lookahead_alpha = float(variant.lookahead_alpha)
    cfg.num_envs = int(eval_batch_size)

    ctrl = create_controller(env, cfg)
    run_batch = make_eval_runner(env, ctrl, policy, eval_batch_size)
    n_batches = int(math.ceil(episodes_per_seed / eval_batch_size))

    per_seed: list[dict[str, Any]] = []
    all_costs: list[np.ndarray] = []
    all_success: list[np.ndarray] = []

    for seed in seeds:
        rng = jax.random.key(int(seed))
        costs_seed: list[np.ndarray] = []
        qpos_seed: list[np.ndarray] = []
        t0 = time.time()
        for _ in range(n_batches):
            rng, sim_rng = jax.random.split(rng)
            out = run_batch(sim_rng)
            _, _, _, _, _, j_inst, _, _, _, qpos, _, _, _ = out
            costs_seed.append(np.asarray(jnp.sum(j_inst, axis=1)))
            qpos_seed.append(np.asarray(qpos))
        wall_s = time.time() - t0

        ep_costs = np.concatenate(costs_seed, axis=0)[:episodes_per_seed]
        qpos = np.concatenate(qpos_seed, axis=0)[:episodes_per_seed]
        success, collision, final_dist = trajectory_success_flags(env, qpos)

        all_costs.append(ep_costs)
        all_success.append(success)

        per_seed.append(
            {
                "seed": int(seed),
                "episode_cost_mean": float(np.mean(ep_costs)),
                "episode_cost_std": float(np.std(ep_costs)),
                "success_rate": float(np.mean(success)),
                "collision_rate": float(np.mean(collision)),
                "final_distance_mean": float(np.mean(final_dist)),
                "wall_seconds": float(wall_s),
            }
        )

    ep_costs_all = np.concatenate(all_costs, axis=0)
    ep_success_all = np.concatenate(all_success, axis=0)

    agg: dict[str, Any] = {
        "name": variant.name,
        "setting": asdict(variant),
        "num_seeds": int(len(seeds)),
        "episodes_per_seed": int(episodes_per_seed),
        "num_episodes_total": int(ep_costs_all.size),
    }
    for key in ["episode_cost_mean", "success_rate", "collision_rate", "final_distance_mean", "wall_seconds"]:
        vals = np.asarray([row[key] for row in per_seed], dtype=np.float64)
        agg[f"{key}_mean_over_seeds"] = float(np.mean(vals))
        agg[f"{key}_std_over_seeds"] = float(np.std(vals))

    return VariantResult(
        name=variant.name,
        aggregate=agg,
        per_seed=per_seed,
        episode_costs=ep_costs_all.astype(np.float32).tolist(),
        episode_success=ep_success_all.astype(np.float32).tolist(),
    )


def build_variants(args) -> list[Variant]:
    variants: list[Variant] = [
        Variant(
            name="baseline_full",
            chunked_spc=False,
            chunk_size=args.chunk_size,
            tau=args.tau_main,
            resample_pre=True,
            resample_post=True,
            resample_post_last=True,
        ),
        Variant(
            name="chunk_double_tau_low",
            chunked_spc=True,
            chunk_size=args.chunk_size,
            tau=args.tau_low,
            resample_pre=True,
            resample_post=True,
            resample_post_last=True,
        ),
        Variant(
            name="chunk_double_tau_main",
            chunked_spc=True,
            chunk_size=args.chunk_size,
            tau=args.tau_main,
            resample_pre=True,
            resample_post=True,
            resample_post_last=True,
        ),
        Variant(
            name="chunk_single_tau_main",
            chunked_spc=True,
            chunk_size=args.chunk_size,
            tau=args.tau_main,
            resample_pre=False,
            resample_post=True,
            resample_post_last=False,
        ),
        Variant(
            name="chunk_single_sys_ess_tau_main",
            chunked_spc=True,
            chunk_size=args.chunk_size,
            tau=args.tau_main,
            resample_pre=False,
            resample_post=True,
            resample_post_last=False,
            resample_scheme="systematic",
            ess_threshold=args.ess_threshold,
            lookahead_alpha=0.0,
        ),
    ]

    for alpha in args.lookahead_alphas:
        variants.append(
            Variant(
                name=f"chunk_single_sys_ess_tau_main_lookahead_{alpha:g}",
                chunked_spc=True,
                chunk_size=args.chunk_size,
                tau=args.tau_main,
                resample_pre=False,
                resample_post=True,
                resample_post_last=False,
                resample_scheme="systematic",
                ess_threshold=args.ess_threshold,
                lookahead_alpha=float(alpha),
            )
        )

    return variants


def pairwise_vs_baseline(
    results_by_name: dict[str, VariantResult],
    baseline_name: str,
    n_boot: int,
    ci_alpha: float,
) -> dict[str, dict[str, float]]:
    base = results_by_name[baseline_name]
    base_cost = np.asarray(base.episode_costs, dtype=np.float64)
    base_success = np.asarray(base.episode_success, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(0)

    for name, res in results_by_name.items():
        if name == baseline_name:
            continue
        cost = np.asarray(res.episode_costs, dtype=np.float64)
        success = np.asarray(res.episode_success, dtype=np.float64)
        if cost.shape != base_cost.shape:
            raise ValueError(f"Mismatched episode array sizes for baseline vs {name}.")

        d_cost = cost - base_cost
        d_succ = success - base_success

        cost_mean, cost_lo, cost_hi = bootstrap_ci_mean(d_cost, n_boot, ci_alpha, rng)
        succ_mean, succ_lo, succ_hi = bootstrap_ci_mean(d_succ, n_boot, ci_alpha, rng)

        out[name] = {
            "delta_cost_mean": cost_mean,
            "delta_cost_ci_low": cost_lo,
            "delta_cost_ci_high": cost_hi,
            "delta_success_mean": succ_mean,
            "delta_success_ci_low": succ_lo,
            "delta_success_ci_high": succ_hi,
            "cost_better_than_baseline": float(cost_hi) < 0.0,
            "success_better_than_baseline": float(succ_lo) > 0.0,
        }

    return out


def hypothesis_checks(pairwise: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def _cmp(a: str, b: str) -> tuple[float, float]:
        # Improvement from a -> b using baseline-relative deltas.
        da = pairwise[a]
        db = pairwise[b]
        # More negative cost delta is better; more positive success delta is better.
        return (
            float(db["delta_cost_mean"] - da["delta_cost_mean"]),
            float(db["delta_success_mean"] - da["delta_success_mean"]),
        )

    # H2: single resample better than double at tau_main.
    if "chunk_double_tau_main" in pairwise and "chunk_single_tau_main" in pairwise:
        d_cost, d_succ = _cmp("chunk_double_tau_main", "chunk_single_tau_main")
        out["H2_single_vs_double"] = {
            "delta_cost_single_minus_double": d_cost,
            "delta_success_single_minus_double": d_succ,
            "supports_hypothesis": (d_cost < 0.0 and d_succ > 0.0),
        }

    # H3: ESS+systematic better than single multinomial at tau_main.
    if "chunk_single_tau_main" in pairwise and "chunk_single_sys_ess_tau_main" in pairwise:
        d_cost, d_succ = _cmp("chunk_single_tau_main", "chunk_single_sys_ess_tau_main")
        out["H3_sysess_vs_single"] = {
            "delta_cost_sysess_minus_single": d_cost,
            "delta_success_sysess_minus_single": d_succ,
            "supports_hypothesis": (d_cost < 0.0 and d_succ > 0.0),
        }

    # H1: best lookahead variant better than no-lookahead sys+ess.
    lookahead_names = [k for k in pairwise if "lookahead_" in k]
    if "chunk_single_sys_ess_tau_main" in pairwise and lookahead_names:
        best = min(
            lookahead_names,
            key=lambda k: (pairwise[k]["delta_cost_mean"], -pairwise[k]["delta_success_mean"]),
        )
        d_cost, d_succ = _cmp("chunk_single_sys_ess_tau_main", best)
        out["H1_lookahead_vs_no_lookahead"] = {
            "best_lookahead_variant": best,
            "delta_cost_best_lookahead_minus_no_lookahead": d_cost,
            "delta_success_best_lookahead_minus_no_lookahead": d_succ,
            "supports_hypothesis": (d_cost < 0.0 and d_succ > 0.0),
        }

    return out


def save_figures(out_dir: Path, ordered: list[VariantResult], pairwise: dict[str, dict[str, float]]) -> None:
    names = [r.name for r in ordered]
    x = np.arange(len(names))

    succ_means = np.array([r.aggregate["success_rate_mean_over_seeds"] for r in ordered], dtype=np.float64)
    succ_stds = np.array([r.aggregate["success_rate_std_over_seeds"] for r in ordered], dtype=np.float64)
    cost_means = np.array([r.aggregate["episode_cost_mean_mean_over_seeds"] for r in ordered], dtype=np.float64)
    cost_stds = np.array([r.aggregate["episode_cost_mean_std_over_seeds"] for r in ordered], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x, succ_means, yerr=succ_stds, capsize=3)
    ax.set_xticks(x, labels=names, rotation=30, ha="right")
    ax.set_ylabel("Success rate (mean over seeds)")
    ax.set_title("Chunked SPC hypothesis test: success")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "success_by_variant.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x, cost_means, yerr=cost_stds, capsize=3)
    ax.set_xticks(x, labels=names, rotation=30, ha="right")
    ax.set_ylabel("Episode cost (mean over seeds)")
    ax.set_title("Chunked SPC hypothesis test: cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_by_variant.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(cost_means, succ_means, s=90)
    for name, cx, sy in zip(names, cost_means, succ_means, strict=True):
        ax.annotate(name, (cx, sy), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Episode cost (lower better)")
    ax.set_ylabel("Success rate (higher better)")
    ax.set_title("Cost-success frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_success_frontier.png", dpi=180)
    plt.close(fig)

    # Paired deltas vs baseline.
    names_d = list(pairwise.keys())
    idx = np.arange(len(names_d))
    d_cost = np.array([pairwise[n]["delta_cost_mean"] for n in names_d], dtype=np.float64)
    d_cost_lo = np.array([pairwise[n]["delta_cost_ci_low"] for n in names_d], dtype=np.float64)
    d_cost_hi = np.array([pairwise[n]["delta_cost_ci_high"] for n in names_d], dtype=np.float64)
    d_succ = np.array([pairwise[n]["delta_success_mean"] for n in names_d], dtype=np.float64)
    d_succ_lo = np.array([pairwise[n]["delta_success_ci_low"] for n in names_d], dtype=np.float64)
    d_succ_hi = np.array([pairwise[n]["delta_success_ci_high"] for n in names_d], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(idx, d_cost)
    axes[0].errorbar(
        idx,
        d_cost,
        yerr=[d_cost - d_cost_lo, d_cost_hi - d_cost],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Delta cost vs baseline")
    axes[0].set_title("Paired bootstrap deltas vs baseline (95% CI)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(idx, d_succ)
    axes[1].errorbar(
        idx,
        d_succ,
        yerr=[d_succ - d_succ_lo, d_succ_hi - d_succ],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Delta success vs baseline")
    axes[1].set_xticks(idx, labels=names_d, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_deltas_vs_baseline.png", dpi=180)
    plt.close(fig)

    # Lookahead sweep plot (if any).
    lookahead = [
        (n, float(n.split("lookahead_")[-1]))
        for n in names_d
        if "lookahead_" in n
    ]
    if lookahead:
        lookahead = sorted(lookahead, key=lambda t: t[1])
        alphas = np.array([a for _, a in lookahead], dtype=np.float64)
        succ = np.array([pairwise[n]["delta_success_mean"] for n, _ in lookahead], dtype=np.float64)
        succ_lo = np.array([pairwise[n]["delta_success_ci_low"] for n, _ in lookahead], dtype=np.float64)
        succ_hi = np.array([pairwise[n]["delta_success_ci_high"] for n, _ in lookahead], dtype=np.float64)
        cost = np.array([pairwise[n]["delta_cost_mean"] for n, _ in lookahead], dtype=np.float64)
        cost_lo = np.array([pairwise[n]["delta_cost_ci_low"] for n, _ in lookahead], dtype=np.float64)
        cost_hi = np.array([pairwise[n]["delta_cost_ci_high"] for n, _ in lookahead], dtype=np.float64)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(alphas, succ, marker="o")
        axes[0].fill_between(alphas, succ_lo, succ_hi, alpha=0.2)
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("lookahead alpha")
        axes[0].set_ylabel("Delta success vs baseline")
        axes[0].set_title("Lookahead sweep: success")
        axes[0].grid(alpha=0.25)

        axes[1].plot(alphas, cost, marker="o")
        axes[1].fill_between(alphas, cost_lo, cost_hi, alpha=0.2)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("lookahead alpha")
        axes[1].set_ylabel("Delta cost vs baseline")
        axes[1].set_title("Lookahead sweep: cost")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_dir / "lookahead_sweep.png", dpi=180)
        plt.close(fig)


def write_report(
    out_dir: Path,
    args,
    ordered: list[VariantResult],
    pairwise: dict[str, dict[str, float]],
    hypotheses: dict[str, dict[str, Any]],
) -> None:
    lines: list[str] = [
        f"# Chunked SPC Hypothesis Test ({args.controller_type})",
        "",
        "## Setup",
        "",
        f"- Task: avoid_u_trap",
        f"- Controller type: {args.controller_type}",
        f"- Horizon: {args.horizon}",
        f"- Knots: {args.num_knots}",
        f"- Samples: {args.num_samples}",
        f"- SPC iterations: {args.spc_iterations}",
        f"- Chunk size: {args.chunk_size} (active chunking = {args.chunk_size < args.num_knots})",
        f"- Tau low/main: {args.tau_low} / {args.tau_main}",
        f"- ESS threshold: {args.ess_threshold}",
        f"- Lookahead alphas: {args.lookahead_alphas}",
        f"- Seeds: {args.seeds}",
        f"- Episodes per seed: {args.episodes_per_seed}",
        f"- Paired bootstrap CI: {int((1.0 - args.ci_alpha) * 100)}% ({args.bootstrap_samples} samples)",
        "",
        "## Variant Summary",
        "",
        "| variant | mean cost | std cost | success | std success |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in ordered:
        a = r.aggregate
        lines.append(
            "| {name} | {c:.4f} | {cs:.4f} | {s:.4f} | {ss:.4f} |".format(
                name=r.name,
                c=a["episode_cost_mean_mean_over_seeds"],
                cs=a["episode_cost_mean_std_over_seeds"],
                s=a["success_rate_mean_over_seeds"],
                ss=a["success_rate_std_over_seeds"],
            )
        )

    lines += [
        "",
        "## Paired Deltas vs Baseline",
        "",
        "| variant | delta cost | CI low | CI high | delta success | CI low | CI high |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, d in pairwise.items():
        lines.append(
            "| {name} | {dc:.4f} | {dcl:.4f} | {dch:.4f} | {ds:.4f} | {dsl:.4f} | {dsh:.4f} |".format(
                name=name,
                dc=d["delta_cost_mean"],
                dcl=d["delta_cost_ci_low"],
                dch=d["delta_cost_ci_high"],
                ds=d["delta_success_mean"],
                dsl=d["delta_success_ci_low"],
                dsh=d["delta_success_ci_high"],
            )
        )

    lines += [
        "",
        "## Hypothesis Checks",
        "",
    ]

    for key, payload in hypotheses.items():
        lines.append(f"- {key}: {json.dumps(payload)}")

    lines += [
        "",
        "## Critique",
        "",
        "- Strength: same seeds/episodes across variants enable paired deltas.",
        "- Strength: ablation isolates schedule (double vs single), sampler (multinomial vs systematic+ESS), and score (lookahead alpha).",
        "- Limitation: single task geometry (u_trap) and one sample budget in this run.",
        "- Limitation: no direct internal ancestry/ESS logging from controller state; conclusions use outcome-level evidence.",
        "",
        "## Figures",
        "",
        "- success_by_variant.png",
        "- cost_by_variant.png",
        "- cost_success_frontier.png",
        "- paired_deltas_vs_baseline.png",
        "- lookahead_sweep.png (if lookahead variants present)",
    ]

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> Path:
    out_dir = args.out_dir / f"{args.controller_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_base_config(args)
    env = create_environment(cfg)
    policy = build_eval_policy(env, cfg)

    variants = build_variants(args)
    ordered_results: list[VariantResult] = []
    seeds = parse_int_list(args.seeds)

    try:
        for i, variant in enumerate(variants, 1):
            print(f"[{i}/{len(variants)}] Evaluating {variant.name}", flush=True)
            res = evaluate_variant(
                env=env,
                base_cfg=cfg,
                policy=policy,
                variant=variant,
                seeds=seeds,
                episodes_per_seed=args.episodes_per_seed,
                eval_batch_size=args.eval_batch_size,
            )
            ordered_results.append(res)
            a = res.aggregate
            print(
                f"  cost={a['episode_cost_mean_mean_over_seeds']:.4f} "
                f"success={a['success_rate_mean_over_seeds']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    results_by_name = {r.name: r for r in ordered_results}
    pairwise = pairwise_vs_baseline(
        results_by_name=results_by_name,
        baseline_name="baseline_full",
        n_boot=args.bootstrap_samples,
        ci_alpha=args.ci_alpha,
    )
    hypotheses = hypothesis_checks(pairwise)

    payload = {
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "variants": [asdict(v) for v in variants],
        "results": [
            {
                "name": r.name,
                "aggregate": r.aggregate,
                "per_seed": r.per_seed,
                "episode_costs": r.episode_costs,
                "episode_success": r.episode_success,
            }
            for r in ordered_results
        ],
        "pairwise_vs_baseline": pairwise,
        "hypotheses": hypotheses,
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_figures(out_dir, ordered_results, pairwise)
    write_report(out_dir, args, ordered_results, pairwise, hypotheses)

    print(f"Saved hypothesis test artifacts to {out_dir}", flush=True)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/avoid_u_trap.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/avoid_chunked_hypothesis_tests"))
    parser.add_argument("--controller-type", type=str, choices=["cem", "uniform"], default="cem")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--num-knots", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--spc-iterations", type=int, default=1)
    parser.add_argument("--num-elites", type=int, default=16)
    parser.add_argument("--sigma-start", type=float, default=2.0)
    parser.add_argument("--sigma-min", type=float, default=0.025)
    parser.add_argument("--noise-level", type=float, default=0.2)

    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--tau-low", type=float, default=0.1)
    parser.add_argument("--tau-main", type=float, default=5.0)
    parser.add_argument("--ess-threshold", type=float, default=0.6)
    parser.add_argument("--lookahead-alphas", type=str, default="0.25,0.5,1.0")

    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--episodes-per-seed", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)

    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--ci-alpha", type=float, default=0.05)

    args = parser.parse_args()
    args.lookahead_alphas = parse_float_list(args.lookahead_alphas)

    if args.episodes_per_seed % args.eval_batch_size != 0:
        raise ValueError("episodes-per-seed must be divisible by eval-batch-size")

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
