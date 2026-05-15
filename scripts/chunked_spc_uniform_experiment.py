#!/usr/bin/env python3
"""Chunked-SPC uniform experiment helpers (run shard + aggregate).

Designed for the avoid_u_trap task with short budgets and chunk-temperature sweeps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
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
from scripts.avoid_value_guidance_experiment import HJValueArtifacts, load_hj_artifacts, make_hj_terminal_value


VALID_RESAMPLE_SCHEMES = {"multinomial", "systematic", "residual_systematic"}


def parse_int_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated int list.")
    return out


def parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated float list.")
    return out


def parse_text_list(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated text list.")
    return out


def tau_slug(tau: float) -> str:
    s = f"{tau:g}"
    return s.replace("-", "m").replace(".", "p")


def scheme_slug(scheme: str) -> str:
    return str(scheme).replace("-", "_")


def strategy_key(resample_scheme: str, ess_threshold: float) -> str:
    return f"{resample_scheme}_ess{ess_threshold:g}"


def parse_strategy_profiles(text: str) -> list[tuple[str, float, str]]:
    profiles: list[tuple[str, float, str]] = []
    for token in parse_text_list(text):
        if ":" not in token:
            raise ValueError(
                "Each strategy profile must be formatted as 'scheme:ess', e.g. 'systematic:0.6'."
            )
        scheme, ess_text = token.split(":", 1)
        scheme = scheme.strip()
        if scheme not in VALID_RESAMPLE_SCHEMES:
            raise ValueError(
                "Unsupported resample scheme in strategy profile: "
                f"{scheme}. Allowed: {sorted(VALID_RESAMPLE_SCHEMES)}"
            )
        ess = float(ess_text.strip())
        profiles.append((scheme, ess, strategy_key(scheme, ess)))

    dedup: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for scheme, ess, key in profiles:
        if key in seen:
            continue
        dedup.append((scheme, ess, key))
        seen.add(key)
    if not dedup:
        raise ValueError("Expected at least one strategy profile.")
    return dedup


def _build_base_config(config_path: Path, horizon: float, batch_size: int) -> TrainingConfig:
    cfg = TrainingConfig.from_yaml(config_path)
    cfg.task_name = "avoid"
    cfg.task_variant = "u_trap"
    cfg.method = "gpc"
    cfg.controller_type = "uniform"
    cfg.iterations = 1
    cfg.use_wandb = False
    cfg.num_policy_samples = 0
    cfg.strategy = "best"
    cfg.record_training_videos = False
    cfg.record_eval_videos = False
    cfg.proposal_overlay = False
    cfg.proposal_video_trace_points = 1
    cfg.num_knots = 4
    cfg.plan_horizon = float(horizon)
    cfg.num_envs = int(batch_size)
    return cfg


def _build_eval_policy(env, base_cfg: TrainingConfig) -> Policy:
    net = create_network(env, base_cfg)
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    return Policy(net, normalizer, env.task.u_min, env.task.u_max)


def _make_eval_runner(env, ctrl, policy: Policy, batch_size: int, value_alpha: float):
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
            value_alpha,
            jnp.array(1.0),
            1,
        )

    return run_batch


def _trajectory_metrics(env, qpos: np.ndarray, variant: str = "u_trap") -> dict[str, float]:
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
    return {
        "success_rate": float(np.mean(success)),
        "collision_rate": float(np.mean(collision)),
        "final_distance_mean": float(np.mean(final_dist)),
        "final_distance_std": float(np.std(final_dist)),
        "min_sdf_mean": float(np.mean(min_sdf)),
    }


def add_obstacle_patch(ax, obstacle: np.ndarray, variant: str) -> None:
    ox, oy = float(obstacle[0]), float(obstacle[1])
    if variant == "sphere":
        from matplotlib import patches

        ax.add_patch(patches.Circle((ox, oy), radius=0.1, facecolor="black", alpha=0.18, edgecolor="black"))
        return
    if variant == "vertical_block":
        from matplotlib import patches

        ax.add_patch(
            patches.Rectangle((ox - 0.02, oy - 0.12), 0.04, 0.24, facecolor="black", alpha=0.18, edgecolor="black")
        )
        return

    from matplotlib import patches

    specs = [
        ((ox + 0.04 - 0.02, oy - 0.10), 0.04, 0.20),
        ((ox - 0.01 - 0.05, oy + 0.10 - 0.02), 0.10, 0.04),
        ((ox - 0.01 - 0.05, oy - 0.10 - 0.02), 0.10, 0.04),
    ]
    for xy, w, h in specs:
        ax.add_patch(patches.Rectangle(xy, w, h, facecolor="black", alpha=0.18, edgecolor="black"))


def evaluate_setting(
    env,
    base_cfg: TrainingConfig,
    policy: Policy,
    num_samples: int,
    chunked_spc: bool,
    chunk_size: int,
    tau: float,
    strategy_profile: str,
    resample_scheme: str,
    ess_threshold: float,
    chunk_resample_pre: bool,
    chunk_resample_post: bool,
    chunk_resample_post_last: bool,
    hj_artifacts: HJValueArtifacts | None,
    value_alpha: float,
    hj_use_base_terminal: bool,
    seeds: list[int],
    episodes_per_seed: int,
    traj_dir: Path,
) -> dict[str, Any]:
    cfg = copy.copy(base_cfg)
    cfg.num_samples = int(num_samples)
    cfg.chunked_spc = bool(chunked_spc)
    cfg.chunk_size = int(chunk_size)
    cfg.chunk_temperature = float(tau)
    cfg.chunk_resample_scheme = str(resample_scheme)
    cfg.chunk_ess_threshold = float(ess_threshold)
    cfg.chunk_resample_pre = bool(chunk_resample_pre)
    cfg.chunk_resample_post = bool(chunk_resample_post)
    cfg.chunk_resample_post_last = bool(chunk_resample_post_last)
    cfg.num_envs = int(base_cfg.num_envs)

    ctrl = create_controller(env, cfg)
    if hj_artifacts is not None and float(value_alpha) != 0.0:
        ctrl.value_fn = make_hj_terminal_value(env, hj_artifacts)
        ctrl.use_task_terminal_cost = bool(hj_use_base_terminal)
        ctrl.value_alpha = jnp.asarray(value_alpha, dtype=jnp.float32)
    run_batch = _make_eval_runner(env, ctrl, policy, cfg.num_envs, float(value_alpha))
    n_batches = int(math.ceil(episodes_per_seed / cfg.num_envs))

    per_seed: list[dict[str, Any]] = []
    qpos_rel_path = None

    for seed in seeds:
        rng = jax.random.key(int(seed))
        episode_costs: list[np.ndarray] = []
        qpos_chunks: list[np.ndarray] = []
        t0 = time.time()
        for _ in range(n_batches):
            rng, sim_rng = jax.random.split(rng)
            out = run_batch(sim_rng)
            _, _, _, _, _, j_inst, _, _, _, qpos, _, _, _ = out
            episode_costs.append(np.asarray(jnp.sum(j_inst, axis=1)))
            qpos_chunks.append(np.asarray(qpos))
        wall_s = time.time() - t0

        ep = np.concatenate(episode_costs, axis=0)[:episodes_per_seed]
        qpos = np.concatenate(qpos_chunks, axis=0)[:episodes_per_seed]
        tm = _trajectory_metrics(env, qpos)
        per_seed.append(
            {
                "seed": int(seed),
                "episode_cost_mean": float(np.mean(ep)),
                "episode_cost_std": float(np.std(ep)),
                "episode_cost_median": float(np.median(ep)),
                "episode_cost_min": float(np.min(ep)),
                "episode_cost_max": float(np.max(ep)),
                "wall_seconds": float(wall_s),
                **tm,
            }
        )

        if seed == seeds[0]:
            setting_slug = (
                f"N{num_samples}_full"
                if not chunked_spc
                else (
                    f"N{num_samples}_chunk{chunk_size}_tau{tau_slug(tau)}"
                    f"_{scheme_slug(strategy_profile)}"
                )
            )
            fname = f"{setting_slug}_seed{seed}.npz"
            np.savez_compressed(traj_dir / fname, qpos=qpos)
            qpos_rel_path = str(Path("qpos") / fname)

    agg: dict[str, Any] = {
        "num_samples": int(num_samples),
        "chunked_spc": bool(chunked_spc),
        "chunk_size": int(chunk_size),
        "tau": float(tau),
        "resample_scheme": str(resample_scheme) if chunked_spc else "baseline",
        "ess_threshold": float(ess_threshold) if chunked_spc else -1.0,
        "strategy_profile": str(strategy_profile) if chunked_spc else "baseline",
        "chunk_resample_pre": bool(chunk_resample_pre) if chunked_spc else False,
        "chunk_resample_post": bool(chunk_resample_post) if chunked_spc else False,
        "chunk_resample_post_last": bool(chunk_resample_post_last) if chunked_spc else False,
        "value_alpha": float(value_alpha),
        "uses_hj_terminal_value": bool(hj_artifacts is not None and float(value_alpha) != 0.0),
        "hj_use_base_terminal": bool(hj_use_base_terminal),
        "label": (
            f"uniform_full_K4_N{num_samples}"
            if not chunked_spc
            else (
                f"uniform_chunked_N{num_samples}_chunk{chunk_size}_tau{tau:g}"
                f"_{strategy_profile}"
            )
        ),
        "num_chunks": int(max(1, (base_cfg.num_knots + chunk_size - 1) // chunk_size)),
        "episodes_per_seed": int(episodes_per_seed),
        "num_seeds": int(len(seeds)),
        "seeds": [int(s) for s in seeds],
        "qpos_rel_path": qpos_rel_path,
        "per_seed": per_seed,
    }
    for key in [
        "episode_cost_mean",
        "success_rate",
        "collision_rate",
        "final_distance_mean",
        "min_sdf_mean",
        "wall_seconds",
    ]:
        vals = np.asarray([x[key] for x in per_seed], dtype=np.float64)
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    return agg


def cmd_run_shard(args: argparse.Namespace) -> int:
    budgets = parse_int_list(args.budgets)
    chunk_sizes = parse_int_list(args.chunk_sizes)
    taus = parse_float_list(args.taus)
    seeds = parse_int_list(args.seeds)
    if args.strategy_profiles:
        strategy_profiles = parse_strategy_profiles(args.strategy_profiles)
    else:
        strategy_profiles = []
        seen: set[str] = set()
        for scheme in parse_text_list(args.resample_schemes):
            if scheme not in VALID_RESAMPLE_SCHEMES:
                raise ValueError(
                    f"Unsupported resample scheme: {scheme}. Allowed: {sorted(VALID_RESAMPLE_SCHEMES)}"
                )
            key = strategy_key(scheme, float(args.ess_threshold))
            if key in seen:
                continue
            strategy_profiles.append((scheme, float(args.ess_threshold), key))
            seen.add(key)

    resample_schemes = [scheme for scheme, _, _ in strategy_profiles]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "qpos"
    traj_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_config(args.config, args.horizon, args.batch_size)
    env = create_environment(base_cfg)
    policy = _build_eval_policy(env, base_cfg)
    hj_artifacts: HJValueArtifacts | None = None

    rows: list[dict[str, Any]] = []
    try:
        if float(args.value_alpha) != 0.0:
            if args.hj_dir is None:
                raise ValueError("--hj-dir is required when --value-alpha is nonzero.")
            hj_artifacts = load_hj_artifacts(args.hj_dir, base_cfg, env)
            print(
                f"Loaded HJ terminal value artifacts from {args.hj_dir} (value_alpha={float(args.value_alpha):g})",
                flush=True,
            )

        total = 0
        if args.include_baselines:
            total += len(budgets)
        total += len(budgets) * len(chunk_sizes) * len(taus) * len(strategy_profiles)
        k = 0

        if args.include_baselines:
            for n in budgets:
                k += 1
                print(f"[{k}/{total}] baseline N={n}", flush=True)
                row = evaluate_setting(
                    env=env,
                    base_cfg=base_cfg,
                    policy=policy,
                    num_samples=n,
                    chunked_spc=False,
                    chunk_size=4,
                    tau=0.0,
                    strategy_profile="baseline",
                    resample_scheme="baseline",
                    ess_threshold=-1.0,
                    chunk_resample_pre=False,
                    chunk_resample_post=False,
                    chunk_resample_post_last=False,
                    hj_artifacts=hj_artifacts,
                    value_alpha=float(args.value_alpha),
                    hj_use_base_terminal=bool(args.hj_use_base_terminal),
                    seeds=seeds,
                    episodes_per_seed=args.episodes,
                    traj_dir=traj_dir,
                )
                rows.append(row)
                print(
                    f"  success={row['success_rate_mean']:.3f} cost={row['episode_cost_mean_mean']:.3f}",
                    flush=True,
                )

        for n in budgets:
            for c in chunk_sizes:
                for tau in taus:
                    for scheme, ess, profile_key in strategy_profiles:
                        k += 1
                        print(
                            f"[{k}/{total}] chunked N={n} chunk={c} tau={tau:g} strategy={profile_key}",
                            flush=True,
                        )
                        row = evaluate_setting(
                            env=env,
                            base_cfg=base_cfg,
                            policy=policy,
                            num_samples=n,
                            chunked_spc=True,
                            chunk_size=c,
                            tau=tau,
                            strategy_profile=profile_key,
                            resample_scheme=scheme,
                            ess_threshold=ess,
                            chunk_resample_pre=args.chunk_resample_pre,
                            chunk_resample_post=args.chunk_resample_post,
                            chunk_resample_post_last=args.chunk_resample_post_last,
                            hj_artifacts=hj_artifacts,
                            value_alpha=float(args.value_alpha),
                            hj_use_base_terminal=bool(args.hj_use_base_terminal),
                            seeds=seeds,
                            episodes_per_seed=args.episodes,
                            traj_dir=traj_dir,
                        )
                        rows.append(row)
                        print(
                            f"  success={row['success_rate_mean']:.3f} cost={row['episode_cost_mean_mean']:.3f}",
                            flush=True,
                        )
    finally:
        env.close()

    payload = {
        "meta": {
            "config": str(args.config),
            "horizon": float(args.horizon),
            "budgets": budgets,
            "chunk_sizes": chunk_sizes,
            "taus": taus,
            "resample_schemes": resample_schemes,
            "ess_threshold": float(args.ess_threshold),
            "strategy_profiles": [
                {
                    "scheme": scheme,
                    "ess_threshold": float(ess),
                    "key": key,
                }
                for scheme, ess, key in strategy_profiles
            ],
            "chunk_resample_pre": bool(args.chunk_resample_pre),
            "chunk_resample_post": bool(args.chunk_resample_post),
            "chunk_resample_post_last": bool(args.chunk_resample_post_last),
            "hj_dir": str(args.hj_dir) if args.hj_dir is not None else None,
            "value_alpha": float(args.value_alpha),
            "hj_use_base_terminal": bool(args.hj_use_base_terminal),
            "seeds": seeds,
            "episodes": int(args.episodes),
            "batch_size": int(args.batch_size),
            "include_baselines": bool(args.include_baselines),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rows": rows,
    }
    with open(out_dir / "results_shard.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved shard results to {out_dir / 'results_shard.json'}", flush=True)
    return 0


def key_from_row(row: dict[str, Any]) -> tuple[int, bool, int, str, str]:
    if bool(row["chunked_spc"]):
        if "strategy_profile" in row:
            profile = str(row["strategy_profile"])
        else:
            profile = strategy_key(str(row.get("resample_scheme", "multinomial")), float(row.get("ess_threshold", 0.0)))
    else:
        profile = "baseline"
    return (
        int(row["num_samples"]),
        bool(row["chunked_spc"]),
        int(row["chunk_size"]),
        f"{float(row['tau']):.12g}",
        profile,
    )


def load_shards(run_root: Path, shard_glob: str) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    shard_dirs = sorted(run_root.glob(shard_glob))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories matching '{shard_glob}' under {run_root}")

    for shard_dir in shard_dirs:
        shard_json = shard_dir / "results_shard.json"
        if not shard_json.exists():
            continue
        payload = json.loads(shard_json.read_text())
        for row in payload["rows"]:
            row = dict(row)
            row["_source_dir"] = str(shard_dir)
            all_rows.append(row)
    if not all_rows:
        raise FileNotFoundError("No shard rows found.")
    return all_rows


def get_row(
    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]],
    n: int,
    chunked: bool,
    chunk_size: int,
    tau: float,
    strategy_profile: str = "baseline",
) -> dict[str, Any] | None:
    key_profile = str(strategy_profile) if bool(chunked) else "baseline"
    return merged.get((int(n), bool(chunked), int(chunk_size), f"{float(tau):.12g}", key_profile))


def plot_heatmaps(
    out_dir: Path,
    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]],
    budgets: list[int],
    chunk_sizes: list[int],
    taus: list[float],
    strategy_profiles: list[str],
) -> None:
    for profile in strategy_profiles:
        profile_file = scheme_slug(profile)
        for n in budgets:
            for metric, cmap in [
                ("success_rate_mean", "viridis"),
                ("episode_cost_mean_mean", "magma_r"),
            ]:
                mat = np.full((len(chunk_sizes), len(taus)), np.nan, dtype=np.float32)
                for i, c in enumerate(chunk_sizes):
                    for j, tau in enumerate(taus):
                        row = get_row(merged, n, True, c, tau, strategy_profile=profile)
                        if row is not None:
                            mat[i, j] = float(row[metric])

                fig, ax = plt.subplots(figsize=(9.0, 4.8))
                im = ax.imshow(mat, aspect="auto", cmap=cmap)
                ax.set_xticks(np.arange(len(taus)), labels=[f"{t:g}" for t in taus])
                ax.set_yticks(np.arange(len(chunk_sizes)), labels=[str(c) for c in chunk_sizes])
                ax.set_xlabel("tau (chunk_temperature)")
                ax.set_ylabel("chunk_size")
                ax.set_title(f"strategy={profile} | N={n} | {metric}")
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        v = mat[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(metric)
                fig.tight_layout()
                fig.savefig(out_dir / f"heatmap_{profile_file}_N{n}_{metric}.png", dpi=180)
                plt.close(fig)


def resolve_qpos_path(row: dict[str, Any]) -> Path | None:
    rel = row.get("qpos_rel_path")
    if not rel:
        return None
    src = Path(row["_source_dir"])
    p = src / rel
    return p if p.exists() else None


def plot_trajectory_matrices(
    out_dir: Path,
    config_path: Path,
    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]],
    budgets: list[int],
    chunk_sizes: list[int],
    taus: list[float],
    strategy_profiles: list[str],
    max_traj: int,
) -> None:
    cfg = TrainingConfig.from_yaml(config_path)
    cfg.task_name = "avoid"
    cfg.task_variant = "u_trap"
    env = create_environment(cfg)
    try:
        pointmass_body_id = env.task.mj_model.body("pointmass").id
        obstacle_body_id = env.task.mj_model.body("obstacle").id
        offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)
        obstacle = np.asarray(env.task.mj_model.body_pos[obstacle_body_id, :2], dtype=np.float32)
        goal = np.array([0.25, 0.0], dtype=np.float32)
        variant = "u_trap"

        for profile in strategy_profiles:
            profile_file = scheme_slug(profile)
            for n in budgets:
                n_rows = len(taus)
                n_cols = 1 + len(chunk_sizes)
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(4.6 * n_cols, max(2.6 * n_rows, 8.0)),
                    sharex=True,
                    sharey=True,
                )
                axes = np.asarray(axes)
                if axes.ndim == 1:
                    axes = axes[None, :]

                for i, tau in enumerate(taus):
                    row_full = get_row(merged, n, False, 4, 0.0, strategy_profile="baseline")
                    row_c1 = get_row(merged, n, True, chunk_sizes[0], tau, strategy_profile=profile)
                    row_c2 = get_row(merged, n, True, chunk_sizes[1], tau, strategy_profile=profile)
                    row_pack = [row_full, row_c1, row_c2]
                    col_titles = ["Uniform full K=4", f"Chunk size={chunk_sizes[0]}", f"Chunk size={chunk_sizes[1]}"]

                    for j, row in enumerate(row_pack):
                        ax = axes[i, j]
                        if row is None:
                            ax.set_axis_off()
                            continue
                        qpos_path = resolve_qpos_path(row)
                        if qpos_path is None:
                            ax.set_axis_off()
                            continue
                        qpos = np.load(qpos_path)["qpos"]
                        pos = qpos[: min(max_traj, qpos.shape[0]), :, :2] + offset[None, None, :]
                        for traj in pos:
                            ax.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1.0, color="#1f77b4")
                        add_obstacle_patch(ax, obstacle, variant)
                        ax.plot(goal[0], goal[1], marker="*", color="gold", markeredgecolor="k", markersize=8)
                        ax.set_aspect("equal", adjustable="box")
                        ax.grid(alpha=0.2)
                        if i == 0:
                            ax.set_title(col_titles[j])
                        if j == 0:
                            ax.set_ylabel(f"tau={tau:g}\ny")
                        else:
                            ax.set_ylabel("y")
                        if i == n_rows - 1:
                            ax.set_xlabel("x")
                        ax.text(
                            0.02,
                            0.98,
                            f"succ={row['success_rate_mean']:.2f}\ncost={row['episode_cost_mean_mean']:.2f}",
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=8,
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                        )

                fig.suptitle(
                    f"Trajectory matrix (strategy={profile}, N={n}, rows=tau, cols=baseline/chunk)",
                    y=1.0,
                )
                fig.tight_layout()
                fig.savefig(out_dir / f"trajectory_matrix_{profile_file}_N{n}.png", dpi=180, bbox_inches="tight")
                plt.close(fig)
    finally:
        env.close()


def summarize_best_chunked(
    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]],
    budgets: list[int],
    chunk_sizes: list[int],
    taus: list[float],
    strategy_profiles: list[str],
) -> list[dict[str, Any]]:
    best_meta: list[dict[str, Any]] = []
    for n in budgets:
        b = get_row(merged, n, False, 4, 0.0, strategy_profile="baseline")
        if b is None:
            continue
        candidates = []
        for profile in strategy_profiles:
            for c in chunk_sizes:
                for tau in taus:
                    row = get_row(merged, n, True, c, tau, strategy_profile=profile)
                    if row is not None:
                        candidates.append(row)
        if not candidates:
            continue
        k = sorted(candidates, key=lambda r: (-float(r["success_rate_mean"]), float(r["episode_cost_mean_mean"])))[0]
        beat = (k["success_rate_mean"] > b["success_rate_mean"]) or (
            abs(float(k["success_rate_mean"]) - float(b["success_rate_mean"])) < 1e-12
            and float(k["episode_cost_mean_mean"]) < float(b["episode_cost_mean_mean"])
        )
        best_meta.append(
            {
                "num_samples": n,
                "baseline_success": float(b["success_rate_mean"]),
                "baseline_cost": float(b["episode_cost_mean_mean"]),
                "best_strategy_profile": str(k.get("strategy_profile", "")),
                "best_scheme": str(k["resample_scheme"]),
                "best_ess_threshold": float(k.get("ess_threshold", 0.0)),
                "best_chunk_size": int(k["chunk_size"]),
                "best_tau": float(k["tau"]),
                "best_success": float(k["success_rate_mean"]),
                "best_cost": float(k["episode_cost_mean_mean"]),
                "beats_baseline": bool(beat),
            }
        )
    return best_meta


def plot_tau_lines(
    out_dir: Path,
    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]],
    budgets: list[int],
    chunk_sizes: list[int],
    taus: list[float],
    strategy_profiles: list[str],
) -> None:
    n_cols = len(budgets)
    fig_s, axes_s = plt.subplots(1, n_cols, figsize=(5.3 * n_cols, 4.8), sharey=False)
    fig_c, axes_c = plt.subplots(1, n_cols, figsize=(5.3 * n_cols, 4.8), sharey=False)
    if n_cols == 1:
        axes_s = np.asarray([axes_s])
        axes_c = np.asarray([axes_c])

    for idx, n in enumerate(budgets):
        ax_s = axes_s[idx]
        ax_c = axes_c[idx]
        b = get_row(merged, n, False, 4, 0.0, strategy_profile="baseline")
        if b is None:
            ax_s.set_axis_off()
            ax_c.set_axis_off()
            continue

        ax_s.axhline(float(b["success_rate_mean"]), color="black", linestyle="--", linewidth=1.5, label="baseline")
        ax_c.axhline(float(b["episode_cost_mean_mean"]), color="black", linestyle="--", linewidth=1.5, label="baseline")

        for profile in strategy_profiles:
            for c in chunk_sizes:
                ys_s = []
                ys_c = []
                x = []
                for tau in taus:
                    row = get_row(merged, n, True, c, tau, strategy_profile=profile)
                    if row is not None:
                        x.append(float(tau))
                        ys_s.append(float(row["success_rate_mean"]))
                        ys_c.append(float(row["episode_cost_mean_mean"]))
                if not x:
                    continue
                label = f"{profile}, chunk={c}"
                ax_s.plot(x, ys_s, marker="o", linewidth=1.8, label=label)
                ax_c.plot(x, ys_c, marker="o", linewidth=1.8, label=label)

        ax_s.set_title(f"N={n}")
        ax_s.set_xlabel("tau")
        ax_s.set_ylabel("success rate")
        ax_s.set_ylim(0.0, 1.05)
        ax_s.grid(alpha=0.25)

        ax_c.set_title(f"N={n}")
        ax_c.set_xlabel("tau")
        ax_c.set_ylabel("mean episode cost")
        ax_c.grid(alpha=0.25)

    handles, labels = axes_s[0].get_legend_handles_labels()
    fig_s.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig_s.suptitle("Tau effect on success (baseline dashed)", y=1.04)
    fig_s.tight_layout()
    fig_s.savefig(out_dir / "tau_lines_success_by_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig_s)

    handles, labels = axes_c[0].get_legend_handles_labels()
    fig_c.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig_c.suptitle("Tau effect on cost (baseline dashed)", y=1.04)
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "tau_lines_cost_by_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig_c)


def write_report(
    out_dir: Path,
    budgets: list[int],
    chunk_sizes: list[int],
    taus: list[float],
    strategy_profiles: list[str],
    seeds: list[int],
    episodes: int,
    value_alpha: float,
    hj_dir: Path | None,
    hj_use_base_terminal: bool,
    merged_rows: list[dict[str, Any]],
    best_meta: list[dict[str, Any]],
) -> None:
    lines = [
        "# Chunked SPC Uniform Experiment",
        "",
        "## Setup",
        "",
        "- Task: avoid_u_trap",
        "- Controller: uniform sampling",
        "- Horizon: 1.0s (K=4 knots)",
        f"- Budgets: {budgets}",
        f"- Chunk sizes: {chunk_sizes}",
        f"- Tau sweep: {taus}",
        f"- Strategy profiles (scheme:ess): {strategy_profiles}",
        f"- Seeds: {seeds}",
        f"- Episodes per seed: {episodes}",
        f"- HJ terminal value alpha: {value_alpha:g}",
        f"- HJ artifacts dir: {str(hj_dir) if hj_dir is not None else 'none'}",
        f"- HJ keeps task terminal cost: {bool(hj_use_base_terminal)}",
        "- Criterion to beat baseline: higher success primary, lower cost tie-break.",
        "",
        "## Aggregated Results",
        "",
        "| N | chunked | strategy_profile | scheme | ess | chunk_size | tau | success mean | success std | cost mean | cost std | collision mean |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    rows_sorted = sorted(
        merged_rows,
        key=lambda r: (
            int(r["num_samples"]),
            int(not bool(r["chunked_spc"])),
            str(r.get("strategy_profile", "baseline")),
            str(r.get("resample_scheme", "baseline")),
            float(r.get("ess_threshold", -1.0)),
            int(r["chunk_size"]),
            float(r["tau"]),
        ),
    )

    for r in rows_sorted:
        lines.append(
            "| {N} | {chunked} | {strategy} | {scheme} | {ess:g} | {c} | {tau:g} | {succ:.3f} | {succ_std:.3f} | {cost:.3f} | {cost_std:.3f} | {col:.3f} |".format(
                N=r["num_samples"],
                chunked=int(bool(r["chunked_spc"])),
                strategy=str(r.get("strategy_profile", "baseline")),
                scheme=str(r.get("resample_scheme", "baseline")),
                ess=float(r.get("ess_threshold", -1.0)),
                c=r["chunk_size"],
                tau=r["tau"],
                succ=r["success_rate_mean"],
                succ_std=r["success_rate_std"],
                cost=r["episode_cost_mean_mean"],
                cost_std=r["episode_cost_mean_std"],
                col=r["collision_rate_mean"],
            )
        )

    lines += [
        "",
        "## Best Chunked vs Baseline",
        "",
        "| N | baseline success | baseline cost | best strategy_profile | best scheme | best ess | best chunk_size | best tau | best success | best cost | beats baseline |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for m in sorted(best_meta, key=lambda x: x["num_samples"]):
        lines.append(
            "| {N} | {bs:.3f} | {bc:.3f} | {profile} | {scheme} | {ess:g} | {c} | {tau:g} | {s:.3f} | {cost:.3f} | {beat} |".format(
                N=m["num_samples"],
                bs=m["baseline_success"],
                bc=m["baseline_cost"],
                profile=m["best_strategy_profile"],
                scheme=m["best_scheme"],
                ess=m["best_ess_threshold"],
                c=m["best_chunk_size"],
                tau=m["best_tau"],
                s=m["best_success"],
                cost=m["best_cost"],
                beat=str(m["beats_baseline"]),
            )
        )

    lines += [
        "",
        "## Figures",
        "",
        "- tau_lines_success_by_budget.png",
        "- tau_lines_cost_by_budget.png",
    ]
    for profile in strategy_profiles:
        profile_file = scheme_slug(profile)
        for n in budgets:
            lines.append(f"- heatmap_{profile_file}_N{n}_success_rate_mean.png")
            lines.append(f"- heatmap_{profile_file}_N{n}_episode_cost_mean_mean.png")
            lines.append(f"- trajectory_matrix_{profile_file}_N{n}.png")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_aggregate(args: argparse.Namespace) -> int:
    budgets = parse_int_list(args.budgets)
    chunk_sizes = parse_int_list(args.chunk_sizes)
    taus = parse_float_list(args.taus)
    if args.strategy_profiles:
        strategy_profiles = [key for _, _, key in parse_strategy_profiles(args.strategy_profiles)]
    else:
        strategy_profiles = []
        seen: set[str] = set()
        for scheme in parse_text_list(args.resample_schemes):
            key = strategy_key(scheme, float(args.ess_threshold))
            if key in seen:
                continue
            strategy_profiles.append(key)
            seen.add(key)
    if len(chunk_sizes) != 2:
        raise ValueError("This aggregator expects exactly two chunk sizes for matrix columns.")

    rows = load_shards(args.run_root, args.shard_glob)

    merged: dict[tuple[int, bool, int, str, str], dict[str, Any]] = {}
    for row in rows:
        k = key_from_row(row)
        if k not in merged:
            merged[k] = row

    missing = []
    for n in budgets:
        if get_row(merged, n, False, 4, 0.0, strategy_profile="baseline") is None:
            missing.append(f"baseline N={n}")
        for profile in strategy_profiles:
            for c in chunk_sizes:
                for tau in taus:
                    if get_row(merged, n, True, c, tau, strategy_profile=profile) is None:
                        missing.append(f"chunked N={n}, strategy={profile}, c={c}, tau={tau:g}")
    if missing:
        raise RuntimeError("Missing required settings:\n- " + "\n- ".join(missing[:50]))

    out_dir = args.run_root / "combined_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows = list(merged.values())
    plot_heatmaps(out_dir, merged, budgets, chunk_sizes, taus, strategy_profiles)
    plot_trajectory_matrices(
        out_dir=out_dir,
        config_path=args.config,
        merged=merged,
        budgets=budgets,
        chunk_sizes=chunk_sizes,
        taus=taus,
        strategy_profiles=strategy_profiles,
        max_traj=args.matrix_max_traj,
    )
    plot_tau_lines(out_dir, merged, budgets, chunk_sizes, taus, strategy_profiles)
    best_meta = summarize_best_chunked(merged, budgets, chunk_sizes, taus, strategy_profiles)

    seeds = parse_int_list(args.seeds)
    write_report(
        out_dir,
        budgets,
        chunk_sizes,
        taus,
        strategy_profiles,
        seeds,
        args.episodes,
        float(args.value_alpha),
        args.hj_dir,
        bool(args.hj_use_base_terminal),
        merged_rows,
        best_meta,
    )

    summary = {
        "run_root": str(args.run_root),
        "budgets": budgets,
        "chunk_sizes": chunk_sizes,
        "taus": taus,
        "strategy_profiles": strategy_profiles,
        "ess_threshold": float(args.ess_threshold),
        "hj_dir": str(args.hj_dir) if args.hj_dir is not None else None,
        "value_alpha": float(args.value_alpha),
        "hj_use_base_terminal": bool(args.hj_use_base_terminal),
        "seeds": seeds,
        "episodes": int(args.episodes),
        "best_by_budget": best_meta,
        "rows": merged_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved combined report to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-shard")
    run.add_argument("--config", type=Path, default=Path("configs/avoid_u_trap.yaml"))
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--horizon", type=float, default=1.0)
    run.add_argument("--budgets", type=str, default="16,32,64")
    run.add_argument("--chunk-sizes", type=str, default="1,2")
    run.add_argument("--taus", type=str, required=True)
    run.add_argument("--resample-schemes", type=str, default="multinomial,systematic,residual_systematic")
    run.add_argument("--strategy-profiles", type=str, default="")
    run.add_argument("--ess-threshold", type=float, default=0.6)
    run.add_argument("--chunk-resample-pre", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--chunk-resample-post", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--chunk-resample-post-last", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--hj-dir", type=Path, default=None)
    run.add_argument("--value-alpha", type=float, default=0.0)
    run.add_argument("--hj-use-base-terminal", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--seeds", type=str, default="1")
    run.add_argument("--episodes", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--include-baselines", action="store_true")

    agg = sub.add_parser("aggregate")
    agg.add_argument("--run-root", type=Path, required=True)
    agg.add_argument("--config", type=Path, default=Path("configs/avoid_u_trap.yaml"))
    agg.add_argument("--budgets", type=str, default="16,32,64")
    agg.add_argument("--chunk-sizes", type=str, default="1,2")
    agg.add_argument("--taus", type=str, default="0.1,0.3,1.0,3.0,5.0")
    agg.add_argument("--resample-schemes", type=str, default="multinomial,systematic,residual_systematic")
    agg.add_argument("--strategy-profiles", type=str, default="")
    agg.add_argument("--ess-threshold", type=float, default=0.6)
    agg.add_argument("--hj-dir", type=Path, default=None)
    agg.add_argument("--value-alpha", type=float, default=0.0)
    agg.add_argument("--hj-use-base-terminal", action=argparse.BooleanOptionalAction, default=True)
    agg.add_argument("--seeds", type=str, default="1")
    agg.add_argument("--episodes", type=int, default=64)
    agg.add_argument("--shard-glob", type=str, default="shard_*")
    agg.add_argument("--matrix-max-traj", type=int, default=8)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-shard":
        return cmd_run_shard(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
