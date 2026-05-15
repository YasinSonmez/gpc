#!/usr/bin/env python3
"""Aggregate multiseed iid-noise runs across controller variants.

Outputs:
- mean+/-std success/cost comparison across seeds
- matrix figure with [noise z, Vhat, Uniform traj, CEM traj, CEM no-warm]
- markdown report with seeding sanity checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gpc.hj_solver import _obstacle_mask_xy
from run_experiment import create_environment
from gpc.config import TrainingConfig
from scripts.avoid_value_guidance_experiment import (
    add_obstacle_patch,
    blend_to_correlated_noise,
    positive_correlated_field_from_normal,
    positive_softplus_gaussian_field,
    smooth_noise_field,
)


def parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated float list.")
    return out


def parse_int_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated integer list.")
    return out


def parse_text_list(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected non-empty comma-separated text list.")
    return out


def method_label(method: str) -> str:
    mapping = {
        "uniform": "Uniform",
        "cem": "CEM warm-start",
        "cem_no_warm_start": "CEM no warm-start",
    }
    return mapping.get(method, method)


def latest_child_dir(path: Path) -> Path:
    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No run directory found under {path}")
    return sorted(children)[-1]


def load_noise_results(results_path: Path, case_name: str):
    payload = json.loads(results_path.read_text())
    by_noise = {
        float(k): v[case_name]
        for k, v in payload["results_by_noise"].items()
    }
    seeds_by_noise = {
        float(k): int(v)
        for k, v in payload.get("noise_seed_by_noise", {}).items()
    }
    return payload, by_noise, seeds_by_noise


def resolve_noise_field(
    values: np.ndarray,
    sigma: float,
    noise_seed: int,
    noise_mean: float,
    noise_smooth_passes: int,
    noise_distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_distribution == "blend_correlated":
        if not (0.0 <= float(sigma) <= 1.0):
            raise ValueError("for blend_correlated, sigma must be in [0, 1]")
        vhat = blend_to_correlated_noise(
            values=values,
            noise_scale=float(sigma),
            noise_seed=int(noise_seed),
            smooth_passes=int(noise_smooth_passes),
        )
        z = vhat - values
        return z.astype(np.float32), vhat.astype(np.float32)

    rng = np.random.default_rng(int(noise_seed))
    if sigma == 0.0:
        z = np.full(values.shape, float(noise_mean), dtype=np.float32)
    else:
        z0 = rng.standard_normal(values.shape, dtype=np.float32)
        z0 = smooth_noise_field(z0, smooth_passes=int(noise_smooth_passes))
        z0 = (z0 - np.mean(z0)) / (np.std(z0) + 1e-8)
        if noise_distribution == "softplus_gaussian":
            z = positive_softplus_gaussian_field(z0, mean=float(noise_mean), small_noise_std=float(sigma))
        elif noise_distribution == "lognormal":
            z = positive_correlated_field_from_normal(z0, mean=float(noise_mean), std=float(sigma))
        elif noise_distribution == "clipped_gaussian":
            z = np.maximum(float(noise_mean) + float(sigma) * z0, np.finfo(np.float32).tiny).astype(np.float32)
        else:
            raise ValueError(f"Unknown distribution: {noise_distribution}")
    vhat = values + z * values
    return z.astype(np.float32), vhat.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/avoid_u_trap.yaml"))
    parser.add_argument(
        "--hj-dir",
        type=Path,
        default=Path("experiments/avoid/hj_avoid_binary_u_trap_20260502_131354/hj"),
    )
    parser.add_argument("--methods", type=str, default="uniform,cem,cem_no_warm_start")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--sigmas", type=str, required=True)
    parser.add_argument("--matrix-seed", type=int, default=1)
    parser.add_argument("--case-name", type=str, default="spc_hjV")
    parser.add_argument("--noise-mean", type=float, default=0.1)
    parser.add_argument("--noise-smooth-passes", type=int, default=0)
    parser.add_argument(
        "--noise-distribution",
        type=str,
        default="softplus_gaussian",
        choices=["softplus_gaussian", "lognormal", "clipped_gaussian", "blend_correlated"],
    )
    parser.add_argument("--max-traj", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    methods = parse_text_list(args.methods)
    seeds = parse_int_list(args.seeds)
    sigmas = parse_float_list(args.sigmas)
    sigmas = sorted(sigmas)
    out_dir = args.out_dir or (args.run_root / "combined_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: dict[str, dict[int, Path]] = {m: {} for m in methods}
    payloads: dict[str, dict[int, dict]] = {m: {} for m in methods}
    results: dict[str, dict[int, dict[float, dict]]] = {m: {} for m in methods}
    noise_seeds: dict[str, dict[int, dict[float, int]]] = {m: {} for m in methods}

    for method in methods:
        for seed in seeds:
            seed_dir = args.run_root / "runs" / method / f"seed_{seed}"
            run_dir = latest_child_dir(seed_dir)
            payload, by_noise, seed_map = load_noise_results(run_dir / "results.json", args.case_name)
            missing = [s for s in sigmas if s not in by_noise]
            if missing:
                raise ValueError(f"Missing sigma values for {method}/seed{seed}: {missing}")
            run_dirs[method][seed] = run_dir
            payloads[method][seed] = payload
            results[method][seed] = by_noise
            noise_seeds[method][seed] = seed_map

    # Seeding sanity checks.
    same_noise_across_methods = True
    different_noise_across_sigmas_per_seed = True
    different_noise_across_seeds_per_sigma = True
    reference_method = methods[0]
    mismatch_examples = []

    for seed in seeds:
        per_sigma = []
        for sigma in sigmas:
            ref_seed = noise_seeds[reference_method][seed].get(sigma)
            if ref_seed is None:
                raise ValueError(f"No noise_seed_by_noise entry for sigma={sigma} ({reference_method}, seed={seed})")
            per_sigma.append(ref_seed)
            for method in methods[1:]:
                cur = noise_seeds[method][seed].get(sigma)
                if cur != ref_seed:
                    same_noise_across_methods = False
                    mismatch_examples.append((method, seed, sigma, ref_seed, cur))
        if len(set(per_sigma)) != len(per_sigma):
            different_noise_across_sigmas_per_seed = False

    for sigma in sigmas:
        per_seed = [noise_seeds[reference_method][seed][sigma] for seed in seeds]
        if len(set(per_seed)) != len(per_seed):
            different_noise_across_seeds_per_sigma = False

    # Mean +/- std metrics across seeds.
    summary = {}
    for method in methods:
        summary[method] = {}
        for sigma in sigmas:
            succ = [float(results[method][seed][sigma]["success_rate"]) for seed in seeds]
            cost = [float(results[method][seed][sigma]["episode_cost_mean"]) for seed in seeds]
            summary[method][sigma] = {
                "success_mean": float(np.mean(succ)),
                "success_std": float(np.std(succ)),
                "cost_mean": float(np.mean(cost)),
                "cost_std": float(np.std(cost)),
            }

    # Comparison plot: mean +/- std.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    styles = {
        "uniform": dict(color="#1f77b4", marker="o", linestyle="-"),
        "cem": dict(color="#ff7f0e", marker="s", linestyle="-"),
        "cem_no_warm_start": dict(color="#2ca02c", marker="^", linestyle="--"),
    }
    x_raw = np.asarray(sigmas, dtype=np.float64)
    x = x_raw.copy()
    pos = x_raw[x_raw > 0.0]
    if pos.size == 0:
        raise ValueError("At least one positive sigma is required for log-scale plotting.")
    min_pos = float(np.min(pos))
    x[x <= 0.0] = 0.5 * min_pos
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(float(np.min(x)) * 0.9, float(np.max(x)) * 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in x_raw])

    for method in methods:
        st = styles.get(method, dict(marker="o", linestyle="-"))
        succ_m = np.asarray([summary[method][s]["success_mean"] for s in sigmas])
        succ_sd = np.asarray([summary[method][s]["success_std"] for s in sigmas])
        cost_m = np.asarray([summary[method][s]["cost_mean"] for s in sigmas])
        cost_sd = np.asarray([summary[method][s]["cost_std"] for s in sigmas])

        axes[0].plot(x, succ_m, label=method_label(method), linewidth=2.5, markersize=6.5, **st)
        axes[0].fill_between(x, succ_m - succ_sd, succ_m + succ_sd, alpha=0.18, color=st.get("color"))

        axes[1].plot(x, cost_m, label=method_label(method), linewidth=2.5, markersize=6.5, **st)
        axes[1].fill_between(x, cost_m - cost_sd, cost_m + cost_sd, alpha=0.18, color=st.get("color"))

    axes[0].set_title("Success vs iid noise sigma (mean +/- std across seeds)")
    axes[0].set_xlabel("sigma")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.02, 1.05)

    axes[1].set_title("Mean episode cost vs iid noise sigma (mean +/- std across seeds)")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("mean episode cost")

    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Avoid U-trap iid value-noise: Uniform vs CEM vs CEM no warm-start", fontsize=16.5, y=0.98)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(out_dir / "comparison_success_cost_mean_std.png", dpi=200)
    plt.close(fig)

    # Matrix figure with z, Vhat and trajectories.
    cfg = TrainingConfig.from_yaml(args.config)
    cfg.task_name = "avoid"
    env = create_environment(cfg)
    try:
        coords = np.load(args.hj_dir / "grid_coordinates.npz")
        x_vec = np.asarray(coords["axis_0"], dtype=np.float32)
        y_vec = np.asarray(coords["axis_1"], dtype=np.float32)
        vx_vec = np.asarray(coords["axis_2"], dtype=np.float32)
        vy_vec = np.asarray(coords["axis_3"], dtype=np.float32)
        values = np.asarray(np.load(args.hj_dir / "value_function.npy"), dtype=np.float32)
        metadata = json.loads((args.hj_dir / "metadata.json").read_text())

        ix_vx0 = int(np.argmin(np.abs(vx_vec)))
        ix_vy0 = int(np.argmin(np.abs(vy_vec)))
        obstacle = np.asarray(metadata["obstacle_pos"], dtype=np.float32)
        goal = np.asarray(metadata["goal_pos"], dtype=np.float32)
        variant = str(metadata.get("task_variant", "u_trap"))
        mask = _obstacle_mask_xy(
            x_vec=x_vec,
            y_vec=y_vec,
            variant=variant,
            obstacle_pos=obstacle,
        )
        base_slice = values[:, :, ix_vx0, ix_vy0].T
        base_masked = np.array(base_slice, copy=True)
        base_masked[mask] = np.nan
        free_vals = base_masked[np.isfinite(base_masked)]
        vmin = float(np.percentile(free_vals, 2.0))
        vmax = float(np.percentile(free_vals, 98.0))
        if vmax <= vmin:
            vmin, vmax = float(np.min(free_vals)), float(np.max(free_vals))

        pointmass_body_id = env.task.mj_model.body("pointmass").id
        offset = np.asarray(env.task.mj_model.body_pos[pointmass_body_id, :2], dtype=np.float32)

        n_rows = len(sigmas)
        fig, axes = plt.subplots(n_rows, 5, figsize=(23, max(2.4 * n_rows, 8.5)), sharex=False, sharey=False)
        if n_rows == 1:
            axes = np.asarray([axes])

        z_im = None
        v_im = None
        traj_cols = ["uniform", "cem", "cem_no_warm_start"]
        for i, sigma in enumerate(sigmas):
            noise_seed = noise_seeds[reference_method][args.matrix_seed][sigma]
            z, vhat = resolve_noise_field(
                values=values,
                sigma=float(sigma),
                noise_seed=int(noise_seed),
                noise_mean=float(args.noise_mean),
                noise_smooth_passes=int(args.noise_smooth_passes),
                noise_distribution=str(args.noise_distribution),
            )
            z_slice = np.asarray(z[:, :, ix_vx0, ix_vy0]).T
            v_slice = np.asarray(vhat[:, :, ix_vx0, ix_vy0]).T
            z_slice = np.array(z_slice, copy=True)
            v_slice = np.array(v_slice, copy=True)
            z_slice[mask] = np.nan
            v_slice[mask] = np.nan

            ax_z = axes[i, 0]
            ax_v = axes[i, 1]
            z_im = ax_z.imshow(z_slice, origin="lower", extent=[x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]], aspect="equal", cmap="viridis")
            v_im = ax_v.imshow(v_slice, origin="lower", extent=[x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]], aspect="equal", cmap="magma", vmin=vmin, vmax=vmax)

            if i == 0:
                ax_z.set_title("z(x) slice")
                ax_v.set_title("Vhat(x) slice")
            ax_z.set_ylabel(f"sigma={sigma:g}\ny")
            ax_z.set_xlabel("x")
            ax_v.set_xlabel("x")

            for j, method in enumerate(traj_cols):
                ax_t = axes[i, 2 + j]
                run_dir = run_dirs[method][args.matrix_seed]
                qpos_path = run_dir / f"noise_{sigma:.4f}_{args.case_name}_qpos.npz"
                qpos = np.load(qpos_path)["qpos"]
                pos = qpos[: min(args.max_traj, qpos.shape[0]), :, :2] + offset[None, None, :]
                for traj in pos:
                    ax_t.plot(traj[:, 0], traj[:, 1], alpha=0.55, linewidth=1.0, color="#1f77b4")
                add_obstacle_patch(ax_t, obstacle, variant)
                ax_t.plot(goal[0], goal[1], marker="*", color="gold", markeredgecolor="k", markersize=8)
                ax_t.set_aspect("equal", adjustable="box")
                ax_t.grid(alpha=0.2)
                ax_t.set_xlabel("x")
                if i == 0:
                    ax_t.set_title(method_label(method))

        fig.suptitle(
            f"Noise/Vhat and trajectory matrix (seed={args.matrix_seed}, case={args.case_name})",
            fontsize=15,
            y=0.997,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        if z_im is not None:
            cax1 = fig.add_axes([0.92, 0.55, 0.012, 0.33])
            cbar1 = fig.colorbar(z_im, cax=cax1)
            cbar1.set_label("z")
        if v_im is not None:
            cax2 = fig.add_axes([0.92, 0.13, 0.012, 0.33])
            cbar2 = fig.colorbar(v_im, cax=cax2)
            cbar2.set_label("Vhat")
        fig.savefig(out_dir / "noise_vhat_trajectory_matrix_seed1.png", dpi=180)
        plt.close(fig)
    finally:
        env.close()

    # Report.
    lines = [
        "# IID Noise Multiseed Comparison",
        "",
        "## Seed Sanity Checks",
        "",
        f"- Same noise for all methods at fixed (seed, sigma): `{same_noise_across_methods}`.",
        f"- Different noise across sigma within each seed: `{different_noise_across_sigmas_per_seed}`.",
        f"- Different noise across seeds at fixed sigma: `{different_noise_across_seeds_per_sigma}`.",
        "",
    ]
    if mismatch_examples:
        lines.append("- Method mismatch examples:")
        for ex in mismatch_examples[:10]:
            lines.append(f"  - method={ex[0]}, seed={ex[1]}, sigma={ex[2]} ref={ex[3]} got={ex[4]}")
        lines.append("")

    lines += [
        "## Mean +/- Std Across Seeds",
        "",
        "| sigma | Uniform success | CEM success | CEM no-warm success | Uniform cost | CEM cost | CEM no-warm cost |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sigma in sigmas:
        u = summary["uniform"][sigma]
        c = summary["cem"][sigma]
        n = summary["cem_no_warm_start"][sigma]
        lines.append(
            "| {s:g} | {us:.3f} +/- {uss:.3f} | {cs:.3f} +/- {css:.3f} | {ns:.3f} +/- {nss:.3f} | {uc:.3f} +/- {ucs:.3f} | {cc:.3f} +/- {ccs:.3f} | {nc:.3f} +/- {ncs:.3f} |".format(
                s=sigma,
                us=u["success_mean"],
                uss=u["success_std"],
                cs=c["success_mean"],
                css=c["success_std"],
                ns=n["success_mean"],
                nss=n["success_std"],
                uc=u["cost_mean"],
                ucs=u["cost_std"],
                cc=c["cost_mean"],
                ccs=c["cost_std"],
                nc=n["cost_mean"],
                ncs=n["cost_std"],
            )
        )

    lines += [
        "",
        "## Figures",
        "",
        "- comparison_success_cost_mean_std.png",
        "- noise_vhat_trajectory_matrix_seed1.png",
        "",
        "## Run Directories",
        "",
    ]
    for method in methods:
        for seed in seeds:
            lines.append(f"- {method} seed {seed}: {run_dirs[method][seed]}")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "run_root": str(args.run_root),
        "methods": methods,
        "seeds": seeds,
        "sigmas": sigmas,
        "same_noise_across_methods": same_noise_across_methods,
        "different_noise_across_sigmas_per_seed": different_noise_across_sigmas_per_seed,
        "different_noise_across_seeds_per_sigma": different_noise_across_seeds_per_sigma,
        "mismatch_examples": mismatch_examples,
        "summary": summary,
        "run_dirs": {
            m: {str(seed): str(run_dirs[m][seed]) for seed in seeds}
            for m in methods
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Saved aggregate report to {out_dir}")
    print(
        "Seed checks:",
        {
            "same_noise_across_methods": same_noise_across_methods,
            "different_noise_across_sigmas_per_seed": different_noise_across_sigmas_per_seed,
            "different_noise_across_seeds_per_sigma": different_noise_across_seeds_per_sigma,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
