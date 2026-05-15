#!/usr/bin/env python3
"""Prepare deterministic avoid_u_trap HJ value-noise directories.

This script recreates the HJ value-function folders used by the final
chunked-SPC noise matrix runs from a single base HJ artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpc.config import TrainingConfig
from run_experiment import create_environment
from scripts.avoid_value_guidance_experiment import load_hj_artifacts, perturb_hj_values


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated int list.")
    return values


def sigma_label(sigma: float) -> str:
    return str(sigma).replace("-", "m").replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-hj-dir",
        type=Path,
        default=Path("reproducibility/avoid_u_trap_latest/hj_base"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/avoid_u_trap.yaml"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("reproducibility/avoid_u_trap_latest/hj_noise_levels"),
    )
    parser.add_argument("--noise-levels", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument(
        "--noise-seeds",
        type=str,
        default="0,1123,2123,3123",
        help="One seed per noise level, same order as --noise-levels.",
    )
    parser.add_argument("--noise-mean", type=float, default=0.1)
    parser.add_argument("--noise-smooth-passes", type=int, default=4)
    parser.add_argument(
        "--noise-distribution",
        type=str,
        default="softplus_gaussian",
        choices=["softplus_gaussian", "lognormal", "clipped_gaussian", "blend_correlated"],
    )
    args = parser.parse_args()

    noise_levels = parse_float_list(args.noise_levels)
    noise_seeds = parse_int_list(args.noise_seeds)
    if len(noise_levels) != len(noise_seeds):
        raise ValueError("--noise-levels and --noise-seeds must have the same length.")

    args.out_root.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig.from_yaml(args.config)
    config.task_name = "avoid"
    config.task_variant = "u_trap"
    config.method = "gpc"
    config.num_policy_samples = 0
    config.use_wandb = False

    env = create_environment(config)
    try:
        base_artifacts = load_hj_artifacts(args.base_hj_dir, config, env)
    finally:
        env.close()

    mapping: dict[str, str] = {}
    for sigma, noise_seed in zip(noise_levels, noise_seeds, strict=True):
        label = f"noise_{sigma_label(sigma)}"
        out_hj_dir = args.out_root / label / "hj"
        out_hj_dir.mkdir(parents=True, exist_ok=True)

        if abs(sigma) < 1e-12:
            values = np.asarray(base_artifacts.values, dtype=np.float32)
            metadata = dict(base_artifacts.metadata)
            metadata.update(
                {
                    "value_noise_model": "none",
                    "value_noise_distribution": "none",
                    "value_noise_sigma": 0.0,
                    "value_noise_mean": 0.0,
                    "value_noise_smooth_passes": 0,
                    "value_noise_seed": int(noise_seed),
                }
            )
        else:
            noisy = perturb_hj_values(
                artifacts=base_artifacts,
                config=config,
                noise_sigma=float(sigma),
                noise_seed=int(noise_seed),
                noise_mean=float(args.noise_mean),
                smooth_passes=int(args.noise_smooth_passes),
                noise_distribution=str(args.noise_distribution),
            )
            values = np.asarray(noisy.values, dtype=np.float32)
            metadata = dict(noisy.metadata)
            metadata["value_noise_seed"] = int(noise_seed)

        np.save(out_hj_dir / "value_function.npy", values)
        np.savez(
            out_hj_dir / "grid_coordinates.npz",
            domain_lo=np.asarray(base_artifacts.grid.domain.lo),
            domain_hi=np.asarray(base_artifacts.grid.domain.hi),
            axis_0=np.asarray(base_artifacts.grid.coordinate_vectors[0]),
            axis_1=np.asarray(base_artifacts.grid.coordinate_vectors[1]),
            axis_2=np.asarray(base_artifacts.grid.coordinate_vectors[2]),
            axis_3=np.asarray(base_artifacts.grid.coordinate_vectors[3]),
        )
        (out_hj_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        mapping[f"{sigma:g}"] = str(out_hj_dir)
        print(f"Wrote {label}: sigma={sigma:g}, seed={noise_seed}", flush=True)

    (args.out_root / "noise_hj_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote mapping: {args.out_root / 'noise_hj_mapping.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
