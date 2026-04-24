#!/usr/bin/env python3
"""Run ablation experiments with multiple seeds.

Usage:
    python scripts/run_ablation.py configs/ablations/walker_ablation.yaml
    python scripts/run_ablation.py configs/ablations/walker_ablation.yaml \
        --dry-run
    python scripts/run_ablation.py configs/ablations/walker_ablation.yaml \
        --variants v0_baseline v1_replay
    python scripts/run_ablation.py configs/ablations/walker_ablation.yaml \
        --seeds 1 --num-iters 5
"""
import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def load_ablation_config(path: str) -> dict:
    """Load ablation configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_variant_config(
    base_config_path: str,
    global_overrides: dict,
    overrides: dict,
    seed: int,
) -> dict:
    """Create a merged config for one variant and seed."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    
    # Apply global overrides (from top level of ablation config)
    for key, value in global_overrides.items():
        if key not in [
            "base_config",
            "num_seeds",
            "num_iters",
            "variants",
            "grid",
        ]:
            config[key] = value

    # Apply variant-specific overrides (skip 'description' key)
    for key, value in overrides.items():
        if key != "description":
            config[key] = value
    
    # Set seed
    config["seed"] = seed
    
    return config


def expand_variants(ablation: dict) -> dict:
    """Expand explicit variants plus an optional Cartesian-product grid."""
    variants = dict(ablation.get("variants", {}))
    grid = ablation.get("grid", {})
    if not grid:
        return variants

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo, strict=True))
        variant_name = "_".join(
            f"{key}{str(value).replace('.', 'p')}"
            for key, value in overrides.items()
        )
        overrides["description"] = ", ".join(
            f"{key}={value}" for key, value in overrides.items()
        )
        variants[variant_name] = overrides

    return variants


def run_experiment(
    config: dict,
    exp_name: str,
    output_dir: str,
    dry_run: bool = False,
    parallel: bool = False,
) -> bool:
    """Run a single experiment with the given config."""
    # Write temporary config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
        temp_config_path = f.name
    
    # Build command
    cmd = [
        sys.executable,
        "run_experiment.py",
        "train",
        "--config",
        temp_config_path,
        "--exp-name",
        exp_name,
        "--output-dir",
        output_dir,
    ]
    
    env = os.environ.copy()
    if parallel:
        # Disable pre-allocation to allow multiple processes on one GPU
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True
    
    print(f"  Starting: {exp_name}")
    try:
        result = subprocess.run(cmd, check=True, env=env)
        print(f"  Finished: {exp_name}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Experiment {exp_name} failed with code {e.returncode}")
        return False
    finally:
        Path(temp_config_path).unlink(missing_ok=True)


def main() -> int:  # noqa: PLR0915
    """Run all ablation variants."""
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("config", type=str, help="Path to ablation config YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None, help="Run only these variants"
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="Override number of seeds"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Run one explicit seed"
    )
    parser.add_argument(
        "--num-iters", type=int, default=None, help="Override num_iters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of experiments to run in parallel",
    )
    parser.add_argument(
        "--noise-level",
        "--noise_level",
        dest="noise_level",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        "--chunk_size",
        dest="chunk_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--chunk-temperature",
        "--chunk_temperature",
        dest="chunk_temperature",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    
    # Load ablation config
    ablation = load_ablation_config(args.config)
    base_config_path = ablation["base_config"]
    num_seeds = args.seeds or ablation.get("num_seeds", 3)
    seed_values = (
        [args.seed] if args.seed is not None else list(range(num_seeds))
    )
    num_iters = args.num_iters or ablation.get("num_iters", 20)
    variants = expand_variants(ablation)

    # W&B sweep mode: `${args}` can pass these scalar overrides directly.
    cli_overrides = {
        key: value
        for key, value in {
            "noise_level": args.noise_level,
            "chunk_size": args.chunk_size,
            "chunk_temperature": args.chunk_temperature,
        }.items()
        if value is not None
    }
    if cli_overrides:
        variants = {
            "wandb_sweep": {
                **cli_overrides,
                "description": ", ".join(
                    f"{k}={v}" for k, v in cli_overrides.items()
                ),
            }
        }
    
    # Filter variants if specified
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    
    # Get ablation name from config filename
    ablation_name = Path(args.config).stem
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_name}")
    print(f"Base config: {base_config_path}")
    print(f"Variants: {len(variants)}")
    print(f"Seeds per variant: {len(seed_values)}")
    print(f"Iterations: {num_iters}")
    print(f"Parallel: {args.parallel}")
    print(f"Total experiments: {len(variants) * len(seed_values)}")
    print(f"{'='*60}\n")
    
    # List variants
    for name, overrides in variants.items():
        desc = overrides.get("description", "")
        print(f"  {name}: {desc}")
    print()
    
    # Prepare all experiments
    all_experiments = []
    for variant_name, overrides in variants.items():
        for seed in seed_values:
            exp_name = f"{ablation_name}_{variant_name}_seed{seed}"
            config = create_variant_config(
                base_config_path, ablation, overrides, seed
            )
            config["num_iters"] = num_iters
            config["experiment_name"] = exp_name
            all_experiments.append((config, exp_name))

    # Run experiments
    success_count = 0
    total_count = len(all_experiments)
    
    if args.parallel > 1:
        print(
            f"Running {total_count} experiments "
            f"with parallelism {args.parallel}..."
        )
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_exp = {
                executor.submit(
                    run_experiment,
                    config,
                    exp_name,
                    args.output_dir,
                    args.dry_run,
                    parallel=True,
                ): exp_name
                for config, exp_name in all_experiments
            }
            for future in as_completed(future_to_exp):
                exp_name = future_to_exp[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as exc:
                    print(
                        f"  Experiment {exp_name} "
                        f"generated an exception: {exc}"
                    )
    else:
        for config, exp_name in all_experiments:
            if run_experiment(config, exp_name, args.output_dir, args.dry_run):
                success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{total_count} experiments")
    print(f"{'='*60}")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
