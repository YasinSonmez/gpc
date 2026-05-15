#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final avoid_u_trap corrected HJ-noise matrix run.
# This is a sequential runner for clarity/reproducibility.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/avoid_u_trap.yaml"
HJ_LEVEL_ROOT="${HJ_LEVEL_ROOT:-reproducibility/avoid_u_trap_latest/hj_noise_levels}"
NOISE_LABELS=("noise_0p0" "noise_0p05" "noise_0p1" "noise_0p2")

BUDGETS="16,32,64"
CHUNK_SIZES="1,2"
TAUS="0.1,0.3,1.0,3.0,5.0"
STRATEGY_PROFILES="multinomial:0.0,systematic:0.0,systematic:0.6"
SEEDS="1"
EPISODES=64
BATCH_SIZE=16
HORIZON=1.0
VALUE_ALPHA=1.0

RUN_ROOT_DEFAULT="experiments/avoid_full_corrected_noise_matrix_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${1:-$RUN_ROOT_DEFAULT}"
mkdir -p "$RUN_ROOT"

echo "Run root: $RUN_ROOT"
echo "HJ level root: $HJ_LEVEL_ROOT"

for label in "${NOISE_LABELS[@]}"; do
  hj_dir="$HJ_LEVEL_ROOT/$label/hj"
  if [[ ! -f "$hj_dir/value_function.npy" ]]; then
    echo "Missing HJ value file: $hj_dir/value_function.npy"
    echo "Run scripts/prepare_avoid_hj_noise_levels.py first."
    exit 1
  fi

  out_root="$RUN_ROOT/$label"
  shard_dir="$out_root/shard_1"
  mkdir -p "$shard_dir"

  echo "[$label] run-shard"
  uv run python scripts/chunked_spc_uniform_experiment.py run-shard \
    --config "$CONFIG" \
    --out-dir "$shard_dir" \
    --horizon "$HORIZON" \
    --budgets "$BUDGETS" \
    --chunk-sizes "$CHUNK_SIZES" \
    --taus "$TAUS" \
    --strategy-profiles "$STRATEGY_PROFILES" \
    --chunk-resample-pre=false \
    --chunk-resample-post=true \
    --chunk-resample-post-last=false \
    --hj-dir "$hj_dir" \
    --value-alpha "$VALUE_ALPHA" \
    --hj-use-base-terminal \
    --seeds "$SEEDS" \
    --episodes "$EPISODES" \
    --batch-size "$BATCH_SIZE" \
    --include-baselines

  echo "[$label] aggregate"
  uv run python scripts/chunked_spc_uniform_experiment.py aggregate \
    --run-root "$out_root" \
    --config "$CONFIG" \
    --budgets "$BUDGETS" \
    --chunk-sizes "$CHUNK_SIZES" \
    --taus "$TAUS" \
    --strategy-profiles "$STRATEGY_PROFILES" \
    --hj-dir "$hj_dir" \
    --value-alpha "$VALUE_ALPHA" \
    --hj-use-base-terminal \
    --seeds "$SEEDS" \
    --episodes "$EPISODES" \
    --shard-glob "shard_*" \
    --matrix-max-traj 16
done

printf "label,hj_dir\n" > "$RUN_ROOT/noise_hj_manifest.csv"
for label in "${NOISE_LABELS[@]}"; do
  printf "%s,%s\n" "$label" "$HJ_LEVEL_ROOT/$label/hj" >> "$RUN_ROOT/noise_hj_manifest.csv"
done

echo "Completed final matrix run under: $RUN_ROOT"
