#!/usr/bin/env bash
set -euo pipefail

# Run Uniform/CEM/CEM-no-warm iid-noise sweeps in parallel across seeds,
# then aggregate into comparison plots/report.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/avoid_u_trap.yaml"
HJ_DIR="experiments/avoid/hj_avoid_binary_u_trap_20260502_131354/hj"
SIGMAS="1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2,5e2,1e3"
SEEDS=(1 2 3 4 5)
METHODS=(uniform cem cem_no_warm_start)

EPISODES=64
BATCH_SIZE=16
CEM_NUM_SAMPLES=256
CEM_ITERATIONS=4
NOISE_MEAN=0.1
NOISE_SEED=123
NOISE_SMOOTH=0
NOISE_DIST="softplus_gaussian"

RUN_ROOT_DEFAULT="experiments/avoid_iid_noise_multiseed_parallel_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${1:-$RUN_ROOT_DEFAULT}"

mkdir -p "$RUN_ROOT/runs"

run_method() {
  local method="$1"
  local log_file="$RUN_ROOT/log_${method}.txt"
  local extra_args=()
  if [[ "$method" == "cem" || "$method" == "cem_no_warm_start" ]]; then
    extra_args=(--num-samples "$CEM_NUM_SAMPLES" --iterations "$CEM_ITERATIONS")
  fi
  : > "$log_file"
  echo "[$method] starting seed loop" | tee -a "$log_file"
  for seed in "${SEEDS[@]}"; do
    local out_dir="$RUN_ROOT/runs/${method}/seed_${seed}"
    mkdir -p "$out_dir"
    echo "[$method] seed=${seed} -> $out_dir" | tee -a "$log_file"
    uv run python scripts/avoid_value_guidance_experiment.py \
      --config "$CONFIG" \
      --hj-dir "$HJ_DIR" \
      --out-dir "$out_dir" \
      --episodes "$EPISODES" \
      --batch-size "$BATCH_SIZE" \
      --seed "$seed" \
      --controller-type "$method" \
      "${extra_args[@]}" \
      --cases spc_hjV \
      --noise-analysis \
      --noise-levels "$SIGMAS" \
      --noise-mean "$NOISE_MEAN" \
      --noise-seed "$NOISE_SEED" \
      --noise-seed-mode per_seed_sigma \
      --noise-smooth-passes "$NOISE_SMOOTH" \
      --noise-distribution "$NOISE_DIST" \
      >> "$log_file" 2>&1
  done
  echo "[$method] done" | tee -a "$log_file"
}

run_method "uniform" &
PID_UNIFORM=$!
run_method "cem" &
PID_CEM=$!
run_method "cem_no_warm_start" &
PID_NOWARM=$!

echo "Launched parallel method workers:" \
  "uniform=$PID_UNIFORM" \
  "cem=$PID_CEM" \
  "cem_no_warm_start=$PID_NOWARM"

# Verify all three workers are active right after launch.
ps -p "${PID_UNIFORM},${PID_CEM},${PID_NOWARM}" -o pid,stat,etime,cmd

wait "$PID_UNIFORM"
wait "$PID_CEM"
wait "$PID_NOWARM"

echo "All three method workers completed. Aggregating plots/report..."

uv run python scripts/aggregate_iid_noise_multiseed.py \
  --run-root "$RUN_ROOT" \
  --config "$CONFIG" \
  --hj-dir "$HJ_DIR" \
  --methods "uniform,cem,cem_no_warm_start" \
  --seeds "1,2,3,4,5" \
  --sigmas "$SIGMAS" \
  --matrix-seed 1 \
  --case-name spc_hjV \
  --noise-mean "$NOISE_MEAN" \
  --noise-smooth-passes "$NOISE_SMOOTH" \
  --noise-distribution "$NOISE_DIST"

echo "Done. Outputs under: $RUN_ROOT"
