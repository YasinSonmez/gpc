# Avoid U-trap Reproducibility (Latest HJ + Chunked SPC)

This folder captures the exact artifact and commands used for the latest `avoid_u_trap` runs involving:

- uniform sampling baselines,
- chunked resampling method sweeps,
- HJ value-function augmentation,
- HJ value-noise scaling experiments.

## Canonical HJ Value Function

Base HJ artifact used in recent runs is copied here:

- `reproducibility/avoid_u_trap_latest/hj_base/value_function.npy`
- `reproducibility/avoid_u_trap_latest/hj_base/grid_coordinates.npz`
- `reproducibility/avoid_u_trap_latest/hj_base/metadata.json`

Reference checksums are in:

- `reproducibility/avoid_u_trap_latest/reference_checksums.json`

## 1) Recreate the Final Noise-Level HJ Artifacts

```bash
uv run python scripts/prepare_avoid_hj_noise_levels.py \
  --base-hj-dir reproducibility/avoid_u_trap_latest/hj_base \
  --out-root reproducibility/avoid_u_trap_latest/hj_noise_levels \
  --noise-levels 0.0,0.05,0.1,0.2 \
  --noise-seeds 0,1123,2123,3123 \
  --noise-mean 0.1 \
  --noise-smooth-passes 4 \
  --noise-distribution softplus_gaussian
```

Optional checksum check (should match `reference_checksums.json`):

```bash
sha256sum reproducibility/avoid_u_trap_latest/hj_noise_levels/noise_0p*/hj/value_function.npy
```

## 2) Reproduce the Final Chunked Matrix (Uniform + HJ Value)

```bash
bash scripts/run_avoid_full_corrected_noise_matrix.sh
```

This runs all four HJ-noise folders with:

- controller: `uniform`
- budgets: `16,32,64`
- chunk sizes: `1,2`
- taus: `0.1,0.3,1.0,3.0,5.0`
- profiles: `multinomial:0.0,systematic:0.0,systematic:0.6`
- resampling schedule: pre=`false`, post=`true`, post_last=`false`
- `value_alpha=1.0`, `hj_use_base_terminal=true`
- seed `1`, episodes `64`

## 3) Reproduce HJ Noise-Scaling Controller Comparisons

Primary scripts:

- `scripts/run_iid_noise_multiseed_parallel.sh`
- `scripts/run_iid_noise_blend_seed123_smallsigmas_parallel.sh`
- `scripts/aggregate_iid_noise_multiseed.py`

Example:

```bash
bash scripts/run_iid_noise_blend_seed123_smallsigmas_parallel.sh
```

## 4) Minimal Verification Targets

After step (2), verify each noise folder has:

- `shard_1/results_shard.json`
- `combined_report/summary.json`
- `combined_report/report.md`

After step (3), verify each run root has:

- `combined_report/comparison_success_cost_mean_std.png`
- `combined_report/report.md`
