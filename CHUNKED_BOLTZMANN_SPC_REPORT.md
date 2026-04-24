# Chunked Boltzmann SPC Report

## Codebase Summary

This repository implements Generative Predictive Control (GPC): a training
loop that alternates between sampling-based predictive control (SPC) and
supervised flow-matching policy learning. Hydrax supplies the base SPC
algorithms and MuJoCo/MJX rollout machinery; this codebase supplies task
wrappers, policy networks, configuration-driven experiments, logging,
visualization, replay buffers, and policy-augmented SPC.

The main experiment entry point is `run_experiment.py`. It loads a
`TrainingConfig` from YAML, constructs an environment, builds a Hydrax
controller (`PredictiveSampling`, `CEM`, `Evosax`, or `MPPI`), wraps it in
`PolicyAugmentedController` or `ChunkedPAC`, constructs a denoising policy
network, and calls `gpc.train_config.train_with_config`.

The key control abstraction is `PolicyAugmentedController` in
`gpc/augmented.py`. It preserves Hydrax's `SamplingBasedController` API while
adding learned policy samples to the base SPC samples. During each MPC step,
`gpc.training.simulate_episode` obtains policy proposals, stores them in
`PACParams.policy_samples`, calls `ctrl.optimize`, records proposal costs, and
advances the simulator either with the policy proposal or with the best
optimized sequence.

PushT is implemented in `gpc/envs/pusht.py`. `GPCPushT` refines the Hydrax
PushT cost, and `PushTEnv` defines reset and observation logic. The comparison
configs of interest are `configs/pusht_k6_baseline.yaml`,
`configs/pusht_k12_baseline.yaml`, and `configs/pusht_k12_chunked.yaml`.

## Proposed Algorithm

Let `N` be the number of particles, `H` the number of control steps in the
planning horizon, `K` the number of action knots, `I` the number of SPC
iterations, and `M = ceil(K / chunk_size)` the number of chunks. Each particle
represents a partial trajectory prefix. At chunk boundary `m`, particle `i`
has endpoint state `x_i^m`, cumulative cost `C_i^m`, and prefix controls.

For costs, define Boltzmann weights

```text
w_i^m = exp(-(C_i^m - min_j C_j^m) / tau)
p_i^m = w_i^m / sum_j w_j^m,
```

where `tau > 0` is the chunk temperature. The subtraction of the minimum cost
is a numerical stabilization and does not change the categorical distribution.

One optimization iteration is:

1. Initialize `N` particles at the current MPC state.
2. For chunk `m = 0`, sample one proposal segment per particle and roll out
   those segments.
3. For each later chunk `m > 0`, sample `N` parent indices from the previous
   particle population using `p_i^(m-1)`.
4. From each selected parent endpoint, sample one fresh proposal segment and
   roll it out.
5. After every extension, including the last one, resample the population using
   the updated cumulative costs.
6. Add terminal costs, return the resulting `N` full trajectories and their
   costs, and let the base controller update its parameters from this normal
   Hydrax `Trajectory` batch.

This is a sequential Monte Carlo view of SPC. Ordinary SPC samples complete
open-loop candidates and only selects at the end. Chunked Boltzmann SPC applies
selection pressure at intermediate horizons, so low-cost prefixes receive more
offspring and high-cost prefixes stop consuming later-horizon simulation
budget.

## Why It Could Help

For long horizons, naive full-horizon sampling can waste most samples on bad
early prefixes. If early costs are informative about eventual success, chunked
resampling increases the effective number of samples allocated to promising
regions of the state-control tree. This can make longer horizons or more knots
usable without increasing the number of complete particles.

The method is most likely to help when:

- Early prefix cost is correlated with final trajectory quality.
- The task has branching structure where committing early to a good contact or
  approach mode matters, as in PushT.
- The base SPC sampler has enough local diversity that resampled endpoints can
  be extended in meaningfully different ways.

It can fail or hurt when:

- Early costs are misleading and good long-horizon trajectories require a bad
  prefix.
- `tau` is too small, causing particle impoverishment and duplicate branches.
- `tau` is too large, making resampling nearly uniform and reducing the method
  to ordinary random allocation.
- The chunk size is too short, so costs are too noisy, or too long, so the
  method approaches ordinary full-horizon SPC.

## Budget Accounting

For a fixed horizon, ordinary SPC with `N` samples, `I` iterations, and `R`
domain randomizations costs approximately

```text
R * I * N * H
```

MJX model steps per MPC decision. Chunked Boltzmann SPC evaluates `N`
particles over all chunks, whose segment lengths sum to `H`, so it has the
same leading simulator-step budget:

```text
R * I * N * (H_1 + ... + H_M) = R * I * N * H.
```

Thus the method does not make a fixed long-horizon rollout cheaper. It changes
the allocation of later-horizon evaluations toward particles with better
intermediate cost. When comparing `K=6, horizon=0.5s` to `K=12, horizon=1.0s`,
the equal `num_samples * iterations` configs are equal in proposal count but
not equal in simulator-step budget; the longer horizon costs about twice as
many dynamics steps. A strict compute-matched comparison should either double
the short-horizon sample/iteration budget or report both proposal budget and
simulator-step budget.

## Implementation Plan

The implementation should be controller-level and task-agnostic:

1. Add `gpc.chunked_spc.ChunkedPAC` as a `PolicyAugmentedController` subclass.
   This preserves the `init_params`, `optimize`, `get_action_sequence`, and
   rollout return contract expected by `simulate_episode` and
   `train_with_config`.
2. Reuse `sample_knots` from `PolicyAugmentedController`, so the extension is
   plug-and-play on top of Hydrax SPC algorithms and still accepts learned
   policy proposals and `exploration_floor`.
3. Override only the rollout generation inside `optimize`: sample proposal
   knots, interpolate them to full controls, apply only the current chunk from
   each candidate, resample endpoints with Boltzmann weights, and assemble a
   standard Hydrax `Trajectory`.
4. Keep final parameter learning delegated to `base_ctrl.update_params` via
   `PolicyAugmentedController.update_params`, so each base algorithm still
   decides how to update its sampling distribution.
5. Validate `chunk_size >= 1` and `chunk_temperature > 0` in `TrainingConfig`
   and in the controller constructor.
6. Test the Boltzmann sampler, config validation, JIT compatibility, returned
   trajectory shapes, finite costs, and compatibility with a small Hydrax
   `PredictiveSampling` controller on the Particle task.

## PushT Experiment Plan

Primary commands:

```bash
uv run python run_experiment.py train --config configs/pusht_k6_baseline.yaml
uv run python run_experiment.py train --config configs/pusht_k12_baseline.yaml
uv run python run_experiment.py train --config configs/pusht_k12_chunked.yaml
```

Recommended first sweep:

- Keep `num_samples=256`, `iterations=4`, `num_envs=16`.
- For `K=12`, compare ordinary full-horizon SPC to chunked SPC with
  `chunk_size=6`, then sweep `chunk_size in {3, 4, 6}`.
- Sweep `chunk_temperature in {0.1, 0.3, 1.0, 3.0}`.
- Report episode cost, SPC proposal cost, final evaluation cost, wall-clock
  time, and simulator-step budget.

The success criterion is not merely lower proposal cost. The method is useful
if the `K=12` chunked controller yields better episode/evaluation cost than
the `K=12` non-chunked baseline at the same long-horizon simulator budget, and
if the longer-horizon chunked controller beats or matches the `K=6` baseline
under a clearly stated compute budget.
