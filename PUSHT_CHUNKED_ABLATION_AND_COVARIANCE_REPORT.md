# PushT Chunked SPC Ablation and Covariance Directions

## Data Analyzed

I aggregated the 48 completed runs under `experiments/pusht` matching
`pusht_k12_chunked_ablation_*`. Each run used one seed and one training
iteration with `num_eval_episodes: 0`. Therefore the primary metrics are
data-generation metrics from the training rollout, not final policy evaluation
metrics.

The full per-run CSV summary is saved at
`experiments/pusht/chunked_ablation_summary.csv`.

## Main Empirical Findings

The best observed run was:

```text
noise_level = 0.2
chunk_size = 6
chunk_temperature = 0.01
episode_cost = 204.33 +/- 120.61
spc_cost = 0.23
```

The next best runs were also dominated by `noise_level = 0.2`, with
`chunk_size` in `{4, 6}` and low-to-moderate temperature. Averaged over the
grid, increasing proposal noise was the largest effect:

```text
noise_level = 0.05: mean episode_cost = 334.44
noise_level = 0.10: mean episode_cost = 251.38
noise_level = 0.20: mean episode_cost = 222.64
```

Temperature also mattered:

```text
chunk_temperature = 0.01: mean episode_cost = 241.53
chunk_temperature = 0.10: mean episode_cost = 258.67
chunk_temperature = 1.00: mean episode_cost = 285.82
chunk_temperature = 10.0: mean episode_cost = 291.91
```

Chunk size was weaker as a marginal effect, but larger chunks were faster:

```text
chunk_size = 2: mean episode_cost = 273.75, mean total_time = 201.4s
chunk_size = 3: mean episode_cost = 262.62, mean total_time = 182.9s
chunk_size = 4: mean episode_cost = 273.11, mean total_time = 171.3s
chunk_size = 6: mean episode_cost = 268.46, mean total_time = 161.2s
```

The best two-factor average was also clear:

```text
noise_level = 0.2, chunk_size = 6:
mean episode_cost = 212.10
mean spc_cost = 0.24
mean total_time = 144.9s
```

## Interpretation

Within this chunked-SPC grid, the method appears sensitive to exploration
scale. The lowest noise value, `0.05`, is consistently poor. This suggests that
the particle-resampling mechanism cannot compensate for an extension proposal
distribution that is too local. Once a particle prefix is resampled, the suffix
still needs enough stochastic spread to discover useful continuations.

Low Boltzmann temperature was usually better. This means intermediate running
cost is informative on PushT: aggressively allocating suffix samples to
low-cost prefixes helps. Very high temperature approaches uniform resampling
and performs worse, consistent with the sequential Monte Carlo interpretation.

The evidence that chunking itself helps is suggestive but not yet conclusive.
Earlier one-off runs at `noise_level=0.1` had roughly:

```text
chunk_size = 12, temperature = 1.0: episode_cost ~= 275.97
chunk_size = 6,  temperature = 1.0: episode_cost ~= 229.28
chunk_size = 3,  temperature = 1.0: episode_cost ~= 236.68
```

That suggests intermediate resampling can improve over a no-effective-chunking
case. However, the current full suite did not include a matched non-chunked
K=12 baseline or K=6 baseline in the same launch, and all runs have one seed.
The rigorous conclusion is:

```text
Chunked Boltzmann SPC is promising on PushT, especially with higher extension
noise and low temperature, but the present data is not sufficient to claim a
robust advantage over matched-budget non-chunked SPC.
```

The next decisive experiment should run at least 3 seeds for:

```text
K=6 baseline, horizon=0.5
K=12 non-chunked baseline, horizon=1.0
K=12 chunked best setting: noise=0.2, chunk_size=6, temperature=0.01
K=12 chunked near-best setting: noise=0.2, chunk_size=4, temperature=0.1
```

Use `num_eval_episodes > 0`, disable or separate training videos for speed, and
report both proposal count and simulator-step budget.

## Why Trajectories Look Discrete and Kinky

There are three likely causes.

First, the current configs use `spline_type: zero` by default. This is
zero-order hold over knots, so the control is piecewise constant. With many
receding-horizon replans, piecewise-constant actions naturally look kinked.

Second, particle resampling creates ancestry switches. A final particle is a
stitched object: prefix from one ancestor, suffix from descendants selected by
Boltzmann resampling. Even if every segment is valid, the selected full control
sequence need not be globally smooth across chunk boundaries.

Third, predictive sampling has fixed isotropic noise. It does not learn which
directions in action-knot space are useful or smooth. It can repeatedly sample
high-frequency knot perturbations, especially when `noise_level` must be large
for exploration.

## Standard CEM in This Framework

Hydrax already provides diagonal-covariance CEM. Let a full knot sequence be

```text
U in R^(K x d)
```

and let the proposal distribution at optimization iteration `r` be

```text
q_r(U) = N(U; mu_r, diag(sigma_r^2)).
```

For samples `U_i ~ q_r` with rollout cost `J_i`, CEM chooses the elite set

```text
E_r = indices of the L lowest J_i.
```

Then it updates

```text
mu_{r+1} = (1 / L) sum_{i in E_r} U_i
sigma_{r+1} = max(std({U_i : i in E_r}), sigma_min).
```

In the current `ChunkedPAC` implementation, this already composes with CEM in a
basic way: set `controller_type: cem`, and the chunked rollout returns full
stitched trajectories to `CEM.update_params`. CEM then fits `mu` and diagonal
`sigma` to the best final trajectories.

The limitation is that this is final-trajectory CEM, not truly chunk-local CEM.
The covariance is updated only after the chunked particle rollout completes.
It does not adapt the suffix proposal covariance immediately after each
intermediate Boltzmann selection.

## Chunked CEM: Rigorous Formulation

A more principled chunked CEM would maintain a proposal over segments. Let
chunk `m` contain knot block `U^(m)`. A factorized proposal is

```text
q(U) = product_{m=0}^{M-1} q_m(U^(m))
q_m(U^(m)) = N(U^(m); mu_m, Sigma_m).
```

During chunked rollout, particle `i` has prefix cost `C_i^m`. Define either
elite indicators

```text
e_i^m = 1{i is among the L lowest C_i^m}
```

or soft weights

```text
w_i^m = exp(-(C_i^m - min_j C_j^m) / tau_cem)
w_i^m <- w_i^m / sum_j w_j^m.
```

Then update the segment proposal by weighted maximum likelihood:

```text
mu_m <- sum_i w_i^m U_i^(m)
Sigma_m <- sum_i w_i^m (U_i^(m) - mu_m)(U_i^(m) - mu_m)^T + epsilon I.
```

For diagonal covariance, keep only the diagonal of `Sigma_m`. For smoothness,
use a structured covariance:

```text
Sigma_m = D_m K_m D_m
K_m[a,b] = exp(-|t_a - t_b|^2 / (2 ell^2)).
```

Here `K_m` couples nearby knots and `ell` controls temporal smoothness. Sampling
then becomes

```text
U_i^(m) = mu_m + D_m L_m z_i,
z_i ~ N(0, I),
L_m L_m^T = K_m.
```

This is the cleanest way to keep both mean and covariance while avoiding
independent, kink-producing perturbations at each knot.

## Promising Directions

1. **Use CEM as the base controller immediately.**
   This requires no new algorithmic machinery. Configure
   `controller_type: cem`, choose `num_elites`, `sigma_start`, and `sigma_min`,
   and run the same chunked wrapper. It tests whether covariance adaptation
   alone improves smoothness and cost.

2. **Use soft-CEM rather than hard elites.**
   Replace the elite set by Boltzmann weights over final or prefix costs. This
   interpolates between MPPI and CEM:

   ```text
   mu <- sum_i w_i U_i
   Sigma <- sum_i w_i (U_i - mu)(U_i - mu)^T + epsilon I.
   ```

   It should be less brittle than hard top-`L` selection when costs are noisy.

3. **Sample temporally correlated noise.**
   Instead of independent knot noise, sample from a Gaussian process or
   low-pass basis over time. This directly targets the kinkiness. A simple
   implementation is an RBF covariance over knot index, or sampling in a DCT /
   B-spline basis with fewer coefficients than knots.

4. **Optimize increments rather than absolute controls.**
   Let the sampled variable be `Delta U`, with

   ```text
   U_k = U_{k-1} + Delta U_k.
   ```

   Penalize or bound `Delta U`. This makes smoothness an invariant of the
   parameterization rather than an after-the-fact penalty.

5. **Add a control variation penalty to the rollout cost.**
   Use

   ```text
   J_smooth = J_task + lambda_1 sum_k ||u_k - u_{k-1}||^2
                    + lambda_2 sum_k ||u_{k+1} - 2u_k + u_{k-1}||^2.
   ```

   This is simple but changes the objective, so it should be reported
   separately from pure task-cost improvements.

6. **Use resample-move particles.**
   After Boltzmann resampling, duplicate particles reduce diversity. Add a
   "move" step by perturbing suffix controls with a smooth covariance kernel.
   This is standard sequential Monte Carlo logic: resampling focuses mass,
   mutation restores diversity.

7. **Use spline interpolation intentionally.**
   Test `spline_type: linear` and `spline_type: cubic`. This is the cheapest
   smoothness experiment. It may improve videos immediately, although it does
   not solve covariance adaptation.

## Recommended Next Experiment Suite

The next suite should be smaller and more diagnostic than the 48-run grid:

```text
common:
  num_seeds: 3
  num_eval_episodes: 8
  record_training_videos: false
  record_eval_videos: true

baselines:
  K=6 predictive_sampling, horizon=0.5
  K=12 predictive_sampling, horizon=1.0

chunked predictive sampling:
  noise_level=0.2, chunk_size=6, chunk_temperature=0.01
  noise_level=0.2, chunk_size=4, chunk_temperature=0.1

chunked CEM:
  controller_type=cem
  num_samples=256
  num_elites in {16, 32, 64}
  sigma_start in {0.2, 0.5, 1.0}
  sigma_min in {0.02, 0.05}
  chunk_size in {4, 6}
  chunk_temperature in {0.01, 0.1}

smoothness:
  spline_type in {zero, linear, cubic}
```

The evaluation metrics should include:

```text
episode_cost
evaluation_cost_mean
evaluation_cost_std
spc_cost
wall_clock_time
mean ||u_k - u_{k-1}||^2
mean ||u_{k+1} - 2u_k + u_{k-1}||^2
```

The last two metrics are important because "smoother" must be measured, not
only inspected visually.

## Bottom Line

The current ablation supports three working hypotheses:

```text
1. Chunked Boltzmann resampling is useful only if suffix proposals remain
   sufficiently exploratory.
2. PushT intermediate costs are informative enough that low-temperature
   resampling can help.
3. The current predictive-sampling proposal is too rough; smooth covariance
   structure or CEM-style covariance adaptation is the next most promising
   improvement.
```

The most practical next step is to test chunked CEM and linear/cubic splines
before implementing a custom covariance-kernel CEM. If standard CEM already
improves smoothness and cost, then a chunk-local soft-CEM with temporally
correlated covariance is the clean generalization.

## References

- Cross-Entropy Method MPC and improved CEM variants such as iCEM motivate
  elite-based moment fitting, temporally correlated noise, and memory.
- MPPI and path-integral policy improvement motivate Boltzmann weighting over
  costs rather than hard elite truncation.
- Covariance-adaptive path-integral methods such as PI2-CMA and covariance
  scheduling for MPPI motivate learning or scheduling the proposal covariance
  instead of hand-tuning isotropic noise.
