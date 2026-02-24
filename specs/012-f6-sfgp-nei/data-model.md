# Data Model: F6 Week 7 — SFGP Matérn-1.5 + NEI

**Feature**: 012-f6-sfgp-nei
**Date**: 2026-02-24

---

## E-01: Week 7 Raw Data

| Field | Type | Constraints |
|-------|------|-------------|
| X_raw | ndarray (27, 5) | All values in [0, 1] |
| y_raw | ndarray (27,) | All-negative reals; range [-2.571, -0.206] |

**Source**: `data/f6/updated_inputs - Week 7.npy`, `data/f6/updated_outputs - Week 7.npy`
**Relationships**: Input to E-02 (GP model — `Standardize(m=1)` handles transform internally)

---

## E-02: Training Tensors

| Field | Type | Constraints |
|-------|------|-------------|
| X_train | Tensor (27, 5) | `torch.tensor(X_raw, dtype=torch.double)` |
| Y_train | Tensor (27, 1) | `torch.tensor(y_raw, dtype=torch.double).unsqueeze(-1)` |

**Note**: No manual transform needed. Raw outputs are passed directly — `Standardize(m=1)` inside `SingleTaskGP` handles the z-scoring internally. This is the key difference from F5 (which required manual log1p + z-score).

**Validation**: `X_train.min() >= 0.0`, `X_train.max() <= 1.0`, `Y_train.max() < 0` (all negative)
**Relationships**: Input to E-03 (GP model)

---

## E-03: GP Surrogate Model

| Field | Type | Description |
|-------|------|-------------|
| kernel | ScaleKernel(MaternKernel) | **nu=1.5**, ARD with 5 lengthscales |
| likelihood | GaussianLikelihood | noise_constraint=GreaterThan(1e-2) |
| model | SingleTaskGP | Default `Standardize(m=1)` outcome transform; fitted on (X_train, Y_train) |

**Initial Hyperparameters**:
| Parameter | Init Value | Learnable | Notes |
|-----------|-----------|----------|-------|
| ℓ₁–ℓ₅ (lengthscales) | 0.5 | Yes | Broader uncertainty for exploration |
| σ²_f (output scale) | 1.0 | Yes | Matches standardised variance |
| σ²_n (noise) | 0.2 | Yes | 20% of standardised Var(y)≈1.0; aggressive exploration init; see RES-003 |

**Critical note**: Noise init is `0.2` (not `0.1 * y_raw.var()` = 0.033) because `Standardize(m=1)` ensures internal training targets have Var≈1.0, and the higher init discourages the exact-interpolation local minimum that caused the x4=0 boundary trap.

**Relationships**: Trained via E-04 (MLL); input to E-05 (acquisition); posterior used for E-08 (visualisation)

---

## E-04: MLL Training State

| Field | Type | Description |
|-------|------|-------------|
| N_RESTARTS | int | 15 random restarts |
| best_loss | float | Lowest negative MLL across restarts |
| best_model | SingleTaskGP | deepcopy of best restart |

**State Transition**: For each restart seed 0..14: construct → init HPs (ℓ=0.5, noise=0.2, outputscale=1.0) → fit_gpytorch_mll → score → keep if best

---

## E-05: Fitted Hyperparameters

| Parameter | Accessor | Format |
|-----------|----------|--------|
| ℓ₁ | `model.covar_module.base_kernel.lengthscale[0,0]` | 6 decimal places |
| ℓ₂ | `model.covar_module.base_kernel.lengthscale[0,1]` | 6 decimal places |
| ℓ₃ | `model.covar_module.base_kernel.lengthscale[0,2]` | 6 decimal places |
| ℓ₄ | `model.covar_module.base_kernel.lengthscale[0,3]` | 6 decimal places |
| ℓ₅ | `model.covar_module.base_kernel.lengthscale[0,4]` | 6 decimal places |
| σ²_f | `model.covar_module.outputscale` | 6 decimal places |
| σ²_n | `model.likelihood.noise[0]` | 6 decimal places |

**Relationships**: Extracted from E-03 after training; reported in output; ℓ values determine E-08 (visualisation dims)

---

## E-06: NEI Acquisition Function

| Field | Type | Description |
|-------|------|-------------|
| acq_fn | qLogNoisyExpectedImprovement | q=4, prune_baseline=True |
| sampler | SobolQMCNormalSampler | sample_shape=torch.Size([512]) |
| bounds | Tensor (2, 5) | [[0.01,0.01,0.01,0.01,0.10],[1,1,1,1,1]] — feasibility-constrained |
| num_restarts | int | 50 |
| raw_samples | int | 3000 |

**Relationships**: Uses E-03 (fitted model); produces E-07 (candidates)

---

## E-07: Candidate Points

| Field | Type | Constraints |
|-------|------|-------------|
| candidates | Tensor (4, 5) | All values in [0, 0.999999]; q=4 batch |
| posterior_means | Tensor (4,) | In **original** space (auto-untransformed by Standardize) |
| distances | Tensor (4,) | Min Euclidean distance from each candidate to X_train |
| best_idx | int | Distance-based: farthest from data among candidates with mean ≥ median(means) |
| best_point | Tensor (5,) | The selected candidate for submission |

**Selection Logic**: (1) Filter candidates where posterior_mean ≥ median(posterior_means), (2) among filtered, select the one farthest from training data (max min-distance to X_train). This balances exploitation (above-median predicted quality) with exploration (spatial diversity).

**Note**: Unlike F5, posterior means are already in original space — no manual `expm1` inverse transform needed.

**Relationships**: Input to E-09 (submission query) and E-08 (visualisation overlay)

---

## E-08: Surrogate Visualisation

| Field | Type | Description |
|-------|------|-------------|
| top2_dims | tuple(int, int) | 2 dims with shortest lengthscales (most important) |
| fix_dims | list[int] | 3 dims fixed at best_point values |
| grid_res | int | 80 × 80 grid for contour |
| mean_surface | ndarray (80, 80) | GP posterior mean (original scale — automatic) |
| std_surface | ndarray (80, 80) | GP posterior std (original scale — automatic) |
| relevance | ndarray (5,) | 1/ℓ for each dim (normalised) |

**Note**: `model.posterior(X_grid)` returns values in original space automatically via `Standardize(m=1)` untransform. No manual inverse transform needed (contrast with F5's `expm1` requirement).

**Relationships**: Uses E-03 (model), E-05 (lengthscales for dim selection), E-07 (best_point for overlay)

---

## E-09: Submission Query

| Field | Type | Constraints |
|-------|------|-------------|
| query_string | str | Format: `x1-x2-x3-x4-x5` |
| components | list[str] | 5 values, each `0.xxxxxx` (6 decimal places) |
| values | list[float] | Each in [0.0, 0.999999] after clamping |

**Validation**: len(parts) == 5, all parts parseable as float, all values in [0, 1]
**Clamping**: `torch.clamp(candidates, 0.0, 0.999999)` before formatting

---

## E-10: Convergence Data

| Field | Type | Description |
|-------|------|-------------|
| running_best | ndarray (27,) | `np.maximum.accumulate(y_raw)` |
| boundary | float | 26.5 (Week 6→7 boundary at sample 26/27) |

**Relationships**: Uses E-01 (raw outputs)
