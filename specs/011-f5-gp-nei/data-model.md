# Data Model: F5 Week 7 — GP Matérn-5/2 + NEI

**Feature**: 011-f5-gp-nei
**Date**: 2026-02-23

---

## E-01: Week 7 Raw Data

| Field | Type | Constraints |
|-------|------|-------------|
| X_raw | ndarray (27, 4) | All values in [0, 1] |
| y_raw | ndarray (27,) | Positive reals; range ~0.1 to ~3395 |

**Source**: `data/f5/updated_inputs - Week 7.npy`, `data/f5/updated_outputs - Week 7.npy`
**Relationships**: Input to E-02 (transform pipeline)

---

## E-02: Transformed Training Data

| Field | Type | Constraints |
|-------|------|-------------|
| y_log | ndarray (27,) | `np.log1p(y_raw)`; compressed range |
| y_log_mean | float | Mean of y_log (for z-score) |
| y_log_std | float | Std of y_log (for z-score) |
| y_std | ndarray (27,) | `(y_log - mean) / std`; mean=0, var=1 |
| X_train | Tensor (27, 4) | `torch.tensor(X_raw, dtype=torch.double)` |
| Y_train | Tensor (27, 1) | `torch.tensor(y_std, dtype=torch.double).unsqueeze(-1)` |

**Validation**: `y_std.mean() ≈ 0`, `y_std.std() ≈ 1`
**Relationships**: Input to E-03 (GP model); E-02 stats needed for E-06 (inverse transform)

---

## E-03: GP Surrogate Model

| Field | Type | Description |
|-------|------|-------------|
| kernel | ScaleKernel(MaternKernel) | nu=2.5, ARD with 4 lengthscales |
| likelihood | GaussianLikelihood | noise_constraint=GreaterThan(1e-6) |
| model | SingleTaskGP | Fitted on (X_train, Y_train) |

**Initial Hyperparameters**:
| Parameter | Init Value | Learnable |
|-----------|-----------|-----------|
| ℓ₁–ℓ₄ (lengthscales) | 0.25 | Yes |
| σ²_f (output scale) | Var(y_std) ≈ 1.0 | Yes |
| σ²_n (noise) | 0.03 | Yes |

**Relationships**: Trained via E-04 (MLL); input to E-05 (acquisition)

---

## E-04: MLL Training State

| Field | Type | Description |
|-------|------|-------------|
| N_RESTARTS | int | 15 random restarts |
| best_loss | float | Lowest negative MLL across restarts |
| best_model | SingleTaskGP | deepcopy of best restart |

**State Transition**: For each restart seed 0..14: construct → init HPs → fit_gpytorch_mll → score → keep if best

---

## E-05: Fitted Hyperparameters

| Parameter | Accessor | Format |
|-----------|----------|--------|
| ℓ₁ | `model.covar_module.base_kernel.lengthscale[0,0]` | 6 decimal places |
| ℓ₂ | `model.covar_module.base_kernel.lengthscale[0,1]` | 6 decimal places |
| ℓ₃ | `model.covar_module.base_kernel.lengthscale[0,2]` | 6 decimal places |
| ℓ₄ | `model.covar_module.base_kernel.lengthscale[0,3]` | 6 decimal places |
| σ²_f | `model.covar_module.outputscale` | 6 decimal places |
| σ²_n | `model.likelihood.noise[0]` | 6 decimal places |

**Relationships**: Extracted from E-03 after training; reported in output; ℓ values determine E-09 (visualisation dims)

---

## E-06: NEI Acquisition Function

| Field | Type | Description |
|-------|------|-------------|
| acq_fn | qLogNoisyExpectedImprovement | q=2, prune_baseline=True |
| sampler | SobolQMCNormalSampler | sample_shape=512 |
| bounds | Tensor (2, 4) | [[0,0,0,0],[1,1,1,1]] |
| num_restarts | int | 50 |
| raw_samples | int | 3000 |

**Relationships**: Uses E-03 (fitted model); produces E-07 (candidates)

---

## E-07: Candidate Points

| Field | Type | Constraints |
|-------|------|-------------|
| candidates | Tensor (2, 4) | All values in [0, 1] |
| posterior_means | ndarray (2,) | In standardised space |
| posterior_means_orig | ndarray (2,) | Inverse-transformed to original scale |
| best_idx | int | Index of candidate with highest posterior mean |
| best_point | ndarray (4,) | The selected candidate for submission |

**Relationships**: Input to E-08 (submission query) and E-09 (visualisation overlay)

---

## E-08: Submission Query

| Field | Type | Constraints |
|-------|------|-------------|
| query_string | str | Format: `x1-x2-x3-x4` |
| components | list[str] | 4 values, each `0.xxxxxx` (6 decimal places) |
| values | list[float] | Each in [0.0, 1.0] |

**Validation**: len(parts) == 4, all parts start with "0.", all float values in [0, 1]

---

## E-09: Surrogate Visualisation

| Field | Type | Description |
|-------|------|-------------|
| top2_dims | tuple(int, int) | 2 dims with shortest lengthscales |
| fix_dims | tuple(int, int) | 2 dims fixed at best_point values |
| grid_res | int | 80 × 80 grid for contour |
| mean_surface | ndarray (80, 80) | GP posterior mean (original scale) |
| std_surface | ndarray (80, 80) | GP posterior std (original scale) |
| relevance | ndarray (4,) | 1/ℓ for each dim (normalised) |

**Relationships**: Uses E-03 (model), E-05 (lengthscales for dim selection), E-07 (best_point for overlay)

---

## E-10: Convergence Data

| Field | Type | Description |
|-------|------|-------------|
| running_best | ndarray (27,) | `np.maximum.accumulate(y_raw)` |
| boundary | float | 26.5 (Week 6→7 boundary at sample 26/27) |

**Relationships**: Uses E-01 (raw outputs)
