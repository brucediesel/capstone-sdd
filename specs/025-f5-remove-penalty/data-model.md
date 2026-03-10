# Data Model: F5 Week 9 — Kernel, Standardize & Raw Samples

**Branch**: `026-f5-kernel-standardize` | **Date**: 2026-03-09

## Entities (unchanged from data files)

| Entity | Shape | Type | Source |
|--------|-------|------|--------|
| `X_raw` | (29, 4) | float64 | `updated_inputs - Week 9.npy` + `initial_inputs.npy` |
| `y_raw` | (29,) | float64 | `updated_outputs - Week 9.npy` + `initial_outputs.npy` |
| `X_train` | (29, 4) | torch.double | Tensor from `X_raw` |
| `Y_train` | (29, 1) | torch.double | `log1p(y_raw)` — **no manual z-score** |

## Constants (after changes)

| Constant | Value | Changed? |
|----------|-------|----------|
| `N_INITIAL` | 20 | No |
| `N_TOTAL` | 29 | No |
| `N_DIMS` / `DIM` | 4 | No |
| `N_SUBMISSIONS` | 9 | No |
| `N_RESTARTS` | 15 | No |
| `STALLING_WINDOW` | 3 | No |
| `STALLING_REL_THRESHOLD` | 0.05 | No |

## GP Configuration (changes highlighted)

| Parameter | Before | After | Changed? |
|-----------|--------|-------|----------|
| Kernel | `MaternKernel(nu=2.5, ard_num_dims=4)` | `MaternKernel(nu=1.5, ard_num_dims=4)` | **YES** |
| Outer kernel | `ScaleKernel(...)` | `ScaleKernel(...)` | No |
| Likelihood | `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))` | Same | No |
| `outcome_transform` | `None` | `Standardize(m=1)` | **YES** |
| MLL restarts | 15 | 15 | No |
| Lengthscale init | 0.5 | 0.5 | No |
| Noise init | `0.1 * Y_train.var()` | `0.1 * Y_train.var()` | No |
| Outputscale init | 1.0 | 1.0 | No |

## Acquisition Configuration (changes highlighted)

| Parameter | Before | After | Changed? |
|-----------|--------|-------|----------|
| Function | `qLogNoisyExpectedImprovement` | Same | No |
| q | 4 | 4 | No |
| MC samples | 512 | 512 | No |
| `num_restarts` | 50 | 50 | No |
| `raw_samples` | 3000 | 5000 | **YES** |
| Selection | Distance-based (mean ≥ median, farthest) | Same | No |

## Transform Pipeline (changes highlighted)

| Step | Before | After |
|------|--------|-------|
| 1. Raw → log | `y_log = np.log1p(y_raw)` | Same |
| 2. Z-score | Manual: `(y_log - mean) / std` | **Automatic via `Standardize(m=1)`** |
| 3. To tensor | `Y_train = tensor(y_std).unsqueeze(-1)` | `Y_train = tensor(y_log).unsqueeze(-1)` |
| 4. Inverse (posterior) | `expm1(pred * y_std_val + y_mean)` | `expm1(posterior.mean)` |

## Removed Entities

| Entity | Reason |
|--------|--------|
| `y_mean` | No longer needed — Standardize handles internally |
| `y_std_val` | No longer needed — Standardize handles internally |
| `y_std` | No longer needed — Y_train receives `y_log` directly |
