# Data Model: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Date**: 2026-02-23 | **Branch**: `010-f4-mfgp-nei`

## Entity Definitions

### E-01: Raw Inputs (NumPy)

| Field | Type | Shape | Constraint |
|-------|------|-------|------------|
| `X_raw` | `np.ndarray[float64]` | (37, 4) | All values in [0, 1] |

Source: `data/f4/updated_inputs - Week 7.npy` (cumulative: 30 initial + 7 weekly).

### E-02: Raw Outputs (NumPy)

| Field | Type | Shape | Constraint |
|-------|------|-------|------------|
| `y_raw` | `np.ndarray[float64]` | (37,) | Real-valued (float64) |

Source: `data/f4/updated_outputs - Week 7.npy` (cumulative).

### E-03: Standardised Outputs

| Field | Type | Shape | Derivation |
|-------|------|-------|------------|
| `y_mean` | `float64` | scalar | `y_raw.mean()` |
| `y_std` | `float64` | scalar | `y_raw.std()` |
| `y_std_np` | `np.ndarray[float64]` | (37,) | `(y_raw - y_mean) / y_std` |

Manual z-score standardisation for explicitness.

### E-04: Training Tensors

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `X_train` | `torch.Tensor[float64]` | (37, 4) | From E-01, `torch.tensor(X_raw)` |
| `Y_train` | `torch.Tensor[float64]` | (37, 1) | From E-03, `.unsqueeze(-1)` |
| `fidelity_col` | `torch.Tensor[float64]` | (37, 1) | All 1.0 |
| `X_mf` | `torch.Tensor[float64]` | (37, 5) | `torch.cat([X_train, fidelity_col], dim=-1)` |

### E-05: MFGP Hyperparameters (Initial)

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| `nu` | 2.5 | Matérn-5/2, PE winner for F4 |
| `linear_truncated` | True | LinearTruncatedFidelityKernel, PE winner |
| `noise_lb` | 1e-4 | Noise floor from user specification |
| MLL restarts | 15 | Validated in F3 Week 7 pattern |

### E-06: MFGP Hyperparameters (Fitted)

| Field | Type | Source |
|-------|------|--------|
| `ℓ₁, ℓ₂, ℓ₃, ℓ₄` | float64 | `model.covar_module.base_kernel.kernels[0].lengthscale` |
| `σ²_f` | float64 | `model.covar_module.outputscale` |
| `σ²_n` | float64 | `model.likelihood.noise` |
| `power` | float64 | LinearTruncatedFidelityKernel power parameter |
| `neg_mll` | float64 | Best negative MLL across restarts |

### E-07: Acquisition Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Acquisition | `qLogNoisyExpectedImprovement` | Log-space NEI |
| `q` | 4 | Batch of 4 candidates |
| MC samples | 64 | Via `SobolQMCNormalSampler` |
| `num_restarts` | 20 | Optimisation restarts |
| `raw_samples` | 512 | Sobol initialisation points |
| `fixed_features` | `{4: 1.0}` | Pin fidelity to 1.0 |
| `sequential` | False | Joint optimisation |

### E-08: Candidate Batch

| Field | Type | Shape | Constraint |
|-------|------|-------|------------|
| `candidates` | `torch.Tensor[float64]` | (4, 5) | 4 spatial coords + fidelity=1.0 |
| `next_points` | `torch.Tensor[float64]` | (4, 4) | Spatial coords only, `candidates[:, :4]` |
| `acq_value` | `torch.Tensor[float64]` | scalar | Joint NEI acquisition value |

### E-09: Primary Submission Point

| Field | Type | Shape | Constraint |
|-------|------|-------|------------|
| `best_point` | `np.ndarray[float64]` | (4,) | Selected from E-08 by highest posterior mean |
| `submission_query` | `str` | — | `"0.XXXXXX-0.YYYYYY-0.ZZZZZZ-0.WWWWWW"` |

### E-10: Surrogate Visualisation Grid

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `top2_dims` | `tuple[int, int]` | — | 2 dims with shortest ARD lengthscales |
| `fix_dims` | `tuple[int, int]` | — | Remaining 2 dims, fixed at best_point values |
| `grid_res` | `int` | 80 | Grid resolution per axis |
| `grid_mu` | `np.ndarray[float64]` | (80, 80) | Posterior mean on grid (de-standardised) |
| `grid_sigma` | `np.ndarray[float64]` | (80, 80) | Posterior std on grid (de-standardised) |

### E-11: Convergence Data

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `running_best` | `np.ndarray[float64]` | (37,) | `np.maximum.accumulate(y_raw)` |
| `boundary` | float | 30.5 | Separates initial (30) from weekly submissions |

## Relationships

```
E-01 (X_raw) ──→ E-04 (X_train) ──→ E-04 (X_mf) ──→ MFGP ──→ E-06 (fitted HPs)
E-02 (y_raw) ──→ E-03 (y_std)   ──→ E-04 (Y_train)            ↓
                                                          E-07 (NEI config)
                                                                ↓
                                                          E-08 (candidates)
                                                                ↓
                                                          E-09 (submission)
E-06 (ℓ values) ──→ E-10 (top2_dims selection)
E-02 (y_raw) ──→ E-11 (convergence)
```
