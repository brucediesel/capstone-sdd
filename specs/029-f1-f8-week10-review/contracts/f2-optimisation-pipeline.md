# Contract: F2 Optimisation Pipeline

**Feature**: F2 Week 10 — SFGP Optimisation Run  
**Date**: 2026-03-11

## Pipeline Interface

Sequential notebook pipeline. Each "contract" describes the inputs, outputs, and postconditions of each notebook cell group. Key differences from F1: no log transform, Standardize(m=1) outcome transform, wider LS bounds, more MLL restarts, fewer RAW_SAMPLES.

### Cell Group 1: Imports & Configuration

**Preconditions**: Existing cells 1–12 have executed. Variables `inputs`, `outputs`, `N_INITIAL`, `N_DIMS`, `n_total` are in scope.

**Outputs**:
- All BoTorch/GPyTorch imports available
- All hyperparameter constants defined (see F2OptimisationConfig in data-model.md)
- `import copy`, `import warnings` available

**Postconditions**:
- No computation performed
- All constants are named and commented
- No LOG_EPSILON constant (F2 does not use log transform)

---

### Cell Group 2: Data Preparation (Tensor Conversion)

**Preconditions**: `inputs` (ndarray N×2), `outputs` (ndarray N,) in scope. Configuration constants defined.

**Inputs**:
- `inputs`: ndarray, shape (20, 2), values in [0, 1]
- `outputs`: ndarray, shape (20,), values in [0.25, 0.67]

**Outputs**:
- `X_train`: Tensor, shape (20, 2), dtype float64
- `Y_train`: Tensor, shape (20, 1), dtype float64 — raw values (no transform)

**Postconditions**:
- `Y_train` contains no NaN or Inf
- `Y_train` values approximately in [0.25, 0.67]
- No log transform applied — Standardize(m=1) handles conditioning during GP construction
- Print: data shape summary and Y_train range

---

### Cell Group 3: GP Fitting (Multi-restart MLL + Standardize)

**Preconditions**: `X_train`, `Y_train` tensors available. Configuration constants defined.

**Inputs**:
- `X_train`: Tensor (20, 2)
- `Y_train`: Tensor (20, 1)
- `N_MLL_RESTARTS` (50), `KERNEL_NU` (2.5), `ARD_NUM_DIMS` (2), `LS_LOWER` (0.005), `LS_UPPER` (10.0), `NOISE_LB` (1e-4)

**Outputs**:
- `best_model`: SingleTaskGP — best model by MLL loss, with Standardize(m=1) outcome transform
- `best_loss`: float — lowest negative MLL

**Postconditions**:
- Model constructed with `outcome_transform=Standardize(m=1)`
- Model is in eval mode
- Lengthscales in [LS_LOWER, LS_UPPER]
- Noise ≥ NOISE_LB (in standardised space)
- Print: fitted hyperparameters (lengthscales, noise, outputscale, best_loss)

---

### Cell Group 4: Acquisition Optimisation & Candidate Selection

**Preconditions**: `best_model` fitted and in eval mode. `X_train` available.

**Inputs**:
- `best_model`: SingleTaskGP (with Standardize — predictions auto-untransform)
- `X_train`: Tensor (20, 2)
- `MC_SAMPLES` (512), `Q_BATCH` (4), `NUM_RESTARTS` (20), `RAW_SAMPLES` (4096)

**Outputs**:
- `candidates`: Tensor (Q_BATCH, 2) — all q=4 candidates
- `acqf`: qLogNoisyExpectedImprovement — fitted acquisition function (retained in scope for CG5)
- `x_new`: ndarray (2,) — distance-selected best candidate
- `proposed_query`: str — formatted "x1-x2"
- `is_duplicate`: bool

**Postconditions**:
- `x_new` values in [0.0, 0.999999]
- `proposed_query` matches format `\d\.\d{6}-\d\.\d{6}`
- Duplicate status printed
- Print: all 4 candidates, selection rationale, final submission string

---

### Cell Group 5: 3-Panel Surrogate Visualisation

**Preconditions**: `best_model`, `acqf` (acquisition function), `X_train`, `x_new` available.

**Inputs**:
- `best_model`: fitted GP (posteriors auto-untransform via Standardize)
- `acqf`: qLogNEI acquisition function
- Grid: 50×50 over [0,1]²

**Outputs**:
- Rendered matplotlib figure with 3 panels

**Postconditions**:
- Panel 1: GP posterior mean (viridis), overlaid points (blue=initial, orange=submissions, green star=proposed)
- Panel 2: GP posterior std (YlOrRd), same overlays
- Panel 3: Acquisition surface (plasma), same overlays
- Figure has appropriate titles and colorbars
- Y-axis scale is linear (not log) — F2 outputs are in normal range

---

### Cell Group 6: Updated Convergence Plot

**Preconditions**: `outputs`, `running_best`, `x_new`, `best_model` available.

**Inputs**:
- `outputs`: ndarray (20,) — all historical outputs
- `x_new`: ndarray (2,) — proposed point
- `best_model`: GP — for predicting at proposed point (predictions auto-untransform)

**Outputs**:
- Rendered matplotlib figure with convergence trajectory + proposed point

**Postconditions**:
- Y-axis in **linear** scale (not log — F2 outputs in [0.25, 0.67])
- Running best shown as line
- Proposed point marked with green star
- Initial/submission regions visually distinct
