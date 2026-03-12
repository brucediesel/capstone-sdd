# Notebook Cell Contracts: F4–F8 Week 10 Optimisation

**Date**: 2026-03-12 | **Branch**: `031-f4-f8-week10-optimisation`

This document defines the cell-by-cell contract for each notebook. Since these are Jupyter notebooks (not REST APIs), contracts are defined as cell input/output specifications.

---

## Common Cell Contract (all 5 notebooks)

Each notebook appends ~7 cells after the existing 12 review cells.

### Cell 13 — Markdown: Section Header

**Input**: None  
**Output**: Markdown heading "Step 6 — Week 10 Optimisation Run" with bullet list of strategy changes from week 9 and rationale.

### Cell 14 — Code: Imports & Configuration

**Input**: None (self-contained)  
**Output**: All hyperparameter constants printed to stdout.

**Contract**:
- All constants defined as UPPER_CASE module-level variables
- All imports included (torch, botorch, gpytorch, numpy, matplotlib, copy)
- `torch.set_default_dtype(torch.float64)` called
- Constants table printed with `print()` for auditability

### Cell 15 — Code: Data Preparation

**Input**: `.npy` files from `../../data/fX/`  
**Output**: `X_train` (n, d) tensor, `Y_train` (n, 1) tensor, printed shape and range summary.

**Contract**:
- Load `updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy`
- Convert to `torch.float64` tensors
- Apply function-specific output transform (F5: log)
- Print raw and transformed output ranges
- Print training data shape

### Cell 16 — Code: Surrogate Fitting

**Input**: `X_train`, `Y_train` from Cell 15  
**Output**: Fitted `model` object, printed hyperparameters.

**Contract** (GP functions — F4, F5, F6, F8):
- Construct fresh model per MLL restart
- Randomise hyperparameters before each restart
- Fit with `fit_gpytorch_mll`
- Track and restore best model state
- Print: best loss, number of converged restarts, final lengthscales, noise, outputscale

**Contract** (NN function — F7):
- Train NN with SGD for configured epochs
- Print training loss curve summary
- Enable MC dropout for uncertainty estimation

### Cell 17 — Code: Acquisition & Selection

**Input**: Fitted `model`, `X_train`, `Y_train`  
**Output**: Selected candidate `x_new`, printed submission query.

**Contract** (GP functions — F4, F5, F6, F8):
- Construct acquisition function (qLogNEI)
- Call `optimize_acqf` with configured parameters
- Apply distance-based selection (if q > 1)
- Apply interior penalty (F6 only)
- Clamp to [0.0, 0.999999]
- Duplicate check against existing observations
- Print formatted submission: `x1-x2-...-xn`

**Contract** (NN function — F7):
- Generate random candidates
- MC dropout forward passes for uncertainty
- Compute blended score (mean + EI)
- Apply interior penalty
- Select best candidate
- Print formatted submission

### Cell 18 — Code: 2D Contour Visualisation

**Input**: Fitted `model`, `X_train`, `x_new`  
**Output**: Matplotlib figure with 2D contour slices.

**Contract**:
- Create grid of input dimension pairs
- Plot GP posterior mean (or NN prediction) as contour
- Overlay training points (initial vs submitted in different colours)
- Mark proposed point with distinct marker
- Title each subplot with dimension pair names

### Cell 19 — Code: Convergence Plot

**Input**: All observation `Y` values, proposed `y_new` prediction  
**Output**: Matplotlib figure with convergence trajectory.

**Contract**:
- Plot running best objective value across all observations
- Mark initial vs submitted observations
- Mark proposed point with predicted value
- X-axis: observation index, Y-axis: objective value
- F1 special: use log scale (not applicable here)

---

## Function-Specific Contract Variations

### F4: Surrogate & Acquisition Overhaul

| Parameter | Week 9 | Week 10 | 
|-----------|--------|---------|
| Surrogate | SingleTaskMultiFidelityGP | SingleTaskGP |
| Kernel | Matérn-2.5 + LinTrunc | Matérn-2.5 ARD (4D) |
| Input dims | 5 (4 + fidelity) | 4 |
| Outcome transform | Manual z-score | Standardize(m=1) |
| Acquisition | qLogNEI (MF) | qLogNEI (standard) |
| noise_lb | 1e-4 | 1e-3 |
| MLL restarts | 15 | ≥30 |
| q | 4 | 4 |
| raw_samples | 512 | 2048 |
| MC samples | 64 | 512 |

### F5: Exploration Tuning & Transform Simplification

| Parameter | Week 9 | Week 10 |
|-----------|--------|---------|
| Output transform | log1p + Standardize | log + Standardize |
| raw_samples | 5000 | 8000 |
| num_restarts (acq) | 50 | 60 |
| MLL restarts | 15 | 15 |
| Distance selection | median gate | relaxed gate (25th pct or all) |

### F6: Incremental Refinement

| Parameter | Week 9 | Week 10 |
|-----------|--------|---------|
| milk threshold | ≥ 0.10 | ≥ 0.12 |
| noise_lb | 1e-2 | 1e-3 |
| raw_samples | 3000 | 5000 |

### F7: Exploration Boost with MC Dropout

| Parameter | Week 9 | Week 10 |
|-----------|--------|---------|
| EXPLOITATION_WEIGHT | 0.7 | 0.5 |
| STEEPNESS | 0.05 | 0.02 |
| N_CANDIDATES | 20000 | 50000 |
| MC_SAMPLES | 30 | ≥50 |

### F8: Exploration & Numerical Stability

| Parameter | Week 9 | Week 10 |
|-----------|--------|---------|
| Acquisition | qEI | qLogNEI |
| XI | 0.01 | N/A (qLogNEI) |
| MC samples | 256 | 512 |
| raw_samples | 4096 | 8192 |
| noise_lb | 1e-7 | 1e-7 (verify stability) |
