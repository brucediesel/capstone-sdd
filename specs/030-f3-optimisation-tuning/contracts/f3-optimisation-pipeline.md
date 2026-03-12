# Contract: F3 Optimisation Pipeline

**Feature**: F3 Week 10 — Optimisation Tuning  
**Date**: 2026-03-12

## Pipeline Interface

Sequential notebook pipeline. Each "contract" describes the inputs, outputs, and postconditions of each notebook cell group. Key differences from F1/F2: 3D inputs (3 ARD lengthscales), shift transform instead of log/Standardize, q=3 instead of q=4, 2048 raw samples, 40 MLL restarts, 2D contour slices (3 pairs) instead of 3-panel contour.

### Cell Group 1: Section Header (Markdown)

**Content**: Markdown cell titled "## Step 6 — Week 10 Optimisation Run" with brief description of the 5 changes from week 9.

**Postconditions**: No code executed. Documents the optimisation strategy changes.

---

### Cell Group 2: Imports & Configuration

**Preconditions**: Existing cells 1–12 have executed. Variables `inputs`, `outputs`, `N_INITIAL`, `N_DIMS`, `n_total` are in scope.

**Outputs**:
- All BoTorch/GPyTorch imports available
- All hyperparameter constants defined (see OptimisationConfig in data-model.md)
- `import copy`, `import warnings`, `import matplotlib.pyplot as plt` available

**Postconditions**:
- No computation performed
- All constants are named and commented with week 9 → week 10 change justification
- No LOG_EPSILON or Standardize — F3 uses manual shift transform

---

### Cell Group 3: Data Preparation (Tensor Conversion + Shift Transform)

**Preconditions**: `inputs` (ndarray 25×3), `outputs` (ndarray 25,) in scope. Configuration constants defined.

**Inputs**:
- `inputs`: ndarray, shape (25, 3), values in [0, 1]
- `outputs`: ndarray, shape (25,), values in [-0.399, -0.031]

**Outputs**:
- `X_train`: Tensor, shape (25, 3), dtype float64
- `Y_train`: Tensor, shape (25, 1), dtype float64 — shift-transformed values in [0, 0.368]
- `y_min`: float — minimum observed output (≈ -0.399), stored for reverse-transform

**Postconditions**:
- `Y_train = outputs - y_min`, all values ≥ 0
- No NaN or Inf in Y_train
- Print: data shape summary, raw output range, shifted output range, y_min value

---

### Cell Group 4: GP Fitting (Multi-restart MLL, no outcome transform)

**Preconditions**: `X_train`, `Y_train` tensors available. Configuration constants defined.

**Inputs**:
- `X_train`: Tensor (25, 3)
- `Y_train`: Tensor (25, 1) — shift-transformed, range [0, 0.368]
- `N_MLL_RESTARTS` (40), `KERNEL_NU` (2.5), `ARD_NUM_DIMS` (3), `NOISE_LB` (1e-4)

**Outputs**:
- `best_model`: SingleTaskGP — best model by MLL loss, NO outcome_transform
- `best_loss`: float — lowest negative MLL

**Postconditions**:
- Model constructed WITHOUT outcome_transform (shift is pre-applied to Y_train)
- Model is in eval mode
- Noise ≥ NOISE_LB (1e-4)
- 40 restarts completed; best loss reported
- Print: fitted hyperparameters (3 lengthscales, noise, outputscale, best_loss)
- Print: number of restarts that converged to same optimum (within 0.1 of best_loss)

---

### Cell Group 5: Acquisition Optimisation & Candidate Selection

**Preconditions**: `best_model` fitted and in eval mode. `X_train` available.

**Inputs**:
- `best_model`: SingleTaskGP (no outcome_transform — predictions in shifted space)
- `X_train`: Tensor (25, 3)
- `MC_SAMPLES` (512), `Q_BATCH` (3), `NUM_RESTARTS` (20), `RAW_SAMPLES` (2048)

**Outputs**:
- `candidates`: Tensor (Q_BATCH, 3) — all q=3 candidates in [0,1]³
- `acqf`: qLogNoisyExpectedImprovement — fitted acquisition function (retained for CG6)
- `x_new`: ndarray (3,) — distance-selected best candidate
- `proposed_query`: str — formatted "x1-x2-x3"
- `is_duplicate`: bool

**Postconditions**:
- `x_new` values in [0.0, 0.999999]
- `proposed_query` matches format `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`
- Duplicate check against all 25 existing samples
- Print: all 3 candidates with posterior means (shifted), selection rationale, final submission string

---

### Cell Group 6: 2D Contour Slice Visualisation

**Preconditions**: `best_model`, `acqf`, `X_train`, `x_new` available.

**Inputs**:
- `best_model`: fitted GP (predictions in shifted space)
- `acqf`: qLogNEI acquisition function
- `x_new`: ndarray (3,) — proposed point (for fixing slice dimensions)
- Grid: 50×50 over [0,1]² for each pair

**Outputs**:
- Rendered matplotlib figure with 2 rows × 3 columns (6 panels total)

**Postconditions**:
- Row 1: GP posterior mean for pairs (d0,d1), (d0,d2), (d1,d2) — fix remaining dim at x_new's coordinate
- Row 2: Acquisition surface for same pairs
- Each panel: contourf with appropriate colormap
- Point overlays: initial (blue), submissions (orange), proposed (green star) — projected onto each pair
- Figure has titles, colorbars, axis labels identifying dimensions
- Values displayed in shifted space (no reverse-transform for visualisation)

---

### Cell Group 7: Updated Convergence Plot

**Preconditions**: `outputs`, `running_best`, `x_new`, `best_model`, `y_min` available.

**Inputs**:
- `outputs`: ndarray (25,) — all historical outputs (original scale)
- `x_new`: ndarray (3,) — proposed point
- `best_model`: GP — for predicting at proposed point (predictions in shifted space)
- `y_min`: float — for reverse-transforming prediction to original scale

**Outputs**:
- Rendered matplotlib figure with convergence trajectory + proposed point

**Postconditions**:
- Y-axis in **linear** scale (F3 outputs in [-0.399, -0.031])
- Running best shown as line in original scale
- Proposed point marked with green star, with predicted value reverse-shifted: `pred_original = pred_shifted + y_min`
- Initial samples (blue region) and submissions (orange region) visually distinct
- Print: predicted shifted value and reverse-shifted original value at proposed point
