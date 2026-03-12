# Data Model: F3 Week 10 — Optimisation Tuning

**Feature**: 030-f3-optimisation-tuning  
**Date**: 2026-03-12

## Entities

### FunctionData (existing — from review cells)

Variables already in scope from the existing 12 cells.

| Field | Type | Value/Shape | Description |
|-------|------|-------------|-------------|
| inputs | numpy.ndarray (25×3) | Values in [0,1] | All 25 input samples (15 initial + 10 submissions) |
| outputs | numpy.ndarray (25,) | Range [-0.399, -0.031] | All 25 output values (all negative) |
| N_INITIAL | int | 15 | Number of initial samples |
| FUNC_NUM | int | 3 | Function number |
| N_DIMS | int | 3 | Input dimensionality |
| DATA_DIR | str | `../../data/f3/` | Path to data folder |
| WEEK | int | 10 | Current week number |
| n_total | int | 25 | Total sample count |
| n_submissions | int | 10 | Number of weekly submissions |
| running_best | numpy.ndarray (25,) | From np.maximum.accumulate | Running best trajectory |

### OptimisationConfig

Hyperparameter constants for the F3 SFGP + qLogNEI optimisation run. All declared as named constants in a dedicated configuration cell. Key difference from F1/F2: shift transform instead of log/Standardize, 3D ARD, q=3, 2048 raw samples, 40 MLL restarts.

| Field | Type | Default | Description | Change from Week 9 |
|-------|------|---------|-------------|---------------------|
| KERNEL_NU | float | 2.5 | Matérn kernel smoothness parameter | Unchanged |
| ARD_NUM_DIMS | int | 3 | Number of ARD lengthscales (= N_DIMS) | Unchanged |
| NOISE_LB | float | 1e-4 | Noise variance lower bound | Was 1e-6 (relaxed to prevent overfitting) |
| N_MLL_RESTARTS | int | 40 | Number of MLL fitting restarts | Was 15–20 (increased for 3D multi-modal likelihood) |
| MC_SAMPLES | int | 512 | Monte Carlo samples for qLogNEI sampler | Unchanged |
| Q_BATCH | int | 3 | Number of candidates per acquisition batch | Was 1 (increased for better 3D coverage) |
| NUM_RESTARTS | int | 20 | L-BFGS restarts for acquisition optimisation | Unchanged |
| RAW_SAMPLES | int | 2048 | Sobol seed points for acquisition optimisation | Was 512 (increased for 3D search) |
| GRID_RES | int | 50 | Resolution of visualisation grid (50×50) | Unchanged |

### ShiftTransformedOutputs

Derived entity representing the shift-transformed training targets. This replaces the Standardize(m=1) used in week 9.

| Field | Type | Description |
|-------|------|-------------|
| y_raw | Tensor (25×1) | Original output values, range [-0.399, -0.031] |
| y_min | float | Minimum of y_raw (≈ -0.399) — stored for reverse-transform |
| Y_shifted | Tensor (25×1) | `y_raw - y_min`, range [0, 0.368] — non-negative targets for GP |

**Validation rules**:
- Y_shifted contains no NaN or Inf values
- Y_shifted values are all ≥ 0
- y_min is finite
- Reverse-transform: `y_original = Y_shifted + y_min`

### SFGPModel

Fitted Gaussian Process surrogate. No outcome_transform (shift is applied manually before construction).

| Field | Type | Description |
|-------|------|-------------|
| model | SingleTaskGP | BoTorch GP with Matérn-2.5 ARD kernel, no outcome_transform |
| train_X | Tensor (25×3) | Input data (unchanged) |
| train_Y | Tensor (25×1) | Shift-transformed output data |
| best_loss | float | Lowest negative MLL across all 40 restarts |
| lengthscales | Tensor (3,) | Fitted ARD lengthscales per dimension |
| noise | float | Fitted noise variance |
| outputscale | float | Fitted output scale |

**Validation rules**:
- lengthscales are finite and positive
- noise ≥ NOISE_LB (1e-4)
- outputscale > 0

### AcquisitionCandidates

Output of the acquisition optimisation step.

| Field | Type | Description |
|-------|------|-------------|
| candidates | Tensor (Q_BATCH × 3) | q=3 candidate points in [0,1]³ |
| acq_value | float | Best acquisition function value |
| pred_means | Tensor (Q_BATCH,) | Posterior mean at each candidate (shifted space) |
| selected_idx | int | Index of the distance-selected candidate |
| x_new | ndarray (3,) | Final selected point for submission |

### SubmissionPoint

Formatted output for challenge submission.

| Field | Type | Description |
|-------|------|-------------|
| query | str | Formatted as `x1-x2-x3`, 6 decimal places |
| is_duplicate | bool | Whether query matches any existing sample |
| x1 | float | First coordinate, clamped to [0.0, 0.999999] |
| x2 | float | Second coordinate, clamped to [0.0, 0.999999] |
| x3 | float | Third coordinate, clamped to [0.0, 0.999999] |

## Relationships

```
FunctionData ─────uses────> ShiftTransformedOutputs (y_raw → Y_shifted via y - y_min)
ShiftTransformedOutputs ──trains──> SFGPModel (GP fitted on shifted targets)
SFGPModel ────────drives──> AcquisitionCandidates (qLogNEI proposes q=3 points)
AcquisitionCandidates ──selects──> SubmissionPoint (distance-based → single point)
```

## State Transitions

```
[Data Loaded (existing cells)] → [Shift Transform Applied] → [GP Fitted (40 restarts)]
  → [qLogNEI Optimised (q=3)] → [Distance Selection] → [Submission Formatted]
```
