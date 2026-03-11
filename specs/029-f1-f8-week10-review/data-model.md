# Data Model: Week 10 Performance Review & Visualisation

**Feature**: 029-f1-f8-week10-review  
**Date**: 2026-03-11

## Entities

### FunctionData

Represents the loaded input/output data for a single black-box function.

| Field | Type | Description |
|-------|------|-------------|
| func_num | int | Function number (1-8) |
| n_dims | int | Number of input dimensions |
| n_initial | int | Number of initial sample points |
| inputs | numpy.ndarray (N × d) | All input samples, shape (total_samples, n_dims) |
| outputs | numpy.ndarray (N × 1) | All output values, shape (total_samples, 1) |
| n_total | int | Total number of samples (derived from data shape) |
| n_submissions | int | n_total - n_initial |

**Validation rules**:
- All inputs in range [0, 1]
- outputs is 1-dimensional (second axis = 1)
- n_total > n_initial

### FunctionConfig

Static configuration parameters per function. Defined as constants at top of each notebook.

| Field | Type | Description |
|-------|------|-------------|
| FUNC_NUM | int | Function number |
| N_DIMS | int | Input dimensionality |
| N_INITIAL | int | Count of initial samples |
| DATA_DIR | str | Path to data folder (e.g., `../../data/f1/`) |
| WEEK | int | Current week number (10) |
| USE_LOG_SCALE | bool | True only for F1 |

**Static values by function**:

| Constant | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|----------|----|----|----|----|----|----|----|----|
| N_DIMS | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |
| N_INITIAL | 10 | 10 | 15 | 30 | 20 | 20 | 30 | 40 |
| USE_LOG_SCALE | True | False | False | False | False | False | False | False |

### StrategyInfo

Static text documenting the week 9 strategy for each function. Used in the performance evaluation markdown cell.

| Field | Type | Description |
|-------|------|-------------|
| surrogate | str | Name and key hyperparameters of week 9 surrogate model |
| acquisition | str | Name and key hyperparameters of week 9 acquisition function |
| stalling | bool | Whether the function is stalling (no new best in ≥3 consecutive submissions) |
| improvements | str | Fraction of submissions that yielded improvement (e.g., "2/9") |

## Relationships

```
FunctionConfig ──1:1──> FunctionData (config determines which data files to load)
FunctionConfig ──1:1──> StrategyInfo (config determines which strategy description to include)
```

## State Transitions

No state transitions — these notebooks are read-only visualisations with no model fitting or data mutation.

---

## F1 Optimisation Run — Additional Entities

**Spec**: [spec-f1-optimisation.md](spec-f1-optimisation.md)

### OptimisationConfig

Hyperparameter constants for the F1 SFGP + qLogNEI optimisation run. All declared as named constants in a dedicated configuration cell.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| KERNEL_NU | float | 2.5 | Matérn kernel smoothness parameter |
| ARD_NUM_DIMS | int | 2 | Number of ARD lengthscales (= N_DIMS) |
| LS_LOWER | float | 0.01 | Lower bound on lengthscale constraint |
| LS_UPPER | float | 2.0 | Upper bound on lengthscale constraint |
| NOISE_LB | float | 1e-4 | Noise variance lower bound |
| N_MLL_RESTARTS | int | 15 | Number of MLL fitting restarts |
| LOG_EPSILON | float | 1e-300 | Clamping floor for log transform |
| MC_SAMPLES | int | 512 | Monte Carlo samples for qLogNEI sampler |
| Q_BATCH | int | 4 | Number of candidates per acquisition batch |
| NUM_RESTARTS | int | 20 | L-BFGS restarts for acquisition optimisation |
| RAW_SAMPLES | int | 10000 | Sobol seed points for acquisition optimisation |
| GRID_RES | int | 50 | Resolution of visualisation grid (50×50) |

### LogTransformedOutputs

Derived entity representing the log-transformed training targets.

| Field | Type | Description |
|-------|------|-------------|
| y_raw | Tensor (N×1) | Original output values from data |
| y_log | Tensor (N×1) | `log(max(y_raw, LOG_EPSILON))`, range approx [-690, -35] |

**Validation rules**:
- y_log contains no NaN or Inf values
- y_log values are all negative (since all F1 outputs < 1)

### SFGPModel

Fitted Gaussian Process surrogate.

| Field | Type | Description |
|-------|------|-------------|
| model | SingleTaskGP | BoTorch GP with Matérn-2.5 ARD kernel |
| train_X | Tensor (N×2) | Normalised input data |
| train_Y | Tensor (N×1) | Log-transformed output data |
| best_loss | float | Lowest negative MLL across all restarts |
| lengthscales | Tensor (2,) | Fitted ARD lengthscales per dimension |
| noise | float | Fitted noise variance |
| outputscale | float | Fitted output scale |

**Validation rules**:
- lengthscales in [LS_LOWER, LS_UPPER]
- noise ≥ NOISE_LB
- outputscale > 0

### AcquisitionCandidates

Output of the acquisition optimisation step.

| Field | Type | Description |
|-------|------|-------------|
| candidates | Tensor (Q_BATCH × 2) | q=4 candidate points in [0,1]² |
| acq_value | float | Best acquisition function value |
| pred_means | Tensor (Q_BATCH,) | Posterior mean at each candidate |
| selected_idx | int | Index of the distance-selected candidate |
| x_new | ndarray (2,) | Final selected point for submission |

### SubmissionPoint

Formatted output for challenge submission.

| Field | Type | Description |
|-------|------|-------------|
| query | str | Formatted as `x1-x2`, 6 decimal places |
| is_duplicate | bool | Whether query matches any existing sample |
| x1 | float | First coordinate, clamped to [0.0, 0.999999] |
| x2 | float | Second coordinate, clamped to [0.0, 0.999999] |

## F1 Optimisation Relationships

```
FunctionData ──uses──> LogTransformedOutputs (y_raw → y_log via log transform)
LogTransformedOutputs ──trains──> SFGPModel (GP fitted on log-space targets)
SFGPModel ──drives──> AcquisitionCandidates (qLogNEI proposes q=4 points)
AcquisitionCandidates ──selects──> SubmissionPoint (distance-based → single point)
```

## F1 Optimisation State Transitions

```
[Data Loaded] → [Log Transform Applied] → [GP Fitted (15 restarts)]
  → [qLogNEI Optimised] → [Distance Selection] → [Submission Formatted]
```
