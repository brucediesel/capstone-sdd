# Data Model: Week 12 Bayesian Optimisation Loop (F1–F8)

**Feature**: 035-f1-f8-week12-optimisation
**Date**: 2026-03-18

## Entities

### FunctionData

Represents the observed input/output dataset for a single black-box function.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| func_num | int | Function identifier (1–8) | 1 ≤ func_num ≤ 8 |
| n_dims | int | Input dimensionality | {2, 2, 3, 4, 4, 5, 6, 8} per function |
| n_initial | int | Count of initial (non-submission) samples | Fixed per function |
| week | int | Current data week number | 11 (latest available) |
| inputs | ndarray [n_total, n_dims] | All input observations in [0, 1]^d | 0 ≤ xᵢ ≤ 1 |
| outputs | ndarray [n_total] | All output observations (1D real) | Maximisation objective |
| n_total | int (derived) | len(outputs) | n_initial + n_submissions |
| n_submissions | int (derived) | n_total − n_initial | Number of weekly submissions |
| best_output | float (derived) | max(outputs) | Running optimisation target |

### SurrogateConfig

Per-function configuration for the surrogate model. Immutable within a notebook execution.

| Field | Type | Description |
|-------|------|-------------|
| surrogate_type | str | "SFGP" or "NN" |
| kernel_nu | float | Matérn smoothness (2.5 or 1.5); null for NN |
| ard_num_dims | int | ARD dimensionality; null for NN |
| ls_lower | float or null | Lengthscale lower bound (Interval); null = unconstrained |
| ls_upper | float or null | Lengthscale upper bound (Interval); null = unconstrained |
| noise_lb | float | Noise variance floor |
| n_mll_restarts | int | Multi-restart MLL fitting count; null for NN |
| output_transform | str | "log", "shift", "log+standardize", "zscore", or "none" |
| outcome_transform | str | "Standardize(m=1)" or "none" |

### AcquisitionConfig

Per-function configuration for the acquisition function and candidate selection.

| Field | Type | Description |
|-------|------|-------------|
| acq_type | str | "qLogNEI" or "blended_mean_ei" |
| mc_samples | int | Monte Carlo samples for acquisition |
| q_batch | int | Number of candidates per batch |
| num_restarts | int | L-BFGS multi-start restarts (GP only) |
| raw_samples | int | Sobol/random seed points |
| selection_method | str | "median_gate", "p25_gate", "rank_based", "argmax", or "none" |
| has_interior_penalty | bool | Whether IP is applied |
| steepness | float or null | IP steepness parameter |
| floor | float or null | IP minimum weight |

### FittedHyperparameters

Output of surrogate fitting — varies per execution.

| Field | Type | Description |
|-------|------|-------------|
| lengthscales | list[float] | Fitted per-dimension lengthscales (GP only) |
| noise | float | Fitted noise variance |
| outputscale | float | Fitted output scale (GP only) |
| best_mll | float | Best negative MLL across restarts |

### SubmissionQuery

Final output of each notebook — the proposed sample point.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| func_num | int | Function identifier | 1 ≤ func_num ≤ 8 |
| coordinates | list[float] | Proposed input coordinates | 0 ≤ xᵢ ≤ 0.999999 |
| is_duplicate | bool | Whether point matches existing observation | Must be false |
| formatted_query | str | "x1-x2-...-xn" format for submission portal | 6 decimal places |

## Relationships

```
FunctionData 1──1 SurrogateConfig    (each function has one fixed config)
FunctionData 1──1 AcquisitionConfig  (each function has one fixed acq config)
SurrogateConfig 1──* FittedHyperparameters  (each restart produces params; best kept)
AcquisitionConfig 1──1 SubmissionQuery  (acquisition produces one selected point)
```

## State Transitions

Each notebook follows a linear pipeline with no branching:

```
Load Data → Visualise (convergence + pairs) → Evaluate Performance
    → Fit Surrogate → Optimise Acquisition → Select Candidate
    → Visualise Surrogate → Updated Convergence → Output Submission Query
```

## Per-Function Sample Counts (Week 11 Data)

| Function | n_dims | n_initial | n_total | n_submissions |
|----------|--------|-----------|---------|---------------|
| F1 | 2 | 10 | 21 | 11 |
| F2 | 2 | 10 | 21 | 11 |
| F3 | 3 | 15 | 26 | 11 |
| F4 | 4 | 30 | 41 | 11 |
| F5 | 4 | 20 | 31 | 11 |
| F6 | 5 | 20 | 31 | 11 |
| F7 | 6 | 30 | 41 | 11 |
| F8 | 8 | 40 | 51 | 11 |
