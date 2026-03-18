# Data Model: Week 11 Performance Review & Feedback

**Feature**: 033-week11-feedback
**Date**: 2026-03-17

## Entities

### FunctionData

Represents the loaded input/output data for a single black-box function.

| Field | Type | Description |
|-------|------|-------------|
| func_num | int | Function number (1-8) |
| n_dims | int | Number of input dimensions |
| n_initial | int | Number of initial sample points |
| inputs | numpy.ndarray (N × d) | All input samples, shape (total_samples, n_dims) |
| outputs | numpy.ndarray (N,) | All output values, shape (total_samples,) |
| n_total | int | Total number of samples (derived from data shape) |
| n_submissions | int | n_total - n_initial |
| best_idx | int | Index of sample with highest output value: `np.argmax(outputs)` |

**Validation rules**:
- All inputs in range [0, 1]
- outputs is 1-dimensional
- n_total > n_initial

### FunctionConfig

Static configuration parameters per function. Defined as constants at top of each notebook.

| Field | Type | Description |
|-------|------|-------------|
| FUNC_NUM | int | Function number |
| N_DIMS | int | Input dimensionality |
| N_INITIAL | int | Count of initial samples |
| DATA_DIR | str | Path to data folder (e.g., `../../data/f1/`) |
| WEEK | int | Current week number (11) |
| USE_LOG_SCALE | bool | True only for F1 |

**Static values by function**:

| Constant | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|----------|----|----|----|----|----|----|----|----|
| N_DIMS | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |
| N_INITIAL | 10 | 10 | 15 | 30 | 20 | 20 | 30 | 40 |
| N_TOTAL | 21 | 21 | 26 | 41 | 31 | 31 | 41 | 51 |
| USE_LOG_SCALE | True | False | False | False | False | False | False | False |
| PAIR_PLOTS | 1 | 1 | 3 | 6 | 6 | 10 | 15 | 28 |

### BestOutputMarker

Visual marker for the best-performing sample on pair plots.

| Field | Type | Description |
|-------|------|-------------|
| marker_style | str | `'*'` (star) |
| marker_color | str | `'green'` — high contrast against both blue (initial) and orange (submissions) |
| marker_size | int | 500 — scatter `s` parameter (area in points²) |
| zorder | int | 5 — renders above all other scatter points |
| label | str | `'Best'` — legend entry text |
| legend_markersize | int | 15 — `Line2D` markersize (independent of scatter `s`) |
| best_idx | int | `np.argmax(outputs)` — index of overall best sample |
| best_input | numpy.ndarray (d,) | `inputs[best_idx]` — input coordinates of best sample |
| best_output | float | `outputs[best_idx]` — highest output value |

### StrategyInfo

Static text documenting the Week 10 strategy for each function. Used in the performance evaluation markdown cell.

| Field | Type | Description |
|-------|------|-------------|
| surrogate | str | Name and key hyperparameters of Week 10 surrogate model |
| acquisition | str | Name and key hyperparameters of Week 10 acquisition function |
| stalling | bool | Whether the function is stalling (no new best in ≥3 consecutive submissions) |
| improvements | str | Fraction of submissions that yielded improvement |

## Relationships

```
FunctionConfig ──1:1──> FunctionData (config determines which data files to load)
FunctionConfig ──1:1──> StrategyInfo (config determines which strategy description to include)
FunctionData ──1:1──> BestOutputMarker (derived from outputs array)
```

## State Transitions

No state transitions — these notebooks are read-only visualisations with no model fitting or data mutation.
