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
