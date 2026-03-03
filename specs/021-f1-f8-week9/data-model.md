# Data Model: F1-F8 Week 9

**Feature**: 021-f1-f8-week9 | **Date**: 2026-03-02

## Entities

### 1. FunctionConfig

Defines the per-function configuration carried from Week 8.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| func_id | int | Function identifier (1-8) | 1 <= func_id <= 8 |
| n_dims | int | Input dimensionality | {2, 3, 4, 5, 6, 8} |
| n_initial | int | Number of initial samples | F1=10, F2=10, F3=15, F4=30, F5=20, F6=20, F7=30, F8=40 |
| n_total | int | Total samples in Week 9 data | F1=19, F2=19, F3=24, F4=39, F5=29, F6=29, F7=39, F8=49 |
| n_submissions | int | Number of weekly submissions | Always 9 (n_total - n_initial) |
| surrogate_type | str | Surrogate model class | One of: hurdle, sfgp, mfgp, gp, nn |
| acquisition_type | str | Acquisition function class | One of: weighted_ucb, qLogNEI, mf_qNEI, mc_dropout_ei, qEI |
| has_interior_penalty | bool | Whether interior penalty is applied | F1, F5, F6, F7 = True |
| penalty_steepness | float | Interior penalty steepness | 0.1 (F1, F7), 1.0 (F5, F6) |
| penalty_floor | float | Interior penalty floor | 0.01 (all) |

### 2. WeekData

The loaded data arrays for a single function.

| Field | Type | Shape | Validation |
|-------|------|-------|------------|
| inputs | ndarray | (n_total, n_dims) | All values in [0.0, 1.0] |
| outputs | ndarray | (n_total,) | No NaN values |
| initial_inputs | ndarray | (n_initial, n_dims) | Slice of inputs[:n_initial] |
| initial_outputs | ndarray | (n_initial,) | Slice of outputs[:n_initial] |
| submission_inputs | ndarray | (n_submissions, n_dims) | Slice of inputs[n_initial:] |
| submission_outputs | ndarray | (n_submissions,) | Slice of outputs[n_initial:] |
| best_idx | int | scalar | Index of max output |
| best_value | float | scalar | outputs[best_idx] |
| best_point | ndarray | (n_dims,) | inputs[best_idx] |

### 3. PerformanceMetrics

Computed by the performance evaluation section.

| Field | Type | Description |
|-------|------|-------------|
| best_trajectory | ndarray(9,) | Running max after each submission |
| per_submission_delta | ndarray(9,) | Improvement from previous best per submission |
| new_best_flags | ndarray(9,) bool | Whether each submission found a new best |
| consecutive_no_improvement | int | Trailing streak of no-new-best (counted from most recent submission backwards) |
| initial_best | float | Max of initial outputs |
| final_best | float | Max of all outputs |
| relative_improvement | float | (final_best - initial_best) / abs(initial_best) |
| stalling_consecutive | bool | consecutive_no_improvement >= 3 |
| stalling_relative | bool | relative_improvement < 0.05 |
| stalling | bool | stalling_consecutive OR stalling_relative |

### 4. ExplorationMetrics

Computed from submission point positions.

| Field | Type | Description |
|-------|------|-------------|
| mean_pairwise_distance | float | Mean Euclidean distance between all submission pairs |
| max_nn_distance | float | Maximum nearest-neighbour distance among submissions |
| min_nn_distance | float | Minimum nearest-neighbour distance (tightest cluster) |

### 5. LOOMetrics

Leave-one-out surrogate prediction accuracy.

| Field | Type | Description |
|-------|------|-------------|
| loo_predictions | ndarray(9,) | Model prediction for each held-out submission |
| loo_actuals | ndarray(9,) | Actual output for each submission |
| loo_errors | ndarray(9,) | Absolute error per fold |
| loo_mae | float | Mean absolute error across 9 folds |
| loo_rmse | float | Root mean squared error across 9 folds |

## Relationships

```
FunctionConfig 1──1 WeekData
WeekData 1──1 PerformanceMetrics
WeekData 1──1 ExplorationMetrics
WeekData 1──1 LOOMetrics
```

All entities are computed fresh within each notebook. No cross-notebook dependencies.

## State Transitions

Each notebook progresses through these states:

```
DATA_LOADED → SURROGATE_FITTED → ACQUISITION_OPTIMISED → 
POINT_PROPOSED → VISUALISED → PERFORMANCE_EVALUATED → STRATEGY_ASSESSED
```

The final two states (PERFORMANCE_EVALUATED, STRATEGY_ASSESSED) are new for Week 9.
