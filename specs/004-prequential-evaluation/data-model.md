# Data Model: F6 Prequential Evaluation — NN, SFGP & MFGP

**Feature**: 004-prequential-evaluation (F6 extension)
**Date**: 2025-07-15

## Entities

### 1. NN Config

| Field | Type | Validation | Notes |
|-------|------|------------|-------|
| `n_layers` | int | {1, 2, 3} | Number of hidden layers |
| `n_nodes` | int | {4, 5, 6} | Nodes per hidden layer |
| `lr` | float | {0.001, 0.005, 0.01, 0.05, 0.1} | Learning rate for Adam |
| `label` | str | Format: `NN: {n_layers}L×{n_nodes}N, lr={lr}` | Human-readable identifier |

**Total combinations**: 3 × 3 × 5 = **45 configs**

### 2. SFGP Config

| Field | Type | Validation | Notes |
|-------|------|------------|-------|
| `kernel_type` | str | {'matern_2.5', 'matern_1.5', 'matern_0.5', 'rbf'} | Base kernel for ScaleKernel wrapping |
| `output_transform` | str | {'raw', 'standardise'} | Manual z-score or no transform |
| `noise_lb` | float | {1e-4, 1e-5, 1e-6, 1e-7, 1e-8} | GaussianLikelihood noise floor |
| `label` | str | Format: `SF: {kernel}, {transform}, noise>={noise_lb:.0e}` | Human-readable identifier |

**Total combinations**: 4 × 2 × 5 = **40 configs**

### 3. MFGP Config (unchanged)

| Field | Type | Validation | Notes |
|-------|------|------------|-------|
| `nu` | float | {0.5, 1.5, 2.5} | Matérn smoothness |
| `linear_truncated` | bool | {True, False} | Fidelity kernel type |
| `output_transform` | str | {'raw', 'standardise'} | Manual z-score or no transform |
| `noise_lb` | float | {1e-4, 1e-5, 1e-6, 1e-7} + {5e-5} | Noise floor |
| `label` | str | Format: `MF: nu={nu}, {kernel}, {transform}, noise>={noise_lb:.0e}` | Human-readable identifier |

**Total combinations**: 3 × 2 × 2 × 4 = 48 + 2 extras = **50 configs**

### 4. Prequential Result

| Field | Type | Notes |
|-------|------|-------|
| `predictions` | list[float] | Length = n_steps (6) |
| `actuals` | list[float] | Length = n_steps (6) |
| `stds` | list[float] | Length = n_steps (6), floored at 1e-10 |
| `metrics` | dict | Contains MAE, NLP, Coverage_95 |

### 5. HP DataFrame (nn_hp_df / sfgp_hp_df / mfgp_hp_df)

| Column | Type | Notes |
|--------|------|-------|
| `label` | str | Config identifier |
| `MAE` | float | Mean Absolute Error (lower is better) |
| `NLP` | float | Negative Log Predictive density (lower is better) |
| `Coverage_95` | float | 95% CI coverage (target = 0.95) |

**Row counts**: nn_hp_df = 45, sfgp_hp_df = 40, mfgp_hp_df = 50

### 6. Comparison DataFrame (comparison_df)

| Column | Type | Notes |
|--------|------|-------|
| `Model` | str | {'NN', 'SFGP', 'MFGP'} |
| `Configuration` | str | Best config label from that family |
| `MAE` | float | Best config MAE |
| `NLP` | float | Best config NLP |
| `Coverage_95` | float | Best config coverage |

**Row count**: 3 (one per family)

### 7. Full Ranked DataFrame (full_ranked)

| Column | Type | Notes |
|--------|------|-------|
| `Model` | str | {'NN', 'SFGP', 'MFGP'} |
| `Configuration` | str | Config label |
| `MAE` | float | MAE |
| `NLP` | float | NLP (sort key, ascending) |
| `Coverage_95` | float | Coverage |

**Row count**: 135 (45 + 40 + 50)
**Index**: 1-based rank (sorted by NLP ascending)

## Relationships

```
NN Config (45) ──→ nn_prequential_with_config() ──→ nn_hp_df (45 rows)
                                                         │
SFGP Config (40) ──→ sfgp_prequential_with_config() ──→ sfgp_hp_df (40 rows)
                                                         │
MFGP Config (50) ──→ mfgp_prequential_with_config() ──→ mfgp_hp_df (50 rows)
                                                         │
                    ┌─── best_nn (1 row) ←── min NLP from nn_hp_df
                    ├─── best_sfgp (1 row) ←── min NLP from sfgp_hp_df
                    └─── best_mfgp (1 row) ←── min NLP from mfgp_hp_df
                              │
                              ▼
                    comparison_df (3 rows)
                              │
                              ▼
                    full_ranked (135 rows, sorted by NLP)
```

## State Transitions

No entity state transitions — this is a stateless evaluation pipeline. Each config is evaluated independently, producing a single result row.

## Validation Rules

1. All DataFrames must have no NaN values in metrics columns (failed configs are caught by try/except)
2. Coverage_95 must be in [0.0, 1.0]
3. MAE must be ≥ 0
4. NLP can be any real number (negative = well-calibrated, positive = poorly-calibrated)
5. Predictions and actuals lists must have exactly 6 elements (n_steps = 26 − 20)
