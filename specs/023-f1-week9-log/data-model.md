# Data Model: F1 Week 9 — Hurdle Model with log Transform (No Penalties)

**Date**: 2026-03-09  
**Feature**: 023-f1-week9-log

## Entities

### 1. Training Data

| Field | Type | Description |
|-------|------|-------------|
| X | ndarray (19, 2) | Input features, all values in [0.0, 1.0] |
| y | ndarray (19,) | Scalar outputs; mix of positive, zero, and negative values |
| X_initial | ndarray (10, 2) | First 10 rows of X (initial samples) |
| X_submissions | ndarray (9, 2) | Rows 10-18 of X (weekly submissions) |
| y_binary | ndarray (19,) bool | Binary labels: y > 0 |
| y_pos | ndarray (N_pos,) | Subset of y where y > 0 |
| y_pos_log | ndarray (N_pos,) | `np.log(y_pos)` — training targets for Stage 2 RF |

### 2. Hurdle Model (Two-Stage Surrogate)

**Stage 1 — Classifier**

| Field | Type | Description |
|-------|------|-------------|
| stage1 | CalibratedClassifierCV | Wraps LogisticRegression; produces P(y > 0) |
| p(x) | float in [0, 1] | Calibrated probability that candidate x yields positive output |

**Stage 2 — RF Regressor** (only trained when N_pos >= MIN_POSITIVE)

| Field | Type | Description |
|-------|------|-------------|
| stage2 | RandomForestRegressor | Trained on log(y) for positive outputs |
| mu(x) | float | Mean RF prediction in log-space (range ~[-565, -35]) |
| sigma_RF(x) | float | Std across tree predictions in log-space |
| FALLBACK_MODE | bool | True if N_pos < MIN_POSITIVE; Stage 2 skipped |

### 3. Acquisition Components

| Field | Type | Description |
|-------|------|-------------|
| a_ucb(x) | float | p(x)·mu(x) + kappa·p(x)·sigma_RF(x) |
| a_final(x) | float | a_ucb(x) (no penalization applied) |

### 4. Hyperparameters

| Constant | Value | Entity |
|----------|-------|--------|
| N_INITIAL | 10 | Data split |
| MIN_POSITIVE | 3 | Fallback threshold |
| C_STAGE1 | 1.0 | Classifier L2 regularisation |
| N_ESTIMATORS | 100 | RF ensemble size |
| MAX_DEPTH | 3 | RF max tree depth |
| KAPPA | 0.5 | UCB exploitation weight (exploitation-focused) |
| N_CANDIDATES | 20000 | Acquisition candidate count |
| GRID_RES | 50 | Visualisation grid resolution |

## Relationships

```
Training Data (X, y)
  ├── y_binary = y > 0 ──→ Stage 1 Classifier (input: X, target: y_binary)
  │                           └── p(x) for all candidates
  └── y_pos = y[y > 0]
      └── y_pos_log = log(y_pos) ──→ Stage 2 RF (input: X[y>0], target: y_pos_log)
                                       ├── mu(x) in log-space
                                       └── sigma_RF(x) in log-space

Acquisition: a_final(x) = p(x)·mu(x) + kappa·p(x)·sigma_RF(x)
  └── argmax over N_CANDIDATES random candidates ──→ next_x (proposed sample point)
```

## State Transitions

| State | Condition | Behaviour |
|-------|-----------|-----------|
| NORMAL_MODE | N_pos >= MIN_POSITIVE | Both stages trained; full acquisition |
| FALLBACK_MODE | N_pos < MIN_POSITIVE | Stage 2 skipped; mu=0, sigma=1; pure exploration |

## Validation Rules

- All X values must be in [0.0, 1.0]
- No NaN or Inf in y
- y_pos must be strictly positive (y > 0, not y >= 0) before applying log
- Proposed point clipped to [0.0, 0.999999]
- Warning if proposed point within 0.05 Euclidean distance of any existing observation
