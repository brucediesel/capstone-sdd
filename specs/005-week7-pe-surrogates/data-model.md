# Data Model: Week 7 — F1 Hurdle Model

**Branch**: `005-week7-pe-surrogates`  
**Phase**: 1  
**Date**: 2026-02-22

This feature adds cells to a Jupyter notebook. There is no persistent database. The "data model" here describes the in-memory variables that flow between cells in the Week 7 section.

---

## Input Data

| Variable | Shape | dtype | Source | Description |
|----------|-------|-------|--------|-------------|
| `X_w7` | `(17, 2)` | `float64` | `data/f1/updated_inputs - Week 7.npy` | Cumulative inputs; both columns ∈ [0.0, 0.999999] |
| `y_w7` | `(17,)` | `float64` | `data/f1/updated_outputs - Week 7.npy` | Cumulative outputs; maximisation target |

---

## Derived Variables (cell-level flow)

### Data validation
| Variable | Description |
|----------|-------------|
| `y_binary` | `(17,)` bool array — `y_w7 > 0`; binary label for Stage 1 |
| `n_positive` | `int` — count of `True` in `y_binary` |
| `X_pos`, `y_pos` | Subset of `X_w7`, `y_w7` where `y_w7 > 0` |
| `y_pos_log` | `np.log1p(y_pos)` — log-scale target for Stage 2 |

### Stage 1 — Classifier
| Variable | Type | Description |
|----------|------|-------------|
| `stage1_base` | `LogisticRegression` | Base estimator; `max_iter=1000`, `C=C_STAGE1`, `class_weight='balanced'` |
| `stage1_clf` | `CalibratedClassifierCV` | Fitted calibrator; `cv=3`, `method='sigmoid'` |
| `p_train` | `(17,)` float64 | `stage1_clf.predict_proba(X_w7)[:, 1]` — probability P(y>0) for each training point |

### Stage 2 — Regressor
| Variable | Type | Description |
|----------|------|-------------|
| `stage2_rf` | `RandomForestRegressor` | Fitted on `(X_pos, y_pos_log)`; `n_estimators=N_ESTIMATORS`, `max_depth=MAX_DEPTH`, `random_state=42` |

### Candidate evaluation
| Variable | Shape | Description |
|----------|-------|-------------|
| `X_cand` | `(N_CANDIDATES, 2)` | Uniform random candidates in [0.0, 0.999999]² |
| `p_cand` | `(N_CANDIDATES,)` | Stage 1 probabilities for candidates |
| `tree_preds` | `(N_ESTIMATORS, N_CANDIDATES)` | Per-tree Stage 2 predictions on log scale |
| `mu_log_cand` | `(N_CANDIDATES,)` | Mean Stage 2 prediction (log scale) |
| `sigma_rf_cand` | `(N_CANDIDATES,)` | Std across trees (log scale) |
| `mu_cand` | `(N_CANDIDATES,)` | `np.expm1(mu_log_cand)` — back-transformed conditional mean |
| `hurdle_pred` | `(N_CANDIDATES,)` | Combined prediction: `p_cand * mu_cand` |
| `acq_raw` | `(N_CANDIDATES,)` | Weighted UCB before penalization: `p_cand*mu_cand + KAPPA*p_cand*sigma_rf_cand` |
| `penalty` | `(N_CANDIDATES,)` | Multiplicative Gaussian penalty mask ∈ (0, 1] |
| `acq_penalized` | `(N_CANDIDATES,)` | `acq_raw * penalty` |
| `best_idx` | `int` | `np.argmax(acq_penalized)` |
| `next_x` | `(2,)` | `X_cand[best_idx]` clipped to [0.000000, 0.999999] |

### Visualization grid
| Variable | Shape | Description |
|----------|-------|-------------|
| `x1_grid`, `x2_grid` | `(GRID_RES,)` | Linspace [0.0, 0.999999] — 50 points each |
| `X_grid` | `(GRID_RES², 2)` | Meshgrid reshaped for model input |
| `grid_hurdle` | `(GRID_RES, GRID_RES)` | Hurdle prediction on grid, reshaped |
| `grid_uncertainty` | `(GRID_RES, GRID_RES)` | `p_grid * sigma_rf_grid`, reshaped |
| `grid_ucb` | `(GRID_RES, GRID_RES)` | Penalized UCB on grid, reshaped |

---

## Hyperparameter Constants

All defined in a single Python cell at the start of the Week 7 section:

| Constant | Value | Rationale |
|----------|-------|-----------|
| `C_STAGE1` | `1.0` | Default LR regularisation; balanced between bias and variance at small N |
| `N_ESTIMATORS` | `100` | Standard RF default; sufficient diversity on a 2D dataset |
| `MAX_DEPTH` | `3` | Constrains tree depth to prevent total overfitting on ≤10 positive samples |
| `KAPPA` | `3.0` | Exploration-focused; same as Week 6 kappa, justified by continued absence of improvement |
| `PENALTY_RADIUS` | `0.15` | ~10.6% of input space diagonal; comfortable margin above SC-003 minimum of 0.05 |
| `N_CANDIDATES` | `20_000` | Matches Week 5/6 pattern; sufficient coverage of 2D space |
| `GRID_RES` | `50` | Matches Week 5/6 pattern; 50×50 = 2 500 grid points |
| `MIN_POSITIVE` | `3` | Minimum positive samples required before Stage 2 is fitted |

---

## State Transitions

```
.npy files
    │
    ▼
X_w7, y_w7          ← data load + validation
    │
    ├──► y_binary, n_positive   ← class balance check
    │
    ├──► stage1_clf.fit()       ← Stage 1 train
    │       │
    │       └──► p_cand         ← Stage 1 predict on candidates
    │
    ├──► stage2_rf.fit()        ← Stage 2 train (if n_positive >= MIN_POSITIVE)
    │       │
    │       └──► mu_log_cand, sigma_rf_cand  ← Stage 2 predict on candidates
    │
    ├──► acq_raw                ← weighted UCB
    │
    ├──► penalty                ← local penalization mask
    │
    ├──► acq_penalized          ← argmax → next_x
    │
    ├──► visualizations         ← grid evaluation + 3-panel plot + convergence
    │
    └──► "x1.xxxxxx-x2.xxxxxx" ← formatted submission string
```
