# Cell Contracts: Week 7 F1 Section

**Branch**: `005-week7-pe-surrogates`  
**File**: `functions/f1/f1.ipynb` (append after cell 54, before cell 55)  
**Date**: 2026-02-22

Each cell below defines its **preconditions** (what must be true before it runs), **outputs** (what it produces), and **side effects** (prints, plots, state mutations).

---

## Cell W7-01 — Section Header
**Type**: Markdown  
**Content**: `## Week 7 — Hurdle Model with Weighted UCB and Local Penalization`  
Followed by a paragraph explaining the exploration rationale and the three components: hurdle model, weighted UCB, local penalization.

**Preconditions**: None (Markdown only)  
**Outputs**: Rendered heading visible in notebook  

---

## Cell W7-02 — Load & Validate Week 7 Data
**Type**: Code  
**Preconditions**: `numpy` imported as `np`; Week 7 `.npy` files exist at `data/f1/`  
**Outputs**:
- `X_w7 : ndarray (17, 2)` — inputs
- `y_w7 : ndarray (17,)` — outputs
- `y_binary : ndarray (17,) bool` — `y_w7 > 0`
- `n_positive : int`
- `X_pos : ndarray (n_positive, 2)`
- `y_pos : ndarray (n_positive,)`
- `y_pos_log : ndarray (n_positive,)` — `np.log1p(y_pos)`

**Printed side effects**:
```
Week 7 Data — 17 samples, 2 inputs each
Input range: [x_min, x_max] ✓ within [0.0, 0.999999]
Outputs — min: X  max: X  mean: X
Positive outputs (y > 0): N/17
⚠ WARNING: fewer than 3 positive samples — Stage 2 fallback active  (conditional)
```
**Fallback**: If `n_positive < MIN_POSITIVE`, sets `FALLBACK_MODE = True`; all Stage 2 cells are skipped or replaced by random exploration.

---

## Cell W7-03 — Hyperparameter Rationale
**Type**: Markdown  
Lists every hyperparameter as `**NAME = value**: one-sentence rationale`. Covers: `C_STAGE1`, `N_ESTIMATORS`, `MAX_DEPTH`, `KAPPA`, `PENALTY_RADIUS`, `N_CANDIDATES`, `GRID_RES`, `MIN_POSITIVE`.

**Preconditions**: None  
**Outputs**: Rendered table visible in notebook  

---

## Cell W7-04 — Hyperparameter Constants
**Type**: Code  
**Preconditions**: None (no Python state required)  
**Outputs** (module-level variables):
- `C_STAGE1 = 1.0`
- `N_ESTIMATORS = 100`
- `MAX_DEPTH = 3`
- `KAPPA = 3.0`
- `PENALTY_RADIUS = 0.15`
- `N_CANDIDATES = 20_000`
- `GRID_RES = 50`
- `MIN_POSITIVE = 3`

**Printed side effects**: One `print` per constant, e.g. `"  C_STAGE1: 1.0"`.

---

## Cell W7-05 — Fit Stage 1: Classifier
**Type**: Code  
**Preconditions**: `X_w7`, `y_binary`, `C_STAGE1` defined; `sklearn` imported  
**Outputs**:
- `stage1_base : LogisticRegression`
- `stage1_clf : CalibratedClassifierCV` (fitted)
- `p_train : ndarray (17,)` — training-set probability estimates

**Printed side effects**:
```
Stage 1 — Logistic Regression (CalibratedClassifierCV)
Training accuracy: X.XX
Positive-class probabilities (training): [p1, p2, ...]
```

---

## Cell W7-06 — Fit Stage 2: Regressor
**Type**: Code  
**Preconditions**: `X_pos`, `y_pos_log`, `N_ESTIMATORS`, `MAX_DEPTH` defined; `FALLBACK_MODE` is `False`  
**Outputs**:
- `stage2_rf : RandomForestRegressor` (fitted on `X_pos`, `y_pos_log`)

**Printed side effects**:
```
Stage 2 — Random Forest Regressor on log1p(y) for y > 0
Training R²: X.XX  (on log scale)
Number of trees: 100  Max depth: 3
```

---

## Cell W7-07 — Weighted UCB with Local Penalization
**Type**: Code  
**Preconditions**: `stage1_clf`, `stage2_rf`, `X_w7`, `KAPPA`, `PENALTY_RADIUS`, `N_CANDIDATES` defined  
**Outputs**:
- `X_cand : (N_CANDIDATES, 2)`
- `p_cand : (N_CANDIDATES,)`
- `mu_cand : (N_CANDIDATES,)` — back-transformed (expm1 scale)
- `sigma_rf_cand : (N_CANDIDATES,)`
- `acq_raw : (N_CANDIDATES,)`
- `penalty : (N_CANDIDATES,)`
- `acq_penalized : (N_CANDIDATES,)`
- `next_x : (2,)` — clipped to [0.000000, 0.999999]

**Printed side effects**:
```
Weighted UCB with Local Penalization
  κ = 3.0 (exploration-focused)
  Penalization radius r = 0.15 (all 17 evaluated points)
  Best penalized UCB score: X.XXXX
  Next candidate: [x1, x2]
  Min distance to existing data: X.XXXX  (must be ≥ 0.05)
```

---

## Cell W7-08 — Surrogate & Acquisition Surface Plot (3 panels)
**Type**: Code  
**Preconditions**: `stage1_clf`, `stage2_rf`, `X_w7`, `y_w7`, `next_x`, `KAPPA`, `PENALTY_RADIUS`, `GRID_RES` defined  
**Outputs**:
- `grid_hurdle : (GRID_RES, GRID_RES)` — p(x)·expm1(μ(x))
- `grid_uncertainty : (GRID_RES, GRID_RES)` — p(x)·σ_RF(x)
- `grid_ucb : (GRID_RES, GRID_RES)` — penalized UCB on grid

**Plot side effect**: `plt.subplots(1, 3, figsize=(18, 5))` — 3-panel contourf  
- Panel 1 colormap: `viridis`, title: `'Hurdle Mean Prediction (ŷ = p·expm1(μ))'`  
- Panel 2 colormap: `YlOrRd`, title: `'Hurdle Uncertainty (p·σ_RF)'`  
- Panel 3 colormap: `plasma`, title: `f'Penalized UCB Acquisition (κ={KAPPA})'`  
- All panels: training points as red/blue scatter (positive/non-positive, matching Week 5/6 visual style), proposed point as yellow star `s=200`

---

## Cell W7-09 — Convergence Plot
**Type**: Code  
**Preconditions**: `y_w7`, `next_x` defined  
**Plot side effect**: `plt.figure(figsize=(10, 5))`  
- Running maximum line `'b-o'`
- Individual observations gray scatter
- Vertical dashed red line at x=10.5 (initial → weekly boundary)
- Title: `'Function 1 — Convergence Plot (Week 7)'`
- x-axis: `'Observation Number'`, y-axis: `'Objective Value'`

---

## Cell W7-10 — Format Submission Query
**Type**: Code  
**Preconditions**: `next_x` defined  
**Printed side effects**:
```
Week 7 Submission Query:
  x1 = X.XXXXXX  x2 = X.XXXXXX
  Formatted: X.XXXXXX-X.XXXXXX
```
Both values clipped to [0.000000, 0.999999] before formatting.
