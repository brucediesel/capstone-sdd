# Notebook Cell Contracts: 006-sfgp-mfgp-pe

**Branch**: `006-sfgp-mfgp-pe`  
**Date**: 2026-02-22  
**Notebook**: `functions/f2/preq-eval-f2.ipynb`

This document specifies the input/output contract for each logical code cell in the rewritten notebook. "Contract" means: what variables must exist before the cell runs (preconditions) and what new variables or outputs the cell produces (postconditions).

---

## Cell C-01 — Imports

**Type**: Code  
**Preconditions**: Kernel started  
**Postconditions**:
- `np`, `torch`, `plt`, `pd`, `warnings` are importable aliases in scope  
- `SingleTaskGP`, `MultiTaskGP`, `fit_gpytorch_mll`, `ExactMarginalLogLikelihood` in scope  
- `MaternKernel`, `RBFKernel`, `ScaleKernel`, `GaussianLikelihood`, `GreaterThan` in scope  
- `np.random.seed(42)`, `torch.manual_seed(42)` called  
- Prints: `'All imports successful.'`

---

## Cell C-02 — Load Data and Fidelity Split

**Type**: Code  
**Preconditions**: C-01 complete; `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy` exist  
**Postconditions**:
- `WEEK = 7` (int)
- `N_INIT = 10` (int)
- `X_all` → `np.ndarray` shape `(17, 2)`
- `y_all` → `np.ndarray` shape `(17,)`, flattened
- `n_total = 17` (int)
- `n_steps = 7` (int)
- `LF_TASK = 0`, `HF_TASK = 1` (int constants)
- Prints: data summary including shape, output range, fidelity split counts

**Error contract**: If `.npy` file does not exist, `assert` raises `AssertionError` with a descriptive message before any load.

---

## Cell C-03 — `compute_metrics()` Definition

**Type**: Code  
**Preconditions**: C-01 complete  
**Postconditions**:
- `compute_metrics(predictions, actuals, pred_means, pred_stds) → dict` is defined and callable  
- Returns keys: `'MAE'`, `'NLP'`, `'Coverage_95'`, `'nlp_values'`, `'errors'`, `'in_interval'`  
- **Function body is identical to the existing notebook**; no modifications  
- Prints: `'compute_metrics() defined.'`

---

## Cell C-04 — `plot_prequential_results()` Definition

**Type**: Code  
**Preconditions**: C-01, C-03 complete  
**Postconditions**:
- `plot_prequential_results(results, model_name) → None` is defined and callable  
- Expects `results` dict with keys: `'actuals'`, `'pred_means'`, `'pred_stds'`, `'metrics'`  
- Renders a 3-panel matplotlib figure (predictions vs actuals, absolute errors, NLP per step)  
- **Function body is identical to the existing notebook**; no modifications  
- Prints: `'plot_prequential_results() defined.'`

---

## Cell C-05 — `sfgp_prequential_evaluation()` Definition and Default Run

**Type**: Code  
**Preconditions**: C-01, C-02, C-03 complete  
**Postconditions**:
- `sfgp_prequential_evaluation(X_all, y_all, n_init) → dict` is defined  
- `sfgp_default_results → dict` with keys `'predictions'`, `'actuals'`, `'pred_means'`, `'pred_stds'`, `'metrics'`  
- `plot_prequential_results()` called → 3-panel figure rendered  
- Prints per-step progress and final MAE/NLP/Coverage

---

## Cell C-06 — `sfgp_prequential_with_config()` Definition

**Type**: Code  
**Preconditions**: C-01, C-03 complete  
**Postconditions**:
- `sfgp_prequential_with_config(X_all, y_all, n_init, config) → dict` is defined  
- `config` dict keys consumed: `kernel_type`, `noise_lb`, `ard`, `log_transform`, `input_normalize`  
- Returns same structure as `sfgp_prequential_evaluation()` plus `'MAE_original'` key  
- Prints: `'sfgp_prequential_with_config() defined.'`

---

## Cell C-07 — `sfgp_configs` List Definition (50 entries)

**Type**: Code  
**Preconditions**: None (pure data declaration)  
**Postconditions**:
- `sfgp_configs` → `list[dict]`, length exactly 50  
- Each entry has keys: `label` (str), `kernel_type`, `noise_lb`, `ard`, `log_transform`, `input_normalize`  
- Prints: `'50 SFGP configurations defined.'`

---

## Cell C-08 — SFGP 50-Config Sweep

**Type**: Code  
**Preconditions**: C-02, C-06, C-07 complete  
**Postconditions**:
- `sfgp_hp_results → list[dict]` length 50 (NaN entries allowed for failed configs)  
- `sfgp_hp_df → pd.DataFrame` shape `(50, 4)` with columns `label`, `MAE`, `NLP`, `Coverage_95`  
- DataFrame displayed inline  
- Prints running progress: `Config i/50: <label>  MAE=... NLP=... Coverage=...`

---

## Cell C-09 — Best SFGP Selection

**Type**: Code  
**Preconditions**: C-08 complete  
**Postconditions**:
- `best_sfgp_idx → int` (row index of best config)  
- `best_sfgp → pd.Series` with index keys `label`, `MAE`, `NLP`, `Coverage_95`  
- Prints: `'Best SFGP by NLP:'`, label, MAE, NLP, Coverage

---

## Cell C-10 — Best SFGP Prediction Plot

**Type**: Code  
**Preconditions**: C-02, C-06, C-09 complete  
**Postconditions**:
- `best_sfgp_results → dict` (full detail including per-step data)  
- `plot_prequential_results()` called → 3-panel figure rendered  
- Title includes best SFGP config label

---

## Cell C-11 — `mfgp_prequential_with_config()` Definition

**Type**: Code  
**Preconditions**: C-01, C-03 complete  
**Postconditions**:
- `mfgp_prequential_with_config(X_all, y_all, n_init, config) → dict` is defined  
- `config` dict keys consumed: `kernel_type`, `rank`, `noise_lb`, `output_standardize`  
- At `step=0` (0 HF training points): uses `SingleTaskGP` on `X_all[:n_init]` (fallback)  
- At `step>=1`: builds task-augmented `X_train`, fits `MultiTaskGP(X_train, y_train, task_feature=-1, rank=config['rank'])`  
- Posterior extracted at `(X_test, task=HF_TASK)` using `model.posterior(X_test_aug)`  
- Returns: same dict structure as `sfgp_prequential_with_config()`  
- On exception: appends `NaN` for `pred_mean` and `pred_std` for that step  
- Prints: `'mfgp_prequential_with_config() defined.'`

---

## Cell C-12 — `mfgp_configs` List Definition (50 entries)

**Type**: Code  
**Preconditions**: None  
**Postconditions**:
- `mfgp_configs → list[dict]` length exactly 50  
- Each entry has keys: `label`, `kernel_type`, `rank`, `noise_lb`, `output_standardize`, `step0_fallback`  
- `step0_fallback` is always `'lf_sfgp'`  
- Prints: `'50 MFGP configurations defined.'`

---

## Cell C-13 — MFGP Default Run

**Type**: Code  
**Preconditions**: C-02, C-11, C-12 complete  
**Postconditions**:
- `mfgp_default_results → dict`  
- `plot_prequential_results()` called → 3-panel figure rendered

---

## Cell C-14 — MFGP 50-Config Sweep

**Type**: Code  
**Preconditions**: C-02, C-11, C-12 complete  
**Postconditions**:
- `mfgp_hp_results → list[dict]` length 50  
- `mfgp_hp_df → pd.DataFrame` shape `(50, 4)` with columns `label`, `MAE`, `NLP`, `Coverage_95`  
- DataFrame displayed inline

---

## Cell C-15 — Best MFGP Selection

**Type**: Code  
**Preconditions**: C-14 complete  
**Postconditions**:
- `best_mfgp_idx → int`  
- `best_mfgp → pd.Series`  
- Prints: `'Best MFGP by NLP:'`, label, MAE, NLP, Coverage

---

## Cell C-16 — Best MFGP Prediction Plot

**Type**: Code  
**Preconditions**: C-02, C-11, C-15 complete  
**Postconditions**:
- `best_mfgp_results → dict`  
- `plot_prequential_results()` called → 3-panel figure rendered

---

## Cell C-17 — 2-Way Comparison Table

**Type**: Code  
**Preconditions**: C-09, C-15 complete  
**Postconditions**:
- `comparison_df → pd.DataFrame` shape `(2, 5)`, columns: `Model`, `Configuration`, `MAE`, `NLP`, `Coverage_95`  
- Prints metric winners and overall winner sentence  
- `best_overall → pd.Series` (best of the two by NLP; MAE tiebreak; Coverage tiebreak)

---

## Cell C-18 — 2-Way Comparison Bar Chart

**Type**: Code  
**Preconditions**: C-17 complete  
**Postconditions**:
- 3-panel matplotlib figure rendered: MAE, NLP, Coverage  
- 2 bars per panel (SFGP=#2196F3, MFGP=#FF9800)  
- Value labels on each bar  
- 0.95 reference line on Coverage panel  
- Title: `'F2: Best SFGP vs Best MFGP — 2-Way Comparison'`

---

## Cell C-19 — 100-Config Sensitivity Chart

**Type**: Code  
**Preconditions**: C-08, C-14 complete  
**Postconditions**:
- Horizontal bar chart with 100 rows rendered  
- All SFGP rows = blue (#2196F3), all MFGP rows = orange (#FF9800)  
- Sorted by NLP (ascending, best at top)  
- Title: `'F2: All 100 Configurations — Hyperparameter Sensitivity'`

---

## Cell C-20 — Full Ranked Results Table

**Type**: Code  
**Preconditions**: C-08, C-14 complete  
**Postconditions**:
- `full_summary → pd.DataFrame` shape `(100, 5)` columns: `Model`, `Configuration`, `MAE`, `NLP`, `Coverage_95`  
- Sorted by NLP ascending, index is 1-based rank  
- Displayed inline

---

## Cell C-21 — Winner Detail Prediction Plot

**Type**: Code  
**Preconditions**: C-17 complete (best_overall defined)  
**Postconditions**:
- `plot_prequential_results()` called for winner  
- Title includes: winner model name + configuration label  
- Prints prominent winner announcement

---

## Variable Dependency Graph

```
C-01 (imports)
  └─► C-02 (data) ─────────────────────────────────────────────┐
  └─► C-03 (compute_metrics)                                    │
  └─► C-04 (plot_prequential_results)                           │
  └─► C-05 (sfgp default) ◄───────── C-02, C-03                │
  └─► C-06 (sfgp config fn)                                     │
  └─► C-07 (sfgp config list)                                   │
  └─► C-08 (sfgp sweep) ◄──────────── C-02, C-06, C-07         │
  └─► C-09 (best sfgp) ◄───────────── C-08                     │
  └─► C-10 (best sfgp plot) ◄──────── C-02, C-06, C-09         │
  └─► C-11 (mfgp config fn) ◄──────── C-03                     │
  └─► C-12 (mfgp config list)                                   │
  └─► C-13 (mfgp default) ◄────────── C-02, C-11, C-12         │
  └─► C-14 (mfgp sweep) ◄──────────── C-02, C-11, C-12         │
  └─► C-15 (best mfgp) ◄───────────── C-14                     │
  └─► C-16 (best mfgp plot) ◄──────── C-02, C-11, C-15         │
  └─► C-17 (comparison) ◄──────────── C-09, C-15               │
  └─► C-18 (bar chart) ◄───────────── C-17                     │
  └─► C-19 (sensitivity chart) ◄───── C-08, C-14               │
  └─► C-20 (ranked table) ◄────────── C-08, C-14               │
  └─► C-21 (winner plot) ◄──────────── C-17                    │
```
