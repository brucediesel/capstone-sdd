# Data Model: 006-sfgp-mfgp-pe

**Branch**: `006-sfgp-mfgp-pe`  
**Date**: 2026-02-22

---

## Entities

### 1. `SFGPConfig`

A single-fidelity GP hyperparameter configuration.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Human-readable name, e.g. `'matern25, noise>=1e-5, ard=True'` |
| `kernel_type` | `str` | One of `'matern05'`, `'matern15'`, `'matern25'`, `'rbf'` |
| `noise_lb` | `float` | Lower bound on the noise variance: `1e-6`, `1e-5`, `1e-4`, or `1e-3` |
| `ard` | `bool` | Whether to use Automatic Relevance Determination (per-dimension lengthscales) |
| `log_transform` | `bool` | Whether to apply `log(|y| + eps)` to outputs before fitting |
| `input_normalize` | `bool` | Whether to prepend a `Normalize` input transform |

**Represented as**: Python `dict` in the `sfgp_configs` list.

---

### 2. `MFGPConfig`

A multi-fidelity GP hyperparameter configuration.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Human-readable name, e.g. `'matern25, rank=1, noise>=1e-5, standardize=True'` |
| `kernel_type` | `str` | One of `'matern15'`, `'matern25'`, `'rbf'` |
| `rank` | `int` | ICM rank parameter for `MultiTaskGP`: 1 or 2 |
| `noise_lb` | `float` | Lower bound on noise variance applied to both tasks: `1e-6`, `1e-5`, `1e-4`, or `1e-3` |
| `output_standardize` | `bool` | Whether to apply BoTorch's `Standardize` output transform |
| `step0_fallback` | `str` | Strategy when 0 HF training points are available; always `'lf_sfgp'` |

**Represented as**: Python `dict` in the `mfgp_configs` list.

---

### 3. `PrequentialResultRecord`

The output of one run of the prequential evaluation loop for a single config.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Copied from the config's `label` |
| `MAE` | `float` | Mean Absolute Error across the 7 evaluation steps; `NaN` if run failed |
| `NLP` | `float` | Mean Negative Log Predictive Density across the 7 steps; `NaN` if run failed |
| `Coverage_95` | `float` | Proportion of actuals within 95% prediction interval; `NaN` if run failed |

**Represented as**: a row in a `pd.DataFrame` (`sfgp_hp_df` or `mfgp_hp_df`).

---

### 4. `PrequentialStep`

The intermediate data for a single one-step-ahead prediction.

| Field | Type | Description |
|-------|------|-------------|
| `step_index` | `int` | Position in the prequential sequence (0-based; 0 = predicting sample at index `N_INIT`) |
| `n_train` | `int` | Number of training points used (`N_INIT + step_index`) |
| `pred_mean` | `float` | Posterior mean at the test point |
| `pred_std` | `float` | Posterior standard deviation at the test point |
| `actual` | `float` | True observed value at the test point |

**Represented as**: values accumulated in per-loop lists; not stored persistently.

---

### 5. `FidelitySplit`

The fixed partitioning of the 17-sample dataset for the MFGP.

| Field | Type | Value | Description |
|-------|------|-------|-------------|
| `lf_indices` | `range` | `range(0, 10)` | Low-fidelity indices (initial samples) |
| `hf_indices` | `range` | `range(10, 17)` | High-fidelity indices (weekly submissions) |
| `n_lf` | `int` | `10` | Number of LF observations |
| `n_hf` | `int` | `7` | Number of HF observations |
| `task_lf` | `int` | `0` | Task index for LF samples in `MultiTaskGP` |
| `task_hf` | `int` | `1` | Task index for HF samples in `MultiTaskGP` |

**Represented as**: module-level constants in the data-loading cell.

---

## State Transitions

```
Data loaded (17 points)
    │
    ├── SFGP sweep: sfgp_configs[0..49]
    │       │
    │       ├── each config → prequential loop (7 steps)
    │       │       └── PrequentialResultRecord (MAE, NLP, Coverage)
    │       │
    │       └── sfgp_hp_df (50 rows)
    │               └── best_sfgp (1 row)
    │
    ├── MFGP sweep: mfgp_configs[0..49]
    │       │
    │       ├── step 0: fallback to SingleTaskGP on LF data
    │       ├── steps 1–6: MultiTaskGP on LF + t HF points
    │       │       └── PrequentialResultRecord (MAE, NLP, Coverage)
    │       │
    │       └── mfgp_hp_df (50 rows)
    │               └── best_mfgp (1 row)
    │
    └── comparison_df (2 rows) → winner declared
```

---

## Validation Rules

- `noise_lb > 0` — strictly positive; typical range `[1e-6, 1e-3]`
- `rank ∈ {1, 2}` — for 2-task MFGP, rank > 2 is not meaningful
- `n_train ≥ 3` before fitting any GP (satisfied by design: starts at 10)
- For MFGP step `t ≥ 1`: `n_hf_train = t ≥ 1` before calling `MultiTaskGP`; step `t = 0` uses fallback
- `pred_std` is clipped to `1e-10` minimum in `compute_metrics()` to avoid division by zero in NLP
- `NaN` results do not propagate — `idxmin()` skips NaN naturally in pandas
