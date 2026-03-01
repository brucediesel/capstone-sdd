# Data Model: F2–F8 Week 8 Bayesian Optimisation

**Feature**: `020-f2-f8-week8` | **Date**: 2026-03-01

## Entities

### 1. Observation Data (`X`, `y`)

Per-function training data loaded from `.npy` files.

| Field | Type | Description |
|-------|------|-------------|
| `X` | `ndarray (n, d)` | Input observations, d ∈ {2,3,4,5,6,8} |
| `y` | `ndarray (n,)` | Output observations (scalar per point) |

**Validation rules**:
- All `X[i,j]` ∈ [0.0, 1.0]
- No NaN in `y`
- Shape `n` matches expected sample count per function
- Shape `d` matches expected dimensionality

**Sample counts**: F2=18, F3=23, F4=38, F5=28, F6=28, F7=38, F8=48

### 2. Surrogate Model

Function-specific surrogate trained on observation data.

| Field | Type | Description |
|-------|------|-------------|
| `model` | GP or NN | Fitted surrogate (type varies per function) |
| `kernel` | Matérn 1.5 or 2.5 | GP covariance function (F2-F6, F8) |
| `lengthscales` | `tensor (d,)` | ARD lengthscales (GP functions) |
| `noise` | `float` | Observation noise variance |
| `y_mean`, `y_std` | `float` | Z-score normalisation stats (F3, F4, F5, F7) |

**State transitions**:
- UNTRAINED → FITTED (after MLL optimisation or NN training)
- FITTED → PREDICTED (after posterior evaluation)

### 3. Acquisition Result

Output of acquisition function optimisation.

| Field | Type | Description |
|-------|------|-------------|
| `candidates` | `tensor (q, d)` | Proposed sample points, q ∈ {1, 4} |
| `acq_values` | `tensor (q,)` | Acquisition function values |
| `next_x` | `ndarray (d,)` | Selected best candidate |
| `next_x_clipped` | `ndarray (d,)` | Clipped to [0.0, 0.999999] |

**Selection rules** (vary by function):
- F2, F8: Single candidate (q=1), direct selection
- F3: Single candidate, direct selection
- F4: Best of q=4 by posterior mean
- F5, F6: Best of q=4 by distance-based exploration + interior penalty re-scoring
- F7: Best of 20 000 by MC Dropout EI × interior penalty

### 4. Interior Penalty (F5, F6, F7 only)

Boundary suppression weights.

| Field | Type | Description |
|-------|------|-------------|
| `STEEPNESS` | `float` | Controls penalty decay speed |
| `FLOOR` | `float` | Minimum penalty value (prevents zero) |
| `w(x)` | `float ∈ [FLOOR, 1.0]` | Per-candidate boundary weight |

**Formula**: $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{d} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$

| Function | STEEPNESS | FLOOR |
|----------|-----------|-------|
| F5 | 1.0 | 0.01 |
| F6 | 1.0 | 0.01 |
| F7 | 0.1 | 0.01 |

### 5. Submission Query

Formatted string for challenge submission.

| Field | Type | Description |
|-------|------|-------------|
| `formatted_query` | `str` | `0.xxxxxx-0.xxxxxx-...-0.xxxxxx` |
| `n_dims` | `int` | Number of hyphen-separated values |

**Validation rules**:
- Each component starts with `0.`
- Each component has exactly 6 decimal places
- All values ∈ [0.000000, 0.999999]
- Number of components matches function dimensionality

### 6. Visualisation Outputs

Per-function plot outputs.

| Output | Type | Functions |
|--------|------|-----------|
| 3-panel surrogate plot | `matplotlib figure` | All |
| Convergence plot | `matplotlib figure` | All |
| Interior penalty plot | `matplotlib figure` | F5, F6 |
| Feature importance | `matplotlib figure` | F7, F8 |
| Training loss | `matplotlib figure` | F7 |

## Relationships

```
Observation Data (X, y)
    └──> Surrogate Model (trained on data)
            └──> Acquisition Result (optimised from model)
                    ├──> Interior Penalty (applied to candidates, F5/F6/F7)
                    └──> Submission Query (formatted from next_x)
            └──> Visualisation Outputs (generated from model + data)
```

## Per-Function Output Transform Pipeline

| Function | Raw y → Model input | Model output → Display |
|----------|---------------------|----------------------|
| F2 | raw y → tensor | posterior → raw y |
| F3 | z-score: (y-μ)/σ | unstandardise: pred×σ+μ |
| F4 | z-score: (y-μ)/σ | unstandardise: pred×σ+μ |
| F5 | log1p → z-score | unstandardise → expm1 |
| F6 | auto (Standardize) | auto-untransform |
| F7 | z-score (numpy) | unnormalise: pred×σ+μ |
| F8 | auto (Standardize) | auto-untransform |
