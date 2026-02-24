# Data Model: F6 Interior Penalty

**Feature**: 016-f6-interior-penalty  
**Date**: 2025-02-24

---

## Entities

### 1. Penalty Hyperparameters (Constants)

| Field | Type | Value | Description |
|-------|------|-------|-------------|
| `STEEPNESS` | `float` | `1.0` | Controls transition width; higher = narrower interior band |
| `FLOOR` | `float` | `0.01` | Minimum weight at boundary; prevents zero-weight degeneracy |
| `D` | `int` | `5` | Dimensionality (inferred from `candidates.shape[1]`) |

**Validation**: `STEEPNESS > 0`, `0 < FLOOR < 1`.

### 2. Interior Weight Vector

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `interior_weight` | `np.ndarray[float64]` | `(4,)` | Per-candidate penalty weight ∈ [FLOOR, 1.0] |

**Formula**: `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`

**Validation**:
- All values in `[FLOOR, 1.0]`.
- Boundary candidates (any `xᵢ ≈ 0` or `xᵢ ≈ 1`) yield `w(x) ≈ FLOOR`.
- Centre point `(0.5, …, 0.5)` yields `w(x) = 1.0`.

### 3. Rank-Based Scores

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `rank_mean` | `np.ndarray[int]` | `(4,)` | Rank by posterior mean (1=worst, 4=best) |
| `rank_weight` | `np.ndarray[int]` | `(4,)` | Rank by interior weight (1=most boundary, 4=most interior) |
| `combined_score` | `np.ndarray[int]` | `(4,)` | Sum of ranks; range [2, 8] |

**Formula**:
```python
rank_mean = np.argsort(np.argsort(pred_means)) + 1
rank_weight = np.argsort(np.argsort(interior_weight)) + 1
combined_score = rank_mean + rank_weight
```

**Validation**:
- Each rank array contains values `{1, 2, 3, 4}` (or with ties, fractional ranks via argsort).
- `combined_score` ∈ `[2, 8]`.
- Higher `combined_score` = more desirable (better mean AND more interior).

### 4. Selection Outputs

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `above_median` | `np.ndarray[bool]` | `(4,)` | Candidates with `combined_score >= median` |
| `best_ip_idx` | `int` | scalar | Index of selected candidate (above-median, farthest from data) |
| `best_point` | `np.ndarray[float64]` | `(5,)` | Final selected point in normalised [0,1]⁵ space |

**Validation**:
- `above_median.sum() >= 1` (at least one candidate passes filter).
- `best_ip_idx ∈ {0, 1, 2, 3}`.
- `best_point` satisfies feasibility bounds: all `≥ 0.01`, `x₄ ≥ 0.10`, all `≤ 1.0`.

### 5. Visualisation Grid (Panel 3)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `top2` | `np.ndarray[int]` | `(2,)` | Indices of 2 most relevant dims (highest 1/ℓ) |
| `grid` | `np.ndarray[float64]` | `(80,)` | Linspace from 0 to 1 for heatmap axes |
| `w_grid` | `np.ndarray[float64]` | `(6400,)` | Penalty weight on 80×80 grid, reshaped to `(80,80)` for plotting |

**Validation**:
- `w_grid.min() ≈ FLOOR`, `w_grid.max() ≈ 1.0`.
- `w_grid` is symmetric about `(0.5, 0.5)` in the projected 2D plane.

---

## Comparison: F5 vs F6

| Aspect | F5 (015) | F6 (016) |
|--------|----------|----------|
| Dimensions | 4 | 5 |
| Output sign | Positive | All negative |
| Re-scoring | `pred_means_orig * w(x)` (multiplicative) | Rank-based `combined_score` |
| Variable name | `pred_means_orig` | `pred_means` |
| Transform | Manual `expm1` + z-score inverse | `Standardize(m=1)` auto-untransform |
| Bounds | `[[0,0,0,0],[1,1,1,1]]` uniform | `[[0.01,…,0.10],[1,…,1]]` feasibility |
| Median filter input | `weighted_means` (float) | `combined_score` (int, rank-based) |
| scipy dependency | No | No |
| Viz Panel 3 | Penalty heatmap (2D, x₀ vs x₁) | Penalty heatmap (2D, top-2 ARD dims) |

---

## State Transitions

```text
candidates (4,5) ──► interior_weight (4,) ──► rank_weight (4,) ──┐
                                                                   ├──► combined_score (4,) ──► above_median (4,) ──► best_ip_idx ──► best_point (5,)
pred_means (4,)  ──────────────────────────► rank_mean (4,)   ──┘
```
