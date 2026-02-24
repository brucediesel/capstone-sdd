# Data Model: F5 Interior Penalty

**Feature**: 015-f5-interior-penalty  
**Date**: 2026-02-24

## Entities

### Interior Penalty Function

A smooth multiplicative weight applied to the posterior mean at each candidate point, used to re-rank the `optimize_acqf` batch.

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `STEEPNESS` | `float` (> 0) | Controls boundary suppression width. Higher = steeper transition, narrower penalty band. Default: 1.0 (reduced from F1's 2.0 due to 4D multiplicative effect) |
| `FLOOR` | `float` (0 < FLOOR < 1) | Minimum penalty value at boundaries. Prevents exact zeros. Default: 0.01 |

**Computed values** (per candidate point):
| Field | Type | Description |
|-------|------|-------------|
| `interior_weight` | `np.ndarray` shape `(4,)` | The penalty factor w(x) ∈ [FLOOR, 1.0] for each of the 4 candidates |
| `weighted_means` | `np.ndarray` shape `(4,)` | `pred_means_orig * interior_weight` — penalty-weighted posterior mean per candidate |

**Formula**:
```
w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ₌₁⁴ sin(π · xᵢ)^(2 · STEEPNESS)
```

**Relationships**:
- Depends on: `candidates` (4×4 tensor from Cell 54/optimize_acqf), `pred_means_orig` (shape (4,) from Cell 54)
- Feeds into: distance-based selection (median filter uses `weighted_means` instead of `pred_means_orig`), visualisation grid, submission query

### Existing Entities (unchanged, referenced)

| Entity | Source Cell | Type | Role in this feature |
|--------|-----------|------|---------------------|
| `X_raw` | Cell 51 | `np.ndarray` (27, 4) | Training input coordinates; used for min-distance validation and scatter plots |
| `y_raw` | Cell 51 | `np.ndarray` (27,) | Training outputs; used for convergence plot |
| `X_train` | Cell 53 | `torch.Tensor` (27, 4) | Torch training inputs; used for posterior evaluation |
| `Y_train` | Cell 53 | `torch.Tensor` (27, 1) | Torch training outputs (standardised) |
| `best_model` | Cell 53 | `SingleTaskGP` | Fitted GP model; used for posterior mean/std on vis grid |
| `ls` | Cell 53 | `np.ndarray` (4,) | Fitted lengthscales; used to select top-2 dims for vis |
| `y_mean` | Cell 53 | `float` | Log-transform mean; used for inverse transform |
| `y_std_val` | Cell 53 | `float` | Log-transform std; used for inverse transform |
| `candidates` | Cell 54 | `torch.Tensor` (4, 4) | Batch of 4 candidate points from optimize_acqf |
| `pred_means_orig` | Cell 54 | `np.ndarray` (4,) | Posterior mean at candidates in original scale |
| `dists` | Cell 54 | `torch.Tensor` (4,) | Min distance from each candidate to training data |
| `best_point` | Cell 54 | `np.ndarray` (4,) | Selected point from Week 7 (pre-penalty); used to fix viz slice dims |
| `nei` | Cell 54 | `qLogNoisyExpectedImprovement` | Acquisition function object (not directly used — kept for audit) |

## State Transitions

This feature has no state machine — it is a single-pass computation:

```
[Existing W7 variables] → Compute interior penalty for 4 candidates
                        → Multiply with pred_means_orig (re-score)
                        → Apply median filter on weighted_means
                        → Distance-based selection among filtered candidates
                        → Visualise (3-panel with penalised mean surface)
                        → Convergence plot
                        → Format submission query
```

## Validation Rules

- `STEEPNESS > 0` (zero disables the penalty; negative is undefined for sin^power)
- `0 < FLOOR < 1` (FLOOR=0 defeats the purpose; FLOOR=1 disables the penalty)
- `interior_weight` values must be in `[FLOOR, 1.0]` for all candidates
- Selected point must have all 4 coordinates in `[0.0, 0.999999]`
- Selected point must be ≥ 0.05 from any existing data point (SC-001, FR-009)

## Key Differences from F1 (014)

| Aspect | F1 (014) | F5 (015) |
|--------|----------|----------|
| Dimensions | 2D | 4D |
| Candidates | 20,000 random uniform | 4 from `optimize_acqf` (q=4) |
| Penalty applied to | `acq_penalized` (UCB × local penalty) | `pred_means_orig` (posterior mean) |
| STEEPNESS default | 2.0 | 1.0 (reduced for 4D) |
| Selection method | argmax of penalised acq | median filter → farthest from data |
| Surrogate framework | scikit-learn (LR + RF hurdle) | BoTorch (SingleTaskGP) |
| Viz Panel 3 | Penalised acquisition heatmap | Penalised mean heatmap (proxy for penalised acquisition) |
