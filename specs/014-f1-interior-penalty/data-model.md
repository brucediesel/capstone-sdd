# Data Model: F1 Interior Penalty

**Feature**: 014-f1-interior-penalty  
**Date**: 2026-02-24

## Entities

### Interior Penalty Function

A smooth multiplicative weight applied to the acquisition score at each candidate point.

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `STEEPNESS` | `float` (> 0) | Controls boundary suppression width. Higher = steeper transition, narrower penalty band |
| `FLOOR` | `float` (0 < FLOOR < 1) | Minimum penalty value at boundaries. Prevents exact zeros |

**Computed values** (per candidate point):
| Field | Type | Description |
|-------|------|-------------|
| `interior_weight` | `np.ndarray` shape `(N_CANDIDATES,)` | The penalty factor w(x) ∈ [FLOOR, 1.0] for each candidate |
| `acq_with_interior` | `np.ndarray` shape `(N_CANDIDATES,)` | Final acquisition = `acq_penalized * interior_weight` |

**Formula**:
```
w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(π · xᵢ)^(2 · STEEPNESS)
```

**Relationships**:
- Depends on: `X_cand` (candidate points from W7-07), `acq_penalized` (existing weighted UCB with local penalization from W7-07)
- Feeds into: candidate selection (argmax), visualisation grid, submission query

### Existing Entities (unchanged, referenced)

| Entity | Source Cell | Role in this feature |
|--------|-----------|---------------------|
| `X_w7` | W7-02 | Training data coordinates; used for min-distance validation |
| `y_w7` | W7-02 | Training data outputs; used for convergence plot |
| `stage1_clf` | W7-05 | Classifier P(y>0); used to recompute acquisition on viz grid |
| `stage2_rf` | W7-06 | RF regressor; used to recompute acquisition on viz grid |
| `acq_penalized` | W7-07 | UCB × local penalty; input to interior penalty multiplication |
| `KAPPA` | W7-04 | UCB exploration parameter; reused in new acquisition computation |
| `PENALTY_RADIUS` | W7-04 | Local penalty radius; reused in new acquisition computation |
| `N_CANDIDATES` | W7-04 | Candidate count; reused |
| `GRID_RES` | W7-04 | Visualisation grid resolution; reused |
| `FALLBACK_MODE` | W7-02 | Degenerate data guard; controls Stage 2 availability |
| `y_binary` | W7-02 | Positive mask for scatter plot colours |

## State Transitions

This feature has no state machine — it is a single-pass computation:

```
[Existing W7 variables] → Compute interior penalty → Multiply with acquisition → Select best → Visualise → Format query
```

## Validation Rules

- `STEEPNESS > 0` (zero disables the penalty; negative is undefined for sin^power)
- `0 < FLOOR < 1` (FLOOR=0 defeats the purpose; FLOOR=1 disables the penalty)
- `interior_weight` values must be in `[FLOOR, 1.0]` for all candidates
- Selected point must have all coordinates in `[0.0, 0.999999]`
- Selected point must be ≥ 0.05 from any existing data point (SC-001, FR-009)
