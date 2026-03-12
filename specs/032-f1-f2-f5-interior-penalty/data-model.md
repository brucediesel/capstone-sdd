# Data Model: F1, F2 & F5 Interior Penalty

**Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)

## Entities

### 1. Interior Penalty Constants

| Field | Type | Value | Description |
|-------|------|-------|-------------|
| `STEEPNESS` | `float` | `0.02` | Controls boundary suppression aggressiveness. Exponent = 2×STEEPNESS = 0.04 |
| `FLOOR` | `float` | `0.01` | Minimum penalty weight; prevents zero acquisition values at boundaries |

**Location**: Defined in the imports/configuration cell of each notebook (FR-009).

### 2. Interior Penalty Weight

| Field | Type | Shape | Range | Description |
|-------|------|-------|-------|-------------|
| `interior_weight` | `np.ndarray` | `(n_survivors,)` | `[0.01, 1.0]` | Per-candidate penalty weight |

**Formula**: `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`

**Computed from**: Candidate coordinates `x ∈ [0, 1]^d` where d=2 (F1, F2) or d=4 (F5).

**Behaviour**:
- At boundary (any xᵢ = 0 or 1): sin(πxᵢ) = 0 → w(x) = FLOOR = 0.01
- At interior (all xᵢ ≈ 0.5): sin(πxᵢ) ≈ 1 → w(x) ≈ 1.0
- With STEEPNESS=0.02: w(x) ≈ 1.0 everywhere except within ~0.01 of boundary

### 3. Penalised Acquisition Value

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `acq_values` | `torch.Tensor` | `(n_survivors,)` | Per-candidate acquisition values from qLogNEI |
| `penalised_acq` | `np.ndarray` | `(n_survivors,)` | `acq_values * interior_weight` |

**Selection rule**: `best_idx = argmax(penalised_acq)` among distance-filter survivors.

## State Transitions

```
optimize_acqf(q=4)
      │
      ▼
  [4 candidates]
      │
      ▼
  Distance filter (median/25th %ile gate + max min-distance)
      │
      ▼
  [n survivors] ← NEW: compute interior_weight for each
      │
      ▼
  Evaluate acqf per-candidate → acq_values
      │
      ▼
  penalised_acq = acq_values * interior_weight
      │
      ▼
  Select argmax(penalised_acq)
      │
      ▼
  x_new → clamp to [0, 0.999999] → submission
```

## Validation Rules

- `interior_weight[i] ∈ [FLOOR, 1.0]` for all candidates
- `penalised_acq[i] ≥ 0` (acquisition values from qLogNEI are non-negative)
- Final `x_new` values in `[0.0, 0.999999]`
- Submission dimensionality: 2D for F1/F2, 4D for F5
