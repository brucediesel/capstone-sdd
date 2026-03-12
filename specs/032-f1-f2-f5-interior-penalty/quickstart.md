# Quickstart: F1, F2 & F5 Interior Penalty

**Date**: 2026-03-12 | **Branch**: `031-f4-f8-week10-optimisation`

## What This Feature Does

Adds a very shallow interior penalty to the F1, F2, and F5 week 10 notebooks. The penalty softly discourages candidates near domain boundaries ([0,1]^d edges) by multiplicatively re-scoring acquisition values after the existing distance-based selection filter.

## Key Parameters

```python
STEEPNESS = 0.02    # Very shallow — near-no-op except at exact boundaries
FLOOR     = 0.01    # Minimum penalty weight (boundary candidates keep 1% of acq value)
```

## Formula

```
w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)
penalised_acq = acquisition_value × w(x)
```

## Integration Pattern (per notebook)

### Step 1: Add constants to the configuration cell

```python
# Interior penalty hyperparameters
STEEPNESS = 0.02    # boundary suppression steepness (very shallow)
FLOOR     = 0.01    # minimum penalty weight
```

### Step 2: After existing distance-based selection, add a new cell

```python
# ── Interior Penalty Re-Scoring ──────────────────────────────────
# Compute penalty weight for each distance-filter survivor
survivors = candidates[qualified_idx]          # or qualified_indices for F5
surv_np   = survivors.detach().cpu().numpy()

interior_weight = FLOOR + (1 - FLOOR) * np.prod(
    np.sin(np.pi * surv_np) ** (2 * STEEPNESS), axis=1
)

# Evaluate acquisition function per-candidate
with torch.no_grad():
    acq_values = torch.tensor([
        acqf(survivors[i:i+1].unsqueeze(0)).item()
        for i in range(len(survivors))
    ])

# Penalised re-scoring
penalised_acq = acq_values.numpy() * interior_weight
best_pen_idx  = int(np.argmax(penalised_acq))

# Report effect
original_best = 0  # index of distance-selected best
print(f"Interior penalty weights: {interior_weight}")
print(f"Original selection: candidate {original_best}, "
      f"Penalised selection: candidate {best_pen_idx}")
print(f"Penalty changed selection: {best_pen_idx != original_best}")

# Update x_new
x_new = surv_np[best_pen_idx]
x_new = np.clip(x_new, 0.0, 0.999999)
```

## Files Modified

| File | Change |
|------|--------|
| `functions/f1/f1 - week 10.ipynb` | Add STEEPNESS/FLOOR constants + interior penalty cell |
| `functions/f2/f2 - week 10.ipynb` | Add STEEPNESS/FLOOR constants + interior penalty cell |
| `functions/f5/f5 - week 10.ipynb` | Add STEEPNESS/FLOOR constants + interior penalty cell |

## Verification

Run each notebook end-to-end:
1. All penalty weights should be in [0.01, 1.0]
2. Final submission format: `0.XXXXXX-0.XXXXXX` (2D) or `0.XXXXXX-0.XXXXXX-0.XXXXXX-0.XXXXXX` (4D)
3. Penalty effect (whether selection changed) is printed
