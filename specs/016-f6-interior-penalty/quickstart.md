# Quickstart: F6 Interior Penalty

**Feature**: 016-f6-interior-penalty  
**Date**: 2025-02-24

---

## Prerequisites

1. **Conda environment**: `sdd-dev` activated (`conda activate sdd-dev`)
2. **Python 3.11** with: `torch`, `botorch`, `gpytorch`, `numpy`, `matplotlib`
3. **Data files**: `data/f6/updated_inputs - Week 7.npy` and `data/f6/updated_outputs - Week 7.npy` present
4. **No new dependencies** — uses only NumPy for ranking (no scipy)

## Running

1. Open `functions/f6/f6.ipynb` in VS Code or Jupyter.
2. Run all cells top-to-bottom (Ctrl+Shift+Enter or "Run All").
3. New cells 60–65 execute after existing Week 7 cells.

## Hyperparameters

| Parameter | Value | Cell | Tunable |
|-----------|-------|------|---------|
| `STEEPNESS` | `1.0` | 61 | Yes — increase for narrower interior band |
| `FLOOR` | `0.01` | 61 | Yes — increase to soften boundary penalty |

## Verification Checklist

After running all cells, confirm:

- [ ] **Cell 61**: `interior_weight` printed with 4 values in `[0.01, 1.0]`
- [ ] **Cell 62**: Comparison table printed with ranks, combined scores; one candidate marked `◄`
- [ ] **Cell 62**: `best_point` has shape `(5,)`, all values ≥ 0.01, x₄ ≥ 0.10
- [ ] **Cell 63**: 3-panel figure displayed — mean heatmap, std heatmap, penalty heatmap
- [ ] **Cell 63**: Panel 3 shows `w(x)` colour range from ~0.01 (red, boundary) to ~1.0 (green, interior)
- [ ] **Cell 63**: `best_point` marked with star on all panels
- [ ] **Cell 64**: Convergence plot shows historical points + IP-selected predicted mean
- [ ] **Cell 65**: Submission point printed with 4 decimal places
- [ ] **Cell 65**: Feasibility assertions pass (no `AssertionError`)
- [ ] **Cell 65**: Penalty metadata printed (`STEEPNESS`, `FLOOR`, weight, score)
- [ ] **No existing cells modified** — cells 0–59 unchanged

## Key Differences from F5

| Aspect | F5 | F6 |
|--------|----|----|
| Re-scoring | Multiplicative `pred_means_orig * w(x)` | Rank-based `combined_score` |
| Output sign | Positive | All negative |
| Dimensions | 4 | 5 |
| Bounds | `[0,1]⁴` uniform | Feasibility: x₄ ≥ 0.10 |
| Transform | Manual `expm1` + z-score | Auto (`Standardize(m=1)`) |
| Panel 3 dim selection | Fixed x₀ vs x₁ | Top-2 ARD dims (dynamic) |

## Troubleshooting

- **`AssertionError` in Cell 65**: Check that `best_point[4] >= 0.10`. If the GP pushes milk to boundary, the penalty should catch this — investigate `interior_weight` values.
- **All `interior_weight ≈ FLOOR`**: All 4 candidates are near boundaries. Consider reducing `STEEPNESS` to 0.5 for a softer penalty.
- **`combined_score` ties**: The distance-based tiebreaker in Cell 62 handles this — the candidate farthest from training data wins.
