# Quickstart: F5 Interior Penalty

**Feature**: 015-f5-interior-penalty  
**Branch**: `015-f5-interior-penalty`

## Prerequisites

1. Ensure you are on branch `015-f5-interior-penalty`:
   ```bash
   git checkout 015-f5-interior-penalty
   ```

2. The conda environment `sdd-dev` must be active with: numpy, torch, botorch, gpytorch, matplotlib

3. Week 7 data files must exist:
   - `data/f5/updated_inputs - Week 7.npy` (shape: 27×4)
   - `data/f5/updated_outputs - Week 7.npy` (shape: 27)

## Running

1. Open `functions/f5/f5.ipynb` in Jupyter / VS Code
2. Run **all cells from the beginning** through the existing Week 7 section (cells 0–57)
   - This loads data, trains the GP Matérn-5/2, runs qLogNEI acquisition (q=4), visualises, and produces the Week 7 submission
3. Continue running the **new Interior Penalty section** (cells 58–63):
   - Cell 58: Markdown header — hyperparameter documentation (no execution needed)
   - Cell 59: Hyperparameter constants — sets `STEEPNESS=1.0`, `FLOOR=0.01`
   - Cell 60: Interior penalty computation + candidate re-scoring + selection → prints re-ranked candidates and selected point
   - Cell 61: 3-panel visualisation → GP mean, GP std, penalised mean surface
   - Cell 62: Convergence plot → running best observed value
   - Cell 63: Submission query → prints formatted `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`

## Adjusting Hyperparameters

To change the boundary avoidance strength, edit Cell 59:

```python
STEEPNESS = 1.0   # ↑ = stronger boundary avoidance (try 0.5 for gentler, 2.0 for aggressive)
                   # NOTE: 1.0 in 4D ≈ 2.0 in 2D due to multiplicative effect
FLOOR     = 0.01  # Minimum penalty at boundary (keep > 0)
```

Then re-run cells 59–63.

## Expected Output

- The selected point should have all four coordinates at least 0.05 away from both 0 and 1 (i.e. in [0.05, 0.95])
- Panel 3 of the visualisation should show near-zero penalised mean values along all four edges of the 2D slice
- The submission query should be a valid `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` string

## Key Difference from F1 (014)

| Aspect | F1 | F5 |
|--------|----|----|
| Penalty applied to | 20K random candidate scores | 4 candidates from `optimize_acqf` |
| Scoring basis | Weighted UCB × local penalty × interior | Posterior mean × interior penalty |
| Selection | argmax | Median filter → farthest from data |
| STEEPNESS default | 2.0 | 1.0 (4D needs less per-dim steepness) |
| Viz Panel 3 | Penalised acquisition surface | Penalised mean surface |

## Verification

After running all cells, check:
- [ ] No error output in any cell
- [ ] Selected point coordinates printed in Cell 60
- [ ] All 4 coordinates in [0.05, 0.95]
- [ ] Min distance to existing data ≥ 0.05
- [ ] 3-panel plot shows boundary suppression in Panel 3
- [ ] Submission query formatted correctly in Cell 63
