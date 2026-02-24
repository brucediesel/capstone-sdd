# Quickstart: F1 Interior Penalty

**Feature**: 014-f1-interior-penalty  
**Branch**: `014-f1-interior-penalty`

## Prerequisites

1. Ensure you are on branch `014-f1-interior-penalty`:
   ```bash
   git checkout 014-f1-interior-penalty
   ```

2. The conda/venv environment `sdd-dev` must be active with: numpy, matplotlib, scikit-learn, botorch, gpytorch

3. Week 7 data files must exist:
   - `data/f1/updated_inputs - Week 7.npy` (shape: 17×2)
   - `data/f1/updated_outputs - Week 7.npy` (shape: 17)

## Running

1. Open `functions/f1/f1.ipynb` in Jupyter / VS Code
2. Run **all cells from the beginning** through the existing Week 7 section (cells 1–65)
   - This trains the hurdle model, computes the weighted UCB acquisition, and sets all required kernel variables
3. Continue running the **new Interior Penalty section** (cells 66–71):
   - Cell 66: Markdown header (no execution needed)
   - Cell 67: Hyperparameter constants — sets `STEEPNESS=2.0`, `FLOOR=0.01`
   - Cell 68: Interior penalty computation + candidate selection → prints selected point
   - Cell 69: 3-panel visualisation → shows surrogate with interior-penalised acquisition
   - Cell 70: Convergence plot → shows running best
   - Cell 71: Submission query → prints formatted `0.xxxxxx-0.xxxxxx`

## Adjusting Hyperparameters

To change the boundary avoidance strength, edit Cell 67:

```python
STEEPNESS = 2.0   # ↑ = stronger boundary avoidance (try 1.0 for gentler, 5.0 for aggressive)
FLOOR     = 0.01  # Minimum penalty at boundary (keep > 0)
```

Then re-run cells 67–71.

## Expected Output

- The selected point should have both coordinates in approximately [0.05, 0.95]
- Panel 3 of the visualisation should show near-zero acquisition values along all four edges
- The submission query should be a valid `0.xxxxxx-0.xxxxxx` string

## Verification

After running all cells, check:
- [ ] No error output in any cell
- [ ] Selected point coordinates printed in Cell 68
- [ ] Min distance to existing data ≥ 0.05
- [ ] 3-panel plot shows boundary suppression in Panel 3
- [ ] Submission query formatted correctly in Cell 71
