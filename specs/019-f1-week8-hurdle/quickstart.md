# Quickstart: F1 Week 8 — Hurdle Model Bayesian Optimisation

**Feature**: `019-f1-week8-hurdle`  
**Date**: 2026-03-01

## Prerequisites

- Python 3.x with Jupyter kernel available
- Required packages: `numpy`, `matplotlib`, `scikit-learn`
- Data files present in `./data/f1/`:
  - `updated_inputs - Week 8.npy` (18×2 array)
  - `updated_outputs - Week 8.npy` (18-element array)

## Running

1. **Check out the feature branch**:
   ```bash
   git checkout 019-f1-week8-hurdle
   ```

2. **Open the notebook**:
   ```
   functions/f1/f1 - week 8.ipynb
   ```

3. **Run all cells** — the notebook is self-contained and executes top-to-bottom without external dependencies.

4. **Collect the submission query** from the final code cell output. It will be in the format `0.xxxxxx-0.xxxxxx`.

## Expected Outputs

| Output | Description |
|--------|-------------|
| Data table | 18 rows showing all inputs and outputs with best highlighted |
| 3-panel contour plot | Hurdle mean, uncertainty, and penalised acquisition surface |
| Convergence plot | Running maximum across 18 observations |
| Submission query | Formatted string `0.xxxxxx-0.xxxxxx` for Week 9 submission |

## Key Hyperparameters

All defined in a single cell near the top of the notebook. To adjust:

| Parameter | Default | Effect of Increasing |
|-----------|---------|---------------------|
| KAPPA | 3.0 | More exploration (wider search) |
| PENALTY_RADIUS | 0.15 | Larger exclusion zone around existing points |
| STEEPNESS | 0.1 | Stronger boundary suppression |
| N_CANDIDATES | 20000 | Better acquisition surface coverage |

## Fallback Behaviour

If fewer than 3 positive outputs (`y > 0`) exist in the data, the notebook automatically enters **fallback mode** — the random forest regressor is skipped and a pure exploration strategy is used. This is documented in a markdown cell and signalled in the printed output.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` on data load | Ensure Week 8 data files exist in `./data/f1/` |
| All candidates too close to existing points | Increase `N_CANDIDATES` or reduce `PENALTY_RADIUS` |
| Proposed point on boundary | Increase `STEEPNESS` (e.g., to 0.5) |
