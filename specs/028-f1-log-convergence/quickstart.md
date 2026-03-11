# Quickstart: F1 Log-Scale Convergence Plot

**Feature**: 028-f1-log-convergence
**Date**: 2026-03-11 (updated after user clarification)

## Prerequisites

- Branch `028-f1-log-convergence` checked out
- Python environment `sdd-dev` activated
- Weekly data files present in `data/f1/`

## What Changed

**Single cell modified**: Cell 13 of `functions/results/process_results.ipynb`

Two additions inside the convergence graph loop, both guarded by `if fn == 'f1'`:

1. **Before plotting** — clip negative outputs to zero:
```python
if fn == 'f1':
    out = np.maximum(out, 0)
```

2. **After axis setup** — switch to log scale:
```python
if fn == 'f1':
    ax.set_yscale('log')
    ax.set_ylabel('Output Value (log)')
```

**No other cells or files are modified.**

## Verification

1. Run all cells of `process_results.ipynb` (enter any valid week number when prompted)
2. Check the convergence graph output:
   - **F1 subplot**: y-axis should show logarithmic tick spacing (e.g., 1e-228, 1e-170, 1e-54, 1e-25)
   - **F2–F8 subplots**: should look identical to before (linear y-axis)
3. Confirm no error messages appear during Cell 13 execution
4. Note: zero and negative F1 values will not appear on the plot — this is expected

## Rollback

Revert to `master` branch or remove both `if fn == 'f1':` blocks from Cell 13.
