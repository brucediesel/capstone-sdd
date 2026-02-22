# Quickstart: Week 7 — F1 Hurdle Model

**Branch**: `005-week7-pe-surrogates`  
**Date**: 2026-02-22

## What is being built

A new "Week 7" section appended to `functions/f1/f1.ipynb`. The section replaces the polynomial surrogate (Weeks 5–6) with a two-stage hurdle model and introduces a weighted UCB acquisition function with local penalization.

**No new files are created.** All work is new cells inserted into the existing notebook.

## Prerequisites

```bash
# From repo root — confirm on the correct branch
git branch   # should show * 005-week7-pe-surrogates

# Confirm the Week 7 data files exist
ls data/f1/ | grep "Week 7"
# Expected: updated_inputs - Week 7.npy  updated_outputs - Week 7.npy

# Confirm required packages are installed
python -c "import numpy, matplotlib, sklearn; print('OK')"
```

## Implementation steps (in order)

### Step 1 — Open the notebook

Open `functions/f1/f1.ipynb` in VS Code (or Jupyter). Scroll to the bottom of the Week 6 section (after the "Week 6 — Format Submission Query" cell). All new cells go after this point, before the large research-note cell.

### Step 2 — Add cells in order

Follow [contracts/cell-contracts.md](contracts/cell-contracts.md) for exact cell content and variable contracts. Cell order:

| # | Type | Content summary |
|---|------|----------------|
| W7-01 | Markdown | Section header + rationale |
| W7-02 | Code | Load & validate Week 7 data |
| W7-03 | Markdown | Hyperparameter rationale table |
| W7-04 | Code | Hyperparameter constants |
| W7-05 | Code | Stage 1: fit CalibratedClassifierCV |
| W7-06 | Code | Stage 2: fit RandomForestRegressor on log1p(y) |
| W7-07 | Code | Weighted UCB + local penalization → next_x |
| W7-08 | Code | 3-panel surrogate + acquisition surface plot |
| W7-09 | Code | Convergence plot |
| W7-10 | Code | Format submission query |

### Step 3 — Run the section

Run cells W7-02 through W7-10 in order. Expected outputs:

- **W7-02**: prints data shape, input range validation, positive/negative class balance
- **W7-04**: prints all 8 hyperparameter values
- **W7-05**: prints Stage 1 training accuracy
- **W7-06**: prints Stage 2 training R²
- **W7-07**: prints best UCB score, proposed candidate, minimum distance to existing data (must be ≥ 0.05)
- **W7-08**: renders 3-panel plot (hurdle surface, uncertainty surface, penalized UCB surface)
- **W7-09**: renders convergence plot
- **W7-10**: prints formatted submission string

### Step 4 — Verify acceptance criteria

Check each item before submitting:

- [ ] All cells execute without errors
- [ ] W7-07 output confirms `Min distance to existing data ≥ 0.050`
- [ ] W7-10 output is exactly `X.XXXXXX-X.XXXXXX` (6 decimal places, no spaces)
- [ ] Both values in submission are within [0.000000, 0.999999]
- [ ] No existing cells (cells 1–55) were modified or deleted
- [ ] The 3 visualization panels have the correct titles, colorbars, and the yellow star at the proposed point

## Key hyperparameters and rationale

| Hyperparameter | Value | One-line rationale |
|---|---|---|
| `C_STAGE1` | `1.0` | Default LR regularisation; balanced for 17 samples |
| `N_ESTIMATORS` | `100` | Standard RF; sufficient diversity for 2D |
| `MAX_DEPTH` | `3` | Prevents overfitting on ≤10 positive samples |
| `KAPPA` | `3.0` | Exploration-focused; matches Week 6 value, justified by zero improvement after 3 submissions |
| `PENALTY_RADIUS` | `0.15` | ~10.6% of input space diagonal; safely above SC-003 minimum of 0.05 |
| `N_CANDIDATES` | `20_000` | Matches prior weeks; sufficient coverage of the 2D space |
| `GRID_RES` | `50` | Matches prior weeks; 50×50 grid for visualization |
| `MIN_POSITIVE` | `3` | Minimum positive samples for Stage 2 to be fitted |

## Fallback behavior

If fewer than 3 positive outputs are present in the Week 7 data:

1. Cell W7-02 prints a `⚠ WARNING` and sets `FALLBACK_MODE = True`
2. Cell W7-06 is skipped
3. Cell W7-07 uses pure random exploration (uniform sampling over the penalized space) instead of the hurdle acquisition

## Submission

Copy the output from Cell W7-10 exactly. Example format: `0.123456-0.654321`
