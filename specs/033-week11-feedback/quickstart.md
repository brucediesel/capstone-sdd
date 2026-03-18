# Quickstart: Week 11 Performance Review & Feedback

**Feature**: 033-week11-feedback
**Branch**: `033-week11-feedback`

## Prerequisites

- Python environment with `numpy` and `matplotlib` installed (conda `sdd-dev`)
- Week 11 data files present in `./data/f1/` through `./data/f8/`
- Jupyter notebook runtime available

## Running a Single Notebook

```bash
# From repository root
cd functions/f1
jupyter notebook "f1 - week 11.ipynb"
# Run All Cells
```

Each notebook is self-contained. Run any function's notebook independently.

## Expected Output Per Notebook

1. **Data Summary** — Table showing loaded sample counts and value ranges
2. **Convergence Plot** — Running best (maximum) objective over all samples
   - Blue region: initial samples
   - Orange region: weekly submissions
   - F1 only: logarithmic y-axis (negatives zeroed)
3. **2D Pair Plots** — One subplot per unique pair of input dimensions
   - Blue dots: initial samples (unmarked)
   - Orange dots: submission samples (numbered by week 3–11)
   - Green star: overall best output location (no annotation)
   - Legend: Initial, Submissions, Best
   - Grid layout scales with dimensionality
4. **Performance Evaluation** — Markdown cell summarising:
   - Current Week 10 strategy (surrogate + acquisition)
   - Best value achieved
   - Improvement trajectory
   - Stalling detection (≥3 consecutive no-improvement = stalling)
5. **Strategy Improvements** — Markdown cell with specific, actionable suggestions

## Notebook Inventory

| Notebook | Location | Input Dims | Pairs | Total Samples |
|----------|----------|-----------|-------|---------------|
| `f1 - week 11.ipynb` | `functions/f1/` | 2 | 1 | 21 |
| `f2 - week 11.ipynb` | `functions/f2/` | 2 | 1 | 21 |
| `f3 - week 11.ipynb` | `functions/f3/` | 3 | 3 | 26 |
| `f4 - week 11.ipynb` | `functions/f4/` | 4 | 6 | 41 |
| `f5 - week 11.ipynb` | `functions/f5/` | 4 | 6 | 31 |
| `f6 - week 11.ipynb` | `functions/f6/` | 5 | 10 | 31 |
| `f7 - week 11.ipynb` | `functions/f7/` | 6 | 15 | 41 |
| `f8 - week 11.ipynb` | `functions/f8/` | 8 | 28 | 51 |

## Verification

After running a notebook, verify:
- All cells execute without errors
- Convergence plot shows clear initial/submission distinction
- Pair plot point numbers match expected week range (3–11)
- Green star appears on each pair plot at the overall best sample location
- Legend contains "Initial", "Submissions", and "Best" entries
- Performance evaluation markdown is populated with specific metrics
- Improvement suggestions reference the current Week 10 strategy by name
