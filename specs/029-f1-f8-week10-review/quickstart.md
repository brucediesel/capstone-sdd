# Quickstart: Week 10 Performance Review & Visualisation

**Feature**: 029-f1-f8-week10-review  
**Branch**: `029-f1-f8-week10-review`

## Prerequisites

- Python environment with `numpy` and `matplotlib` installed
- Week 10 data files present in `./data/f1/` through `./data/f8/`
- Jupyter notebook runtime available

## Running a Single Notebook

```bash
# From repository root
cd functions/f1
jupyter notebook "f1 - week 10.ipynb"
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
   - Orange dots: submission samples (numbered by week 3–10)
   - Grid layout scales with dimensionality
4. **Performance Evaluation** — Markdown cell summarising:
   - Current week 9 strategy (surrogate + acquisition)
   - Best value achieved
   - Improvement trajectory
   - Stalling detection
5. **Strategy Improvements** — Markdown cell with specific, actionable suggestions

## Notebook Inventory

| Notebook | Location | Input Dims | Pairs |
|----------|----------|-----------|-------|
| `f1 - week 10.ipynb` | `functions/f1/` | 2 | 1 |
| `f2 - week 10.ipynb` | `functions/f2/` | 2 | 1 |
| `f3 - week 10.ipynb` | `functions/f3/` | 3 | 3 |
| `f4 - week 10.ipynb` | `functions/f4/` | 4 | 6 |
| `f5 - week 10.ipynb` | `functions/f5/` | 4 | 6 |
| `f6 - week 10.ipynb` | `functions/f6/` | 5 | 10 |
| `f7 - week 10.ipynb` | `functions/f7/` | 6 | 15 |
| `f8 - week 10.ipynb` | `functions/f8/` | 8 | 28 |

## Verification

After running a notebook, verify:
- All cells execute without errors
- Convergence plot shows clear initial/submission distinction
- Pair plot point numbers match expected week range (3–10)
- Performance evaluation markdown is populated with specific metrics
- Improvement suggestions reference the current strategy by name
