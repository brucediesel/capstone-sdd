# Contracts: Week 11 Performance Review & Feedback

No API contracts for this feature — all outputs are Jupyter notebook visualisations and markdown cells.

## Notebook Cell Contract

Each `fX - week 11.ipynb` notebook follows an identical cell structure:

| Cell | Type | Content |
|------|------|---------|
| 0 | Markdown | Title, objective, Week 10 strategy summary |
| 1 | Code | Imports (numpy, matplotlib, itertools, math) |
| 2 | Markdown | Configuration section header |
| 3 | Code | Config constants (FUNC_NUM, N_DIMS, N_INITIAL, WEEK, USE_LOG_SCALE, DATA_DIR) + data loading + summary table |
| 4 | Markdown | Convergence section header |
| 5 | Code | Convergence plot (running max, blue/orange split, log scale for F1) |
| 6 | Markdown | Pair plots section header |
| 7 | Code | Pair plots with numbered submissions + green star best marker + legend |
| 8 | Markdown | Performance evaluation (best value, improvements, stalling) |
| 9 | Code | Stalling detection analysis (quantitative metrics) |
| 10 | Markdown | Strategy proposals for next submission |

**Visual outputs**:
- Convergence plot: 1 figure per notebook
- Pair plots: $\binom{d}{2}$ subplots per notebook (1 to 28 depending on dimensionality)
- Best marker: Green star (`*`, s=500, zorder=5) on each pair plot subplot at `inputs[best_idx]` coordinates
- Legend includes: Initial (blue patch), Submissions (orange patch), Best (green star)
