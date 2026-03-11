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

---

## F1 Optimisation Run — Quickstart

**Spec**: [spec-f1-optimisation.md](spec-f1-optimisation.md)

### Prerequisites

- All review notebook prerequisites (above)
- BoTorch and GPyTorch installed in the `sdd-dev` environment
- Week 10 data present: `data/f1/updated_inputs - Week 10.npy`, `data/f1/updated_outputs - Week 10.npy`

### Running the F1 Optimisation

```bash
# From repository root
cd functions/f1
jupyter notebook "f1 - week 10.ipynb"
# Run All Cells (Kernel > Restart & Run All)
```

The notebook has two sections:
1. **Cells 1–12**: Review & evaluation (existing) — loads data, plots convergence, pair plots, evaluates performance, suggests improvements
2. **Cells 13–19**: Optimisation run (new) — fits SFGP, runs qLogNEI, selects candidate, visualises surrogate, shows updated convergence

### Expected Output from Optimisation Cells

1. **Configuration** — Prints all hyperparameter values
2. **Data Preparation** — Prints tensor shapes and log-transformed output range
3. **GP Fitting** — Prints fitted hyperparameters after 15 MLL restarts:
   - Lengthscales (2 values, one per dimension)
   - Noise variance
   - Output scale
   - Best MLL loss
4. **Acquisition** — Prints all 4 candidates, selection rationale, and formatted submission:
   ```
   >>> SUBMISSION: 0.xxxxxx-0.yyyyyy
   ```
5. **Surrogate Visualisation** — 3-panel contour: mean, uncertainty, acquisition surface
6. **Convergence Plot** — Running best with proposed point marked

### Verification Checklist

- [ ] All cells execute without errors
- [ ] GP hyperparameters are reasonable (lengthscales 0.01–2.0, noise > 0)
- [ ] Submission point is in valid range [0.0, 0.999999] for both dimensions
- [ ] No duplicate warning printed
- [ ] 3-panel contour renders with correct colour overlays
- [ ] Convergence plot shows proposed point in green
