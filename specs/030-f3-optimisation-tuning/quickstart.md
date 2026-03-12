# Quickstart: F3 Week 10 — Optimisation Tuning

**Feature**: 030-f3-optimisation-tuning  
**Branch**: `030-f3-optimisation-tuning`

## Prerequisites

- Python environment (`sdd-dev`) with BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib installed
- Week 10 data present: `data/f3/updated_inputs - Week 10.npy`, `data/f3/updated_outputs - Week 10.npy`
- Jupyter notebook runtime available

## Running the Notebook

```bash
# From repository root
cd functions/f3
jupyter notebook "f3 - week 10.ipynb"
# Kernel > Restart & Run All
```

The notebook has two sections:
1. **Cells 1–12**: Review & evaluation (existing) — loads data, convergence, pair plots, performance evaluation, strategy improvements
2. **Cells 13–19**: Optimisation run (new) — fits SFGP with shift transform, runs qLogNEI q=3, selects candidate, visualises 2D contour slices, shows convergence

## Expected Output from Optimisation Cells

1. **Section Header** — Markdown describing week 10 strategy changes
2. **Configuration** — Prints all hyperparameter values with week 9 → 10 changes noted
3. **Data Preparation** — Prints tensor shapes, raw output range [-0.399, -0.031], shifted range [0, 0.368], y_min value
4. **GP Fitting** — Prints fitted hyperparameters after 40 MLL restarts:
   - 3 lengthscales (one per dimension)
   - Noise variance (≥ 1e-4)
   - Output scale
   - Best MLL loss
   - Number of restarts converging to same optimum
5. **Acquisition** — Prints all 3 candidates with shifted posterior means, selection rationale, and formatted submission:
   ```
   >>> SUBMISSION: 0.xxxxxx-0.yyyyyy-0.zzzzzz
   ```
6. **2D Contour Slices** — 2×3 figure: posterior mean and acquisition surface for all 3 input dimension pairs
7. **Convergence** — Plot with running best in original scale, proposed point as green star with reverse-shifted prediction

## Verification

After running, verify:
- All cells execute without errors
- GP hyperparameters are reasonable (finite lengthscales, noise ≥ 1e-4, outputscale > 0)
- Submission format: 3 values separated by hyphens, 6 decimal places each, all in [0.0, 0.999999]
- Contour slices show 3 pairs with correct dimension labels
- Convergence plot shows proposed point in original (negative) output scale
- No duplicate submission (or duplicate warning printed)
