# Quickstart: Week 12 Bayesian Optimisation Loop (F1–F8)

**Feature**: 035-f1-f8-week12-optimisation
**Branch**: `035-f1-f8-week12-optimisation`

## Prerequisites

1. **Python environment**: pyenv `sdd-dev` activated (Python 3.14.2)
2. **Dependencies installed**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch 2.10.0, NumPy 2.4.1, Matplotlib 3.10.8
3. **Data available**: Week 11 `.npy` files in `./data/f1/` through `./data/f8/`
4. **Branch**: `git checkout 035-f1-f8-week12-optimisation`

## Implementation Approach

Each week 12 notebook is created by cloning the corresponding week 10 optimisation notebook and updating:

1. **WEEK constant**: 10 → 11 (data source week)
2. **Notebook title**: "Week 10" → "Week 12" (submission target)
3. **Submission label**: "WEEK 10 SUBMISSION" → "WEEK 12 SUBMISSION"
4. **Strategy description**: Updated to reference week 11 data context

No strategy, hyperparameter, or structural changes are made. The optimisation code is identical.

## Per-Function Source Mapping

| Target Notebook | Source Template | Key Strategy |
|----------------|----------------|-------------|
| `functions/f1/f1 - week 12.ipynb` | `functions/f1/f1 - week 10.ipynb` | SFGP Matérn-2.5 + log + qLogNEI q=4 + IP |
| `functions/f2/f2 - week 12.ipynb` | `functions/f2/f2 - week 10.ipynb` | SFGP Matérn-2.5 + Standardize + qLogNEI q=4 + IP |
| `functions/f3/f3 - week 12.ipynb` | `functions/f3/f3 - week 10.ipynb` | SFGP Matérn-2.5 + shift + qLogNEI q=3 |
| `functions/f4/f4 - week 12.ipynb` | `functions/f4/f4 - week 10.ipynb` | SFGP Matérn-2.5 + Standardize + qLogNEI q=4 |
| `functions/f5/f5 - week 12.ipynb` | `functions/f5/f5 - week 10.ipynb` | SFGP Matérn-1.5 + log + Standardize + qLogNEI q=4 |
| `functions/f6/f6 - week 12.ipynb` | `functions/f6/f6 - week 10.ipynb` | SFGP Matérn-1.5 + Standardize + rank IP + qLogNEI q=4 |
| `functions/f7/f7 - week 12.ipynb` | `functions/f7/f7 - week 10.ipynb` | NN (6→5→5→1) + MC dropout + blended acq |
| `functions/f8/f8 - week 12.ipynb` | `functions/f8/f8 - week 10.ipynb` | SFGP Matérn-2.5 + Standardize + qLogNEI q=1 |

## Execution Order

Notebooks are independent and can be executed in any order. Suggested sequence for review:

1. **F1, F2** (2D) — fastest to run, have contour visualisations
2. **F3, F4, F5** (3–4D) — moderate runtime
3. **F6** (5D) — rank-based selection adds minor complexity
4. **F8** (8D) — 30 MLL restarts in 8D takes longest for GP
5. **F7** (6D) — NN training (200 epochs) + 50k candidate scoring

## Verification

For each notebook after execution:

1. ✅ All cells execute without errors
2. ✅ Convergence plot renders (log scale for F1)
3. ✅ Pair plots show green star on best sample
4. ✅ Performance evaluation prints stalling metrics
5. ✅ Submission query outputs valid coordinates in [0, 0.999999]
6. ✅ Duplicate check reports "OK — unique point"
7. ✅ Surrogate contour renders (F1, F2 only)
8. ✅ Updated convergence shows proposed point

## Key Implementation Notes

- **F1 log transform**: `log(max(y, 1e-300))` — do NOT use Standardize(m=1) for the optimisation GP (that was only for ARD diagnostic)
- **F3 shift transform**: `y - y_min` is computed at runtime; do NOT hardcode the minimum value
- **F5 log transform**: `np.log(outputs)` — requires all outputs to be strictly positive (confirmed for F5)
- **F6 milk constraint**: Candidates must have milk dimension (index 4) ≥ 0.12; fallback to 0.10 if no candidates qualify
- **F7 z-score**: Manual normalisation using training set mean/std — different from BoTorch Standardize
- **F8 Cholesky check**: Post-fit stability validation for the 8D GP; if it fails, the notebook should report it
