# Quickstart: F4–F8 Week 10 Optimisation

**Branch**: `031-f4-f8-week10-optimisation`

## Prerequisites

- Python 3.14.2 via pyenv (`sdd-dev` environment)
- BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib installed
- Week 10 data files in `./data/fX/` for F4–F8
- Existing week 10 review notebooks in `./functions/fX/`

## Implementation Order

Process functions by priority:

1. **F4** (P1 — HIGH) — Fundamental surrogate overhaul from MFGP to SFGP
2. **F5** (P1 — LOW priority but P1 by spec) — Transform simplification
3. **F8** (P2 — MEDIUM) — qEI→qLogNEI switch
4. **F6** (P2 — LOW) — Conservative tuning
5. **F7** (P2 — HIGH stalling) — Exploration boost with MC dropout

## Per-Notebook Workflow

For each function:

1. Open `functions/fX/fX - week 10.ipynb`
2. Append markdown cell: "Step 6 — Week 10 Optimisation Run" with strategy change rationale
3. Append code cell: imports & all hyperparameter constants
4. Append code cell: load week 10 data, apply transforms, print summary
5. Append code cell: fit surrogate (GP with MLL restarts or NN with SGD)
6. Append code cell: optimise acquisition, select candidate, print submission
7. Append code cell: 2D contour visualisation
8. Append code cell: convergence plot
9. Run all new cells and verify output
10. Validate against success criteria (SC-001 through SC-006)

## Key Configuration Quick Reference

| Function | Surrogate | Acquisition | Key Changes |
|----------|-----------|-------------|-------------|
| F4 | SFGP Matérn-2.5 ARD | qLogNEI q=4 | MFGP→SFGP, noise_lb=1e-3, MLL≥30, Standardize(m=1) |
| F5 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | log1p→log, raw_samples=8000, acq_restarts=60 |
| F6 | SFGP Matérn-1.5 ARD + rank IP | qLogNEI q=4 | milk≥0.12, noise_lb=1e-3, raw_samples=5000 |
| F7 | NN (6→5→5→1) | 50/50 mean/EI blend | MC dropout≥50, STEEPNESS=0.02, 50k candidates |
| F8 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | qEI→qLogNEI, MC=512, raw_samples=8192 |

## Validation Checklist

- [ ] Each notebook executes end-to-end without errors (SC-001)
- [ ] GP convergence: ≥50% of restarts within 10% of best loss (SC-002)
- [ ] Submission format correct: dimensionality, range, 6 decimal places (SC-003)
- [ ] Visualisations render: contour + convergence (SC-004)
- [ ] No duplicate candidates (SC-005)
- [ ] Strategy changes documented in markdown (SC-006)
