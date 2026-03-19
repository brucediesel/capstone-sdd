# Implementation Plan: Week 12 Bayesian Optimisation Loop (F1–F8)

**Branch**: `035-f1-f8-week12-optimisation` | **Date**: 2026-03-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/035-f1-f8-week12-optimisation/spec.md`

## Summary

Create 8 new Jupyter notebooks (`f1 - week 12.ipynb` through `f8 - week 12.ipynb`) that load the latest week 11 data and run the same Bayesian optimisation strategy used in the week 10 optimisation round for each function. Each notebook fits the function-specific surrogate model, optimises the acquisition function, proposes a submission candidate, and provides the same visualisations (convergence plot, 2D pair plots with green-star best marker, performance evaluation, surrogate surfaces, and updated convergence with predicted point) as previous optimisation weeks.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch 2.10.0, NumPy 2.4.1, Matplotlib 3.10.8, scikit-learn (F7 only — not used for GP functions)
**Storage**: NumPy `.npy` files in `./data/fX/` directories
**Testing**: Manual execution — run all cells top-to-bottom, verify zero errors and all visualisations render (per constitution: no unit tests required)
**Target Platform**: macOS, Jupyter/VS Code notebook runtime
**Project Type**: Jupyter notebook collection (one notebook per function per week)
**Performance Goals**: Each notebook must execute completely without errors; GP fitting with multi-restart MLL must converge for at least one restart
**Constraints**: Single machine execution; F7 NN training and F8 8D GP fitting are the most computationally expensive; all inputs in [0, 0.999999]
**Scale/Scope**: 8 notebooks, each ~20–27 cells, replicating established patterns from week 10 optimisation notebooks with updated data week references

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Notebooks replicate existing proven patterns; no new complexity introduced |
| II. Per-Function Isolation | ✅ PASS | Each function gets its own notebook in `./functions/fX/` |
| III. Per-Iteration Notebooks | ✅ PASS | New `fX - week 12.ipynb` notebooks created; existing notebooks untouched |
| IV. Data Organisation | ✅ PASS | Loads from `./data/fX/updated_*- Week 11.npy`; follows naming convention |
| V. BoTorch & PyTorch Stack | ✅ PASS | GP functions use BoTorch SingleTaskGP; F7 uses PyTorch NN; acquisition via BoTorch |
| VI. Documentation & Visualisation | ✅ PASS | Hyperparameters documented; convergence, pair plots, surrogate surfaces provided |
| VII. Maximisation Objective | ✅ PASS | All acquisition functions maximise; running best uses `np.maximum.accumulate` |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/035-f1-f8-week12-optimisation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── notebook-cell-contract.md
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
functions/
├── f1/
│   └── f1 - week 12.ipynb    # NEW — SFGP Matérn-2.5 ARD + log + qLogNEI q=4 + IP
├── f2/
│   └── f2 - week 12.ipynb    # NEW — SFGP Matérn-2.5 ARD + Standardize + qLogNEI q=4 + IP
├── f3/
│   └── f3 - week 12.ipynb    # NEW — SFGP Matérn-2.5 ARD + shift + qLogNEI q=3
├── f4/
│   └── f4 - week 12.ipynb    # NEW — SFGP Matérn-2.5 ARD + Standardize + qLogNEI q=4
├── f5/
│   └── f5 - week 12.ipynb    # NEW — SFGP Matérn-1.5 ARD + log + Standardize + qLogNEI q=4
├── f6/
│   └── f6 - week 12.ipynb    # NEW — SFGP Matérn-1.5 ARD + Standardize + qLogNEI q=4 + rank IP
├── f7/
│   └── f7 - week 12.ipynb    # NEW — NN (6→5→5→1) + MC dropout + blended acquisition
└── f8/
    └── f8 - week 12.ipynb    # NEW — SFGP Matérn-2.5 ARD + Standardize + qLogNEI q=1

data/
├── f1/ ... f8/                # Read-only: updated_inputs/outputs - Week 11.npy
```

**Structure Decision**: Per-function isolation with one new notebook per function. No shared code modules — each notebook is self-contained per constitution Principle III. The week 10 optimisation notebooks serve as the source template for each function's week 12 notebook.
