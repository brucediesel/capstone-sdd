# Implementation Plan: F1 Week 8 — Hurdle Model Bayesian Optimisation Iteration

**Branch**: `019-f1-week8-hurdle` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/019-f1-week8-hurdle/spec.md`

## Summary

Create a new self-contained Jupyter notebook `f1 - week 8.ipynb` that loads the Week 8 data (18 observations), fits the same two-stage hurdle model surrogate used in Week 7 (calibrated logistic classifier + random forest regressor on log1p(y)), maximises a weighted UCB acquisition function with local penalization and interior penalty, and proposes the next sample point for Week 9 submission. Identical strategy, hyperparameters, and visualisation approach to Week 7.

## Technical Context

**Language/Version**: Python 3.x (Jupyter Notebook)
**Primary Dependencies**: numpy, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor)
**Storage**: `.npy` files in `./data/f1/`
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: macOS, local Jupyter kernel
**Project Type**: Single notebook (self-contained iteration)
**Performance Goals**: N/A — single-run notebook with 18 data points and 20 000 candidates
**Constraints**: All inputs/outputs in [0.0, 1.0]; submission values in [0.0, 0.999999]
**Scale/Scope**: 18 observations, 2D input space, 1 notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity | PASS | Notebook uses basic sklearn models; each step explained in markdown cells. No unnecessary complexity. |
| II. Per-Function Isolation | PASS | Only F1 is affected. Notebook in `./functions/f1/`. |
| III. Per-Iteration Notebooks | PASS | New notebook `f1 - week 8.ipynb` created; original `f1.ipynb` NOT modified. Self-contained with all imports, data loading, fitting, acquisition, visualisation, submission. |
| IV. Data Organisation | PASS | Loads from `./data/f1/updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy`. |
| V. BoTorch & PyTorch Stack | PASS | Uses scikit-learn for non-GP surrogate (hurdle model). This is explicitly allowed by constitution: "Additional surrogates MAY use scikit-learn". |
| VI. Documentation & Visualisation | PASS | Hyperparameters documented in dedicated cell with rationale. 3-panel contour plots + convergence plot provided. |
| VII. Maximisation Objective | PASS | Acquisition function maximises weighted UCB. Convergence plot shows running maximum. |

**GATE RESULT: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/019-f1-week8-hurdle/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (N/A — no API)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
functions/
└── f1/
    ├── f1.ipynb              # Original notebook (NOT modified)
    ├── f1 - week 8.ipynb     # NEW — this feature's deliverable
    └── preq-eval-f1.ipynb    # Existing PE notebook (not touched)

data/
└── f1/
    ├── initial_inputs.npy
    ├── initial_outputs.npy
    ├── updated_inputs - Week 8.npy    # Input data (pre-existing)
    └── updated_outputs - Week 8.npy   # Output data (pre-existing)
```

**Structure Decision**: Single self-contained notebook per constitution principle III. No source directory hierarchy needed — the deliverable is one `.ipynb` file.

## Complexity Tracking

> No constitution violations. Table not needed.
