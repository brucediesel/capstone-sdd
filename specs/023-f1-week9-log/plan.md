# Implementation Plan: F1 Week 9 — log Transform (No Penalties)

**Branch**: `023-f1-week9-log` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/023-f1-week9-log/spec.md`

## Summary

Create a self-contained Jupyter notebook for F1 Week 9 that uses a hurdle model (CalibratedClassifierCV + RandomForestRegressor) with `log(y)` instead of `log1p(y)` for Stage 2. The acquisition function is simplified weighted UCB with no local penalization or interior penalty, keeping KAPPA=3.0. The notebook loads 19 samples (10 initial + 9 submissions), fits the surrogate, proposes the next sample point in correct submission format, and provides 3-panel contour visualisation in log-space plus convergence and performance evaluation.

## Technical Context

**Language/Version**: Python 3.x (Jupyter Notebook)
**Primary Dependencies**: numpy, pandas, matplotlib, scikit-learn (CalibratedClassifierCV, LogisticRegression, RandomForestRegressor), scipy (cdist)
**Storage**: NumPy .npy files in `./data/f1/`
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: Local Jupyter / VS Code notebook
**Project Type**: Single self-contained notebook
**Performance Goals**: Notebook executes end-to-end in <30 seconds
**Constraints**: Must run on student laptop; no GPU required
**Scale/Scope**: 19 data points, 2D input space, single notebook deliverable

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Simplicity | **PASS** | Single notebook, scikit-learn only, no complex frameworks. Simplified further by removing penalization terms. |
| II | Per-Function Isolation | **PASS** | Only F1 modified, notebook in `./functions/f1/` |
| III | Per-Iteration Notebooks | **PASS** | New notebook `f1 - week 9.ipynb`, existing notebooks untouched |
| IV | Data Organisation | **PASS** | Loads from `./data/f1/updated_inputs - Week 9.npy` / `updated_outputs - Week 9.npy` |
| V | BoTorch & PyTorch Stack | **PASS** | Uses scikit-learn for tree-based surrogate (permitted by constitution for non-GP surrogates) |
| VI | Documentation & Visualisation | **PASS** | Explicit hyperparameters with rationale, 3-panel contour + convergence plots |
| VII | Maximisation Objective | **PASS** | UCB maximises acquisition; convergence tracks running maximum |

**Result: 7/7 PASS — proceed to Phase 0.**

## Project Structure

### Documentation (this feature)

```text
specs/023-f1-week9-log/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
functions/f1/
└── f1 - week 9.ipynb    # THE DELIVERABLE — self-contained notebook

data/f1/
├── initial_inputs.npy
├── initial_outputs.npy
├── updated_inputs - Week 9.npy
└── updated_outputs - Week 9.npy
```

**Structure Decision**: Single self-contained Jupyter notebook per constitution Principle III. No separate source files, modules, or test directories needed.

## Complexity Tracking

No constitution violations. Table omitted.
