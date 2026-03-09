# Implementation Plan: F1 Week 9 — log Transform, No Penalties

**Branch**: `023-f1-week9-log` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/023-f1-week9-log/spec.md`

## Summary

Create a self-contained Jupyter notebook for F1 Week 9 that switches the Stage 2 RF regressor from `log1p(y)` to `log(y)`, removes local penalization and interior penalty from the acquisition function, and reduces KAPPA from 3.0 to 0.5 (exploitation-focused). The notebook uses a two-stage hurdle model (Calibrated LR classifier + RF regressor) with weighted UCB acquisition, produces 3-panel log-space visualisations, and outputs a formatted submission query.

## Technical Context

**Language/Version**: Python 3.x (Jupyter Notebook)
**Primary Dependencies**: numpy, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), scipy (pdist, squareform)
**Storage**: NumPy `.npy` files in `./data/f1/`
**Testing**: None required (Constitution Principle I — no unit tests)
**Target Platform**: Local Jupyter / VS Code notebook
**Project Type**: Single notebook deliverable
**Performance Goals**: N/A (single-run notebook, <1 min execution)
**Constraints**: All code in one self-contained notebook (Constitution Principle III)
**Scale/Scope**: 19 observations (10 initial + 9 submissions), 2D input space, single function (F1)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity | ✅ PASS | Single notebook, no unnecessary abstractions, sklearn-only surrogate |
| II. Per-Function Isolation | ✅ PASS | Deliverable is `functions/f1/f1 - week 9.ipynb` — F1 only |
| III. Per-Iteration Notebooks | ✅ PASS | New notebook `f1 - week 9.ipynb`, does not modify existing notebooks |
| IV. Data Organisation | ✅ PASS | Loads from `data/f1/updated_inputs - Week 9.npy` / `updated_outputs - Week 9.npy` |
| V. BoTorch & PyTorch Stack | ✅ PASS | Hurdle model uses scikit-learn (permitted for non-GP surrogates); acquisition matches surrogate type |
| VI. Documentation & Visualisation | ✅ PASS | Hyperparameter table with rationale, 3-panel contour + convergence plot, performance evaluation |
| VII. Maximisation Objective | ✅ PASS | argmax acquisition, running maximum convergence, higher = better throughout |

**GATE RESULT**: ✅ ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/023-f1-week9-log/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (N/A — no API)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
functions/f1/
└── f1 - week 9.ipynb    # Single notebook deliverable

data/f1/
├── updated_inputs - Week 9.npy
└── updated_outputs - Week 9.npy
```

**Structure Decision**: Single notebook in the existing `functions/f1/` directory per Constitution Principle II (per-function isolation) and III (per-iteration notebooks). No additional source directories needed.

## Complexity Tracking

No violations. All constitution gates pass.
