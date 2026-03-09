# Implementation Plan: F5 Week 9 — Remove Interior Penalty

**Branch**: `025-f5-remove-penalty` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/025-f5-remove-penalty/spec.md`

## Summary

Remove all interior penalty code from the F5 week 9 notebook (`functions/f5/f5 - week 9.ipynb`). This includes the `PenalisedAcquisition` class, `STEEPNESS`/`FLOOR`/`EPS_BOUND` constants, penalty visualisation cells, and all penalty references in documentation. The notebook will use plain qLogNEI with distance-based selection (the existing Step 3 logic) for candidate proposal. No other changes to the surrogate, acquisition, or evaluation pipeline.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev` environment)
**Primary Dependencies**: BoTorch (SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood), NumPy, Matplotlib, SciPy
**Storage**: NumPy `.npy` files in `data/f5/`
**Testing**: No unit tests (Constitution Principle I)
**Target Platform**: Jupyter notebook (local execution)
**Project Type**: Single notebook modification
**Performance Goals**: N/A — notebook executes end-to-end without errors
**Constraints**: Submission format `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`, all values in [0.0, 0.999999]
**Scale/Scope**: 1 notebook file, ~27 cells, removing ~4 cells and editing ~6 cells

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | PASS | Removing code simplifies the notebook — no complexity added |
| II. Per-Function Isolation | PASS | Only modifies F5 notebook in `./functions/f5/` |
| III. Per-Iteration Notebooks | PASS | Modifies existing `f5 - week 9.ipynb` (same iteration, same notebook) |
| IV. Data Organisation | PASS | No data file changes |
| V. BoTorch & PyTorch Stack | PASS | GP + qLogNEI remain unchanged |
| VI. Documentation & Visualisation | PASS | Hyperparameter table and strategy section updated to reflect removal |
| VII. Maximisation Objective | PASS | No change to objective direction |

No structural violations. All constitution principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/025-f5-remove-penalty/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── checklists/
    └── requirements.md  # Specification quality checklist
```

### Source Code (repository root)

```text
functions/f5/
└── f5 - week 9.ipynb    # The single file modified by this feature
```

**Structure Decision**: Single notebook modification — no new files created. The notebook already exists and contains all the code. This feature only removes and edits cells within it.

## Complexity Tracking

No constitution violations — this section is empty.
