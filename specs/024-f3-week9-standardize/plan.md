# Implementation Plan: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Branch**: `024-f3-week9-standardize` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/024-f3-week9-standardize/spec.md`

## Summary

Replace manual z-score output standardisation with BoTorch `Standardize(m=1)` outcome transform and increase acquisition restarts from 10 to 20 in the F3 week 9 notebook. Interior penalty was initially specified, evaluated during implementation, and **removed per clarification**. The notebook uses plain qLogNEI with a 1×3 posterior mean visualisation layout matching Week 8.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: BoTorch (SingleTaskGP, Standardize, qLogNEI, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood), NumPy, Matplotlib, SciPy  
**Storage**: `.npy` files in `./data/f3/`  
**Testing**: No unit tests (Constitution Principle I)  
**Target Platform**: Jupyter Notebook (macOS local)  
**Project Type**: Single notebook (`functions/f3/f3 - week 9.ipynb`)  
**Performance Goals**: N/A — single execution per week  
**Constraints**: Notebook must execute end-to-end in a few minutes  
**Scale/Scope**: 24 samples, 3D input, 1 output, single GP surrogate

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Simplicity | ✅ PASS | Replacing manual z-score with built-in Standardize reduces code complexity. Removing interior penalty further simplifies. |
| II | Per-Function Isolation | ✅ PASS | Changes only affect `functions/f3/` — no other function notebooks touched |
| III | Per-Iteration Notebooks | ✅ PASS | Modifying existing `f3 - week 9.ipynb` in-place (same iteration, strategy refinement) |
| IV | Data Organisation | ✅ PASS | Uses `data/f3/updated_inputs - Week 9.npy` and `updated_outputs - Week 9.npy` |
| V | BoTorch & PyTorch Stack | ✅ PASS | SingleTaskGP + qLogNEI — standard BoTorch stack |
| VI | Documentation & Visualisation | ✅ PASS | Hyperparameter table with rationale, 1×3 contour plots, convergence plot, LOO analysis |
| VII | Maximisation Objective | ✅ PASS | qLogNEI maximises expected improvement over best observed value |

**Gate result**: 7/7 PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/024-f3-week9-standardize/
├── plan.md              # This file
├── research.md          # Phase 0 output (6 research tasks, R3 removed)
├── data-model.md        # Phase 1 output (entities, state transitions)
├── quickstart.md        # Phase 1 output (7 implementation steps)
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (task breakdown)
```

### Source Code (repository root)

```text
functions/f3/
└── f3 - week 9.ipynb    # Target notebook — only file modified

data/f3/
├── updated_inputs - Week 9.npy   # 24×3 input data (read-only)
└── updated_outputs - Week 9.npy  # 24 output data (read-only)
```

**Structure Decision**: Single notebook modification — no new files created, no source directory structure needed.

No structural violations — single notebook modification requires no additional complexity.
