# Implementation Plan: F5 Week 9 — Kernel, Standardize & Raw Samples

**Branch**: `026-f5-kernel-standardize` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/025-f5-remove-penalty/spec.md` (Clarifications session, FR-012/013/014)

**Note**: This plan covers the 3 tuning changes to be implemented on a new branch from `025-f5-remove-penalty`. The penalty removal is already complete on the parent branch.

## Summary

After removing the interior penalty, the F5 week 9 notebook still proposes a boundary-stuck candidate. Three targeted changes address this: (1) switch kernel from Matérn-5/2 to Matérn-1.5 (rougher assumptions may prevent lengthscale collapse), (2) replace manual z-score with BoTorch `Standardize(m=1)` while keeping `log1p` (simplifies pipeline), (3) increase acquisition `raw_samples` from 3000 to 5000 (better initial coverage in 4D). The notebook is then re-executed end-to-end and results reviewed for further improvements.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `./data/f5/`
**Testing**: No unit tests (constitution). Manual end-to-end execution + LOO validation.
**Target Platform**: macOS (local Jupyter kernel)
**Project Type**: Single Jupyter notebook
**Performance Goals**: Notebook executes end-to-end without errors; produces valid submission query
**Constraints**: Single file change (`functions/f5/f5 - week 9.ipynb`), 23 cells
**Scale/Scope**: Editing 10 cells (5 code, 5 markdown) across the 23-cell notebook; 13 unchanged

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Simplicity | PASS | Replacing manual z-score with `Standardize(m=1)` *reduces* complexity. Kernel swap is a single parameter change (`nu=2.5` → `nu=1.5`). |
| II | Per-Function Isolation | PASS | Only modifies `functions/f5/f5 - week 9.ipynb` |
| III | Per-Iteration Notebooks | PASS | Using existing `f5 - week 9.ipynb`; no new week notebook needed (same week) |
| IV | Data Organisation | PASS | Data files unchanged — reads `updated_inputs - Week 9.npy` / `updated_outputs - Week 9.npy` from `./data/f5/` |
| V | BoTorch & PyTorch Stack | PASS | All changes use BoTorch/GPyTorch APIs |
| VI | Documentation & Visualisation | PASS | Hyperparameter table, title, submission cell, and strategy cell updated to reflect new kernel/transform |
| VII | Maximisation Objective | PASS | qLogNEI maximises expected improvement; no change to objective direction |

**GATE RESULT: ALL PASS — proceed to Phase 0**

## Project Structure

### Documentation (this feature)

```text
specs/025-f5-remove-penalty/
├── spec.md              # Feature spec (with clarifications)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── quickstart.md        # Phase 1 output
```

### Source Code

```text
functions/f5/
└── f5 - week 9.ipynb    # The single file modified (23 cells)

data/f5/
├── updated_inputs - Week 9.npy   # Read-only input
└── updated_outputs - Week 9.npy  # Read-only input
```

**Structure Decision**: Single notebook modification — no new files or directories created.

## Complexity Tracking

> No constitution violations — no justifications required.

## Constitution Check — Post-Design Re-evaluation

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Simplicity | PASS | Standardize(m=1) reduces LOO code by removing per-fold z-score. Kernel swap is a single param. Net reduction in code complexity. |
| II | Per-Function Isolation | PASS | Only `functions/f5/f5 - week 9.ipynb` modified |
| III | Per-Iteration Notebooks | PASS | Same week 9 notebook; no new weekly notebook needed |
| IV | Data Organisation | PASS | No data file changes |
| V | BoTorch & PyTorch Stack | PASS | Standardize is a BoTorch built-in transform; MaternKernel is GPyTorch |
| VI | Documentation & Visualisation | PASS | Title, hyperparams table, submission cell, strategy all updated |
| VII | Maximisation Objective | PASS | qLogNEI maximises; unchanged |

**POST-DESIGN GATE: ALL PASS**
