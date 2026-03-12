# Implementation Plan: F3 Week 10 — Optimisation Tuning

**Branch**: `030-f3-optimisation-tuning` | **Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/030-f3-optimisation-tuning/spec.md`

## Summary

Append new optimisation cells to the existing F3 week 10 review notebook. Tune the existing SFGP Matérn-2.5 ARD pipeline by replacing Standardize(m=1) with a shift transform (y - y_min), increasing batch size to q=3, expanding raw samples to 2048, relaxing noise floor to 1e-4, and increasing MLL restarts to 40. Produce a formatted submission point, 2D contour slice visualisation (3 input pairs), and convergence plot.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev` environment)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib
**Storage**: NumPy `.npy` files in `./data/f3/`
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: macOS, Jupyter notebook
**Project Type**: Single project — Jupyter notebooks
**Performance Goals**: Notebook executes in < 120s including GP fitting with 40 MLL restarts and q=3 acquisition over 3D
**Constraints**: q=3 candidates, 2048 raw samples, 20 restarts, 40 MLL restarts, noise_lb=1e-4, 3D input space
**Scale/Scope**: Single notebook (F3 only), ~7 new cells appended to existing 12-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Each step in its own cell with markdown explanations. Hyperparameters as named constants with comments. |
| II. Per-Function Isolation | ✅ PASS | Only modifies `functions/f3/f3 - week 10.ipynb`. No other functions affected. |
| III. Per-Iteration Notebooks | ✅ PASS | Appending to existing week 10 notebook (same iteration). Extending current review with optimisation run. |
| IV. Data Organisation | ✅ PASS | Reads from `./data/f3/updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy` (already loaded in existing cells). |
| V. BoTorch & PyTorch Stack | ✅ PASS | Uses BoTorch SingleTaskGP + qLogNEI. All GP components from BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | Hyperparameters documented with rationale and week 9 → 10 change justification. 2D contour slices + convergence plot. |
| VII. Maximisation Objective | ✅ PASS | qLogNEI maximises expected improvement. Shift transform preserves ordering. Distance selection keeps highest-mean candidates. |

**GATE RESULT: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/030-f3-optimisation-tuning/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── f3-optimisation-pipeline.md
├── checklists/
│   └── requirements.md  # Specification checklist (complete)
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
functions/f3/
└── f3 - week 10.ipynb    # Target notebook — append ~7 new cells after existing 12
data/f3/
├── updated_inputs - Week 10.npy    # 25×3 input array (already loaded)
└── updated_outputs - Week 10.npy   # 25×1 output array (already loaded)
```

**Structure Decision**: No new files created. All changes are cell appends to the existing notebook `functions/f3/f3 - week 10.ipynb`.

## Complexity Tracking

No constitution violations. No complexity justification needed.

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts (data-model.md, contracts/, quickstart.md) are complete.*

| Principle | Status | Post-Design Notes |
|-----------|--------|-------------------|
| I. Simplicity | ✅ PASS | Pipeline is 7 linear cell groups. No abstractions, no helper modules. Shift transform is a single subtraction. Each cell has one responsibility. |
| II. Per-Function Isolation | ✅ PASS | Only `functions/f3/f3 - week 10.ipynb` is modified. No cross-function imports or dependencies. |
| III. Per-Iteration Notebooks | ✅ PASS | Appending to same iteration (week 10). Existing 12 cells untouched. |
| IV. Data Organisation | ✅ PASS | Uses existing data files loaded by earlier cells. No new data files created. |
| V. BoTorch & PyTorch Stack | ✅ PASS | SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf, fit_gpytorch_mll — all BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | 9 hyperparameters as named constants with comments and week 9 change justifications. 2D contour slices (3 pairs) + convergence plot. Contract specifies all visual outputs. |
| VII. Maximisation Objective | ✅ PASS | qLogNEI maximises expected improvement in shifted space. Shift transform preserves ordering (monotonic). Distance selection keeps highest-mean candidates. |

**POST-DESIGN GATE: ALL PASS** — ready for `/speckit.tasks`.
