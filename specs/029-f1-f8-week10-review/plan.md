# Implementation Plan: F1 Week 10 — SFGP Optimisation Run

**Branch**: `029-f1-f8-week10-review` | **Date**: 2026-03-11 | **Spec**: [spec-f1-optimisation.md](spec-f1-optimisation.md)
**Input**: Feature specification from `/specs/029-f1-f8-week10-review/spec-f1-optimisation.md`

## Summary

Append new optimisation cells to the existing F1 week 10 review notebook. Replace the stalled Hurdle Model + Weighted UCB approach with a BoTorch SFGP (Matérn-2.5 ARD) + qLogNEI (q=4) pipeline, applying log-transform to outputs and seeding acquisition with 10,000 Sobol points. Produce a formatted submission point and 3-panel surrogate visualisation.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev` environment)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib
**Storage**: NumPy `.npy` files in `./data/f1/`
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: macOS, Jupyter notebook
**Project Type**: Single project — Jupyter notebooks
**Performance Goals**: Notebook executes in < 60s including GP fitting and acquisition optimisation
**Constraints**: q=4 candidates, 10,000 Sobol raw samples, 20 restarts, 15 MLL restarts
**Scale/Scope**: Single notebook (F1 only), ~7 new cells appended to existing 12-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Each step in its own cell with markdown explanations. Hyperparameters as named constants. |
| II. Per-Function Isolation | ✅ PASS | Only modifies `functions/f1/f1 - week 10.ipynb`. No other functions affected. |
| III. Per-Iteration Notebooks | ✅ PASS | Appending to existing week 10 notebook (same iteration). Not creating a new week — extending current review with optimisation run. |
| IV. Data Organisation | ✅ PASS | Reads from `./data/f1/updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy` (already loaded in existing cells). |
| V. BoTorch & PyTorch Stack | ✅ PASS | Uses BoTorch SingleTaskGP + qLogNEI. All GP components from BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | Hyperparameters documented with rationale. 3-panel contour + convergence plot provided. |
| VII. Maximisation Objective | ✅ PASS | qLogNEI maximises expected improvement. Distance selection keeps highest-mean candidates. |

**GATE RESULT: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/029-f1-f8-week10-review/
├── spec.md                         # Original review notebooks spec
├── spec-f1-optimisation.md         # F1 optimisation spec (this feature)
├── plan.md                         # This file
├── research.md                     # Phase 0 output
├── data-model.md                   # Phase 1 output
├── quickstart.md                   # Phase 1 output
├── contracts/                      # Phase 1 output
├── checklists/
│   ├── requirements.md             # Original review checklist
│   └── requirements-f1-optimisation.md  # F1 optimisation checklist
└── tasks.md                        # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
functions/f1/
└── f1 - week 10.ipynb    # Target notebook — append ~7 new cells after existing 12
data/f1/
├── updated_inputs - Week 10.npy    # 20×2 input array (already loaded)
└── updated_outputs - Week 10.npy   # 20×1 output array (already loaded)
```

**Structure Decision**: No new files created. All changes are cell appends to the existing notebook `functions/f1/f1 - week 10.ipynb`.

## Complexity Tracking

No constitution violations. No complexity justification needed.

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts (data-model.md, contracts/, quickstart.md) are complete.*

| Principle | Status | Post-Design Notes |
|-----------|--------|-------------------|
| I. Simplicity | ✅ PASS | Pipeline is 6 linear cell groups. No abstractions, no helper modules. Each cell has one responsibility. |
| II. Per-Function Isolation | ✅ PASS | Only `functions/f1/f1 - week 10.ipynb` is modified. No cross-function imports or dependencies. |
| III. Per-Iteration Notebooks | ✅ PASS | Appending to same iteration (week 10). Existing 12 cells untouched. |
| IV. Data Organisation | ✅ PASS | Uses existing data files loaded by earlier cells. No new data files created. |
| V. BoTorch & PyTorch Stack | ✅ PASS | SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf, fit_gpytorch_mll — all BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | 12 hyperparameters as named constants with comments. 3-panel contour + convergence plot. Contract specifies all visual outputs. |
| VII. Maximisation Objective | ✅ PASS | qLogNEI operates on log-space (preserves ordering). Distance selection keeps highest-mean candidates. |

**POST-DESIGN GATE: ALL PASS** — ready for `/speckit.tasks`.
