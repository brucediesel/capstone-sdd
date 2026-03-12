# Implementation Plan: F2 Week 10 — SFGP Optimisation Run

**Branch**: `029-f1-f8-week10-review` | **Date**: 2026-03-11 | **Spec**: [spec-f2-optimisation.md](spec-f2-optimisation.md)
**Input**: Feature specification from `/specs/029-f1-f8-week10-review/spec-f2-optimisation.md`

## Summary

Append new optimisation cells to the existing F2 week 10 review notebook. Replace the stalling SFGP Matérn-1.5 ARD approach with SFGP Matérn-2.5 ARD + Standardize(m=1), wider lengthscale bounds [0.005, 10.0], 50 MLL restarts, and qLogNEI acquisition seeded with 4,096 Sobol points. Produce a formatted submission point and 3-panel surrogate visualisation with colour-coded data points.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev` environment)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib
**Storage**: NumPy `.npy` files in `./data/f2/`
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: macOS, Jupyter notebook
**Project Type**: Single project — Jupyter notebooks
**Performance Goals**: Notebook executes in < 60s including GP fitting and acquisition optimisation
**Constraints**: q=4 candidates, 4,096 Sobol raw samples, 20 restarts, 50 MLL restarts, Standardize(m=1)
**Scale/Scope**: Single notebook (F2 only), ~7 new cells appended to existing 12-cell notebook

### Key Differences from F1 Plan

| Aspect | F1 | F2 |
|--------|----|----|
| Output transform | Manual log(max(y, 1e-300)) | BoTorch Standardize(m=1) |
| Kernel ν | 2.5 (from Hurdle Model) | 2.5 (from Matérn-1.5) |
| LS bounds | [0.01, 2.0] | [0.005, 10.0] (wider) |
| Noise LB | 1e-4 | 1e-4 (reduced from 1e-3) |
| MLL restarts | 15 | 50 |
| RAW_SAMPLES | 10,000 | 4,096 |
| Y-axis scaling | Log scale | Linear |
| Output range | [~1e-245, ~7.7e-16] → log: [-690, -35] | [~0.25, ~0.67] (narrow, normal range) |

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Each step in its own cell with markdown explanations. Hyperparameters as named constants. No abstractions. |
| II. Per-Function Isolation | ✅ PASS | Only modifies `functions/f2/f2 - week 10.ipynb`. No other functions affected. |
| III. Per-Iteration Notebooks | ✅ PASS | Appending to existing week 10 notebook (same iteration). Not creating a new week — extending current review with optimisation run. |
| IV. Data Organisation | ✅ PASS | Reads from `./data/f2/updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy` (already loaded in existing cells). |
| V. BoTorch & PyTorch Stack | ✅ PASS | Uses BoTorch SingleTaskGP + qLogNEI + Standardize(m=1). All GP components from BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | Hyperparameters documented with rationale and week 9 → 10 changes. 3-panel contour + convergence plot provided. |
| VII. Maximisation Objective | ✅ PASS | qLogNEI maximises expected improvement. Distance selection keeps highest-mean candidates. |

**GATE RESULT: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/029-f1-f8-week10-review/
├── spec.md                              # Original review notebooks spec
├── spec-f1-optimisation.md              # F1 optimisation spec (completed)
├── spec-f2-optimisation.md              # F2 optimisation spec (this feature)
├── plan.md                              # F1 optimisation plan
├── plan-f2-optimisation.md              # This file
├── research.md                          # Phase 0 output (extended)
├── data-model.md                        # Phase 1 output (extended)
├── quickstart.md                        # Phase 1 output (extended)
├── contracts/
│   ├── f1-optimisation-pipeline.md      # F1 contracts (completed)
│   └── f2-optimisation-pipeline.md      # F2 contracts (this feature)
├── checklists/
│   ├── requirements.md                  # Original review checklist
│   ├── requirements-f1-optimisation.md  # F1 optimisation checklist
│   └── requirements-f2-optimisation.md  # F2 optimisation checklist
└── tasks-f2-optimisation.md              # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
functions/f2/
└── f2 - week 10.ipynb    # Target notebook — append ~7 new cells after existing 12
data/f2/
├── updated_inputs - Week 10.npy    # 20×2 input array (already loaded)
└── updated_outputs - Week 10.npy   # 20×1 output array (already loaded)
```

**Structure Decision**: No new files created. All changes are cell appends to the existing notebook `functions/f2/f2 - week 10.ipynb`.

## Complexity Tracking

No constitution violations. No complexity justification needed.

---

## Post-Design Constitution Re-Check

*After Phase 1 design artifacts (data-model.md, contracts/f2-optimisation-pipeline.md, quickstart.md).*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Pipeline is 6 cell groups, each self-contained. Standardize(m=1) is simpler than F1's manual log transform — fewer entities, no LOG_EPSILON. |
| II. Per-Function Isolation | ✅ PASS | F2 entities (F2OptimisationConfig, StandardizedOutputs, F2SFGPModel) scoped to F2 only. Contract file is separate from F1. |
| III. Per-Iteration Notebooks | ✅ PASS | All appended to existing week 10 notebook. No new notebook created. |
| IV. Data Organisation | ✅ PASS | Data model shows tensors derived from existing `.npy` data — no new data files. |
| V. BoTorch & PyTorch Stack | ✅ PASS | Contract confirms: SingleTaskGP + Matérn-2.5 ARD + Standardize(m=1) + qLogNEI. All BoTorch/GPyTorch. |
| VI. Documentation & Visualisation | ✅ PASS | Contract CG5 specifies 3-panel visualisation. Quickstart documents all expected outputs. Changes from week 9 documented in plan differences table. |
| VII. Maximisation Objective | ✅ PASS | Contract CG4 confirms qLogNEI maximisation with distance-based selection. CG6 convergence plot on linear scale. |

**POST-DESIGN GATE: ALL PASS** — proceed to Phase 2 (task generation via `/speckit.tasks`).

---

## Phase 0–1 Artifacts Generated

| Artifact | Location | Content |
|----------|----------|---------|
| Research | [research.md](research.md) §R12–R15 | Matérn-2.5 rationale, Standardize interaction, LS bounds analysis, F2 variables in scope |
| Data Model | [data-model.md](data-model.md) §F2 | F2OptimisationConfig, StandardizedOutputs, F2SFGPModel, relationships, state transitions |
| Contract | [contracts/f2-optimisation-pipeline.md](contracts/f2-optimisation-pipeline.md) | 6 cell groups: config, data prep, GP fitting, acquisition, visualisation, convergence |
| Quickstart | [quickstart.md](quickstart.md) §F2 | Prerequisites, running instructions, expected outputs, verification checklist |
