# Implementation Plan: F1, F2 & F5 Interior Penalty

**Branch**: `031-f4-f8-week10-optimisation` | **Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/032-f1-f2-f5-interior-penalty/spec.md`

## Summary

Add a very shallow interior penalty (STEEPNESS=0.02, FLOOR=0.01) to the F1, F2, and F5 week 10 notebooks. The penalty uses the established sin-based formula `w(x) = FLOOR + (1−FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)` applied as a post-hoc multiplicative re-scoring of acquisition values after the existing distance-based selection filter, matching the pattern used in F6 and F7.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `./data/fX/`
**Testing**: Manual end-to-end notebook execution (no unit tests per constitution)
**Target Platform**: macOS (local Jupyter notebooks)
**Project Type**: Jupyter notebooks — per-function, per-iteration
**Performance Goals**: N/A (single-iteration BO notebooks)
**Constraints**: Notebooks must execute end-to-end without errors; submissions clamped to [0, 0.999999]
**Scale/Scope**: 3 notebooks modified (F1: 2D, F2: 2D, F5: 4D)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | ✅ PASS | Interior penalty is ~10 lines of code per notebook; formula is well-documented |
| II. Per-Function Isolation | ✅ PASS | Each function modified in its own notebook in `./functions/fX/` |
| III. Per-Iteration Notebooks | ✅ PASS | Editing existing week 10 notebooks (appending cells, not replacing) |
| IV. Data Organisation | ✅ PASS | No data file changes needed |
| V. BoTorch & PyTorch Stack | ✅ PASS | Interior penalty uses only NumPy (already a dependency); no new libraries |
| VI. Documentation & Visualisation | ✅ PASS | Penalty weights printed with effect documentation (FR-006) |
| VII. Maximisation Objective | ✅ PASS | Higher penalised acquisition = better; consistent with maximisation |

**Gate result**: ALL PASS — no violations.

## Project Structure

### Documentation (this feature)

```text
specs/032-f1-f2-f5-interior-penalty/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # Quality checklist (completed)
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
functions/
├── f1/
│   └── f1 - week 10.ipynb    # Add STEEPNESS/FLOOR constants + IP cell
├── f2/
│   └── f2 - week 10.ipynb    # Add STEEPNESS/FLOOR constants + IP cell
└── f5/
    └── f5 - week 10.ipynb    # Add STEEPNESS/FLOOR constants + IP cell
```

**Structure Decision**: Existing per-function notebook layout. No new files; cells appended to 3 existing notebooks.

## Complexity Tracking

No constitution violations — this section is intentionally empty.
