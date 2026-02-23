# Implementation Plan: F5 Week 7 — GP Matérn-5/2 + NEI

**Branch**: `011-f5-gp-nei` | **Date**: 2026-02-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/011-f5-gp-nei/spec.md`

## Summary

Add a Week 7 section to the F5 notebook (`functions/f5/f5.ipynb`) that loads 27 cumulative samples, fits a Gaussian Process surrogate with Matérn-5/2 ARD kernel on log-transformed outputs, proposes 2 candidate points via Noisy Expected Improvement (NEI), and produces surrogate visualisations and convergence plots matching the Week 6 layout. This replaces the GBT ensemble approach used in weeks 5–6 with a principled GP-based BO pipeline using BoTorch.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `data/f5/` (read-only); notebook cells in `functions/f5/f5.ipynb` (append-only)
**Testing**: No unit tests (per constitution); validation via cell execution and output inspection
**Target Platform**: macOS (local Jupyter kernel via VS Code)
**Project Type**: Single Jupyter notebook (append new cells)
**Performance Goals**: GP training with 15 restarts completes within 60 seconds on 27×4 data
**Constraints**: All inputs in [0, 1]; outputs have large range (~0.1 to ~3395); log1p transform required
**Scale/Scope**: 27 samples × 4 dimensions; 8 new cells appended to 50 existing cells

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Code as simple as possible, clearly explained | PASS | Each cell has a single purpose with markdown documentation |
| 2 | Submitted as Jupyter notebooks | PASS | All code in `functions/f5/f5.ipynb` |
| 3 | No unit tests required | PASS | Validation via cell execution only |
| 4 | Each problem in its own notebook/folder | PASS | F5 in `functions/f5/` |
| 5 | Use BoTorch library | PASS | `SingleTaskGP` + `qNoisyExpectedImprovement` from BoTorch |
| 6 | Weekly changes as new section, week number title | PASS | New "Week 7" section appended after existing content |
| 7 | Existing code cells not replaced | PASS | All 50 existing cells preserved; 8 new cells appended |
| 8 | Hyperparameters explicit with explanations | PASS | Dedicated markdown cell documents all HPs with rationale |

All 8 gates PASS. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/011-f5-gp-nei/
├── plan.md              # This file
├── research.md          # Phase 0: research decisions
├── data-model.md        # Phase 1: entity model
├── quickstart.md        # Phase 1: implementation guide
├── contracts/           # Phase 1: cell acceptance contracts
│   └── week7-cells.md
└── tasks.md             # Phase 2: implementation tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
functions/f5/
└── f5.ipynb             # Existing notebook; 8 new cells appended (cells 51–58)

data/f5/
├── updated_inputs - Week 7.npy   # Input: (27, 4) read-only
└── updated_outputs - Week 7.npy  # Input: (27,) read-only
```

**Structure Decision**: Single notebook, append-only. No new files created; all code added as new cells after the existing 50 cells in `f5.ipynb`.

## Complexity Tracking

No constitution violations. Table not applicable.
