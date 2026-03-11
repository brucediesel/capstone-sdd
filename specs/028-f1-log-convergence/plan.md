# Implementation Plan: F1 Log-Scale Convergence Plot

**Branch**: `028-f1-log-convergence` | **Date**: 2026-03-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/028-f1-log-convergence/spec.md`
**Clarification**: User directed: clip negative F1 outputs to zero, use plain `log` scale. Do not use `symlog`.

## Summary

Change the convergence graph cell in `process_results.ipynb` so the F1 subplot uses a logarithmic y-axis. Negative F1 output values are clipped to zero before plotting; matplotlib's `log` scale naturally omits zero-valued points. F2–F8 subplots remain on linear scale, unchanged.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: matplotlib (plotting), numpy (data manipulation) — both already imported in notebook
**Storage**: `.npy` files in `./data/f1/` through `./data/f8/`
**Testing**: Manual notebook execution — no unit tests per constitution
**Target Platform**: macOS, Jupyter notebook
**Project Type**: Single Jupyter notebook edit
**Performance Goals**: N/A — single plot rendering
**Constraints**: F1 data contains zeros and small negatives; clipping to zero means those points are omitted on log scale
**Scale/Scope**: 1 cell edit in 1 notebook

### F1 Data Characteristics

- 19 outputs total (10 initial + 9 BO submissions)
- Positive values span ~230 orders of magnitude (1e-245 to 7.7e-016)
- Contains exact zero values and small negatives (~-3.6e-003, ~-1.1e-003)
- After clipping: zeros are omitted by matplotlib log scale; all positive values rendered correctly

## Constitution Check — Pre-Design

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Simplicity | PASS | 3-line conditional addition; no new abstractions |
| II | Per-Function Isolation | PASS | Change is in `functions/results/process_results.ipynb`, not in function notebooks |
| III | Per-Iteration Notebooks | PASS | Not applicable — this is the shared results notebook, not a function iteration notebook |
| IV | Data Organisation | PASS | No data file changes |
| V | BoTorch & PyTorch Stack | PASS | Not applicable — matplotlib only |
| VI | Documentation & Visualisation | PASS | Improves visualisation of F1 outputs |
| VII | Maximisation Objective | PASS | Running max line preserved; log scale does not affect maximisation semantics |

**Gate result**: 7/7 PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/028-f1-log-convergence/
├── plan.md              # This file
├── research.md          # Phase 0 output — log-scale approach decision
├── data-model.md        # Phase 1 output — data transformation description
├── quickstart.md        # Phase 1 output — verification steps
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
functions/results/
└── process_results.ipynb   # Cell 13 — convergence graph (only file modified)
```

**Structure Decision**: Single notebook cell edit. No new files or directories required in the source tree.

## Complexity Tracking

No constitution violations — table not required.

## Constitution Check — Post-Design

*Re-evaluated after Phase 1 design artifacts completed.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Simplicity | PASS | Two `if fn == 'f1':` blocks — 5 lines total, no abstractions |
| II | Per-Function Isolation | PASS | Shared results notebook — not a function-specific notebook |
| III | Per-Iteration Notebooks | PASS | Not applicable — results notebook is not an iteration notebook |
| IV | Data Organisation | PASS | No data file changes; reads existing `.npy` files |
| V | BoTorch & PyTorch Stack | PASS | Not applicable — matplotlib/numpy only |
| VI | Documentation & Visualisation | PASS | Improves F1 output visualisation; log scale reveals order-of-magnitude differences |
| VII | Maximisation Objective | PASS | Running max line preserved; log scale does not change maximisation semantics |

**Gate result**: 7/7 PASS — design is constitution-compliant.
