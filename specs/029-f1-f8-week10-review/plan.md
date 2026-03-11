# Implementation Plan: Week 10 Performance Review & Visualisation

**Branch**: `029-f1-f8-week10-review` | **Date**: 2026-03-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/029-f1-f8-week10-review/spec.md`

## Summary

Create 8 new self-contained Jupyter notebooks (`fX - week 10.ipynb`, X=1..8) that load Week 10 data and provide performance review visualisations: convergence plots (log scale for F1 with zeroed negatives) and 2D pair plots of inputs with week-numbered submission points. Each notebook evaluates optimisation performance relative to its week 9 strategy and proposes specific improvements. No optimisation loop is run — notebooks stop after proposing improvements.

## Technical Context

**Language/Version**: Python 3.x (Jupyter notebooks)  
**Primary Dependencies**: numpy, matplotlib, itertools (standard library)  
**Storage**: `.npy` files in `./data/fX/` folders  
**Testing**: None required (per constitution)  
**Target Platform**: macOS (local Jupyter execution)  
**Project Type**: Single project — Jupyter notebooks  
**Performance Goals**: N/A (visualisation-only notebooks)  
**Constraints**: Each notebook must be self-contained and executable independently  
**Scale/Scope**: 8 notebooks, each loading 20–50 data points

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | PASS | Visualisation-only — no surrogate fitting or acquisition. Simple matplotlib plots. |
| II. Per-Function Isolation | PASS | Each function gets its own notebook in `./functions/fX/`. |
| III. Per-Iteration Notebooks | PASS | New notebooks named `fX - week 10.ipynb`, self-contained. Existing notebooks not modified. |
| IV. Data Organisation | PASS | Loads from `./data/fX/updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy`. |
| V. BoTorch & PyTorch Stack | PASS | No surrogate fitting in this feature — only numpy/matplotlib needed. |
| VI. Documentation & Visualisation | PASS | Convergence plots, 2D pair plots, performance evaluation markdown, strategy summary. |
| VII. Maximisation Objective | PASS | Convergence plots show running maximum. Evaluation treats higher values as better. |

**GATE RESULT: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/029-f1-f8-week10-review/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (empty — no API)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
functions/
├── f1/
│   └── f1 - week 10.ipynb    # NEW
├── f2/
│   └── f2 - week 10.ipynb    # NEW
├── f3/
│   └── f3 - week 10.ipynb    # NEW
├── f4/
│   └── f4 - week 10.ipynb    # NEW
├── f5/
│   └── f5 - week 10.ipynb    # NEW
├── f6/
│   └── f6 - week 10.ipynb    # NEW
├── f7/
│   └── f7 - week 10.ipynb    # NEW
└── f8/
    └── f8 - week 10.ipynb    # NEW

data/
├── f1/
│   ├── updated_inputs - Week 10.npy   # EXISTS
│   └── updated_outputs - Week 10.npy  # EXISTS
├── f2/ ... f8/                         # Same pattern
```

**Structure Decision**: Uses existing per-function folder structure. Each new notebook follows the `fX - week Y.ipynb` naming convention from Constitution III.
