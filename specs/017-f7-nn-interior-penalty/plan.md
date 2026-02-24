# Implementation Plan: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Branch**: `017-f7-nn-interior-penalty` | **Date**: 2025-02-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/017-f7-nn-interior-penalty/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add a new Week 7 section to the F7 notebook (6D ML hyperparameter tuning) that trains a compact neural network surrogate (6→5→5→1, lr=0.005), computes Expected Improvement via MC Dropout (50 stochastic forward passes), applies a multiplicative interior penalty w(x) = FLOOR + (1−FLOOR)·∏sin(πxᵢ)^(2·STEEPNESS) to discourage boundary-hugging candidates, and proposes a submission query. Since all F7 outputs are positive (range [0.003, 2.305]), the multiplicative penalty approach works correctly (EI ≥ 0, w ∈ [0.01, 1.0]). Visualisations match the Week 5/6 pattern: 3-panel surrogate plot + convergence plot.

## Technical Context

**Language/Version**: Python 3.11+ (pyenv, macOS)
**Primary Dependencies**: PyTorch (nn, optim), NumPy, Matplotlib — no BoTorch needed for NN surrogate
**Storage**: `.npy` files in `data/f7/` (read-only); notebook state in `functions/f7/f7.ipynb`
**Testing**: Manual execution — run all cells top-to-bottom; no unit tests (per constitution)
**Target Platform**: macOS (local), Jupyter notebook / VS Code
**Project Type**: Single Jupyter notebook (additive section)
**Performance Goals**: New section executes in < 30 seconds (NN training + 50×20k MC forward passes through 71-param network)
**Constraints**: Must not modify any existing cells (cells 1–49); all new cells appended after cell 49
**Scale/Scope**: 8 new cells (1 markdown + 7 code) appended to a 49-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Constitution Rule | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Code as simple as possible, clearly explained | PASS | Each cell has one purpose; markdown documents all parameters |
| 2 | Submitted as Jupyter notebooks | PASS | All code appended to `functions/f7/f7.ipynb` |
| 3 | No unit tests required | PASS | Only inline assertions for validation |
| 4 | Each problem in its own notebook/folder | PASS | Only `functions/f7/f7.ipynb` modified |
| 5 | Weekly changes as new sections, existing cells unchanged | PASS | New section "Week 7 — NN + NEI with Interior Penalty" appended after cell 49; cells 1–49 untouched |
| 6 | Use BoTorch library | PASS | BoTorch is for GP surrogates; NN surrogate uses PyTorch directly (per constitution: "Surrogate models are chosen per function based on problem characteristics") |
| 7 | Data in `./data` folder | PASS | No new data files; reads existing Week 7 `.npy` files |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/017-f7-nn-interior-penalty/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cells.md         # Phase 1 output — cell-by-cell contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code

```text
functions/f7/
└── f7.ipynb             # Existing 49 cells + 8 new cells (50–57)
```

**Structure Decision**: Single notebook, additive section only. No new files created. Matches existing project pattern and previous F7 weekly sections.

## Complexity Tracking

> No constitution violations. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
