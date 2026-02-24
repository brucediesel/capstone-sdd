# Implementation Plan: F6 Interior Penalty on Acquisition Function

**Branch**: `016-f6-interior-penalty` | **Date**: 2026-02-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/016-f6-interior-penalty/spec.md`

## Summary

Add a soft interior penalty to the F6 (5D food recipe) acquisition function to discourage boundary-hugging candidates. The penalty is a smooth multiplicative weight `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)` applied post-hoc to the batch of 4 candidates returned by `optimize_acqf`. Because F6 outputs are all-negative, the penalty cannot simply multiply `pred_means` (that would promote boundary candidates). Instead, a rank-based scoring strategy is used: candidates are ranked by posterior mean (higher is better) and by interior penalty weight (higher is better), then a combined rank is used in the median filter. Two hyperparameters (`STEEPNESS`, `FLOOR`) are exposed as named constants. A new 3-panel visualisation replaces the Week 7 dimension-relevance bar chart with a penalised desirability surface.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: BoTorch (SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf), GPyTorch (ExactMarginalLogLikelihood, Matérn-1.5, GaussianLikelihood), PyTorch, NumPy, Matplotlib  
**Storage**: `.npy` files in `data/f6/` (read-only); notebook state in `functions/f6/f6.ipynb`  
**Testing**: Manual execution — run all cells top-to-bottom; no unit tests (per constitution)  
**Target Platform**: macOS (local), Jupyter notebook / VS Code  
**Project Type**: Single Jupyter notebook (additive section)  
**Performance Goals**: New section executes in < 30 seconds (posterior evaluation on 80×80 grid + penalty on 4 candidates)  
**Constraints**: Must not modify any existing cells; all new cells appended after cell 59 (last markdown analysis cell)  
**Scale/Scope**: 6 new cells (1 markdown + 5 code) appended to a 59-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Constitution Rule | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Code as simple as possible, clearly explained | ✅ PASS | Each cell has one purpose; markdown documents all parameters |
| 2 | Submitted as Jupyter notebooks | ✅ PASS | All code appended to `functions/f6/f6.ipynb` |
| 3 | No unit tests required | ✅ PASS | Only inline assertions for validation |
| 4 | Each problem in its own notebook/folder | ✅ PASS | Only `functions/f6/f6.ipynb` modified |
| 5 | Weekly changes as new sections, existing cells unchanged | ✅ PASS | New section "Week 7 — Interior Penalty" appended after cell 59; cells 0–59 untouched |
| 6 | Use BoTorch library | ✅ PASS | Reuses existing BoTorch `best_model` for posterior evaluation; penalty is pure NumPy/PyTorch on top |
| 7 | Data in `./data` folder | ✅ PASS | No new data files; reads existing Week 7 `.npy` files |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/016-f6-interior-penalty/
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
functions/f6/
└── f6.ipynb             # Existing 59 cells + 6 new cells (60–65)
```

**Structure Decision**: Single notebook, additive section only. No new files created. Matches existing project pattern and F5 interior penalty (015) approach.

## Complexity Tracking

> No constitution violations. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
