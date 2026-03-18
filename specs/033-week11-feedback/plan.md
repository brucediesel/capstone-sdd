# Implementation Plan: Week 11 Best-Marker Visibility Fix (Green Star, Larger Size)

**Branch**: `033-week11-feedback` | **Date**: 2026-03-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/033-week11-feedback/spec.md`

**Scope**: This is the **second incremental fix** to the 033-week11-feedback feature. The original full implementation (41 tasks) and the first marker-size fix (s=200→s=350, 12 tasks) are both complete. This plan addresses the user's continued feedback that the red star best-output marker remains invisible on pair plots, especially on high-dimensional functions (F7: 15 subplots, F8: 28 subplots). Two changes: (1) increase marker size further from s=350, and (2) change marker colour from red to green.

## Summary

The best-output star marker (`marker='*'`, `c='red'`, `s=350`) on 2D pair plots remains insufficiently visible, particularly on dense high-dimensional subplot grids. The fix changes the marker colour from red to green for better contrast against both blue (initial) and orange (submission) scatter points, and increases the marker size from `s=350` to `s=500`. The change affects cell index 7, source line 38 in all 8 notebooks.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv `sdd-dev`)
**Primary Dependencies**: numpy 2.4.1, matplotlib 3.10.8
**Storage**: NumPy `.npy` files in `./data/fX/` folders
**Testing**: Manual — run all notebook cells, visually inspect pair plot outputs
**Target Platform**: macOS, Jupyter notebooks via VS Code
**Project Type**: Single project — 8 Jupyter notebooks
**Performance Goals**: N/A (static visualisation)
**Constraints**: Marker must be clearly visible on grids up to 28 subplots (F8)
**Scale/Scope**: 8 notebooks, 1 line change per notebook (cell[7].source[38])

**Current state**: All 8 notebooks have identical scatter call:
```python
c='red', marker='*', s=350, zorder=5, label='Best'
```

**Target state**:
```python
c='green', marker='*', s=500, zorder=5, label='Best'
```

**Unknowns**: None — all parameters resolved.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Simplicity | PASS | Single parameter change in 8 files — minimal complexity |
| II | Per-Function Isolation | PASS | Each notebook modified independently in its own folder |
| III | Per-Iteration Notebooks | PASS | Modifying current iteration (week 11) notebooks only; no previous-iteration notebooks touched |
| IV | Data Organisation | PASS | No data files modified |
| V | BoTorch & PyTorch Stack | N/A | No surrogate/acquisition changes |
| VI | Documentation & Visualisation | PASS | Improving visualisation clarity per this principle |
| VII | Maximisation Objective | N/A | No optimisation logic changes |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/033-week11-feedback/
├── plan.md              # This file (updated for green star fix)
├── research.md          # Phase 0 output (updated)
├── data-model.md        # Phase 1 output (updated)
├── quickstart.md        # Phase 1 output (updated)
├── contracts/           # Phase 1 output (updated)
│   └── README.md
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
functions/
├── f1/f1 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f2/f2 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f3/f3 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f4/f4 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f5/f5 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f6/f6 - week 11.ipynb   # cell[7] line 38: scatter best marker
├── f7/f7 - week 11.ipynb   # cell[7] line 38: scatter best marker (15 subplots)
└── f8/f8 - week 11.ipynb   # cell[7] line 38: scatter best marker (28 subplots)
```

**Structure Decision**: No structural changes. All edits are in-place parameter changes to existing notebook cells.

## Complexity Tracking

No constitution violations. No complexity justification needed.
