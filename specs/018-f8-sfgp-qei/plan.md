# Implementation Plan: F8 Week 7 — SFGP + qEI Acquisition

**Branch**: `018-f8-sfgp-qei` | **Date**: 2025-02-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/018-f8-sfgp-qei/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add a new Week 7 section to the F8 notebook (8D ML hyperparameter tuning) that fits a BoTorch SingleTaskGP surrogate with Matern 2.5 kernel, ARD lengthscales, output standardisation, and noise floor 1e-07. Compute quasi-Expected Improvement (qEI) via 256 MC samples (SobolQMCNormalSampler) with xi=0.01 improvement threshold. Optimise qEI over [0,1]^8 using 30 restarts and 4096 raw samples to propose the next evaluation point. Provide GP lengthscale-based feature importance, 3-panel surrogate slice, and convergence plot matching the Week 5/6 visualisation pattern. All 47 outputs are positive (range [5.59, 9.95]), so qEI with best_f = y_max + xi is well-defined.

## Technical Context

**Language/Version**: Python 3.14.2 (macOS, conda env `sdd-dev`)
**Primary Dependencies**: BoTorch (SingleTaskGP, qExpectedImprovement), GPyTorch (MaternKernel, GaussianLikelihood, GreaterThan), PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `data/f8/` (read-only); notebook state in `functions/f8/f8.ipynb`
**Testing**: Manual execution — run all cells top-to-bottom; no unit tests (per constitution)
**Target Platform**: macOS (local), Jupyter notebook / VS Code
**Project Type**: Single Jupyter notebook (additive section)
**Performance Goals**: New section executes in < 60 seconds (GP fitting on 47x8 + qEI optimisation with 30 restarts)
**Constraints**: Must not modify any existing cells (cells 1-49); all new cells appended after cell 49
**Scale/Scope**: 8 new cells (1 markdown + 7 code) appended to a 49-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Constitution Rule | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Code as simple as possible, clearly explained | PASS | Each cell has one purpose; markdown documents all parameters |
| 2 | Submitted as Jupyter notebooks | PASS | All code appended to `functions/f8/f8.ipynb` |
| 3 | No unit tests required | PASS | Only inline assertions for validation |
| 4 | Each problem in its own notebook/folder | PASS | Only `functions/f8/f8.ipynb` modified |
| 5 | Weekly changes as new sections, existing cells unchanged | PASS | New section "Week 7 — SFGP + qEI Acquisition" appended after cell 49; cells 1-49 untouched |
| 6 | Use BoTorch library | PASS | SingleTaskGP + qExpectedImprovement are BoTorch classes |
| 7 | Data in `./data` folder | PASS | No new data files; reads existing Week 7 `.npy` files |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/018-f8-sfgp-qei/
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
functions/f8/
└── f8.ipynb             # Existing 49 cells + 8 new cells (50-57)
```

**Structure Decision**: Single notebook, additive section only. No new files created. Matches existing project pattern and previous F8 weekly sections.

## Complexity Tracking

> No constitution violations. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
