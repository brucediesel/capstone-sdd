# Implementation Plan: F5 Week 7 — GP Matérn-5/2 + NEI

**Branch**: `011-f5-gp-nei` | **Date**: 2026-02-23 | **Spec**: `specs/011-f5-gp-nei/spec.md`
**Input**: Feature specification from `/specs/011-f5-gp-nei/spec.md`

## Summary

Add Week 7 section to F5 notebook implementing exploration-focused Bayesian optimisation. Uses a GP surrogate (Matérn-5/2 ARD kernel, log1p + z-score transform, ℓ=0.5 init, noise=0.1·Var(y) init, `outcome_transform=None`) fitted via 15-restart MLL, with `qLogNoisyExpectedImprovement` acquisition (q=4, 3000 Sobol → 50 L-BFGS). Candidate selection filters to posterior mean > median, then picks farthest from existing data. F5 has stalled at a local maximum (~3394.68) for 2 weeks; exploration-promoting hyperparameters aim to escape this region.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `data/f5/` (27 samples × 4 dims)
**Testing**: None required (per constitution)
**Target Platform**: macOS (local Jupyter notebook execution)
**Project Type**: Single project — Jupyter notebooks
**Performance Goals**: N/A (single-run batch BO, no latency targets)
**Constraints**: Append-only notebook cells; no modification of existing Week 1–6 content
**Scale/Scope**: 8 new cells (51–58) appended to existing 50-cell notebook

### Key Technical Decisions

1. **No ξ parameter**: BoTorch's `qLogNoisyExpectedImprovement` has no ξ. The `eta` parameter only affects constraint smoothing (no effect when `constraints=None`). Exploration is achieved via q=4 + distance-based selection + ℓ=0.5 + noise=0.1·Var.
2. **No double-standardization**: Pass `outcome_transform=None` to `SingleTaskGP` since we manually z-score. The default `Standardize(m=1)` would double-standardize.
3. **qLogNEI not qNEI**: Use `qLogNoisyExpectedImprovement` (log-space formulation) for numerical stability, not `qNoisyExpectedImprovement`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Code as simple as possible | PASS | Straightforward sequential cells, each step explained in markdown |
| 2 | All code in Jupyter notebooks | PASS | All work targets `functions/f5/f5.ipynb` |
| 3 | No unit tests required | PASS | No test tasks; tasks.md explicitly states "No tests" |
| 4 | Each problem in its own notebook/folder | PASS | F5 notebook in `functions/f5/` |
| 5 | Weekly sections appended, not replaced | PASS | FR-020: append after cell 50, no existing cells modified |
| 6 | Section title includes week number | PASS | Cell 51 header: "## Week 7 — GP Matérn-5/2 + NEI" |
| 7 | BoTorch as default GP library | PASS | SingleTaskGP + qLogNEI from BoTorch |
| 8 | Hyperparameters documented with rationale | PASS | FR-019, Cell 53: markdown HP table with 14+ entries |
| 9 | Surrogate function visualisation | PASS | FR-017, Cell 56: 3-panel mean/std/relevance |
| 10 | Convergence visualisation | PASS | FR-018, Cell 57: running-best plot with boundary |

**Result**: All 10 gates PASS. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/011-f5-gp-nei/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── week7-cells.md   # Cell acceptance contract
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
functions/f5/
└── f5.ipynb             # Target notebook (50 existing cells + 8 new)

data/f5/
├── updated_inputs - Week 7.npy    # (27, 4) input data
└── updated_outputs - Week 7.npy   # (27,) output data
```

**Structure Decision**: No new source directories. All implementation is 8 appended cells in the existing `functions/f5/f5.ipynb` notebook. Data is read from `data/f5/`.

## Complexity Tracking

> No constitution violations. Table intentionally left empty.
