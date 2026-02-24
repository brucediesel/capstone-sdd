# Implementation Plan: F6 Week 7 — SFGP Matérn-1.5 + NEI

**Branch**: `012-f6-sfgp-nei` | **Date**: 2026-02-24 | **Spec**: `specs/012-f6-sfgp-nei/spec.md`
**Input**: Feature specification from `/specs/012-f6-sfgp-nei/spec.md`

## Summary

Add Week 7 section to F6 notebook implementing exploration-focused Bayesian optimisation. Uses a Single-Fidelity GP surrogate (Matérn-1.5 ARD kernel, `Standardize(m=1)` default outcome transform, ℓ=0.5 init, noise=0.1·Var(y) init with lower bound 1e-8) fitted via 15-restart MLL, with `qLogNoisyExpectedImprovement` acquisition (q=4, 3000 Sobol → 50 L-BFGS). Candidate selection filters to posterior mean ≥ median, then picks farthest from existing data. F6 has all-negative outputs [-2.571, -0.206] with strong x4 (milk) anti-correlation; exploration aims to diversify sampling in the 5D space.

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `data/f6/` (27 samples × 5 dims)
**Testing**: None required (per constitution)
**Target Platform**: macOS (local Jupyter notebook execution)
**Project Type**: Single project — Jupyter notebooks
**Performance Goals**: N/A (single-run batch BO, no latency targets)
**Constraints**: Append-only notebook cells; no modification of existing Week 4–6 content
**Scale/Scope**: 8 new cells (48–55) appended to existing 47-cell notebook

### Key Technical Decisions

1. **Matérn ν=1.5 (not 2.5)**: User-specified. Once-differentiable kernel, rougher than ν=2.5, suits functions with sharp local variations. F6's cake-recipe function may have abrupt ingredient interactions.
2. **Standardize(m=1) default**: F6's output range is only 12.5x ([-2.571, -0.206]) and all-negative. Unlike F5 (30,000x range requiring manual log1p + z-score), BoTorch's built-in `Standardize(m=1)` is sufficient. No manual transform, no `outcome_transform=None`.
3. **Noise lower bound 1e-8 (not 1e-6)**: Tighter constraint allows the GP to interpolate more closely when noise is truly low, per user specification.
4. **5D (not 4D)**: F6 has 5 input dimensions (flour, sugar, eggs, butter, milk), one more than F5.
5. **qLogNEI not qNEI**: Use `qLogNoisyExpectedImprovement` (log-space formulation) for numerical stability with the all-negative output regime.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Code as simple as possible | PASS | Straightforward sequential cells, each step explained in markdown |
| 2 | All code in Jupyter notebooks | PASS | All work targets `functions/f6/f6.ipynb` |
| 3 | No unit tests required | PASS | No test tasks |
| 4 | Each problem in its own notebook/folder | PASS | F6 notebook in `functions/f6/` |
| 5 | Weekly sections appended, not replaced | PASS | FR-019: append after cell 47, no existing cells modified |
| 6 | Section title includes week number | PASS | Cell 48 header: "## Week 7 — SFGP Matérn-1.5 + NEI" |
| 7 | BoTorch as default GP library | PASS | SingleTaskGP + qLogNEI from BoTorch |
| 8 | Hyperparameters documented with rationale | PASS | FR-009, Cell 50: markdown HP table |
| 9 | Surrogate function visualisation | PASS | FR-014, Cell 53: 3-panel mean/std/relevance |
| 10 | Convergence visualisation | PASS | FR-016, Cell 54: running-best plot with boundary |

**Result**: All 10 gates PASS. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/012-f6-sfgp-nei/
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
functions/f6/
└── f6.ipynb             # Target notebook (47 existing cells + 8 new)

data/f6/
├── updated_inputs - Week 7.npy    # (27, 5) input data
└── updated_outputs - Week 7.npy   # (27,) output data
```

**Structure Decision**: No new source directories. All implementation is 8 appended cells in the existing `functions/f6/f6.ipynb` notebook. Data is read from `data/f6/`.

## Complexity Tracking

> No constitution violations. Table intentionally left empty.
