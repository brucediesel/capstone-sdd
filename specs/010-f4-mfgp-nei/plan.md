# Implementation Plan: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Branch**: `010-f4-mfgp-nei` | **Date**: 2026-02-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/010-f4-mfgp-nei/spec.md`

## Summary

Add a Week 7 section to the F4 notebook that loads cumulative data (37 samples), trains a Multi-Fidelity GP surrogate (Matérn-5/2 ARD + LinearTruncatedFidelityKernel, z-score standardised, noise ≥ 1e-4) via MLL with 15 random restarts, proposes 4 candidates via qLogNoisyExpectedImprovement (q=4, fantasies=64) with fidelity fixed at 1.0, selects the best candidate for submission, and visualises surrogate slices + convergence. The MFGP configuration was selected as the PE winner for F4 (best NLP of -1.35 across 45 configs).

## Technical Context

**Language/Version**: Python 3.14 (pyenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch, NumPy, Matplotlib
**Storage**: `.npy` files in `data/f4/` (cumulative weekly snapshots)
**Testing**: Manual cell execution — no unit tests per constitution
**Target Platform**: macOS (local Jupyter notebook)
**Project Type**: Single Jupyter notebook (`functions/f4/f4.ipynb`)
**Performance Goals**: All cells complete within minutes; no GPU required
**Constraints**: 37 training points (small dataset), 4D input space, single-fidelity data with synthetic fidelity column
**Scale/Scope**: 8 new cells appended to existing 52-cell notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Code as simple as possible | ✅ PASS | Reuses proven PE winner config; manual z-score; no abstractions |
| 2 | Jupyter notebook format | ✅ PASS | All code in `functions/f4/f4.ipynb` |
| 3 | No unit tests | ✅ PASS | No test files created |
| 4 | Each problem in own notebook/folder | ✅ PASS | F4 stays in `functions/f4/` |
| 5 | Use BoTorch library | ✅ PASS | `SingleTaskMultiFidelityGP`, `qLogNoisyExpectedImprovement`, `optimize_acqf` |
| 6 | Weekly section with week number | ✅ PASS | New "Week 7" markdown header |
| 7 | Existing cells not replaced | ✅ PASS | Append-only — 8 new cells after cell 52 |
| 8 | All maximisation tasks | ✅ PASS | NEI maximises acquisition; convergence shows running max |

**All 8 gates pass. No violations to justify.**

## Project Structure

### Documentation (this feature)

```text
specs/010-f4-mfgp-nei/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── week7-cells.md   # Cell-by-cell acceptance contract
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
functions/f4/
└── f4.ipynb             # Existing 52-cell notebook; 8 new cells appended

data/f4/
├── updated_inputs - Week 7.npy   # Cumulative inputs (37×4)
└── updated_outputs - Week 7.npy  # Cumulative outputs (37,)
```

**Structure Decision**: No new files created. All code appended to the existing `f4.ipynb` notebook as new cells after the current Week 6 section (cell 52, `#VSC-21b0ced4`).

## Complexity Tracking

No constitution violations. Table intentionally left empty.
