# Implementation Plan: Week 7 — F1 Hurdle Model with Weighted UCB and Local Penalization

**Branch**: `005-week7-pe-surrogates` | **Date**: 2026-02-22 | **Spec**: [spec.md](spec.md)

## Summary

Append a "Week 7" section to `functions/f1/f1.ipynb`. Replace the polynomial surrogate (Weeks 5–6) with a two-stage hurdle model (Stage 1: Logistic Regression + CalibratedClassifierCV for P(y>0); Stage 2: Random Forest Regressor on log1p(y) for positive outputs). Combine with a weighted UCB acquisition function $a(x) = p(x)\cdot\mu(x) + \kappa\cdot p(x)\cdot\sigma_\text{RF}(x)$ and a multiplicative Gaussian local penalization mask applied against all 17 existing data points. Strategy is exploration-focused ($\kappa = 3.0$) due to zero improvement across three consecutive submissions. Output: one submission query in `x1.xxxxxx-x2.xxxxxx` format.

## Technical Context

**Language/Version**: Python 3 (Jupyter notebook, kernel already active in workspace)  
**Primary Dependencies**: `numpy`, `matplotlib`, `scikit-learn` (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor) — all already installed in the project environment  
**Storage**: Read-only `.npy` files from `data/f1/`; no writes except notebook state  
**Testing**: None required (per CONSTITUTION)  
**Target Platform**: Jupyter notebook cell execution (VS Code / JupyterLab)  
**Project Type**: Single notebook — append-only modification  
**Performance Goals**: All cells run in < 60 seconds total; 20 000 candidates × 100 trees on a 2D space is well within this budget  
**Constraints**: Append-only — no existing cells (cells 1–55) may be modified or deleted; all inputs clipped to [0.000000, 0.999999]; submission format exactly `x1.xxxxxx-x2.xxxxxx`  
**Scale/Scope**: 17 training samples, 2D input space, 10 new cells, ~1 notebook section

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Rule | Status | Evidence |
|------|--------|---------|
| All code submitted as Jupyter notebooks | ✅ PASS | New cells appended to `functions/f1/f1.ipynb` |
| Each step clearly explained | ✅ PASS | Markdown rationale cells W7-01, W7-03 required by contract |
| New section with week number as title | ✅ PASS | W7-01: `## Week 7 — Hurdle Model with Weighted UCB and Local Penalization` |
| Existing cells NOT replaced | ✅ PASS | Insert-only; cells 1–55 untouched |
| Hyperparameters documented with rationale | ✅ PASS | W7-03 markdown + W7-04 constants cell; all 8 parameters covered |
| Surrogate surface visualization | ✅ PASS | W7-08: 3-panel plot including hurdle prediction surface |
| Convergence plot | ✅ PASS | W7-09: running maximum line plot |
| scikit-learn for non-GP surrogate | ✅ PASS | LogisticRegression, RandomForestRegressor from scikit-learn |
| Maximization task | ✅ PASS | `argmax` of penalized UCB; convergence shows running maximum |
| Inputs ∈ [0.0, 0.999999] | ✅ PASS | W7-02 validates; W7-10 clips before formatting |
| No unit tests | ✅ PASS | Not applicable |

**Constitution Check POST-DESIGN**: All 11 gates pass. No violations.

## Project Structure

### Documentation (this feature)

```text
specs/005-week7-pe-surrogates/
├── plan.md                   ← this file
├── spec.md                   ← feature specification (with clarifications)
├── research.md               ← Phase 0 decisions (6 decisions, 0 NEEDS CLARIFICATION)
├── data-model.md             ← in-memory variable flow across the 10 new cells
├── quickstart.md             ← step-by-step implementation guide
├── contracts/
│   └── cell-contracts.md     ← per-cell preconditions, outputs, side effects
└── checklists/
    └── requirements.md       ← spec quality checklist (all items pass)
```

### Source Code

```text
functions/f1/
└── f1.ipynb          ← MODIFIED (10 new cells appended after existing cell 54)

data/f1/
├── updated_inputs - Week 7.npy    ← READ (already exists)
└── updated_outputs - Week 7.npy   ← READ (already exists)
```

No new files created in source. No other function notebooks are touched.

## Complexity Tracking

No constitution violations to justify.
