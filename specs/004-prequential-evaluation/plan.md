# Implementation Plan: Prequential Evaluation of Surrogate Models

**Branch**: `004-prequential-evaluation` | **Date**: 2026-02-20 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-prequential-evaluation/spec.md`

## Summary

Fix and complete the auto-generated notebook at `functions/f1/preq-eval-f1.ipynb` (36 cells, 18 markdown + 18 code). The notebook structure is correct but contains 5 bugs that prevent execution: missing `WEEK` parameter (FR-001), `kernel_type` config ignored (FR-008), `noise_lb` config ignored (FR-008), a stray `∏` Unicode character causing SyntaxError, and an inconsistent log-transform inverse. All fixes are in-place cell edits — no structural changes needed.

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv sdd-dev)
**Primary Dependencies**: BoTorch/GPyTorch (GP), PyMC + PyMC-BART (BART), numpy, pandas, matplotlib
**Storage**: `.npy` files in `data/f1/`
**Testing**: No unit tests required (per CONSTITUTION) — manual notebook execution validates
**Target Platform**: Jupyter Notebooks (Google Colab compatible)
**Project Type**: Single project — 1 Jupyter notebook
**Performance Goals**: Full notebook executes in <5 minutes (BART MCMC is bottleneck)
**Constraints**: Append-only project convention; F1-only (no generalisation to f2–f8)
**Scale/Scope**: 1 notebook, 36 cells, 5 bug fixes + 1 import addition

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Constitution Principle | Status | Notes |
|----------------------|--------|-------|
| Code as simple as possible, clearly explained | ✅ PASS | Each step has a markdown explanation cell |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests required | ✅ PASS | N/A |
| 8 separate problems, each in own notebook | ✅ PASS | Notebook in `functions/f1/` |
| Data in `./data` folder structure | ✅ PASS | Loads from `data/f1/` |
| Use BoTorch library | ✅ PASS | GP via BoTorch SingleTaskGP |
| Hyperparameters clearly specified and explained | ✅ PASS | 10 GP configs + 10 BART configs with labels |
| Visualisations provided | ✅ PASS | 3-panel plots + bar charts + ranked table |

**Gate Result**: ✅ ALL PASS — no violations.

## Project Structure

### Documentation (this feature)

```text
specs/004-prequential-evaluation/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
functions/
└── f1/
    └── preq-eval-f1.ipynb   # Target notebook (exists, needs bug fixes)

data/
└── f1/
    ├── updated_inputs - Week 6.npy    # Already exists (16×2)
    └── updated_outputs - Week 6.npy   # Already exists (16,)
```

**Structure Decision**: Fix cells in existing notebook `functions/f1/preq-eval-f1.ipynb`. No new files created (except spec artifacts).

## Steps

1. **Add `WEEK` variable and parameterize data paths** (cell 5) — Add `WEEK = 6` at the top of the data-loading cell. Replace hardcoded `'../../data/f1/updated_inputs - Week 6.npy'` with `f'../../data/f1/updated_inputs - Week {WEEK}.npy'` (and same for outputs). Update cell 4 markdown to mention the parameter. *(Fixes FR-001)*

2. **Add missing imports** (cell 3) — Add `import gpytorch`, `from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel` for kernel switching, and verify all existing imports are present.

3. **Implement `kernel_type` switching in `gp_prequential_with_config()`** (cell 15) — Inside the training loop, add a conditional that constructs a `ScaleKernel(MaternKernel(nu=2.5))` when `config['kernel_type'] == 'matern'` or `ScaleKernel(RBFKernel())` when `'rbf'`, and passes it as the `covar_module` to `SingleTaskGP`. Currently the function always uses the default kernel. *(Fixes FR-008 — kernel_type)*

4. **Implement `noise_lb` constraint in `gp_prequential_with_config()`** (cell 15) — After constructing the GP model, apply the noise lower bound from config via `model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(config['noise_lb']))` or equivalent BoTorch pattern. *(Fixes FR-008 — noise_lb)*

5. **Remove stray `∏` character** (cell 25) — Delete the Unicode product symbol `∏` between the `bart_configs` list closing `]` and the `print(...)` call. This causes a `SyntaxError` and blocks all BART HP evaluation. *(Critical fix)*

6. **Fix log-transform inverse consistency** (cell 15) — Replace `np.sign(predictions) * (np.exp(np.abs(predictions)) - 1e-300)` with a consistent inverse, e.g., use `np.log1p(np.abs(y))` / `np.expm1(np.abs(pred))` pair, or ensure the epsilon handling matches in both directions. *(Minor accuracy fix)*

7. **Run full notebook and validate** — Execute all 36 cells top-to-bottom and confirm the 7 verification checks below pass.

## Verification

- **SC-001**: All 36 cells execute without errors
- **SC-002**: GP results DataFrame has 10 rows; BART results DataFrame has 10 rows (20 total)
- **SC-003**: Final comparison table identifies best GP and best BART by NLP
- **SC-004**: Every row in both DataFrames has MAE, NLP, Coverage_95 (no unexpected NaN)
- **SC-005**: All plots render with labels and titles
- **Kernel check**: RBF configs produce different NLP values than Matérn configs (confirms switching works)
- **Noise check**: Different `noise_lb` values produce different results for same kernel/transform combo

## Decisions

- WEEK parameterized with `WEEK = 6` variable (clarify Q1: option A)
- Ranking by NLP only, lower is better (clarify Q2: option A)
- F1-only notebook, no FUNCTION parameter (clarify Q3: option B)
- Fix existing notebook in-place rather than regenerating from scratch

## Complexity Tracking

No constitution violations — this section is not needed.


---

# Implementation Plan: F2 — Prequential Evaluation (GP vs BART vs Random Forest)

**Date**: 2026-02-20 | **Spec**: [spec.md](spec.md) (F2 section)

## Summary

Create a new notebook `functions/f2/preq-eval-f2.ipynb` from scratch (based on the F1 notebook pattern) that performs prequential one-step-ahead evaluation of three surrogate model families on Function 2 (noisy log-likelihood maximisation, 2D). The notebook includes GP (BoTorch), BART (PyMC-BART), and Random Forest (scikit-learn) with 10 configurations each (30 total), plus a three-way comparison.

## Technical Context (F2)

**Language/Version**: Python 3.14.2 (pyenv sdd-dev)
**Primary Dependencies**: BoTorch/GPyTorch (GP), PyMC + PyMC-BART (BART), scikit-learn (RF), numpy, pandas, matplotlib
**Storage**: `.npy` files in `data/f2/`
**Testing**: No unit tests required (per CONSTITUTION) — manual notebook execution validates
**Target Platform**: Jupyter Notebooks
**Project Type**: Single project — 1 new Jupyter notebook
**Performance Goals**: Full notebook executes in <10 minutes (30 configs, BART MCMC is bottleneck)
**Scale/Scope**: 1 notebook, ~48 cells (18 markdown + 18 code from F1 pattern + ~12 new cells for RF)

## Constitution Check (F2)

| Constitution Principle | Status | Notes |
|----------------------|--------|-------|
| Code as simple as possible | ✅ PASS | Each step has a markdown explanation cell |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests required | ✅ PASS | N/A |
| 8 separate problems, each in own notebook | ✅ PASS | Notebook in `functions/f2/` |
| Data in `./data` folder structure | ✅ PASS | Loads from `data/f2/` |
| Use BoTorch library | ✅ PASS | GP via BoTorch SingleTaskGP |
| scikit-learn for RF/GBT | ✅ PASS | RF via RandomForestRegressor |
| Hyperparameters documented | ✅ PASS | 10 GP + 10 BART + 10 RF configs with labels |
| Visualisations provided | ✅ PASS | 3-panel plots + bar charts + ranked table |

**Gate Result**: ✅ ALL PASS

## Project Structure (F2)

### Source Code

```text
functions/
└── f2/
    ├── f2.ipynb               # Existing BO notebook (not modified)
    └── preq-eval-f2.ipynb     # NEW — prequential evaluation notebook

data/
└── f2/
    ├── updated_inputs - Week 6.npy    # Already exists (16×2)
    └── updated_outputs - Week 6.npy   # Already exists (16,)
```

## Notebook Cell Structure (F2)

| Cell | Type | Content |
|------|------|---------|
| 1 | md | Title & overview (F2, 2D, maximise, 3 surrogates) |
| 2 | md | Evaluation metrics (MAE, NLP, Coverage) |
| 3 | code | Imports (BoTorch, PyMC, sklearn, numpy, etc.) |
| 4 | md | Step 1: Load Data |
| 5 | code | Load F2 data, WEEK=6, N_INIT=10 |
| 6 | md | Step 2: Evaluation Metrics |
| 7 | code | `compute_metrics()` function |
| 8 | md | Step 3: GP Prequential Evaluation |
| 9 | code | `gp_prequential_evaluation()` default function |
| 10 | md | Run GP Default |
| 11 | code | Execute GP default |
| 12 | md | GP Default Visualisation |
| 13 | code | `plot_prequential_results()` + plot GP default |
| 14 | md | GP HP Optimisation (10 configs) |
| 15 | code | `gp_prequential_with_config()` + hp_configs + run all |
| 16 | md | Best GP Configuration |
| 17 | code | Select best GP by NLP |
| 18 | md | Step 4: BART Prequential Evaluation |
| 19 | code | `bart_prequential_evaluation()` function |
| 20 | md | Run BART Default |
| 21 | code | Execute BART default |
| 22 | md | BART Default Visualisation |
| 23 | code | Plot BART default |
| 24 | md | BART HP Optimisation (10 configs) |
| 25 | code | bart_configs + run all 10 |
| 26 | md | Best BART Configuration |
| 27 | code | Select best BART by NLP |
| 28 | md | **Step 5: Random Forest Prequential Evaluation** |
| 29 | code | `rf_prequential_evaluation()` function |
| 30 | md | Run RF Default |
| 31 | code | Execute RF default |
| 32 | md | RF Default Visualisation |
| 33 | code | Plot RF default |
| 34 | md | RF HP Optimisation (10 configs) |
| 35 | code | `rf_prequential_with_config()` + rf_configs + run all 10 |
| 36 | md | Best RF Configuration |
| 37 | code | Select best RF by NLP |
| 38 | md | **Step 6: 3-Way Comparison — GP vs BART vs RF** |
| 39 | code | Build 3-row comparison table, metric winners |
| 40 | md | Visual Comparison |
| 41 | code | 3-panel bar chart (3 bars per metric) |
| 42 | md | Hyperparameter Sensitivity — All Configurations |
| 43 | code | 3-panel horizontal bar chart, all 30 configs |
| 44 | md | Full Results Table |
| 45 | code | Ranked table of all 30 configs sorted by NLP |
| 46 | md | Conclusions |

## Steps (F2)

1. **Create notebook** with all cells from scratch following the F1 pattern
2. **Adapt data loading** for F2 paths and 2D inputs
3. **Reuse GP/BART functions** from F1 (adapted for F2 output range — no log-transform needed by default but kept as config option)
4. **Add RF evaluation** — `rf_prequential_evaluation()` uses sklearn `RandomForestRegressor`, uncertainty from individual tree predictions
5. **Add RF HP optimisation** — 10 configs varying n_estimators, max_depth, min_samples_leaf, bootstrap
6. **Extend comparison** from 2-way to 3-way (GP vs BART vs RF)
7. **Extend ranked table** to 30 rows
8. **Run and validate** — all cells execute without errors

## Verification (F2)

- **SC-F2-001**: All cells execute without errors
- **SC-F2-002**: 6 one-step-ahead predictions per config (30 configs × 6 = 180 total predictions)
- **SC-F2-003**: Final 3-way comparison identifies best surrogate for F2
- **SC-F2-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration
- **SC-F2-005**: Visualisations clear, labelled
- **SC-F2-006**: Code is simple, each step explained

## Decisions (F2)

- Build new notebook from scratch (not modifying F1)
- Random Forest uncertainty via individual tree prediction variance (tree_preds.std)
- Keep log-transform as GP config option even though F2 likely doesn't need it
- Same prequential protocol as F1 (train on first 10, predict steps 11–16)
- Ranking by NLP (lower is better), same as F1
