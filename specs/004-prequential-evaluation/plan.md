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

---

# Implementation Plan: F3 — Prequential Evaluation (GP vs BART vs Random Forest)

**Date**: 2026-02-20 | **Spec**: [spec.md](spec.md) (F3 section)

## Summary

Create a new notebook `functions/f3/preq-eval-f3.ipynb` from scratch (based on the F2 notebook pattern) that performs prequential one-step-ahead evaluation of three surrogate model families on Function 3 (drug discovery — minimise adverse reactions from 3 compounds, 3D input). The notebook includes GP (BoTorch), BART (PyMC-BART), and Random Forest (scikit-learn) with **15 configurations each** (45 total), plus a three-way comparison.

## Technical Context (F3)

**Language/Version**: Python 3.14.2 (pyenv sdd-dev)
**Primary Dependencies**: BoTorch/GPyTorch (GP), PyMC + PyMC-BART (BART), scikit-learn (RF), numpy, pandas, matplotlib
**Storage**: `.npy` files in `data/f3/`
**Testing**: No unit tests required (per CONSTITUTION) — manual notebook execution validates
**Target Platform**: Jupyter Notebooks
**Project Type**: Single project — 1 new Jupyter notebook
**Performance Goals**: Full notebook executes in <15 minutes (45 configs, BART MCMC is bottleneck)
**Scale/Scope**: 1 notebook, ~46 cells (23 markdown + 23 code)

## Constitution Check (F3)

| Constitution Principle | Status | Notes |
|----------------------|--------|-------|
| Code as simple as possible | ✅ PASS | Each step has a markdown explanation cell |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests required | ✅ PASS | N/A |
| 8 separate problems, each in own notebook | ✅ PASS | Notebook in `functions/f3/` |
| Data in `./data` folder structure | ✅ PASS | Loads from `data/f3/` |
| Use BoTorch library | ✅ PASS | GP via BoTorch SingleTaskGP |
| scikit-learn for RF | ✅ PASS | RF via RandomForestRegressor |
| Hyperparameters documented | ✅ PASS | 15 GP + 15 BART + 15 RF configs with labels |
| Visualisations provided | ✅ PASS | 3-panel plots + bar charts + ranked table |

**Gate Result**: ✅ ALL PASS

## Project Structure (F3)

### Source Code

```text
functions/
└── f3/
    ├── f3.ipynb               # Existing BO notebook (not modified)
    └── preq-eval-f3.ipynb     # NEW — prequential evaluation notebook

data/
└── f3/
    ├── updated_inputs - Week 6.npy    # Already exists (16×3)
    └── updated_outputs - Week 6.npy   # Already exists (16,)
```

## Key Differences from F2

| Aspect | F2 | F3 |
|--------|----|----|
| Input dimensions | 2 | **3** (compound concentrations) |
| Problem domain | Log-likelihood | **Drug discovery** (adverse reactions) |
| HP configs per family | 10 | **15** |
| Total configurations | 30 | **45** |
| GP kernels | Matérn 5/2, RBF | Matérn 5/2, **Matérn 3/2**, RBF |
| BART max draws | 500 | **1000** |
| RF max_depth options | None, 5, 10 | None, **3**, 5, 10 |

## Steps (F3)

1. **Create notebook** with all cells following the F2 pattern, adapted for F3
2. **Adapt data loading** for F3 paths (3D inputs, `../../data/f3/`)
3. **Enhance GP configs** — add Matérn 3/2 kernel (less smooth, suits drug response) for 15 configurations
4. **Enhance BART configs** — add draws=1000 for posterior convergence assessment, for 15 configurations
5. **Enhance RF configs** — add max_depth=3 and min_samples_leaf=3 for small-data 3D regime, for 15 configurations
6. **Build 3-way comparison** of best GP vs BART vs RF by NLP
7. **Build 45-row ranked table** sorted by NLP
8. **Run and validate** — all cells execute without errors

## Verification (F3)

- **SC-F3-001**: All cells execute without errors
- **SC-F3-002**: 6 one-step-ahead predictions per config (45 configs × 6 = 270 total predictions)
- **SC-F3-003**: Final 3-way comparison identifies best surrogate for F3
- **SC-F3-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration
- **SC-F3-005**: Visualisations clear, labelled
- **SC-F3-006**: Code is simple, each step explained

## Decisions (F3)

- Build new notebook from scratch following F2 pattern
- Add Matérn 3/2 kernel to GP configs for less-smooth drug response surfaces
- Increase to 15 HP configurations per model (as specified)
- Include BART draws=1000 to test whether more MCMC samples improve calibration
- Include RF max_depth=3 and min_samples_leaf=3 for regularisation in small 3D datasets
- Ranking by NLP (lower is better), consistent with F1/F2

---

# Implementation Plan: F4 — Prequential Evaluation (Single Fidelity GP vs Multi Fidelity GP)

**Date**: 2026-02-20 | **Spec**: [spec.md](spec.md) (F4 section)

## Summary

Create a new notebook `functions/f4/preq-eval-f4.ipynb` from scratch that performs prequential one-step-ahead evaluation of two GP surrogate families and Gradient Boosted Trees on Function 4 (warehouse product placement, 4D input, 30 initial points). The notebook includes Single Fidelity GP (BoTorch `SingleTaskGP`), Multi Fidelity GP (BoTorch `SingleTaskMultiFidelityGP` with autoregressive/co-kriging kernel), and Gradient Boosted Trees (scikit-learn `GradientBoostingRegressor`) with **15 configurations each** (45 total), plus a three-way comparison.

## Technical Context (F4)

**Language/Version**: Python 3.14.2 (pyenv sdd-dev)
**Primary Dependencies**: BoTorch/GPyTorch (SF-GP and MF-GP), scikit-learn (GBT), numpy, pandas, matplotlib
**Storage**: `.npy` files in `data/f4/`
**Testing**: No unit tests required (per CONSTITUTION) — manual notebook execution validates
**Target Platform**: Jupyter Notebooks
**Project Type**: Single project — 1 new Jupyter notebook
**Performance Goals**: Full notebook executes in <15 minutes (45 configs, GBT quantile regression adds moderate overhead)
**Scale/Scope**: 1 notebook, ~52 cells (26 markdown + 26 code)

## Constitution Check (F4)

| Constitution Principle | Status | Notes |
|----------------------|--------|-------|
| Code as simple as possible | ✅ PASS | Each step has a markdown explanation cell |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests required | ✅ PASS | N/A |
| 8 separate problems, each in own notebook | ✅ PASS | Notebook in `functions/f4/` |
| Data in `./data` folder structure | ✅ PASS | Loads from `data/f4/` |
| Use BoTorch library | ✅ PASS | SF-GP via SingleTaskGP, MF-GP via SingleTaskMultiFidelityGP |
| scikit-learn for GBT | ✅ PASS | GBT via GradientBoostingRegressor |
| Hyperparameters documented | ✅ PASS | 15 SF-GP + 15 MF-GP + 15 GBT configs with labels |
| Visualisations provided | ✅ PASS | 3-panel plots + bar charts + ranked table |

**Gate Result**: ✅ ALL PASS

## Project Structure (F4)

### Source Code

```text
functions/
└── f4/
    ├── f4.ipynb               # Existing BO notebook (not modified)
    └── preq-eval-f4.ipynb     # NEW — prequential evaluation notebook

data/
└── f4/
    ├── updated_inputs - Week 6.npy    # Already exists (36×4)
    └── updated_outputs - Week 6.npy   # Already exists (36,)
```

## Key Differences from F3

| Aspect | F3 | F4 |
|--------|----|----|
| Input dimensions | 3 | **4** (warehouse parameters) |
| Problem domain | Drug discovery | **Warehouse product placement** |
| Initial samples | 10 | **30** |
| Total samples | 16 | **36** |
| Surrogate families | GP, BART, RF | **Single Fidelity GP, Multi Fidelity GP, GBT** |
| HP configs per family | 15 | **15** |
| Total configurations | 45 | **45** |
| Output range | Moderate | **Wide negative (-32.6 to 0.5)** |

## Steps (F4)

1. **Create notebook** with all cells following the F3 pattern, adapted for F4
2. **Adapt data loading** for F4 paths (4D inputs, `../../data/f4/`, N_INIT=30)
3. **Implement SF-GP** — reuse GP pattern from F3, add output standardisation option for wide-range outputs
4. **Implement MF-GP** — use BoTorch `SingleTaskMultiFidelityGP` with synthetic fidelity column, vary kernel and fidelity structure
5. **Implement GBT** — use scikit-learn `GradientBoostingRegressor` with quantile regression for uncertainty estimation
6. **Build 15 SF-GP configs** — kernel × transform × noise bounds
7. **Build 15 MF-GP configs** — kernel × fidelity structure × transform × noise bounds
8. **Build 15 GBT configs** — n_estimators × learning_rate × max_depth × subsample
9. **Build 3-way comparison** of best SF-GP vs best MF-GP vs best GBT by NLP
10. **Build 45-row ranked table** sorted by NLP
11. **Run and validate** — all cells execute without errors

## Verification (F4)

- **SC-F4-001**: All cells execute without errors
- **SC-F4-002**: 6 one-step-ahead predictions per config (45 configs × 6 = 270 total predictions)
- **SC-F4-003**: Final 3-way comparison identifies best surrogate for F4
- **SC-F4-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration
- **SC-F4-005**: Visualisations clear, labelled
- **SC-F4-006**: Code is simple, each step explained

## Decisions (F4)

- Build new notebook from scratch following F3 pattern
- Use BoTorch `SingleTaskMultiFidelityGP` for MF-GP with synthetic fidelity dimension
- Use scikit-learn `GradientBoostingRegressor` for GBT with quantile regression for uncertainty
- Include output standardisation as a config option (important for wide output range)
- Include log-transform as a config option for SF-GP (outputs span orders of magnitude)
- N_INIT = 30 (F4 starts with 30 initial samples, unlike F1–F3 which use 10)
- Ranking by NLP (lower is better), consistent with F1–F3

---

# Implementation Plan: F5 — Prequential Evaluation (GP, GBT & MFGP Comparison)

**Date**: 2026-02-21 | **Spec**: [spec.md](spec.md) (F5 section)

## Summary

Extend the notebook `functions/f5/preq-eval-f5.ipynb` to perform prequential one-step-ahead evaluation of three surrogate families on Function 5 (chemical process yield optimisation, 4D input, 20 initial points). The notebook evaluates **15 GP configurations**, **15 GBT configurations**, and **15 MFGP configurations** (45 total), then performs a 3-way comparison to identify the best surrogate for F5.

## Technical Context (F5)

**Language/Version**: Python 3.14.2 (pyenv sdd-dev)
**Primary Dependencies**: BoTorch/GPyTorch (GP, MFGP), scikit-learn (GBT), numpy, pandas, matplotlib
**Storage**: `.npy` files in `data/f5/`
**Testing**: No unit tests required (per CONSTITUTION) — manual notebook execution validates
**Target Platform**: Jupyter Notebooks
**Project Type**: Single project — 1 Jupyter notebook (extend existing)
**Performance Goals**: Full notebook executes in <10 minutes (45 configs across 3 families)
**Scale/Scope**: 1 notebook, ~36 cells (16 markdown + 20 code)

## Constitution Check (F5)

| Constitution Principle | Status | Notes |
|----------------------|--------|-------|
| Code as simple as possible | ✅ PASS | Each step has a markdown explanation cell |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests required | ✅ PASS | N/A |
| 8 separate problems, each in own notebook | ✅ PASS | Notebook in `functions/f5/` |
| Data in `./data` folder structure | ✅ PASS | Loads from `data/f5/` |
| Use BoTorch library | ✅ PASS | GP via BoTorch SingleTaskGP, MFGP via SingleTaskMultiFidelityGP |
| Hyperparameters documented | ✅ PASS | 45 configs (15 per family) with detailed docs |
| Visualisations provided | ✅ PASS | 3-panel plots + sensitivity charts + ranked table + 3-way comparison |

**Gate Result**: ✅ ALL PASS

## Project Structure (F5)

### Source Code

```text
functions/
└── f5/
    ├── f5.ipynb               # Existing BO notebook (not modified)
    └── preq-eval-f5.ipynb     # NEW — prequential evaluation notebook

data/
└── f5/
    ├── updated_inputs - Week 6.npy    # Already exists (26×4)
    └── updated_outputs - Week 6.npy   # Already exists (26,)
```

## Notebook Cell Structure (F5)

| Cell | Type | Content |
|------|------|---------|
| 1 | md | Title & overview (F5, 4D, chemical yield, GP + GBT + MFGP, 45 configs) |
| 2 | md | Evaluation metrics (MAE, NLP, Coverage) |
| 3 | code | Imports (BoTorch, GPyTorch, sklearn, numpy, pandas, matplotlib) |
| 4 | md | Step 1: Load Data |
| 5 | code | Load F5 data, WEEK=6, N_INIT=20 |
| 6 | md | Step 2: Evaluation Metrics |
| 7 | code | `compute_metrics()` function |
| 8 | md | Step 3: GP Prequential with Starting Configuration |
| 9 | code | `gp_prequential_evaluation()` with specified starting config |
| 10 | md | Run GP Default |
| 11 | code | Execute GP with starting config |
| 12 | md | GP Default Visualisation |
| 13 | code | `plot_prequential_results()` + plot GP default |
| 14 | md | Step 4: GP HP Optimisation (15 configs) |
| 15 | code | `gp_prequential_with_config()` + 15 hp_configs + run all |
| 16 | md | Best GP Configuration |
| 17 | code | Select best GP by NLP |
| 18 | md | Step 5: GBT Prequential Evaluation (15 configs) |
| 19 | code | `gbt_prequential_with_config()` + 15 gbt_configs + run all |
| 20 | md | Best GBT Configuration |
| 21 | code | Select best GBT by NLP |
| 22 | md | Step 6: MFGP Prequential Evaluation (15 configs) |
| 23 | code | `mfgp_prequential_with_config()` + 15 mfgp_configs + run all |
| 24 | md | Best MFGP Configuration |
| 25 | code | Select best MFGP by NLP |
| 26 | md | Step 7: 3-Way Comparison (GP vs GBT vs MFGP) |
| 27 | code | Comparison table + bar chart (best-of-family) |
| 28 | md | Step 8: Sensitivity Analysis (all 45 configs) |
| 29 | code | Sensitivity horizontal bar charts — all 45 configs colour-coded |
| 30 | md | Full Results Table |
| 31 | code | Ranked table of all 45 configs sorted by NLP |
| 32 | md | Conclusions |

## Steps (F5)

1. **Update imports** to add sklearn.ensemble.GradientBoostingRegressor and SingleTaskMultiFidelityGP
2. **Expand GP configs** from 10 to 15 configurations (add 5 more combinations)
3. **Add GBT section** — `gbt_prequential_with_config()` + 15 configs + evaluation loop + best selection
4. **Add MFGP section** — `mfgp_prequential_with_config()` + 15 configs + evaluation loop + best selection
5. **Add 3-way comparison** — best GP vs best GBT vs best MFGP comparison table and bar charts
6. **Update sensitivity analysis** — horizontal bar charts with all 45 configs colour-coded by model family
7. **Update ranked table** — all 45 configs sorted by NLP with 1-based ranking
8. **Update conclusions** — 3-way comparison findings, best model for F5
9. **Run and validate** — all cells execute without errors

## Verification (F5)

- **SC-F5-001**: All cells execute without errors
- **SC-F5-002**: 6 one-step-ahead predictions per config (45 configs × 6 = 270 total predictions)
- **SC-F5-003**: Ranked results table identifies best configuration overall and per family
- **SC-F5-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration
- **SC-F5-005**: Visualisations clear, labelled
- **SC-F5-006**: Code is simple, each step explained
- **SC-F5-007**: GP hyperparameter initialisation matches the specified starting config
- **SC-F5-008**: 3-way comparison table shows best-of-family with metric winners
- **SC-F5-009**: Sensitivity bar charts colour-coded by model family (45 bars)

## Decisions (F5)

- Three surrogate families: GP, GBT, MFGP — 15 configs each = 45 total (consistent with F4)
- GP starting configuration: Matérn 5/2, ARD, z-score standardisation, specific initialisations
- GBT uncertainty via quantile regression (same pattern as F4)
- MFGP with constant fidelity column = 1.0 (same pattern as F4)
- N_INIT = 20 (F5 starts with 20 initial samples)
- Ranking by NLP (lower is better), consistent with F1–F4
- Colours: GP = blue (#2196F3), MFGP = pink (#E91E63), GBT = green (#4CAF50)

---

# F6 Plan: Neural Network Prequential Evaluation — 50 Configurations

Create `functions/f6/preq-eval-f6.ipynb` evaluating 50 Neural Network configurations for Function 6 (5D cake-recipe scoring, 26 samples, N_INIT=20, 6 eval steps, output range [−2.57, −0.22]). Uses MC Dropout for uncertainty. Hyperparameters varied: hidden layers (1–2), nodes per layer (5, 8, 16, 32, 64), learning rate (0.001–0.1). The notebook is NN-only (no GP/GBT/MFGP comparison).

## Technical Context (F6)

- **Python Environment**: Same pyenv `sdd-dev` environment as F1–F5
- **Dependencies**: numpy, torch (nn, optim), matplotlib, pandas, warnings
- **No new dependencies** — PyTorch already available from existing F6 BO notebook
- **Data**: `data/f6/updated_inputs - Week 6.npy` (26×5), `data/f6/updated_outputs - Week 6.npy` (26,)

## Constitution Check (F6)

- Simple code with clear explanations ✓ (each cell preceded by markdown)
- Jupyter notebook format ✓
- Stored in `functions/f6/` ✓
- No unit tests ✓

## Project Structure (F6)

```
functions/f6/
├── f6.ipynb               (existing BO notebook)
└── preq-eval-f6.ipynb     (NEW — prequential evaluation notebook)
```

## Notebook Cell Structure (F6)

| # | Type | Content |
|---|------|---------|
| 1 | md | Title: F6 Prequential Evaluation — NN, 50 configs, MC Dropout |
| 2 | md | Evaluation Metrics: MAE, NLP, Coverage |
| 3 | code | Imports: numpy, torch, nn, optim, matplotlib, pandas, warnings |
| 4 | md | Data Loading |
| 5 | code | Load F6 data, set N_INIT=20, print summary |
| 6 | md | Metric function |
| 7 | code | `compute_metrics()` function |
| 8 | md | NN Default: starting config (2 layers, 5 nodes, lr=0.01) |
| 9 | code | `nn_prequential_with_config()` function + default run |
| 10 | md | Default visualisation Header |
| 11 | code | Plot prequential results (3-panel: predictions, error, NLP per step) |
| 12 | md | NN HP Configurations: 50 configs, search space table |
| 13 | code | Generate 50 `nn_configs` + evaluation loop → `nn_hp_df` |
| 14 | md | Best NN Configuration |
| 15 | code | Best selection by NLP, display |
| 16 | md | Sensitivity Analysis: All 50 Configurations |
| 17 | code | Horizontal bar charts (NLP, MAE, Coverage) |
| 18 | md | Full Ranked Results |
| 19 | code | Ranked table sorted by NLP, 1-based ranking |
| 20 | md | Conclusions |

**Total**: ~20 cells

## Steps (F6)

1. **Create notebook** with title, imports, data loading, `compute_metrics()`
2. **Implement `nn_prequential_with_config()`** — builds `FlexibleNN`, trains with Adam, predicts with MC Dropout, returns metrics
3. **Run default config** — 2 layers, 5 nodes, lr=0.01 — visualise prequential results
4. **Generate 50 configs** — systematic grid: layers [1,2] × nodes [5,8,16,32,64] × lr [0.001,0.005,0.01,0.05,0.1]
5. **Evaluate all 50** — loop with try/except, store in `nn_hp_df`
6. **Select best by NLP** — display best config details
7. **Sensitivity analysis** — horizontal bar charts for all 50 configs
8. **Ranked table** — all 50 sorted by NLP, 1-based rank
9. **Conclusions** — key findings, best architecture, implications for BO
10. **Run and validate** — all cells execute without errors

## Verification (F6)

- **SC-F6-001**: All cells execute without errors
- **SC-F6-002**: 6 one-step-ahead predictions per config (50 configs × 6 = 300 total predictions)
- **SC-F6-003**: Ranked results table identifies best NN configuration
- **SC-F6-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration
- **SC-F6-005**: Visualisations clear, labelled
- **SC-F6-006**: Code is simple, each step explained
- **SC-F6-007**: Default NN architecture matches spec (2 layers, 5 nodes, lr=0.01)

## Decisions (F6)

- Single surrogate family: NN with MC Dropout — 50 configs (NN-only, no GP/GBT comparison)
- Configurable architecture: `FlexibleNN(input_dim, n_layers, n_nodes, dropout=0.2)`
- Uncertainty: MC Dropout with 50 forward passes, std floored at 1e-10
- Fixed: activation=ReLU, dropout=0.2, epochs=500, MC_samples=50
- Varied: layers (1–2), nodes (5, 8, 16, 32, 64), lr (0.001, 0.005, 0.01, 0.05, 0.1)
- N_INIT = 20 (F6 starts with 20 initial samples)
- Ranking by NLP (lower is better), consistent with F1–F5
- Colour: NN = orange (#FF9800) — distinct from GP/GBT/MFGP colours
