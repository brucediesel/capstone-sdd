# Tasks: Prequential Evaluation of Surrogate Models

**Input**: Design documents from `/specs/004-prequential-evaluation/`
**Prerequisites**: plan.md ✅, spec.md ✅

**Tests**: Not requested (per CONSTITUTION: no unit tests required).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. Since this feature is a single Jupyter notebook with bug fixes, the "setup" and "foundational" phases handle shared fixes, while each user story phase maps to the notebook sections for that story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- All paths relative to repository root; target file: `functions/f1/preq-eval-f1.ipynb`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Fix imports and parameterize the notebook so all subsequent story phases can execute

- [x] T001 Add `WEEK = 6` variable and parameterize data paths in cell 5 of `functions/f1/preq-eval-f1.ipynb` — replace hardcoded `'Week 6'` strings with f-strings using `{WEEK}`; update cell 4 markdown to mention the parameter (FR-001)
- [x] T002 Add missing imports in cell 3 of `functions/f1/preq-eval-f1.ipynb` — add `import gpytorch`, `from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel` for kernel switching in the HP optimisation phase

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Fix the `gp_prequential_with_config()` function and the BART syntax error — these bugs block US2, US4, and US5

**⚠️ CRITICAL**: No HP optimisation or comparison work can proceed until these fixes are applied

- [x] T003 Implement `kernel_type` switching in `gp_prequential_with_config()` in cell 15 of `functions/f1/preq-eval-f1.ipynb` — add conditional to construct `ScaleKernel(MaternKernel(nu=2.5))` for `'matern'` or `ScaleKernel(RBFKernel())` for `'rbf'`, passing as `covar_module` to `SingleTaskGP` (FR-008)
- [x] T004 Implement `noise_lb` constraint in `gp_prequential_with_config()` in cell 15 of `functions/f1/preq-eval-f1.ipynb` — apply `gpytorch.constraints.GreaterThan(config['noise_lb'])` to the model's noise parameter after construction (FR-008)
- [x] T005 [P] Fix log-transform inverse consistency in `gp_prequential_with_config()` in cell 15 of `functions/f1/preq-eval-f1.ipynb` — replace `np.exp(np.abs(predictions)) - 1e-300` with `np.expm1(np.abs(predictions))` to match `np.log1p` forward transform, or use consistent epsilon handling
- [x] T006 [P] Remove stray `∏` Unicode character in cell 25 of `functions/f1/preq-eval-f1.ipynb` — delete the product symbol between the `bart_configs` list closing `]` and the `print(...)` call (SyntaxError blocker)

**Checkpoint**: Foundation ready — all cells can now execute without syntax or logic errors

---

## Phase 3: User Story 1 — Run GP Prequential Evaluation (Priority: P1) 🎯 MVP

**Goal**: Train GP on initial 10 points, predict 6 remaining one-step-ahead, report MAE/NLP/Coverage, and visualise results

**Independent Test**: Run cells 1–13 of the notebook. Verify 6 prediction steps execute, metrics print, and 3-panel plot renders.

### Implementation for User Story 1

- [x] T007 [US1] Verify `gp_prequential_evaluation()` default function in cell 9 of `functions/f1/preq-eval-f1.ipynb` correctly implements the prequential loop — train on N_INIT, predict next, retrain, repeat 6 times (FR-002, FR-003, FR-005)
- [x] T008 [US1] Verify `compute_metrics()` in cell 7 of `functions/f1/preq-eval-f1.ipynb` correctly computes MAE, NLP with clipped std, and 95% coverage (FR-004)
- [x] T009 [US1] Verify cell 11 runs GP default evaluation and cell 13 `plot_prequential_results()` renders predictions vs actuals with uncertainty bands, absolute error per step, and NLP per step (FR-011)

**Checkpoint**: GP default evaluation runs and visualises correctly

---

## Phase 4: User Story 2 — Optimise GP Hyperparameters (Priority: P1)

**Goal**: Evaluate 10 GP configurations (kernel, log-transform, noise bounds) and identify best by NLP

**Independent Test**: Run cells 14–17. Verify 10-row results DataFrame with MAE/NLP/Coverage, best config identified by lowest NLP.

### Implementation for User Story 2

- [x] T010 [US2] Verify 10 GP `hp_configs` list in cell 15 covers all required combinations in `functions/f1/preq-eval-f1.ipynb` — 2 kernels × {raw, log} × 3 noise bounds minus duplicates = 10 configs (FR-007, FR-008)
- [x] T011 [US2] Verify GP HP optimisation loop in cell 15 of `functions/f1/preq-eval-f1.ipynb` calls `gp_prequential_with_config()` for each config with try/except for NaN on failure (Edge Case: GP fitting failure)
- [x] T012 [US2] Verify best GP selection in cell 17 of `functions/f1/preq-eval-f1.ipynb` picks configuration with lowest NLP and displays results DataFrame (FR-010)

**Checkpoint**: GP HP optimisation produces 10-row ranked results table

---

## Phase 5: User Story 3 — Run BART Prequential Evaluation (Priority: P1)

**Goal**: Train BART on initial 10 points, predict 6 one-step-ahead, report metrics, and visualise

**Independent Test**: Run cells 18–23. Verify BART produces 6 predictions with same metric format as GP.

### Implementation for User Story 3

- [x] T013 [US3] Verify `bart_prequential_evaluation()` in cell 19 of `functions/f1/preq-eval-f1.ipynb` correctly uses PyMC-BART with `pm.Data`, `pmb.BART`, and posterior predictive sampling for uncertainty (FR-006)
- [x] T014 [US3] Verify cell 21 runs BART default and cell 23 reuses `plot_prequential_results()` for the same 3-panel visualisation format (FR-011)

**Checkpoint**: BART default evaluation runs and visualises correctly

---

## Phase 6: User Story 4 — Optimise BART Hyperparameters (Priority: P1)

**Goal**: Evaluate 10 BART configurations (trees, draws, tune) and identify best by NLP

**Independent Test**: Run cells 24–27. Verify 10-row BART results DataFrame with best config identified.

### Implementation for User Story 4

- [x] T015 [US4] Verify 10 BART `bart_configs` list in cell 25 of `functions/f1/preq-eval-f1.ipynb` covers trees (10, 20, 50, 100, 200) × draws (200, 500) with tune variations = 10 configs (FR-007, FR-009)
- [x] T016 [US4] Verify BART HP optimisation loop in cell 25 of `functions/f1/preq-eval-f1.ipynb` calls `bart_prequential_evaluation()` for each config with try/except for NaN on failure (Edge Case: BART divergence)
- [x] T017 [US4] Verify best BART selection in cell 27 of `functions/f1/preq-eval-f1.ipynb` picks configuration with lowest NLP and displays results DataFrame

**Checkpoint**: BART HP optimisation produces 10-row ranked results table

---

## Phase 7: User Story 5 — Compare GP vs BART (Priority: P1)

**Goal**: Side-by-side comparison of best GP vs best BART with bar charts, ranked table of all 20 configs

**Independent Test**: Run cells 28–36. Verify comparison table, bar charts, and full 20-row ranked table display.

### Implementation for User Story 5

- [x] T018 [US5] Verify comparison table in cell 29 of `functions/f1/preq-eval-f1.ipynb` shows best GP and best BART side-by-side with MAE, NLP, Coverage_95, and metric-by-metric winner (FR-010)
- [x] T019 [US5] Verify bar chart visualisation in cell 31 of `functions/f1/preq-eval-f1.ipynb` produces 3 bar charts (MAE, NLP, Coverage) with value annotations and 95% ideal coverage line (FR-011)
- [x] T020 [US5] Verify full ranked results table in cell 35 of `functions/f1/preq-eval-f1.ipynb` combines all 20 configurations and sorts by NLP ascending (FR-012)
- [x] T021 [US5] Verify conclusions in cell 36 markdown of `functions/f1/preq-eval-f1.ipynb` summarise key findings, best model, and next steps (FR-013)

**Checkpoint**: All 5 user stories complete — notebook delivers full GP vs BART comparison

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and cleanup

- [x] T022 Run all 36 cells of `functions/f1/preq-eval-f1.ipynb` top-to-bottom and verify SC-001 (no errors), SC-002 (20 configs × 6 predictions each), SC-004 (all metrics present)
- [x] T023 Verify SC-005: all visualisations are clear, labelled, and suitable for capstone report in `functions/f1/preq-eval-f1.ipynb`
- [x] T024 Verify SC-006: each code step has a preceding markdown explanation cell in `functions/f1/preq-eval-f1.ipynb`
- [x] T025 Spot-check: confirm RBF configs produce different NLP values than Matérn configs (kernel switching works) in `functions/f1/preq-eval-f1.ipynb`
- [x] T026 Spot-check: confirm different `noise_lb` values produce different results for same kernel/transform combo in `functions/f1/preq-eval-f1.ipynb`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001, T002) — BLOCKS all user stories with HP configs
- **US1 (Phase 3)**: Depends on Setup only (T001, T002) — default GP doesn't need the HP fixes
- **US2 (Phase 4)**: Depends on Foundational (T003, T004, T005) — needs kernel/noise fixes
- **US3 (Phase 5)**: Depends on Setup (T001, T002) and T006 fix — default BART doesn't need GP fixes
- **US4 (Phase 6)**: Depends on T006 — needs the `∏` syntax error fixed
- **US5 (Phase 7)**: Depends on US2 and US4 completion (needs both result sets)
- **Polish (Phase 8)**: Depends on all user stories

### User Story Dependencies

- **US1 (GP Default)**: Can start after Setup — no other story dependencies
- **US2 (GP HP Opt)**: Can start after Foundational — no dependency on US1 results (uses own function)
- **US3 (BART Default)**: Can start after Setup + T006 — independent of GP stories
- **US4 (BART HP Opt)**: Can start after T006 — independent of GP stories
- **US5 (Comparison)**: MUST wait for US2 and US4 to complete (needs both DataFrames)

### Parallel Opportunities

- T005 and T006 are marked [P] — they edit different cells and can be done simultaneously
- US1 (GP) and US3 (BART) can proceed in parallel after Setup
- US2 (GP HP) and US4 (BART HP) can proceed in parallel after Foundational fixes

---

## Parallel Example: Foundational Phase

```bash
# These two fixes affect different cells and can be done simultaneously:
Task T005: "Fix log-transform inverse in cell 15"
Task T006: "Remove stray ∏ character in cell 25"
```

## Parallel Example: User Stories 1 & 3

```bash
# GP default and BART default are independent — can be verified in parallel:
Task T007-T009: "Verify GP default evaluation (cells 7-13)"
Task T013-T014: "Verify BART default evaluation (cells 19-23)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T002)
2. Complete Phase 3: User Story 1 — GP Default (T007–T009)
3. **STOP and VALIDATE**: Run cells 1–13, confirm GP produces 6 predictions with plots
4. This gives immediate, visible output even before HP optimisation

### Incremental Delivery

1. Setup + Foundational → All bugs fixed (T001–T006)
2. US1 → GP default works → First visible results
3. US2 → GP HP optimisation works → Best GP identified
4. US3 → BART default works → Second surrogate baseline
5. US4 → BART HP optimisation works → Best BART identified
6. US5 → Comparison → Research question answered
7. Polish → Validated end-to-end → Ready for capstone submission

---

## Notes

- All tasks target a single file: `functions/f1/preq-eval-f1.ipynb`
- No unit tests (per CONSTITUTION)
- "Verify" tasks mean: review the existing auto-generated code, fix any issues found, and confirm correct execution
- The notebook already has 36 cells with mostly correct structure — this is a fix-and-validate workflow, not a build-from-scratch workflow
- Commit after each phase for clean git history


---

# F2 Tasks: Prequential Evaluation — GP vs BART vs Random Forest

**Input**: spec.md (F2 section), plan.md (F2 section)
**Target file**: `functions/f2/preq-eval-f2.ipynb` (NEW — create from scratch)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- All paths relative to repository root

---

## Phase F2-1: Setup (Create Notebook Foundation)

**Purpose**: Create the notebook file with imports, data loading, and shared utility functions

- [x] T-F2-001 Create `functions/f2/preq-eval-f2.ipynb` with title markdown cell (F2 overview: 2D input, maximise, noisy log-likelihood, 3 surrogates, 30 configs) and evaluation metrics markdown cell
- [x] T-F2-002 Add imports cell: numpy, torch, matplotlib, pandas, warnings, botorch/gpytorch (SingleTaskGP, kernels, likelihood, constraints), pymc, pymc_bart, sklearn (RandomForestRegressor). Set seeds.
- [x] T-F2-003 Add data loading cell: `WEEK = 6`, load from `../../data/f2/updated_inputs - Week {WEEK}.npy` and outputs, set `N_INIT = 10`, print data summary (FR-F2-001, FR-F2-002)
- [x] T-F2-004 Add `compute_metrics()` function cell — same as F1: MAE, NLP (clipped std), 95% Coverage (FR-F2-004)
- [x] T-F2-005 Add `plot_prequential_results()` function cell — same as F1: 3-panel (predictions vs actuals, absolute error, NLP per step) (FR-F2-013)

**Checkpoint**: Notebook has 10 cells (5 md + 5 code), imports work, data loads correctly

---

## Phase F2-2: GP Evaluation (User Stories F2-1, F2-2)

**Purpose**: Implement GP default and HP optimisation sections

- [x] T-F2-006 Add `gp_prequential_evaluation()` function (default GP with Matern 5/2) — same as F1 (FR-F2-005, FR-F2-003)
- [x] T-F2-007 Add GP default execution cell and visualisation cell with `plot_prequential_results()` (FR-F2-013)
- [x] T-F2-008 Add `gp_prequential_with_config()` function with kernel switching, log-transform option, noise constraint — same as F1 (FR-F2-005, FR-F2-009)
- [x] T-F2-009 Add 10 GP `hp_configs` list and HP optimisation loop cell (FR-F2-008, FR-F2-009)
- [x] T-F2-010 Add best GP selection cell — select by lowest NLP, display DataFrame

**Checkpoint**: GP section complete — 10 configs evaluated with metrics and plots

---

## Phase F2-3: BART Evaluation (User Stories F2-3, F2-4)

**Purpose**: Implement BART default and HP optimisation sections

- [x] T-F2-011 Add `bart_prequential_evaluation()` function — same as F1 (FR-F2-006, FR-F2-003)
- [x] T-F2-012 Add BART default execution cell and visualisation cell (FR-F2-013)
- [x] T-F2-013 Add 10 `bart_configs` list and HP optimisation loop cell (FR-F2-008, FR-F2-010)
- [x] T-F2-014 Add best BART selection cell — select by lowest NLP, display DataFrame

**Checkpoint**: BART section complete — 10 configs evaluated with metrics and plots

---

## Phase F2-4: Random Forest Evaluation (User Stories F2-5, F2-6) — NEW

**Purpose**: Implement RF default and HP optimisation sections (new for F2)

- [x] T-F2-015 Add `rf_prequential_evaluation()` function — train RF on training set, predict test point, uncertainty via individual tree predictions (tree_preds.mean, tree_preds.std). Print step-by-step results. (FR-F2-007, FR-F2-003)
- [x] T-F2-016 Add RF default execution cell (n_estimators=100, max_depth=None, min_samples_leaf=1) and visualisation cell (FR-F2-013)
- [x] T-F2-017 Add `rf_prequential_with_config()` function — accepts config dict with n_estimators, max_depth, min_samples_leaf, bootstrap. Returns metrics dict. (FR-F2-007, FR-F2-011)
- [x] T-F2-018 Add 10 `rf_configs` list and HP optimisation loop cell (FR-F2-008, FR-F2-011)
- [x] T-F2-019 Add best RF selection cell — select by lowest NLP, display DataFrame

**Checkpoint**: RF section complete — 10 configs evaluated with metrics and plots

---

## Phase F2-5: 3-Way Comparison (User Story F2-7)

**Purpose**: Compare best GP vs best BART vs best RF and rank all 30 configurations

- [x] T-F2-020 Add 3-way comparison table cell — best GP, best BART, best RF side-by-side with metric winners (FR-F2-012)
- [x] T-F2-021 Add 3-panel bar chart cell — MAE, NLP, Coverage with 3 bars per metric (FR-F2-013)
- [x] T-F2-022 Add sensitivity horizontal bar chart cell — all 30 configs grouped by model type (FR-F2-013)
- [x] T-F2-023 Add full ranked table cell — combine all 30 configs, sort by NLP, 1-based rank (FR-F2-014)
- [x] T-F2-024 Add conclusions markdown cell — key findings, best model for F2, implications

**Checkpoint**: Full notebook complete with 3-way comparison and 30-row ranked table

---

## Phase F2-6: Polish & Validation

**Purpose**: End-to-end validation

- [x] T-F2-025 Run all cells of `functions/f2/preq-eval-f2.ipynb` top-to-bottom and verify no errors (SC-F2-001)
- [x] T-F2-026 Verify 30 configs × 6 predictions each = 180 total predictions (SC-F2-002)
- [x] T-F2-027 Verify all visualisations are clear and labelled (SC-F2-005)
- [x] T-F2-028 Verify each code step has a preceding markdown explanation (SC-F2-006)

---

## Dependencies & Execution Order (F2)

- **Phase F2-1** (Setup): No dependencies — start here
- **Phase F2-2** (GP): Depends on Phase F2-1
- **Phase F2-3** (BART): Depends on Phase F2-1; can run in parallel with Phase F2-2
- **Phase F2-4** (RF): Depends on Phase F2-1; can run in parallel with Phases F2-2 and F2-3
- **Phase F2-5** (Comparison): Depends on Phases F2-2, F2-3, and F2-4
- **Phase F2-6** (Polish): Depends on all phases

---

# F3 Tasks: Prequential Evaluation — GP vs BART vs Random Forest

**Input**: spec.md (F3 section), plan.md
**Target file**: `functions/f3/preq-eval-f3.ipynb` (NEW — create from scratch)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- All paths relative to repository root

---

## Phase F3-1: Setup (Create Notebook Foundation)

**Purpose**: Create the notebook file with imports, data loading, and shared utility functions

- [x] T-F3-001 Create `functions/f3/preq-eval-f3.ipynb` with title markdown cell (F3 overview: 3D drug discovery, maximise transformed output, 3 surrogates, 45 configs) and evaluation metrics markdown cell
- [x] T-F3-002 Add imports cell: numpy, torch, matplotlib, pandas, warnings, botorch/gpytorch (SingleTaskGP, kernels, likelihood, constraints), pymc, pymc_bart, sklearn (RandomForestRegressor). Set seeds.
- [x] T-F3-003 Add data loading cell: `WEEK = 6`, load from `../../data/f3/updated_inputs - Week {WEEK}.npy` and outputs, set `N_INIT = 10`, print data summary (FR-F3-001, FR-F3-002)
- [x] T-F3-004 Add `compute_metrics()` function cell — same as F1/F2: MAE, NLP (clipped std), 95% Coverage (FR-F3-004)
- [x] T-F3-005 Add `plot_prequential_results()` function cell — same as F1/F2: 3-panel (predictions vs actuals, absolute error, NLP per step), labelled F3 (FR-F3-013)

**Checkpoint**: Notebook has foundation cells, imports work, data loads correctly

---

## Phase F3-2: GP Evaluation (User Stories F3-1, F3-2)

**Purpose**: Implement GP default and HP optimisation (15 configs including Matérn 3/2)

- [x] T-F3-006 Add `gp_prequential_evaluation()` function (default GP with Matern 5/2, ARD for 3 dims) (FR-F3-005, FR-F3-003)
- [x] T-F3-007 Add GP default execution cell and visualisation cell with `plot_prequential_results()` (FR-F3-013)
- [x] T-F3-008 Add `gp_prequential_with_config()` function with kernel switching (Matern 5/2, Matern 3/2, RBF), log-transform option, noise constraint (FR-F3-005, FR-F3-009)
- [x] T-F3-009 Add 15 GP `hp_configs` list — 3 kernels × {raw, log} × 3 noise bounds = 18, select 15 best combinations (FR-F3-008, FR-F3-009)
- [x] T-F3-010 Add GP HP optimisation loop cell and best GP selection cell (FR-F3-008)

**Checkpoint**: GP section complete — 15 configs evaluated with metrics and plots

---

## Phase F3-3: BART Evaluation (User Stories F3-3, F3-4)

**Purpose**: Implement BART default and HP optimisation (15 configs including higher draws)

- [x] T-F3-011 Add `bart_prequential_evaluation()` function — PyMC-BART with `pm.Data`, posterior predictive sampling (FR-F3-006, FR-F3-003)
- [x] T-F3-012 Add BART default execution cell and visualisation cell (FR-F3-013)
- [x] T-F3-013 Add 15 `bart_configs` list — m (10,20,50,100,200) × draws (200,500,1000) × tune (100,200), select 15 (FR-F3-008, FR-F3-010)
- [x] T-F3-014 Add BART HP optimisation loop cell and best BART selection cell

**Checkpoint**: BART section complete — 15 configs evaluated with metrics and plots

---

## Phase F3-4: Random Forest Evaluation (User Stories F3-5, F3-6)

**Purpose**: Implement RF default and HP optimisation (15 configs with shallow trees for small-data 3D)

- [x] T-F3-015 Add `rf_prequential_evaluation()` function — uncertainty via individual tree predictions (FR-F3-007, FR-F3-003)
- [x] T-F3-016 Add RF default execution cell and visualisation cell (FR-F3-013)
- [x] T-F3-017 Add `rf_prequential_with_config()` function accepting config dict (FR-F3-007, FR-F3-011)
- [x] T-F3-018 Add 15 `rf_configs` list — n_estimators (50,100,200,500) × max_depth (None,3,5,10) × min_samples_leaf (1,2,3,5) × bootstrap, select 15 (FR-F3-008, FR-F3-011)
- [x] T-F3-019 Add RF HP optimisation loop cell and best RF selection cell

**Checkpoint**: RF section complete — 15 configs evaluated with metrics and plots

---

## Phase F3-5: 3-Way Comparison (User Story F3-7)

**Purpose**: Compare best GP vs best BART vs best RF and rank all 45 configurations

- [x] T-F3-020 Add 3-way comparison table cell — best GP, best BART, best RF side-by-side with metric winners (FR-F3-012)
- [x] T-F3-021 Add 3-panel bar chart cell — MAE, NLP, Coverage with 3 bars per metric (FR-F3-013)
- [x] T-F3-022 Add sensitivity horizontal bar chart cell — all 45 configs grouped by model type (FR-F3-013)
- [x] T-F3-023 Add full ranked table cell — combine all 45 configs, sort by NLP, 1-based rank (FR-F3-014)
- [x] T-F3-024 Add conclusions markdown cell — key findings, best model for F3, implications for 3D drug discovery (FR-F3-015)

**Checkpoint**: Full notebook complete with 3-way comparison and 45-row ranked table

---

## Phase F3-6: Polish & Validation

**Purpose**: End-to-end validation

- [x] T-F3-025 Run all cells of `functions/f3/preq-eval-f3.ipynb` top-to-bottom and verify no errors (SC-F3-001)
- [x] T-F3-026 Verify 45 configs × 6 predictions each = 270 total predictions (SC-F3-002)
- [x] T-F3-027 Verify all visualisations are clear and labelled (SC-F3-005)
- [x] T-F3-028 Verify each code step has a preceding markdown explanation (SC-F3-006)

---

## Dependencies & Execution Order (F3)

- **Phase F3-1** (Setup): No dependencies — start here
- **Phase F3-2** (GP): Depends on Phase F3-1
- **Phase F3-3** (BART): Depends on Phase F3-1; can run in parallel with Phase F3-2
- **Phase F3-4** (RF): Depends on Phase F3-1; can run in parallel with Phases F3-2 and F3-3
- **Phase F3-5** (Comparison): Depends on Phases F3-2, F3-3, and F3-4
- **Phase F3-6** (Polish): Depends on all phases

---

# F4 Tasks: Prequential Evaluation — Single Fidelity GP vs Multi Fidelity GP

**Input**: spec.md (F4 section), plan.md (F4 section)
**Target file**: `functions/f4/preq-eval-f4.ipynb` (NEW — create from scratch)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- All paths relative to repository root

---

## Phase F4-1: Setup (Create Notebook Foundation)

**Purpose**: Create the notebook file with imports, data loading, and shared utility functions

- [x] T-F4-001 Create `functions/f4/preq-eval-f4.ipynb` with title markdown cell (F4 overview: 4D warehouse product placement, maximise, 2 GP families, 30 configs) and evaluation metrics markdown cell
- [x] T-F4-002 Add imports cell: numpy, torch, matplotlib, pandas, warnings, botorch/gpytorch (SingleTaskGP, SingleTaskMultiFidelityGP, kernels, likelihood, constraints, transforms). Set seeds.
- [x] T-F4-003 Add data loading cell: `WEEK = 6`, load from `../../data/f4/updated_inputs - Week {WEEK}.npy` and outputs, set `N_INIT = 30`, print data summary (FR-F4-001, FR-F4-002)
- [x] T-F4-004 Add `compute_metrics()` function cell — same as F1–F3: MAE, NLP (clipped std), 95% Coverage (FR-F4-004)
- [x] T-F4-005 Add `plot_prequential_results()` function cell — same as F1–F3: 3-panel (predictions vs actuals, absolute error, NLP per step), labelled F4 (FR-F4-011)

**Checkpoint**: Notebook has foundation cells, imports work, data loads correctly

---

## Phase F4-2: Single Fidelity GP Evaluation (User Stories F4-1, F4-2)

**Purpose**: Implement SF-GP default and HP optimisation (15 configs)

- [x] T-F4-006 Add `sfgp_prequential_evaluation()` function (default SF-GP with Matern 5/2, ARD for 4 dims) (FR-F4-005, FR-F4-003)
- [x] T-F4-007 Add SF-GP default execution cell and visualisation cell with `plot_prequential_results()` (FR-F4-011)
- [x] T-F4-008 Add `sfgp_prequential_with_config()` function with kernel switching (Matern 5/2, Matern 3/2, RBF), output transform options (raw, standardise, log-transform), noise constraint (FR-F4-005, FR-F4-008)
- [x] T-F4-009 Add 15 SF-GP `sfgp_configs` list — kernels × transforms × noise bounds = 15 selected combinations (FR-F4-007, FR-F4-008)
- [x] T-F4-010 Add SF-GP HP optimisation loop cell and best SF-GP selection cell (FR-F4-007)

**Checkpoint**: SF-GP section complete — 15 configs evaluated with metrics and plots

---

## Phase F4-3: Multi Fidelity GP Evaluation (User Stories F4-3, F4-4)

**Purpose**: Implement MF-GP with autoregressive/co-kriging kernel and HP optimisation (15 configs)

- [x] T-F4-011 Add `mfgp_prequential_evaluation()` function — BoTorch `SingleTaskMultiFidelityGP` with synthetic fidelity column, default Matern 5/2 (FR-F4-006, FR-F4-003)
- [x] T-F4-012 Add MF-GP default execution cell and visualisation cell (FR-F4-011)
- [x] T-F4-013 Add `mfgp_prequential_with_config()` function with kernel switching, output transform, noise constraint, fidelity options (FR-F4-006, FR-F4-009)
- [x] T-F4-014 Add 15 MF-GP `mfgp_configs` list — kernels × fidelity × transforms × noise bounds = 15 selected combinations (FR-F4-007, FR-F4-009)
- [x] T-F4-015 Add MF-GP HP optimisation loop cell and best MF-GP selection cell (FR-F4-007)

**Checkpoint**: MF-GP section complete — 15 configs evaluated with metrics and plots

---

## Phase F4-3b: Gradient Boosted Trees Evaluation (User Stories F4-5, F4-6) — NEW

**Purpose**: Implement GBT default and HP optimisation (15 configs) using quantile regression for uncertainty

- [x] T-F4-025 Add `sklearn.ensemble.GradientBoostingRegressor` import to imports cell of `functions/f4/preq-eval-f4.ipynb` (FR-F4-007b)
- [x] T-F4-026 Add GBT section markdown cell explaining GBT as a third surrogate, quantile regression uncertainty, and hyperparameters
- [x] T-F4-027 Add `gbt_prequential_evaluation()` function — train 3 GBT models (mean via `ls`, lower/upper quantiles via `quantile` loss). Predict mean, derive std from quantile spread. Print step-by-step results. (FR-F4-007b, FR-F4-003)
- [x] T-F4-028 Add GBT default execution cell (n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=2, subsample=0.8) and visualisation cell (FR-F4-011)
- [x] T-F4-029 Add `gbt_prequential_with_config()` function — accepts config dict with n_estimators, learning_rate, max_depth, min_samples_leaf, subsample. Returns metrics dict. (FR-F4-007b, FR-F4-009b)
- [x] T-F4-030 Add 15 `gbt_configs` list — n_estimators × learning_rate × max_depth × subsample, select 15 combinations (FR-F4-007, FR-F4-009b)
- [x] T-F4-031 Add GBT HP optimisation loop cell and best GBT selection cell (FR-F4-007)

**Checkpoint**: GBT section complete — 15 configs evaluated with metrics and plots

---

## Phase F4-4: 3-Way Comparison (User Story F4-7)

**Purpose**: Compare best SF-GP vs best MF-GP vs best GBT and rank all 45 configurations

- [x] T-F4-016 Update comparison table cell — best SF-GP, best MF-GP, best GBT side-by-side with metric winners (FR-F4-010) — expand from 2-way to 3-way
- [x] T-F4-017 Update bar chart cell — MAE, NLP, Coverage with 3 bars per metric (FR-F4-011)
- [x] T-F4-018 Update sensitivity horizontal bar chart cell — all 45 configs grouped by model type (FR-F4-011)
- [x] T-F4-019 Update full ranked table cell — combine all 45 configs, sort by NLP, 1-based rank (FR-F4-012)
- [x] T-F4-020 Update conclusions markdown cell — key findings including GBT, best model for F4 across all three families (FR-F4-013)

**Checkpoint**: Full notebook complete with 3-way comparison and 45-row ranked table

---

## Phase F4-5: Polish & Validation

**Purpose**: End-to-end validation

- [x] T-F4-021 Run all cells of `functions/f4/preq-eval-f4.ipynb` top-to-bottom and verify no errors (SC-F4-001)
- [x] T-F4-022 Verify 45 configs × 6 predictions each = 270 total predictions (SC-F4-002)
- [x] T-F4-023 Verify all visualisations are clear and labelled (SC-F4-005)
- [x] T-F4-024 Verify each code step has a preceding markdown explanation (SC-F4-006)

---

## Dependencies & Execution Order (F4)

- **Phase F4-1** (Setup): No dependencies — start here
- **Phase F4-2** (SF-GP): Depends on Phase F4-1
- **Phase F4-3** (MF-GP): Depends on Phase F4-1; can run in parallel with Phase F4-2
- **Phase F4-3b** (GBT): Depends on Phase F4-1; can run in parallel with Phases F4-2 and F4-3
- **Phase F4-4** (Comparison): Depends on Phases F4-2, F4-3, and F4-3b
- **Phase F4-5** (Polish): Depends on all phases

---

# F5 Tasks: Prequential Evaluation — GP, GBT & MFGP Comparison

**Input**: spec.md (F5 section), plan.md (F5 section)
**Target file**: `functions/f5/preq-eval-f5.ipynb` (EXTEND existing notebook)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- All paths relative to repository root

---

## Phase F5-1: Setup (Create Notebook Foundation)

**Purpose**: Create the notebook file with imports, data loading, and shared utility functions

- [x] T-F5-001 Create `functions/f5/preq-eval-f5.ipynb` with title markdown cell (F5 overview: 4D chemical yield, unimodal, GP + GBT + MFGP, 45 configs) and evaluation metrics markdown cell
- [x] T-F5-002 Add imports cell: numpy, torch, matplotlib, pandas, warnings, botorch/gpytorch (SingleTaskGP, kernels, likelihood, constraints, transforms). Set seeds.
- [x] T-F5-003 Add data loading cell: `WEEK = 6`, load from `../../data/f5/updated_inputs - Week {WEEK}.npy` and outputs, set `N_INIT = 20`, print data summary (FR-F5-001, FR-F5-002)
- [x] T-F5-004 Add `compute_metrics()` function cell — same as F1–F4: MAE, NLP (clipped std), 95% Coverage (FR-F5-004)
- [x] T-F5-005 Add `plot_prequential_results()` function cell — same as F1–F4: 3-panel (predictions vs actuals, absolute error, NLP per step), labelled F5 (FR-F5-011)

**Checkpoint**: Notebook has foundation cells, imports work, data loads correctly

---

## Phase F5-2: GP Default Evaluation (User Story F5-1) 🎯 MVP

**Purpose**: Implement GP prequential evaluation with specified starting configuration

- [x] T-F5-006 Add `gp_prequential_evaluation()` function with starting config — Matérn 5/2, ARD for 4 dims, z-score standardisation, lengthscales init 0.25, signal var = Var(y), noise = 0.03·Var(y), jitter 1e-6, multi-restart MLL fitting (FR-F5-005, FR-F5-006, FR-F5-007)
- [x] T-F5-007 Add GP default execution cell and visualisation cell with `plot_prequential_results()` (FR-F5-011)

**Checkpoint**: GP default with starting config runs and visualises correctly

---

## Phase F5-3: GP Hyperparameter Optimisation (User Story F5-2)

**Purpose**: Evaluate 15 GP configurations and identify best by NLP

- [x] T-F5-008 Add `gp_prequential_with_config()` function with configurable kernel, output transform, noise init, lengthscale init, and multi-restart MLL fitting (FR-F5-005, FR-F5-008, FR-F5-009)
- [x] T-F5-009 Expand GP `hp_configs` from 10 to 15 configurations: add 5 more combos of kernels (Matérn 5/2, Matérn 3/2, RBF) × output transforms (z-score, log, raw) × noise inits × lengthscale inits (FR-F5-008, FR-F5-009)
- [x] T-F5-010 GP HP optimisation loop cell with try/except for NaN on failure (FR-F5-008)
- [x] T-F5-011 Best GP selection cell — select by lowest NLP, display DataFrame (FR-F5-010)

**Checkpoint**: GP HP optimisation produces 15-row ranked results table

---

## Phase F5-3b: GBT Prequential Evaluation (User Story F5-4)

**Purpose**: Evaluate 15 GBT configurations and identify best by NLP

- [x] T-F5-019 Add GBT markdown explanation cell — describe GBT approach, quantile regression for uncertainty, HP axes (n_estimators, learning_rate, max_depth, min_samples_leaf, subsample)
- [x] T-F5-020 Add `gbt_prequential_with_config()` function — fits mean + lower/upper quantile GBT models per step, estimates std from quantile spread (FR-F5-014)
- [x] T-F5-021 Add 15 `gbt_configs` list — vary n_estimators (50–500), learning_rate (0.01–0.2), max_depth (3–6), min_samples_leaf (1–5), subsample (0.8–1.0) (FR-F5-014)
- [x] T-F5-022 Add GBT HP optimisation loop cell with try/except — store results in `gbt_hp_df` (FR-F5-014)
- [x] T-F5-023 Add best GBT selection cell — select by lowest NLP, display DataFrame

**Checkpoint**: GBT section complete — 15 configs evaluated with metrics

---

## Phase F5-3c: MFGP Prequential Evaluation (User Story F5-5)

**Purpose**: Evaluate 15 MFGP configurations and identify best by NLP

- [x] T-F5-024 Add MFGP markdown explanation cell — describe Multi-Fidelity GP, fidelity column, LinearTruncated vs ExponentialDecay kernel, HP axes (FR-F5-015)
- [x] T-F5-025 Add `mfgp_prequential_with_config()` function — appends fidelity=1.0 column, builds SingleTaskMultiFidelityGP, varies nu/linear_truncated/output_transform/noise_lb (FR-F5-015)
- [x] T-F5-026 Add 15 `mfgp_configs` list — vary nu (2.5, 1.5), linear_truncated (True/False), output_transform (raw, standardise), noise_lb (1e-4, 1e-5, 1e-6) (FR-F5-015)
- [x] T-F5-027 Add MFGP HP optimisation loop cell with try/except — store results in `mfgp_hp_df` (FR-F5-015)
- [x] T-F5-028 Add best MFGP selection cell — select by lowest NLP, display DataFrame

**Checkpoint**: MFGP section complete — 15 configs evaluated with metrics

---

## Phase F5-4: 3-Way Comparison & Sensitivity (User Stories F5-3, F5-4, F5-5)

**Purpose**: Compare best GP vs best GBT vs best MFGP and rank all 45 configurations

- [x] T-F5-029 Add 3-way comparison markdown cell
- [x] T-F5-030 Add comparison code cell — best GP vs best GBT vs best MFGP side-by-side, identify metric winners for MAE, NLP, Coverage (FR-F5-016)
- [x] T-F5-031 Update sensitivity horizontal bar chart cell — all 45 configs, colour-coded by family (GP=blue, MFGP=pink, GBT=green) (FR-F5-017)
- [x] T-F5-032 Update full ranked table cell — combine all 45 configs, sort by NLP, 1-based rank (FR-F5-010)
- [x] T-F5-033 Update conclusions markdown cell — key findings including 3-way comparison, best model for F5 across all families

**Checkpoint**: Full notebook complete with 3-way comparison and 45-row ranked table

---

## Phase F5-5: Polish & Validation

**Purpose**: End-to-end validation

- [x] T-F5-034 Run all cells of `functions/f5/preq-eval-f5.ipynb` top-to-bottom and verify no errors (SC-F5-001)
- [x] T-F5-035 Verify 45 configs × 6 predictions each = 270 total predictions (SC-F5-002)
- [x] T-F5-036 Verify all visualisations are clear and labelled (SC-F5-005)
- [x] T-F5-037 Verify each code step has a preceding markdown explanation (SC-F5-006)

---

## Dependencies & Execution Order (F5)

- **Phase F5-1** (Setup): No dependencies — start here
- **Phase F5-2** (GP Default): Depends on Phase F5-1
- **Phase F5-3** (GP HP Opt): Depends on Phase F5-2
- **Phase F5-3b** (GBT): Depends on Phase F5-1; can run in parallel with Phases F5-2 and F5-3
- **Phase F5-3c** (MFGP): Depends on Phase F5-1; can run in parallel with Phases F5-2, F5-3, and F5-3b
- **Phase F5-4** (Comparison): Depends on Phases F5-3, F5-3b, and F5-3c
- **Phase F5-5** (Polish): Depends on all phases

---

# F6 Tasks: Prequential Evaluation — Neural Network (50 Configurations)

**Input**: spec.md (F6 section), plan.md (F6 section)
**Target file**: `functions/f6/preq-eval-f6.ipynb` (NEW notebook)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- All paths relative to repository root

---

## Phase F6-1: Setup (Create Notebook Foundation)

**Purpose**: Create the notebook file with imports, data loading, and shared utility functions

- [x] T-F6-001 Create `functions/f6/preq-eval-f6.ipynb` with title markdown cell (F6 overview: 5D cake recipe, NN with MC Dropout, 50 configs) and evaluation metrics markdown cell
- [x] T-F6-002 Add imports cell: numpy, torch, torch.nn, torch.optim, matplotlib, pandas, warnings. Set seeds.
- [x] T-F6-003 Add data loading cell: `WEEK = 6`, load from `../../data/f6/updated_inputs - Week {WEEK}.npy` and outputs, set `N_INIT = 20`, print data summary (FR-F6-001, FR-F6-002)
- [x] T-F6-004 Add `compute_metrics()` function cell — same as F1–F5: MAE, NLP (clipped std), 95% Coverage (FR-F6-003)

**Checkpoint**: Notebook has foundation cells, imports work, data loads correctly

---

## Phase F6-2: NN Default Evaluation (User Story F6-1) 🎯 MVP

**Purpose**: Implement NN prequential evaluation with starting config and visualise results

- [x] T-F6-005 Add NN default markdown cell — describe starting config (2 layers, 5 nodes, lr=0.01), MC Dropout uncertainty approach
- [x] T-F6-006 Add `nn_prequential_with_config()` function — builds FlexibleNN(input_dim, n_layers, n_nodes, dropout=0.2), trains with Adam for 500 epochs, predicts with 50 MC forward passes (FR-F6-004, FR-F6-005, FR-F6-006, FR-F6-007)
- [x] T-F6-007 Run default NN config and print step-by-step results
- [x] T-F6-008 Add `plot_prequential_results()` function and visualise default config (3-panel: predictions vs actuals, absolute error, NLP per step) (FR-F6-013)

**Checkpoint**: Default NN with starting config runs and visualises correctly

---

## Phase F6-3: NN Hyperparameter Optimisation (User Story F6-2)

**Purpose**: Evaluate 50 NN configurations and identify best by NLP

- [x] T-F6-009 Add HP search markdown cell — describe the 50-config grid: layers [1,2] × nodes [5,8,16,32,64] × lr [0.001,0.005,0.01,0.05,0.1]
- [x] T-F6-010 Add 50 `nn_configs` list — generate systematically with descriptive labels (FR-F6-008)
- [x] T-F6-011 Add NN HP optimisation loop cell with try/except for NaN on failure — store results in `nn_hp_df` (FR-F6-009)
- [x] T-F6-012 Add best NN selection cell — select by lowest NLP, display DataFrame (FR-F6-010)

**Checkpoint**: NN HP optimisation produces 50-row ranked results table

---

## Phase F6-4: Sensitivity & Ranking (User Story F6-3)

**Purpose**: Visualise and rank all 50 configurations

- [x] T-F6-013 Add sensitivity horizontal bar chart cell — all 50 configs, NLP + MAE + Coverage subplots (FR-F6-011)
- [x] T-F6-014 Add full ranked table cell — all 50 configs sorted by NLP, 1-based rank (FR-F6-012)
- [x] T-F6-015 Add conclusions markdown cell — key findings, best architecture, implications for BO

**Checkpoint**: Full notebook complete with 50-row ranked table and sensitivity visualisations

---

## Phase F6-5: Polish & Validation

**Purpose**: End-to-end validation

- [x] T-F6-016 Run all cells of `functions/f6/preq-eval-f6.ipynb` top-to-bottom and verify no errors (SC-F6-001)
- [x] T-F6-017 Verify 50 configs × 6 predictions each = 300 total predictions (SC-F6-002)
- [x] T-F6-018 Verify all visualisations are clear and labelled (SC-F6-005)
- [x] T-F6-019 Verify each code step has a preceding markdown explanation (SC-F6-006)

---

## Dependencies & Execution Order (F6)

- **Phase F6-1** (Setup): No dependencies — start here
- **Phase F6-2** (NN Default): Depends on Phase F6-1
- **Phase F6-3** (NN HP Opt): Depends on Phase F6-2
- **Phase F6-4** (Sensitivity): Depends on Phase F6-3
- **Phase F6-5** (Polish): Depends on all phases
