# Tasks: F5 Week 7 — GP Matérn-5/2 + NEI

**Input**: Design documents from `specs/011-f5-gp-nei/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/week7-cells.md

**Tests**: No tests — per constitution, no unit tests are required.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- All tasks target `functions/f5/f5.ipynb` (append-only, no existing cell modifications)

---

## Phase 1: Setup

**Purpose**: Verify environment and notebook readiness

- [X] T001 Verify branch is `011-f5-gp-nei`, notebook `functions/f5/f5.ipynb` has 50 existing cells, and last cell is `#VSC-8f8ac8b4`

**Checkpoint**: Notebook structure confirmed, ready to append cells.

---

## Phase 2: Foundational

**Purpose**: Section header cell that all subsequent cells depend on

- [X] T002 Insert Week 7 markdown header cell after `#VSC-8f8ac8b4` in `functions/f5/f5.ipynb` with heading `## Week 7 — GP Matérn-5/2 + NEI` and rationale paragraph explaining the switch from GBT ensemble (Week 6) to GP-based BO, including comparison table (Week 6: GBT+UCB κ=0.5 vs Week 7: GP Matérn-5/2+NEI q=2)

**Checkpoint**: Week 7 section visible in notebook. All subsequent cells insert after this header.

---

## Phase 3: User Story 1 — Load and Validate Week 7 Data (Priority: P1) 🎯 MVP

**Goal**: Load cumulative Week 7 data (27 samples, 4D) and validate shapes/ranges.

**Independent Test**: Run the data-loading cell; confirm 27 samples, inputs in [0, 1], best value printed.

- [X] T003 [US1] Insert code cell after the Week 7 header in `functions/f5/f5.ipynb` — imports (numpy, torch, copy, matplotlib, botorch.models.SingleTaskGP, botorch.fit.fit_gpytorch_mll, botorch.acquisition.logei.qLogNoisyExpectedImprovement, botorch.optim.optimize_acqf, botorch.sampling.normal.SobolQMCNormalSampler, gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.kernels.ScaleKernel, gpytorch.kernels.MaternKernel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.constraints.GreaterThan) + load `data/f5/updated_inputs - Week 7.npy` and `data/f5/updated_outputs - Week 7.npy` + validate shape (27, 4) and (27,), all inputs in [0, 1], print sample count, input range, output range, best observed value (3394.679933) and its index (#26)

**Checkpoint**: US1 complete — data loaded and validated independently.

---

## Phase 4: User Story 2 — Train GP Surrogate with Specified Hyperparameters (Priority: P1)

**Goal**: Fit SingleTaskGP (Matérn-5/2 ARD, log1p+z-score, noise=0.03, ls=0.25) via 15-restart MLL.

**Independent Test**: Run training cell; confirm fitted HPs printed, noise ≥ 1e-6, no NaN errors.

- [X] T004 [US2] Insert markdown cell after the data loading cell in `functions/f5/f5.ipynb` — hyperparameter documentation table with ≥12 entries (kernel: Matérn-5/2, ARD: True with 4 lengthscales, ls init: 0.25, outputscale init: ~1.0, noise init: 0.03, noise floor: 1e-6, output transform: log1p → z-score, MLL restarts: 15, acquisition: qLogNoisyExpectedImprovement, q: 2, raw_samples: 3000, num_restarts: 50) with rationale for each including why GP replaces GBT
- [X] T005 [US2] Insert code cell after the HP markdown cell in `functions/f5/f5.ipynb` — compute y_log = np.log1p(y_raw), z-score standardise (y_mean, y_std_val, y_std), convert to torch tensors (X_train double (27,4), Y_train double (27,1)), loop 15 restarts with manual seeds creating GaussianLikelihood(noise_constraint=GreaterThan(1e-6)) + ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4)), SingleTaskGP(X_train, Y_train, covar_module, likelihood), init HPs (lengthscale=0.25, noise=0.03, outputscale=1.0), fit_gpytorch_mll, eval mode, score neg MLL, deepcopy best, print per-restart scores, print fitted ℓ₁–ℓ₄, σ²_f, σ²_n with 6 decimal places

**Checkpoint**: US2 complete — GP trained, all hyperparameters reported.

---

## Phase 5: User Story 3 — Propose Next Samples via NEI Acquisition (Priority: P1)

**Goal**: Use qLogNoisyExpectedImprovement (q=2, 3000 Sobol → 50 restarts) to propose 2 candidates and select best for submission.

**Independent Test**: Run acquisition cell; confirm 2 candidates in [0, 1]⁴, best identified by posterior mean.

- [X] T006 [US3] Insert code cell after the GP training cell in `functions/f5/f5.ipynb` — create SobolQMCNormalSampler(sample_shape=512), create qLogNoisyExpectedImprovement(model=best_model, X_baseline=X_train, sampler, prune_baseline=True), set bounds tensor (2, 4) [[0,0,0,0],[1,1,1,1]], optimize_acqf(q=2, num_restarts=50, raw_samples=3000), extract 2 candidates, evaluate posterior mean per candidate (inverse z-score + expm1 to original scale), select best by highest posterior mean, print both candidates with coordinates and posterior means (standardised and original scale)

**Checkpoint**: US3 acquisition complete — 2 candidates proposed, best identified.

---

## Phase 6: User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

**Goal**: Render 3-panel surrogate visualisation and convergence plot matching Week 6 layout.

**Independent Test**: Run both visualisation cells; confirm plots render with correct axes, labels, and colorbars.

- [X] T007 [US4] Insert code cell after the acquisition cell in `functions/f5/f5.ipynb` — identify top-2 dims by shortest ARD lengthscales (np.argsort(lengthscales)[:2]), build 80×80 grid fixing other 2 dims at best_point values, get posterior mean+std (de-standardise via y_std_val*pred + y_mean then expm1), plot 3-panel figure (18×5 inches): Panel 1 mean contour with observed (red dots) + proposed (magenta star) + colourbar, Panel 2 std contour with observed + proposed + colourbar, Panel 3 dimension relevance bar chart (1/ℓ normalised, 4 bars labelled x0–x3), title includes "Week 7" and "GP"
- [X] T008 [US4] Insert code cell after the surrogate plot cell in `functions/f5/f5.ipynb` — compute running_best = np.maximum.accumulate(y_raw), plot vs observation number (1-indexed), add vertical dashed red line at x=26.5 (Week 6→7 boundary), add labels/title/legend/grid, print running best at sample 26 (end of Week 6) and sample 27 (Week 7)

**Checkpoint**: US4 complete — both visualisations rendered.

---

## Phase 7: Polish & Submission

**Purpose**: Format submission query and validate entire notebook

- [X] T009 Insert code cell after the convergence plot in `functions/f5/f5.ipynb` — format best_point as dash-separated string with 6 decimal places (x1-x2-x3-x4), validate (4 parts, all values in [0, 1], 6 decimal places), print submission query prominently with "✓ Submission format validated" confirmation, print summary (surrogate type, acquisition, fitted HPs, query string)
- [X] T010 Run all 8 new cells (51–58) in `functions/f5/f5.ipynb` to validate end-to-end execution, verify no existing cells modified (50 original cells unchanged), verify git diff shows only additions, commit with message "feat(f5): add Week 7 GP Matérn-5/2 + NEI section"

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — verify environment
- **Phase 2 (Foundational)**: Depends on Phase 1 — header cell must be first
- **Phase 3 (US1)**: Depends on Phase 2 — data loading cell follows header
- **Phase 4 (US2)**: Depends on Phase 3 — training needs loaded data
- **Phase 5 (US3)**: Depends on Phase 4 — acquisition needs fitted model
- **Phase 6 (US4)**: Depends on Phase 4 + Phase 5 — visualisation needs model + proposed point
- **Phase 7 (Polish)**: Depends on Phase 6 — submission cell follows convergence plot; validation needs all cells

### User Story Dependencies

- **US1 (Data Loading)**: Independent after foundational
- **US2 (GP Training)**: Depends on US1 (needs data tensors and transform stats)
- **US3 (Acquisition)**: Depends on US2 (needs fitted best_model)
- **US4 (Visualisation)**: Depends on US2 + US3 (needs model, lengthscales, proposed point)

### Within Each Phase

- Markdown cells before code cells (within same story)
- Code cells execute sequentially — each depends on variables from prior cells
- T007 and T008 share model state but render independent plots

### Parallel Opportunities per User Story

```text
Phase 1:  T001
            |
Phase 2:  T002
            |
Phase 3:  T003
            |
Phase 4:  T004 (markdown) -> T005 (training code)
            |
Phase 5:  T006
            |
Phase 6:  T007 --+  (both need model + best_point, but render independent plots)
          T008 --+
            |
Phase 7:  T009 -> T010
```

---

## Implementation Strategy

### MVP First (User Stories 1–3)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Week 7 header cell
3. Complete Phase 3: US1 — data loading
4. Complete Phase 4: US2 — GP training
5. Complete Phase 5: US3 — NEI acquisition
6. **STOP and VALIDATE**: Candidates proposed — can format submission without visualisations

### Full Delivery

7. Complete Phase 6: US4 — surrogate + convergence plots
8. Complete Phase 7: T009 submission formatting + T010 validation + commit

### Incremental Delivery

- After T003: Data loaded and validated (27 samples confirmed)
- After T005: GP trained with Matérn-5/2 ARD, fitted HPs reported
- After T006: 2 candidates proposed via NEI, best selected
- After T009: Submission query formatted and validated
- After T010: Full feature complete, committed

---

## Notes

- All 10 tasks target the same file: `functions/f5/f5.ipynb` (append-only)
- No [P] markers on most tasks because cells must be inserted sequentially in the notebook
- T007 and T008 could theoretically be parallel but ordering in the notebook matters
- Constitution compliance: no unit tests, no existing cell modifications, BoTorch library used
- Transform pipeline: y_raw → log1p → z-score → GP → z-score⁻¹ → expm1 → y_pred
- Key references: research.md (RES-001–RES-007), data-model.md (E-01–E-10), contracts/week7-cells.md (Cells 51–58)
- Total: 10 tasks across 7 phases covering 4 user stories
