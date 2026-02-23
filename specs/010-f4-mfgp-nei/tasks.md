# Tasks: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Input**: Design documents from `specs/010-f4-mfgp-nei/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/week7-cells.md

**Tests**: No tests — per constitution, no unit tests are required.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- All tasks target `functions/f4/f4.ipynb` (append-only, no existing cell modifications)

---

## Phase 1: Setup

**Purpose**: Verify environment and notebook readiness

- [ ] T001 Verify branch is `010-f4-mfgp-nei`, notebook `functions/f4/f4.ipynb` has 52 existing cells, and last cell is `#VSC-21b0ced4`

**Checkpoint**: Notebook structure confirmed, ready to append cells.

---

## Phase 2: Foundational

**Purpose**: Section header cell that all subsequent cells depend on

- [ ] T002 Insert Week 7 markdown header cell after `#VSC-21b0ced4` in `functions/f4/f4.ipynb` with heading `## Week 7 — Multi-Fidelity GP (Matérn-5/2 ARD + LinTrunc) + MF-qNEI` and rationale paragraph (PE winner, acquisition strategy)

**Checkpoint**: Week 7 section visible in notebook. All subsequent cells insert after this header.

---

## Phase 3: User Story 1 — Load and Validate Week 7 Data (Priority: P1) 🎯 MVP

**Goal**: Load cumulative Week 7 data (37 samples, 4D) and validate shapes/ranges.

**Independent Test**: Run the data-loading cell; confirm 37 samples, inputs in [0, 1], best value printed.

- [ ] T003 [US1] Insert code cell after the Week 7 header in `functions/f4/f4.ipynb` — imports (numpy, torch, copy, warnings, matplotlib, botorch.models.SingleTaskMultiFidelityGP, botorch.fit.fit_gpytorch_mll, botorch.acquisition.logei.qLogNoisyExpectedImprovement, botorch.optim.optimize_acqf, botorch.sampling.normal.SobolQMCNormalSampler, gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.constraints.GreaterThan, gpytorch.likelihoods.GaussianLikelihood) + load `data/f4/updated_inputs - Week 7.npy` and `data/f4/updated_outputs - Week 7.npy` + validate shape (37, 4), range [0, 1], print best value, sample count, mean, std

**Checkpoint**: US1 complete — data loaded and validated independently.

---

## Phase 4: User Story 2 — Train MFGP Surrogate (Priority: P1)

**Goal**: Fit MFGP (Matérn-5/2 ARD + LinTrunc, z-score, noise ≥ 1e-4) via 15-restart MLL.

**Independent Test**: Run training cell; confirm fitted HPs printed, noise ≥ 1e-4, no NaN errors.

- [ ] T004 [US2] Insert markdown cell after the data loading cell in `functions/f4/f4.ipynb` — hyperparameter documentation table with ≥ 10 entries (nu=2.5, linear_truncated=True, noise_lb=1e-4, z-score, 15 restarts, q=4, 64 MC samples, 512 raw_samples, 20 acq restarts, fixed_features={4: 1.0}) with justifications
- [ ] T005 [US2] Insert code cell after the HP markdown cell in `functions/f4/f4.ipynb` — z-score standardise outputs, convert to torch tensors, append fidelity column (all 1.0) creating (37, 5) X_mf, loop 15 restarts with manual seeds creating SingleTaskMultiFidelityGP(nu=2.5, linear_truncated=True, data_fidelities=[4], GaussianLikelihood with GreaterThan(1e-4)), fit_gpytorch_mll, track best by neg MLL with deepcopy, print per-restart scores, print fitted ℓ₁–ℓ₄, σ²_f, σ²_n, fidelity power

**Checkpoint**: US2 complete — MFGP trained, all hyperparameters reported.

---

## Phase 5: User Story 3 — Propose Next Sample via MF-qNEI (Priority: P1)

**Goal**: Use qLogNoisyExpectedImprovement (q=4, 64 MC samples, fixed_features={4: 1.0}) to propose 4 candidates and select best for submission.

**Independent Test**: Run acquisition cell; confirm 4 candidates in [0, 0.999999]⁴, best identified.

- [ ] T006 [US3] Insert code cell after the MFGP training cell in `functions/f4/f4.ipynb` — create SobolQMCNormalSampler(64), create qLogNoisyExpectedImprovement(model, X_baseline=X_mf, sampler, prune_baseline=True), set bounds (2, 5) with spatial [0, 0.999999] and fidelity [1.0, 1.0], optimize_acqf(q=4, num_restarts=20, raw_samples=512, fixed_features={4: 1.0}), extract candidates[:, :4], evaluate posterior mean per candidate, select best by highest posterior mean, print all 4 candidates and selected best

**Checkpoint**: US3 complete — submission candidate identified.

---

## Phase 6: User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

**Goal**: Render 2D surrogate slices and convergence plot.

**Independent Test**: Run both visualisation cells; confirm plots render with labels/colorbars.

- [ ] T007 [US4] Insert code cell after the acquisition cell in `functions/f4/f4.ipynb` — identify top-2 dims by shortest ARD lengthscales, build 80×80 grid fixing other 2 dims at best_point values, append fidelity=1.0, get posterior mean+std (de-standardise), plot 2-panel contour figure (mean + std) with observed points (red), proposed point (yellow star), colorbars, axis labels, title
- [ ] T008 [US4] Insert code cell after the surrogate plot cell in `functions/f4/f4.ipynb` — compute running_best = np.maximum.accumulate(y_raw), plot vs observation number, add vertical line at x=30.5 (initial→weekly boundary), add labels/title/legend/grid, print best observed values at boundaries

**Checkpoint**: US4 complete — both visualisations rendered.

---

## Phase 7: Polish & Submission

**Purpose**: Format submission query and validate entire notebook

- [ ] T009 Insert code cell after the convergence plot in `functions/f4/f4.ipynb` — clip best_point to [0, 0.999999], format as dash-separated string with 6 decimal places, validate (4 parts, each in [0, 1]), print submission summary (surrogate type, acquisition, fitted HPs, query string)
- [ ] T010 Run all 8 new cells (53-60) in `functions/f4/f4.ipynb` to validate end-to-end execution, verify no existing cells modified (52 original cells unchanged), verify git diff shows only additions, commit with message "feat(f4): add Week 7 MFGP Matérn-5/2 ARD LinTrunc + MF-qNEI section"

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — verify environment
- **Phase 2 (Foundational)**: Depends on Phase 1 — header cell must be first
- **Phase 3 (US1)**: Depends on Phase 2 — data loading cell follows header
- **Phase 4 (US2)**: Depends on Phase 3 — training needs loaded data
- **Phase 5 (US3)**: Depends on Phase 4 — acquisition needs fitted model
- **Phase 6 (US4)**: Depends on Phase 4 + Phase 5 — visualisation needs model + proposed point
- **Phase 7 (Polish)**: Depends on Phase 5 — submission needs best_point; can run before Phase 6 plots

### User Story Dependencies

- **US1 (Data Loading)**: Independent after foundational
- **US2 (MFGP Training)**: Depends on US1 (needs data tensors)
- **US3 (Acquisition)**: Depends on US2 (needs fitted model)
- **US4 (Visualisation)**: Depends on US2 + US3 (needs model + proposed point)

### Within Each Phase

- Markdown cells before code cells (within same story)
- Code cells execute sequentially — each depends on variables from prior cells
- T007 and T008 are parallelisable [P] within US4 (independent plots) but share model state

### Parallel Opportunities per User Story

```text
Phase 1:  T001
          ↓
Phase 2:  T002
          ↓
Phase 3:  T003
          ↓
Phase 4:  T004 (markdown) → T005 (training code)
          ↓
Phase 5:  T006
          ↓
Phase 6:  T007 ─┐ (both need model + best_point, but render independent plots)
          T008 ─┘
          ↓
Phase 7:  T009 → T010
```

---

## Implementation Strategy

### MVP First (User Stories 1–3)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Week 7 header cell
3. Complete Phase 3: US1 — data loading
4. Complete Phase 4: US2 — MFGP training
5. Complete Phase 5: US3 — MF-qNEI acquisition
6. **STOP and VALIDATE**: Submission query available — can submit without visualisations
7. Complete Phase 7: T009 submission formatting

### Full Delivery

8. Complete Phase 6: US4 — surrogate + convergence plots
9. Complete Phase 7: T010 — end-to-end validation + commit

### Incremental Delivery

- After T003: Data loaded and validated
- After T005: MFGP trained with fitted HPs
- After T006: 4 candidates proposed, best selected
- After T009: Submission query ready
- After T010: Full feature complete, committed

---

## Notes

- All 10 tasks target the same file: `functions/f4/f4.ipynb` (append-only)
- No [P] markers on most tasks because cells must be inserted sequentially in the notebook
- T007 and T008 could theoretically be parallel but ordering in the notebook matters
- Constitution compliance: no unit tests, no existing cell modifications, BoTorch library used
- Total: 10 tasks across 7 phases covering 4 user stories
