# Tasks: F2 Week 10 — SFGP Optimisation Run

**Input**: Design documents from `/specs/029-f1-f8-week10-review/`
**Prerequisites**: plan-f2-optimisation.md (loaded), spec-f2-optimisation.md (loaded), research.md §R12–R15 (loaded), data-model.md §F2 (loaded), contracts/f2-optimisation-pipeline.md (loaded), quickstart.md §F2 (loaded)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. All tasks append cells to the existing `functions/f2/f2 - week 10.ipynb` notebook (cells 1–12 already exist).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify existing notebook and data availability

- [X] T001 Verify existing F2 week 10 notebook has 12 cells and variables `inputs` (20,2), `outputs` (20,), `N_INITIAL`=10, `N_DIMS`=2, `n_total`=20, `running_best`, `stalling`=True are in scope after execution in functions/f2/f2 - week 10.ipynb

---

## Phase 2: Foundational — Markdown Separator & Configuration Cell

**Purpose**: Add the optimisation section header and all imports/constants before any computation cells. These cells MUST exist before any US1/US2/US3 cells can be appended.

**⚠️ CRITICAL**: No user story implementation can begin until these cells are appended.

- [X] T002 Append markdown cell to functions/f2/f2 - week 10.ipynb — section header "## Step 6: Optimisation Run — SFGP Matérn-2.5 ARD + Standardize(m=1)" with brief rationale documenting 5 strategy changes from week 9 (kernel ν 1.5→2.5, Standardize(m=1), LS bounds [0.005,10.0], 50 MLL restarts, 4096 RAW_SAMPLES) per FR-006
- [X] T003 Append code cell (CG1 — Imports & Configuration) to functions/f2/f2 - week 10.ipynb — import torch, botorch (SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf, Standardize), gpytorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood, Interval), copy, warnings; define all named constants: KERNEL_NU=2.5, ARD_NUM_DIMS=2, LS_LOWER=0.005, LS_UPPER=10.0, NOISE_LB=1e-4, N_MLL_RESTARTS=50, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=20, RAW_SAMPLES=4096, GRID_RES=50; each constant with comment explaining value and change from week 9

**Checkpoint**: Configuration cell executes without errors, all constants printed and verified

---

## Phase 3: User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

**Goal**: Fit SFGP with Matérn-2.5 ARD + Standardize(m=1), run qLogNEI acquisition, select best candidate via distance filtering, format submission string

**Independent Test**: Run all cells including new optimisation section; verify a formatted submission point `x1-x2` is printed with values in [0.0, 0.999999]

### Implementation for User Story 1

- [X] T004 [US1] Append code cell (CG2 — Data Preparation) to functions/f2/f2 - week 10.ipynb — convert `inputs` to `X_train` tensor (20,2) float64 and `outputs` to `Y_train` tensor (20,1) float64 with `.unsqueeze(-1)`; NO log transform; print shape summary and Y_train min/max range per contract CG2
- [X] T005 [US1] Append code cell (CG3 — GP Fitting) to functions/f2/f2 - week 10.ipynb — implement multi-restart MLL loop: for each of N_MLL_RESTARTS=50, construct SingleTaskGP with MaternKernel(nu=KERNEL_NU, ard_num_dims=ARD_NUM_DIMS) wrapped in ScaleKernel, lengthscale constraint Interval(LS_LOWER, LS_UPPER), GaussianLikelihood with noise constraint ≥NOISE_LB, outcome_transform=Standardize(m=1); randomise hyperparameters, fit via fit_gpytorch_mll, track best model by lowest MLL loss; set best_model.eval(); print fitted lengthscales, noise, outputscale, best_loss per contract CG3
- [X] T006 [US1] Append code cell (CG4 — Acquisition & Selection) to functions/f2/f2 - week 10.ipynb — construct qLogNoisyExpectedImprovement with best_model, X_observed=X_train, sampler with MC_SAMPLES=512; call optimize_acqf with bounds [[0,0],[1,1]], q=Q_BATCH=4, num_restarts=NUM_RESTARTS=20, raw_samples=RAW_SAMPLES=4096; apply distance-based selection (filter candidates with pred_mean ≥ median, pick max min-distance to X_train); clamp x_new to [0.0, 0.999999]; format proposed_query as `f"{x1:.6f}-{x2:.6f}"`; check is_duplicate against existing inputs; print all 4 candidates, selection rationale, and `>>> SUBMISSION: {proposed_query}` per contract CG4; retain `acqf` in scope for CG5

**Checkpoint**: Submission string printed — US1 is functionally complete. Verify SC-001, SC-002, SC-003

---

## Phase 4: User Story 2 — Visualise New Surrogate and Acquisition Surface (Priority: P2)

**Goal**: 3-panel contour showing GP posterior mean, uncertainty, and acquisition surface with colour-coded point overlays

**Independent Test**: After running all cells, verify 3-panel figure renders with correct surfaces and point colours (blue initial, orange submissions, green star proposed)

### Implementation for User Story 2

- [X] T007 [US2] Append code cell (CG5 — 3-Panel Visualisation) to functions/f2/f2 - week 10.ipynb — create 50×50 meshgrid over [0,1]²; compute posterior mean and std from best_model on grid (linear scale, auto-untransformed by Standardize); compute acquisition values from acqf on grid (unsqueeze for q=1 evaluation); plot 3-panel fig (1×3): Panel 1 GP mean (viridis contourf + colourbar), Panel 2 GP std (YlOrRd contourf + colourbar), Panel 3 Acquisition (plasma contourf + colourbar); overlay on all panels: blue dots for initial samples (X_train[:N_INITIAL]), orange dots for submissions (X_train[N_INITIAL:]), green star for x_new; add titles "Posterior Mean", "Posterior Std Dev", "Acquisition (qLogNEI)"; per contract CG5

**Checkpoint**: 3-panel figure renders correctly — US2 is functionally complete. Verify SC-004

---

## Phase 5: User Story 3 — Display Convergence with Proposed Point (Priority: P3)

**Goal**: Updated convergence plot showing running best trajectory with proposed next point marked

**Independent Test**: After running all cells, verify convergence plot shows all 20 data points plus proposed point with green star marker

### Implementation for User Story 3

- [X] T008 [US3] Append code cell (CG6 — Convergence Plot) to functions/f2/f2 - week 10.ipynb — predict at x_new using best_model.posterior() (auto-untransformed); plot running_best as line (step or connected) with blue region for initial (1–N_INITIAL) and orange for submissions (N_INITIAL+1–n_total); mark proposed point at position n_total+1 with green star and predicted mean; linear y-axis (NOT log — F2 outputs in [0.25, 0.67]); add vertical dashed line at N_INITIAL boundary; title "F2 Convergence — Running Best + Proposed"; xlabel "Sample", ylabel "Output"; per contract CG6

**Checkpoint**: Convergence plot renders with proposed point — US3 is functionally complete. Verify SC-005

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation of the complete notebook

- [X] T009 Run functions/f2/f2 - week 10.ipynb top-to-bottom (Kernel > Restart & Run All) and verify all 19 cells execute without errors per SC-001
- [X] T010 Verify GP hyperparameters are reasonable: lengthscales in [0.005, 10.0], noise ≥ 1e-4, outputscale > 0 per SC-002
- [X] T011 Verify submission point is valid: both values in [0.0, 0.999999], not a duplicate, format matches `\d\.\d{6}-\d\.\d{6}` per SC-003
- [X] T012 Verify all hyperparameter constants have comments documenting value and week 9 change rationale per SC-006 and FR-006

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify existing notebook
- **Foundational (Phase 2)**: Depends on Phase 1 — append markdown + config cell
- **US1 (Phase 3)**: Depends on Phase 2 — CG2→CG3→CG4 sequential (each cell needs prior cell's outputs)
- **US2 (Phase 4)**: Depends on US1 (needs `best_model`, `acqf`, `X_train`, `x_new`)
- **US3 (Phase 5)**: Depends on US1 (needs `best_model`, `x_new`, `running_best`, `outputs`)
- **Polish (Phase 6)**: Depends on Phases 3, 4, 5 — validate complete notebook

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational (Phase 2). Tasks T004→T005→T006 are strictly sequential (data→model→acquisition).
- **User Story 2 (P2)**: Depends on US1 completion (needs `best_model`, `acqf`, `x_new`). Can start once T006 is done.
- **User Story 3 (P3)**: Depends on US1 completion (needs `best_model`, `x_new`). Can start once T006 is done. Independent of US2.

### Parallel Opportunities

**US2 + US3 can run in parallel** after US1 completes:
- T007 (3-panel visualisation) and T008 (convergence plot) use different outputs and produce independent cells
- In practice, since both append to the same notebook, they should be implemented sequentially to maintain cell order (T007 before T008)

**Within US1**: No parallelism — T004→T005→T006 is a strict pipeline (each cell consumes the prior cell's outputs)

**Phase 6**: T009–T012 are sequential validation steps after all cells exist

---

## Parallel Example: After US1 Completion

```
After T006 (acquisition) completes:
  → T007 [US2] (3-panel visualisation) — needs best_model, acqf, X_train, x_new
  → T008 [US3] (convergence plot) — needs best_model, x_new, running_best, outputs
  Both can conceptually run in parallel but append sequentially to maintain notebook order
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Verify notebook (T001)
2. Complete Phase 2: Markdown header + config cell (T002, T003)
3. Complete Phase 3: Data prep → GP fit → acquisition (T004, T005, T006)
4. **STOP and VALIDATE**: Verify submission string is printed and valid
5. Submit point if deadline is imminent

### Incremental Delivery

1. T001–T003 → Foundation ready (config cell executes)
2. T004–T006 → US1 complete: submission point produced
3. T007 → US2 complete: surrogate visualised
4. T008 → US3 complete: convergence updated
5. T009–T012 → Polish: full validation

### Practical Note

Since all tasks modify the same notebook file (`functions/f2/f2 - week 10.ipynb`) by appending cells, implementation is inherently sequential. The task breakdown facilitates tracking and verification of each cell group independently. The most efficient approach is to implement T002–T008 in a single pass, appending 7 cells (1 markdown + 6 code) to the notebook.

---

## Summary

| Metric | Count |
|--------|-------|
| Total tasks | 12 |
| Setup tasks | 1 (T001) |
| Foundational tasks | 2 (T002–T003) |
| US1 tasks (P1 — MVP) | 3 (T004–T006) |
| US2 tasks (P2) | 1 (T007) |
| US3 tasks (P3) | 1 (T008) |
| Polish tasks | 4 (T009–T012) |
| Parallel opportunities | US2 + US3 after US1 (notebook ordering constraint applies) |
| MVP scope | T001–T006 (6 tasks — produces submission point) |
