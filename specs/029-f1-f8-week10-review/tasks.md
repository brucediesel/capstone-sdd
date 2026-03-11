# Tasks: Week 10 Performance Review & Visualisation

**Input**: Design documents from `/specs/029-f1-f8-week10-review/`
**Prerequisites**: plan.md (loaded), spec.md (loaded), research.md (loaded), data-model.md (loaded), quickstart.md (loaded)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify data availability and confirm branch is ready

- [X] T001 Verify Week 10 data files exist in ./data/f1/ through ./data/f8/ (updated_inputs - Week 10.npy & updated_outputs - Week 10.npy)

---

## Phase 2: Foundational — Create Notebook Template Structure

**Purpose**: Establish the common notebook pattern that all 8 function notebooks share. Since all notebooks follow the same structure (data loading → convergence plot → pair plots → evaluation markdown → improvement markdown), create the first notebook as the reference implementation.

**⚠️ CRITICAL**: T002 establishes the pattern. All subsequent [P] notebooks in Phases 3–4 follow this pattern with function-specific constants and strategy text.

- [X] T002 [US1] Create reference notebook functions/f1/f1 - week 10.ipynb with all cells: imports (numpy, matplotlib, itertools), FunctionConfig constants (FUNC_NUM=1, N_DIMS=2, N_INITIAL=10, USE_LOG_SCALE=True, DATA_DIR), data loading from ../../data/f1/updated_inputs - Week 10.npy and updated_outputs - Week 10.npy, data summary display, convergence plot with log y-axis (np.maximum(outputs, 1e-300) before log to avoid -inf, blue initial / orange submissions, vertical dashed line at sample 10), single 2D pair plot (only 1 pair for 2D) with blue initial unmarked / orange submissions numbered by week (3–10), performance evaluation markdown summarising week 9 Hurdle Model + Weighted UCB strategy and evaluating 0/9 improvements as stalling, and improvement suggestions markdown proposing specific changes to the F1 strategy

**Checkpoint**: F1 notebook is complete and serves as the template for F2–F8

---

## Phase 3: User Story 1 — Review Week 10 Data Across All Functions (Priority: P1) 🎯 MVP

**Goal**: Create all 8 notebooks with data loading, convergence plots, and 2D pair plots

**Independent Test**: Open any notebook, run all cells, verify convergence and pair plots render correctly

### Implementation for User Story 1

- [X] T003 [P] [US1] Create notebook functions/f2/f2 - week 10.ipynb — adapt from F1 pattern: FUNC_NUM=2, N_DIMS=2, N_INITIAL=10, USE_LOG_SCALE=False, DATA_DIR=../../data/f2/, linear y-axis convergence plot, 1 pair plot subplot
- [X] T004 [P] [US1] Create notebook functions/f3/f3 - week 10.ipynb — FUNC_NUM=3, N_DIMS=3, N_INITIAL=15, USE_LOG_SCALE=False, DATA_DIR=../../data/f3/, linear convergence plot, 3 pair plot subplots in 1×3 grid
- [X] T005 [P] [US1] Create notebook functions/f4/f4 - week 10.ipynb — FUNC_NUM=4, N_DIMS=4, N_INITIAL=30, USE_LOG_SCALE=False, DATA_DIR=../../data/f4/, linear convergence plot, 6 pair plot subplots in 2×3 grid
- [X] T006 [P] [US1] Create notebook functions/f5/f5 - week 10.ipynb — FUNC_NUM=5, N_DIMS=4, N_INITIAL=20, USE_LOG_SCALE=False, DATA_DIR=../../data/f5/, linear convergence plot, 6 pair plot subplots in 2×3 grid
- [X] T007 [P] [US1] Create notebook functions/f6/f6 - week 10.ipynb — FUNC_NUM=6, N_DIMS=5, N_INITIAL=20, USE_LOG_SCALE=False, DATA_DIR=../../data/f6/, linear convergence plot, 10 pair plot subplots in 2×5 grid
- [X] T008 [P] [US1] Create notebook functions/f7/f7 - week 10.ipynb — FUNC_NUM=7, N_DIMS=6, N_INITIAL=30, USE_LOG_SCALE=False, DATA_DIR=../../data/f7/, linear convergence plot, 15 pair plot subplots in 3×5 grid
- [X] T009 [P] [US1] Create notebook functions/f8/f8 - week 10.ipynb — FUNC_NUM=8, N_DIMS=8, N_INITIAL=40, USE_LOG_SCALE=False, DATA_DIR=../../data/f8/, linear convergence plot, 28 pair plot subplots in 4×7 grid

**Checkpoint**: All 8 notebooks load data and display convergence + pair plots correctly

---

## Phase 4: User Story 2 — Identify Performance Issues & Propose Improvements (Priority: P2)

**Goal**: Add performance evaluation and strategy improvement markdown to each notebook

**Independent Test**: After running any notebook, verify markdown sections at the end evaluate performance and propose specific strategy improvements relative to week 9 configuration

### Implementation for User Story 2

> Note: T002 (F1) already includes both US1 and US2 content. The tasks below add the US2 markdown content to F2–F8.

- [X] T010 [P] [US2] Add performance evaluation markdown to functions/f2/f2 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-1.5 ARD, qLogNEI q=4), evaluate 1/9 improvements, flag stalling and local-optimum trapping
- [X] T011 [P] [US2] Add improvement suggestions markdown to functions/f2/f2 - week 10.ipynb — propose specific changes relative to SFGP Matérn-1.5 (e.g., increase exploration, change kernel, add restart mechanism)
- [X] T012 [P] [US2] Add performance evaluation markdown to functions/f3/f3 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-2.5 ARD, qLogNEI q=1), evaluate 2/9 improvements, note slow but steady progress
- [X] T013 [P] [US2] Add improvement suggestions markdown to functions/f3/f3 - week 10.ipynb — propose specific changes relative to SFGP Matérn-2.5 (e.g., increase q, adjust restarts, consider different acquisition)
- [X] T014 [P] [US2] Add performance evaluation markdown to functions/f4/f4 - week 10.ipynb — summarise week 9 strategy (MFGP Matérn-2.5 + LinearTruncated, MF-qNEI q=4), evaluate 5/9 improvements but recent stalling, flag MFGP on single-fidelity data concern
- [X] T015 [P] [US2] Add improvement suggestions markdown to functions/f4/f4 - week 10.ipynb — propose specific changes relative to MFGP (e.g., switch to SFGP since only single-fidelity data available, adjust noise bounds)
- [X] T016 [P] [US2] Add performance evaluation markdown to functions/f5/f5 - week 10.ipynb — summarise week 9 strategy (GP Matérn-1.5 ARD, qLogNEI q=4, log1p + Standardize), evaluate 9/9 improvements, note strong performance
- [X] T017 [P] [US2] Add improvement suggestions markdown to functions/f5/f5 - week 10.ipynb — propose specific refinements for continued improvement (e.g., fine-tune exploration/exploitation balance, review distance-based candidate selection)
- [X] T018 [P] [US2] Add performance evaluation markdown to functions/f6/f6 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-1.5 ARD, qLogNEI q=4, rank-based IP), evaluate 9/9 improvements, note all-negative but trending towards zero
- [X] T019 [P] [US2] Add improvement suggestions markdown to functions/f6/f6 - week 10.ipynb — propose specific refinements (e.g., adjust milk constraint, review IP steepness, consider output transformation)
- [X] T020 [P] [US2] Add performance evaluation markdown to functions/f7/f7 - week 10.ipynb — summarise week 9 strategy (NN 6→5→5→1, 70% mean + 30% EI, interior penalty), evaluate 4/9 improvements, flag stalling in recent weeks
- [X] T021 [P] [US2] Add improvement suggestions markdown to functions/f7/f7 - week 10.ipynb — propose specific changes relative to NN surrogate (e.g., increase exploration weight, change network architecture, consider switch to GP-based surrogate)
- [X] T022 [P] [US2] Add performance evaluation markdown to functions/f8/f8 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-2.5 ARD, qEI XI=0.01), evaluate 8/9 near-continuous improvement, note positive trajectory
- [X] T023 [P] [US2] Add improvement suggestions markdown to functions/f8/f8 - week 10.ipynb — propose specific refinements for final rounds (e.g., fine-tune XI, increase MC samples if stable, review noise floor)

**Checkpoint**: All 8 notebooks have performance evaluation and improvement suggestion markdown sections

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all notebooks

- [X] T024 Run all 8 notebooks top-to-bottom and verify no execution errors
- [X] T025 Verify convergence plot styling consistency across all notebooks (blue/orange colours, dashed separator line, axis labels)
- [X] T026 Verify pair plot week numbering is correct (3–10) across all notebooks and initial points are unmarked
- [X] T027 Commit all 8 notebooks to branch 029-f1-f8-week10-review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify data exists
- **Foundational (Phase 2)**: Depends on Phase 1 — create F1 reference notebook
- **US1 (Phase 3)**: Depends on Phase 2 (F1 pattern established) — create F2–F8 notebooks
- **US2 (Phase 4)**: Depends on Phase 3 — add evaluation/improvement markdown to F2–F8
- **Polish (Phase 5)**: Depends on Phases 3 and 4 — validate all notebooks

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational (Phase 2). F2–F8 notebooks can all be created in parallel.
- **User Story 2 (P2)**: F1 evaluation is already in T002. F2–F8 evaluation tasks (T010–T023) can run in parallel once their respective notebooks exist.

> **Note**: In practice, each notebook (T003–T009) will be created with all content (US1 + US2) in a single pass, since the evaluation/improvement markdown cells are part of the notebook structure. The task separation is for tracking purposes — all content for a given function can be implemented together.

### Within Each User Story

- Data loading and convergence plot before pair plots (data must be loaded first)
- Pair plots before evaluation markdown (evaluation references plot observations)
- Evaluation before improvement suggestions (improvements reference evaluation findings)

### Parallel Opportunities

**Phase 3 (US1)**: T003–T009 can all run in parallel (7 independent notebooks, different files)

**Phase 4 (US2)**: T010–T023 can all run in parallel (each modifies a different notebook)

**Combined**: Since each notebook is a separate file, a practical approach creates each notebook completely (US1 + US2 content) in a single pass:
- T002 (F1) first as reference
- Then T003+T010+T011 (F2), T004+T012+T013 (F3), T005+T014+T015 (F4), T006+T016+T017 (F5), T007+T018+T019 (F6), T008+T020+T021 (F7), T009+T022+T023 (F8) — all 7 in parallel

---

## Implementation Strategy

### MVP Scope (Recommended)

**MVP = Phase 1 + Phase 2 + Phase 3 (User Story 1 only)**

This delivers all 8 notebooks with data loading and visualisations. The student can visually assess performance before the evaluation markdown is added.

### Incremental Delivery

1. **Increment 1**: F1 notebook complete (T001 + T002) — validates the pattern
2. **Increment 2**: F2–F8 notebooks with visualisations (T003–T009) — all functions visible
3. **Increment 3**: Performance evaluation + improvements for all functions (T010–T023) — strategy decisions enabled
4. **Increment 4**: Validation pass (T024–T027) — quality assurance

### Practical Note

Given that each notebook is self-contained and the content for US1 and US2 is tightly coupled (evaluation references the plots), the most efficient implementation creates each notebook in full (all cells) in a single pass per function. The task breakdown above separates concerns for tracking, but implementation should proceed function-by-function after the F1 reference is established.

---
---

# Tasks: F1 Week 10 — SFGP Optimisation Run

**Input**: Design documents from `/specs/029-f1-f8-week10-review/` (spec-f1-optimisation.md, plan.md, research.md, data-model.md, contracts/f1-optimisation-pipeline.md)
**Prerequisites**: plan.md (loaded), spec-f1-optimisation.md (loaded), research.md (R6–R11), data-model.md (OptimisationConfig, SFGPModel, AcquisitionCandidates, SubmissionPoint), contracts/f1-optimisation-pipeline.md (6 cell groups)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks are grouped by user story. All tasks append new cells to the existing `functions/f1/f1 - week 10.ipynb` notebook (12 existing cells remain untouched).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- All tasks target the single file: `functions/f1/f1 - week 10.ipynb`

---

## Phase 6: Setup (Optimisation Infrastructure)

**Purpose**: Add BoTorch/GPyTorch imports and hyperparameter configuration to the F1 notebook

- [X] T028 Append markdown cell "## Step 6 — Optimisation Run: SFGP + qLogNEI" to functions/f1/f1 - week 10.ipynb after existing Step 5, explaining the strategy change from Hurdle Model to SFGP with Matérn-2.5 ARD and the rationale for each of the 5 changes
- [X] T029 Append code cell with BoTorch/GPyTorch imports to functions/f1/f1 - week 10.ipynb: torch, copy, warnings, SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf, SobolQMCNormalSampler, Normalize, MaternKernel, ScaleKernel, GaussianLikelihood, GreaterThan, Interval, ExactMarginalLogLikelihood
- [X] T030 Append code cell with all hyperparameter constants to functions/f1/f1 - week 10.ipynb: KERNEL_NU=2.5, ARD_NUM_DIMS=2, LS_LOWER=0.01, LS_UPPER=2.0, NOISE_LB=1e-4, N_MLL_RESTARTS=15, LOG_EPSILON=1e-300, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=20, RAW_SAMPLES=10000, GRID_RES=50 — each with explanatory comment

**Checkpoint**: Configuration complete — no computation yet, all constants visible

---

## Phase 7: User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

**Goal**: Fit SFGP on log-transformed F1 data, run qLogNEI acquisition with q=4, select best candidate via distance-based filtering, and produce formatted submission point

**Independent Test**: Run all cells top-to-bottom, verify printed submission string is in format `0.xxxxxx-0.yyyyyy` with values in [0.0, 0.999999]

### Implementation for User Story 1

- [X] T031 [US1] Append markdown cell "### Step 6.1 — Data Preparation & Log Transform" to functions/f1/f1 - week 10.ipynb explaining the log(max(y, ε)) transform and expected output range [-690, -35]
- [X] T032 [US1] Append code cell to functions/f1/f1 - week 10.ipynb that converts `inputs` and `outputs` (from existing cells) to torch tensors X_train (N×2, float64) and Y_train (N×1, float64 log-transformed via torch.log(torch.clamp(y, min=LOG_EPSILON))), prints shapes and Y_train range
- [X] T033 [US1] Append markdown cell "### Step 6.2 — SFGP Fitting (Matérn-2.5 ARD, 15 MLL Restarts)" to functions/f1/f1 - week 10.ipynb explaining multi-restart MLL strategy and kernel configuration
- [X] T034 [US1] Append code cell to functions/f1/f1 - week 10.ipynb implementing 15-restart MLL GP fitting loop: for each seed, construct ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2, lengthscale_constraint=Interval(LS_LOWER,LS_UPPER))), GaussianLikelihood(noise_constraint=GreaterThan(NOISE_LB)), SingleTaskGP with Normalize(d=2), fit via fit_gpytorch_mll, track best_loss and copy.deepcopy best_model; print fitted lengthscales, noise, outputscale, best_loss
- [X] T035 [US1] Append markdown cell "### Step 6.3 — qLogNEI Acquisition & Distance-Based Selection" to functions/f1/f1 - week 10.ipynb explaining qLogNEI with q=4, 10k Sobol seeding, and two-stage distance selection
- [X] T036 [US1] Append code cell to functions/f1/f1 - week 10.ipynb implementing: SobolQMCNormalSampler(MC_SAMPLES), qLogNoisyExpectedImprovement(best_model, X_baseline=X_train, sampler, prune_baseline=True), optimize_acqf(bounds=[[0,0],[1,1]], q=Q_BATCH, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES); then distance-based selection: posterior means for q=4 candidates, quality gate (mean ≥ median), max-min-distance to X_train via torch.cdist; print all 4 candidates, selection rationale, selected point
- [X] T037 [US1] Append code cell to functions/f1/f1 - week 10.ipynb implementing submission formatting: clamp x_new to [0.0, 0.999999], format as "x1-x2" with 6 decimal places, duplicate check against all existing samples, print ">>> SUBMISSION: ..." string

**Checkpoint**: US1 complete — submission point produced and formatted. Notebook can be validated independently at this point.

---

## Phase 8: User Story 2 — Visualise Surrogate and Acquisition Surface (Priority: P2)

**Goal**: Produce 3-panel contour plot (GP mean, GP std, acquisition surface) with point overlays on 50×50 grid

**Independent Test**: Run all cells, verify 3-panel figure renders with correct colormaps (viridis, YlOrRd, plasma) and point overlays (blue, orange, green star)

### Implementation for User Story 2

- [X] T038 [US2] Append markdown cell "### Step 6.4 — Surrogate Visualisation (3-Panel Contour)" to functions/f1/f1 - week 10.ipynb explaining the three surfaces and point overlay convention
- [X] T039 [US2] Append code cell to functions/f1/f1 - week 10.ipynb implementing 3-panel contour visualisation: construct 50×50 grid over [0,1]², evaluate best_model.posterior for mean and std grids, evaluate acqf in batches of 500 (unsqueeze(1) for single-point evaluation) to build acquisition grid; create fig with 3 subplots — Panel 1: contourf(mean, viridis), Panel 2: contourf(std, YlOrRd), Panel 3: contourf(acq, plasma); overlay on all panels: scatter initial (blue, s=40), scatter submissions (orange, s=60), scatter proposed (green star, s=200); add colorbars and titles

**Checkpoint**: US2 complete — 3-panel surrogate visualisation renders correctly

---

## Phase 9: User Story 3 — Display Convergence with Proposed Point (Priority: P3)

**Goal**: Show updated convergence plot with running best, log y-axis, and proposed point marked distinctly

**Independent Test**: Run all cells, verify convergence plot shows green star at position n_total+1 with log y-axis

### Implementation for User Story 3

- [X] T040 [US3] Append markdown cell "### Step 6.5 — Updated Convergence Plot" to functions/f1/f1 - week 10.ipynb explaining the convergence display with proposed point
- [X] T041 [US3] Append code cell to functions/f1/f1 - week 10.ipynb implementing updated convergence plot: predict GP posterior mean at x_new (in log space), plot running_best line (blue initial region, orange submission region), scatter all outputs, mark proposed point with green star at index n_total, use log y-axis with np.maximum(values, 1e-300) clamping, add vertical dashed line at N_INITIAL, axis labels and legend

**Checkpoint**: US3 complete — convergence plot with proposed point renders correctly

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation of the complete notebook

- [X] T042 Run functions/f1/f1 - week 10.ipynb top-to-bottom (Kernel > Restart & Run All) and verify all cells execute without errors in < 60 seconds
- [X] T043 Verify GP hyperparameters are reasonable: lengthscales in [0.01, 2.0], noise > 0, outputscale > 0
- [X] T044 Verify submission point is in valid range [0.0, 0.999999] for both dimensions and is not flagged as duplicate
- [X] T045 Verify 3-panel contour renders with correct colormaps and all three point types visible
- [X] T046 Verify convergence plot shows proposed point in green and uses log y-axis
- [ ] T047 Commit updated notebook to branch 029-f1-f8-week10-review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 6)**: No dependencies on new tasks — requires existing T002 notebook completed (✅)
- **US1 (Phase 7)**: Depends on Phase 6 (imports + config cells appended)
- **US2 (Phase 8)**: Depends on Phase 7 (best_model + acqf + x_new must exist)
- **US3 (Phase 9)**: Depends on Phase 7 (x_new + best_model must exist for prediction)
- **Polish (Phase 10)**: Depends on Phases 7, 8, 9

### User Story Dependencies

- **US1 (P1)**: Sequential within phase — data prep → GP fitting → acquisition → submission formatting (each cell depends on previous)
- **US2 (P2)**: Depends on US1 completion (needs best_model, acqf, x_new). Cannot run in parallel with US1.
- **US3 (P3)**: Depends on US1 completion (needs x_new, best_model). Can run in parallel with US2 (different cells).

### Within User Story 1

Tasks T031–T037 are strictly sequential (each cell depends on variables from previous):
1. T031–T032: Data preparation (X_train, Y_train)
2. T033–T034: GP fitting (best_model)
3. T035–T036: Acquisition (candidates, x_new, acqf)
4. T037: Submission formatting (proposed_query)

### Parallel Opportunities

- **T038–T039 (US2)** and **T040–T041 (US3)** can run in parallel after US1 is complete — they use the same variables (best_model, x_new) but produce independent output cells
- **T042–T046 (Polish)** must run after all user stories complete

---

## Parallel Example: After US1 Completion

```bash
# After T037 completes, both US2 and US3 can start simultaneously:
Task T038-T039: "3-panel contour visualisation cell" (uses best_model, acqf, x_new)
Task T040-T041: "Updated convergence plot cell" (uses x_new, best_model, outputs)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 6: Setup (T028–T030) — imports + config
2. Complete Phase 7: US1 (T031–T037) — data prep → GP → acquisition → submission
3. **STOP and VALIDATE**: Run notebook, verify submission string is produced
4. This delivers the core deliverable: a proposed sample point for submission

### Incremental Delivery

1. **Increment 1**: Setup + US1 (T028–T037) → Submission point available (MVP)
2. **Increment 2**: US2 (T038–T039) → Surrogate visualisation added
3. **Increment 3**: US3 (T040–T041) → Convergence plot with proposal added
4. **Increment 4**: Polish (T042–T047) → Full validation and commit

### Notes

- All tasks target a single file: `functions/f1/f1 - week 10.ipynb`
- Tasks are cell appends — no modification of existing cells 1–12
- Each markdown+code pair (e.g., T031+T032) should be implemented together as they form a logical section
- The notebook should be runnable after each user story checkpoint
