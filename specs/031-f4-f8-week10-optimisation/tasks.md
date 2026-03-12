# Tasks: F4–F8 Week 10 Optimisation Strategy Changes

**Input**: Design documents from `/specs/031-f4-f8-week10-optimisation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Tests**: No tests — constitution §I explicitly states no unit tests required.
**Organization**: Tasks are grouped by user story (one per function) to enable independent implementation and testing of each function.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1=F4, US2=F5, US3=F6, US4=F7, US5=F8)
- Include exact file paths in descriptions

## Path Conventions

- **Notebooks**: `functions/fX/fX - week 10.ipynb`
- **Data**: `data/fX/updated_inputs - Week 10.npy`, `data/fX/updated_outputs - Week 10.npy`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify project prerequisites and confirm existing notebook state

- [X] T001 Verify week 10 data files exist for F4–F8 in data/f4/ through data/f8/
- [X] T002 Verify existing week 10 notebooks have 12 review cells with required variables in scope in functions/f4/ through functions/f8/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No additional foundational tasks — all shared infrastructure (git repo, pyenv, dependencies) already exists. Week 10 review notebooks are in place.

**⚠️ CRITICAL**: Phase 1 verification must pass before proceeding.

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel

---

## Phase 3: User Story 1 — F4 Surrogate & Acquisition Overhaul (Priority: P1) 🎯 MVP

**Goal**: Replace MFGP with SFGP Matérn-2.5 ARD, switch to qLogNEI q=4, add Standardize(m=1), noise_lb=1e-3, MLL restarts ≥30.

**Independent Test**: Run the F4 week 10 notebook end-to-end; verify SFGP with correct kernel, Standardize(m=1), noise_lb=1e-3, ≥30 MLL restarts, qLogNEI q=4, valid 4D submission, contour + convergence plots render.

### Implementation for User Story 1

- [X] T003 [US1] Append markdown cell "Step 6 — Week 10 Optimisation Run" with 5 strategy changes and rationale in functions/f4/f4 - week 10.ipynb
- [X] T004 [US1] Append imports & configuration cell with all constants (KERNEL_NU=2.5, ARD_NUM_DIMS=4, NOISE_LB=1e-3, N_MLL_RESTARTS=30, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=20, RAW_SAMPLES=2048) in functions/f4/f4 - week 10.ipynb
- [X] T005 [US1] Append data preparation cell — load week 10 data (4D, 40 obs), convert to float64 tensors, print shape and output range in functions/f4/f4 - week 10.ipynb
- [X] T006 [US1] Append GP fitting cell — SingleTaskGP with ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4)), GaussianLikelihood(noise_constraint=GreaterThan(1e-3)), Standardize(m=1), ≥30 MLL restarts with random hyperparameter initialisation in functions/f4/f4 - week 10.ipynb
- [X] T007 [US1] Append acquisition & selection cell — qLogNoisyExpectedImprovement with q=4, SobolQMCNormalSampler(512), optimize_acqf(num_restarts=20, raw_samples=2048), distance-based selection, duplicate check, formatted submission output in functions/f4/f4 - week 10.ipynb
- [X] T008 [US1] Append 2D contour visualisation cell — GP posterior mean across input dimension pairs with training points and proposed point in functions/f4/f4 - week 10.ipynb
- [X] T009 [US1] Append convergence plot cell — running best objective with proposed point marked in functions/f4/f4 - week 10.ipynb
- [X] T010 [US1] Run all new cells in F4 notebook and verify: GP converges (≥50% restarts within 10% of best loss), submission format x1-x2-x3-x4 in [0,0.999999], not duplicate, visualisations render in functions/f4/f4 - week 10.ipynb

**Checkpoint**: F4 notebook produces a valid submission — US1 independently verifiable

---

## Phase 4: User Story 2 — F5 Exploration Tuning & Transform Simplification (Priority: P1)

**Goal**: Switch log1p→log, increase raw_samples to 8000, increase acq restarts to 60, relax distance selection, evaluate single vs double transform.

**Independent Test**: Run the F5 week 10 notebook end-to-end; verify log transform, raw_samples=8000, num_restarts=60, relaxed selection, valid 4D submission.

### Implementation for User Story 2

- [X] T011 [P] [US2] Append markdown cell "Step 6 — Week 10 Optimisation Run" with 5 strategy changes and rationale in functions/f5/f5 - week 10.ipynb
- [X] T012 [P] [US2] Append imports & configuration cell with all constants (KERNEL_NU=1.5, ARD_NUM_DIMS=4, NOISE_LB=1e-6, N_MLL_RESTARTS=15, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=60, RAW_SAMPLES=8000) in functions/f5/f5 - week 10.ipynb
- [X] T013 [US2] Append data preparation cell — load week 10 data, apply np.log transform (verify all outputs strictly positive), convert to float64 tensors, print raw and log-space ranges in functions/f5/f5 - week 10.ipynb
- [X] T014 [US2] Append GP fitting cell — SingleTaskGP with Matérn-1.5 ARD, Standardize(m=1), 15 MLL restarts, document log+Standardize vs log-only decision in functions/f5/f5 - week 10.ipynb
- [X] T015 [US2] Append acquisition & selection cell — qLogNEI q=4, num_restarts=60, raw_samples=8000, relaxed distance gate (25th percentile or accept all), duplicate check, formatted submission in functions/f5/f5 - week 10.ipynb
- [X] T016 [P] [US2] Append 2D contour visualisation cell — GP posterior mean across input pairs with training/proposed points in functions/f5/f5 - week 10.ipynb
- [X] T017 [P] [US2] Append convergence plot cell — running best objective with proposed point in functions/f5/f5 - week 10.ipynb
- [X] T018 [US2] Run all new cells in F5 notebook and verify: submission format x1-x2-x3-x4 in [0,0.999999], not duplicate, visualisations render in functions/f5/f5 - week 10.ipynb

**Checkpoint**: F5 notebook produces a valid submission — US2 independently verifiable

---

## Phase 5: User Story 3 — F6 Incremental Refinement (Priority: P2)

**Goal**: Maintain SFGP Matérn-1.5 ARD with rank-based IP, tighten milk constraint to 0.12, reduce noise_lb to 1e-3, increase raw_samples to 5000.

**Independent Test**: Run the F6 week 10 notebook end-to-end; verify milk≥0.12, noise_lb=1e-3, raw_samples=5000, valid 5D submission.

### Implementation for User Story 3

- [X] T019 [P] [US3] Append markdown cell "Step 6 — Week 10 Optimisation Run" with 4 strategy changes and rationale in functions/f6/f6 - week 10.ipynb
- [X] T020 [P] [US3] Append imports & configuration cell with all constants (KERNEL_NU=1.5, ARD_NUM_DIMS=5, NOISE_LB=1e-3, N_MLL_RESTARTS=15, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=50, RAW_SAMPLES=5000, STEEPNESS=1.0, FLOOR=0.01, MILK_THRESHOLD=0.12) in functions/f6/f6 - week 10.ipynb
- [X] T021 [US3] Append data preparation cell — load week 10 data (5D, 30 obs), convert to float64 tensors, print shape and output range in functions/f6/f6 - week 10.ipynb
- [X] T022 [US3] Append GP fitting cell — SingleTaskGP Matérn-1.5 ARD, GaussianLikelihood(noise_constraint=GreaterThan(1e-3)), Standardize(m=1), 15 MLL restarts in functions/f6/f6 - week 10.ipynb
- [X] T023 [US3] Append acquisition & selection cell — qLogNEI q=4, num_restarts=50, raw_samples=5000, rank-based interior penalty (STEEPNESS=1.0, FLOOR=0.01), milk≥0.12 feasibility (fallback to 0.10 if infeasible), duplicate check, formatted submission in functions/f6/f6 - week 10.ipynb
- [X] T024 [P] [US3] Append 2D contour visualisation cell — GP posterior mean across ingredient pairs with training/proposed points in functions/f6/f6 - week 10.ipynb
- [X] T025 [P] [US3] Append convergence plot cell — running best objective with proposed point in functions/f6/f6 - week 10.ipynb
- [X] T026 [US3] Run all new cells in F6 notebook and verify: submission format x1-x2-x3-x4-x5 in [0,0.999999], milk≥0.12 satisfied, not duplicate, visualisations render in functions/f6/f6 - week 10.ipynb

**Checkpoint**: F6 notebook produces a valid submission — US3 independently verifiable

---

## Phase 6: User Story 4 — F7 Exploration Boost with MC Dropout (Priority: P2)

**Goal**: Shift acquisition to 50/50 mean/EI, reduce STEEPNESS to 0.02, expand candidates to 50k, increase MC dropout passes to ≥50.

**Independent Test**: Run the F7 week 10 notebook end-to-end; verify EI weight 50%, STEEPNESS=0.02, 50k candidates, MC dropout ≥50 passes, valid 6D submission.

### Implementation for User Story 4

- [X] T027 [P] [US4] Append markdown cell "Step 6 — Week 10 Optimisation Run" with 4 strategy changes and rationale in functions/f7/f7 - week 10.ipynb
- [X] T028 [P] [US4] Append imports & configuration cell with all constants (N_DIMS=6, DROPOUT=0.05, LR=0.005, EPOCHS=200, MC_SAMPLES=50, N_CANDIDATES=50000, EXPLOITATION_WEIGHT=0.5, STEEPNESS=0.02, FLOOR=0.02) in functions/f7/f7 - week 10.ipynb
- [X] T029 [US4] Append data preparation cell — load week 10 data (6D, 40 obs), manual z-score normalisation, convert to float64 tensors in functions/f7/f7 - week 10.ipynb
- [X] T030 [US4] Append NN training cell — SurrogateNN (6→5→5→1, dropout=0.05), SGD lr=0.005, 200 epochs, print loss summary in functions/f7/f7 - week 10.ipynb
- [X] T031 [US4] Append acquisition & selection cell — generate 50k random candidates, MC dropout ≥50 forward passes for mean+std, compute 50/50 mean/EI blend, apply interior penalty (STEEPNESS=0.02, FLOOR=0.02), select best candidate, duplicate check, formatted submission in functions/f7/f7 - week 10.ipynb
- [X] T032 [P] [US4] Append 2D contour visualisation cell — NN prediction across input pairs with training/proposed points in functions/f7/f7 - week 10.ipynb
- [X] T033 [P] [US4] Append convergence plot cell — running best objective with proposed point in functions/f7/f7 - week 10.ipynb
- [X] T034 [US4] Run all new cells in F7 notebook and verify: submission format x1-x2-x3-x4-x5-x6 in [0,0.999999], not duplicate, visualisations render in functions/f7/f7 - week 10.ipynb

**Checkpoint**: F7 notebook produces a valid submission — US4 independently verifiable

---

## Phase 7: User Story 5 — F8 Exploration & Numerical Stability (Priority: P2)

**Goal**: Switch qEI→qLogNEI, increase MC samples to 512, increase raw_samples to 8192, verify noise_lb=1e-7 stability.

**Independent Test**: Run the F8 week 10 notebook end-to-end; verify qLogNEI, MC samples=512, raw_samples=8192, noise_lb=1e-7 stable, valid 8D submission.

### Implementation for User Story 5

- [X] T035 [P] [US5] Append markdown cell "Step 6 — Week 10 Optimisation Run" with 5 strategy changes and rationale in functions/f8/f8 - week 10.ipynb
- [X] T036 [P] [US5] Append imports & configuration cell with all constants (KERNEL_NU=2.5, ARD_NUM_DIMS=8, NOISE_LB=1e-7, N_MLL_RESTARTS=30, MC_SAMPLES=512, Q_BATCH=1, NUM_RESTARTS=30, RAW_SAMPLES=8192) in functions/f8/f8 - week 10.ipynb
- [X] T037 [US5] Append data preparation cell — load week 10 data (8D, 50 obs), convert to float64 tensors, print shape and output range in functions/f8/f8 - week 10.ipynb
- [X] T038 [US5] Append GP fitting cell — SingleTaskGP Matérn-2.5 ARD (8D), GaussianLikelihood(noise_constraint=GreaterThan(1e-7)), Standardize(m=1), ≥30 MLL restarts, post-fit Cholesky stability check with warning in functions/f8/f8 - week 10.ipynb
- [X] T039 [US5] Append acquisition & selection cell — qLogNoisyExpectedImprovement (replacing qEI), SobolQMCNormalSampler(512), optimize_acqf(num_restarts=30, raw_samples=8192), clamp to [0,0.999999], duplicate check, formatted submission in functions/f8/f8 - week 10.ipynb
- [X] T040 [P] [US5] Append 2D contour visualisation cell — GP posterior mean across input pairs with training/proposed points in functions/f8/f8 - week 10.ipynb
- [X] T041 [P] [US5] Append convergence plot cell — running best objective with proposed point in functions/f8/f8 - week 10.ipynb
- [X] T042 [US5] Run all new cells in F8 notebook and verify: submission format x1-x2-x3-x4-x5-x6-x7-x8 in [0,0.999999], noise_lb stable, not duplicate, visualisations render in functions/f8/f8 - week 10.ipynb

**Checkpoint**: F8 notebook produces a valid submission — US5 independently verifiable

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all 5 notebooks

- [X] T043 Validate all 5 submissions (SC-001 through SC-006): end-to-end execution, GP convergence, format correctness, visualisations, no duplicates, hyperparameter documentation
- [X] T044 Run quickstart.md validation checklist for all notebooks

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify prerequisites immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — confirms infrastructure
- **User Stories (Phase 3–7)**: All depend on Phase 2 completion
  - **US1 (F4)** and **US2 (F5)** are P1 — implement first
  - **US3 (F6)**, **US4 (F7)**, **US5 (F8)** are P2 — implement after P1 or in parallel
  - All 5 user stories are **independent** — no cross-function dependencies
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (F4, P1)**: Can start after Phase 2 — no dependencies on other stories
- **US2 (F5, P1)**: Can start after Phase 2 — no dependencies on other stories
- **US3 (F6, P2)**: Can start after Phase 2 — no dependencies on other stories
- **US4 (F7, P2)**: Can start after Phase 2 — no dependencies on other stories
- **US5 (F8, P2)**: Can start after Phase 2 — no dependencies on other stories

### Within Each User Story

1. Markdown rationale cell (T00x) — first
2. Imports & config cell (T00x) — second
3. Data preparation cell — depends on config
4. Surrogate fitting cell — depends on data
5. Acquisition & selection cell — depends on fitted model
6. Visualisation cells (contour + convergence) — [P] can run in parallel, depend on acquisition
7. Run & verify — last, depends on all cells

### Parallel Opportunities per Story

```
T003 (markdown) ──────────────────────────────────────────────────┐
T004 (config)   ──┐                                              │
                  ├─→ T005 (data) → T006 (GP fit) → T007 (acq) ─┤
                  │                                              ├─→ T010 (verify)
                                                    T008 (vis) ──┤
                                                    T009 (conv) ─┘
```

All 5 user stories can execute in parallel since they modify independent notebooks:

```
Phase 2 ─┬─→ US1 (F4) ─→ T003–T010
         ├─→ US2 (F5) ─→ T011–T018  (parallel with US1)
         ├─→ US3 (F6) ─→ T019–T026  (parallel with US1, US2)
         ├─→ US4 (F7) ─→ T027–T034  (parallel with all above)
         └─→ US5 (F8) ─→ T035–T042  (parallel with all above)
                              │
                              └─→ Phase 8 (T043–T044)
```

---

## Implementation Strategy

### MVP Scope

**MVP = User Story 1 (F4) only** — This addresses the highest-impact issue (MFGP/single-fidelity mismatch causing 7 consecutive stalling submissions). Implementing F4 alone validates the SFGP pattern that F5, F6, and F8 already use, confirming the migration approach before scaling to other functions.

### Incremental Delivery

1. **F4 (US1, P1)**: Fundamental surrogate fix — validates SFGP + qLogNEI pattern
2. **F5 (US2, P1)**: Transform simplification — validates log + relaxed selection pattern
3. **F8 (US5, P2)**: qEI→qLogNEI switch — similar GP pattern to F4
4. **F6 (US3, P2)**: Conservative tuning — lowest risk, smallest changes
5. **F7 (US4, P2)**: NN + MC dropout — most complex, independent of GP functions

### Task Summary

| Phase | Story | Tasks | Parallelizable |
|-------|-------|-------|----------------|
| 1 Setup | — | T001–T002 | 2/2 |
| 2 Foundation | — | — | — |
| 3 US1 (F4) | P1 | T003–T010 | 0/8 (sequential notebook cells) |
| 4 US2 (F5) | P1 | T011–T018 | 4/8 |
| 5 US3 (F6) | P2 | T019–T026 | 4/8 |
| 6 US4 (F7) | P2 | T027–T034 | 4/8 |
| 7 US5 (F8) | P2 | T035–T042 | 4/8 |
| 8 Polish | — | T043–T044 | 0/2 |
| **Total** | | **44 tasks** | **18 parallelizable** |
