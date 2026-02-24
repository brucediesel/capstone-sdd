# Tasks: F6 Week 7 — SFGP Matérn-1.5 + NEI (Exploration Focus)

**Input**: Design documents from `/specs/012-f6-sfgp-nei/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/week7-cells.md, quickstart.md

**Tests**: Not requested — no test tasks included.

**Organization**: Tasks grouped by user story. All implementation is in `functions/f6/f6.ipynb` (cells 50–57, replacing existing first-attempt cells).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- All file paths relative to repository root

---

## Phase 1: Setup

**Purpose**: Verify branch, environment, data, and notebook state before editing cells

- [X] T001 Verify branch `012-f6-sfgp-nei` is checked out and pyenv `sdd-dev` is active
- [X] T002 Verify data files exist: `data/f6/updated_inputs - Week 7.npy` (27×5) and `data/f6/updated_outputs - Week 7.npy` (27,)
- [X] T003 Verify notebook `functions/f6/f6.ipynb` has 58 cells; identify Week 7 cells 50–57 to replace

**Checkpoint**: Environment validated — cell editing can begin

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Insert the section header cell that all subsequent cells depend on for context

**⚠️ CRITICAL**: Cell 50 must be in place before any other Week 7 cells

- [X] T004 [US1] Edit Cell 50 (markdown) in `functions/f6/f6.ipynb`: Week 7 header with `## Week 7 — SFGP Matérn-1.5 + NEI` title, strategy-change rationale, and comparison table (Week 6: NN+MCDropout+UCB κ=0.5 → Week 7: SFGP+NEI q=4, exploration focus) per FR-020
- [X] T005 [US1] Edit Cell 51 (code) in `functions/f6/f6.ipynb`: imports (torch, gpytorch, botorch, numpy, matplotlib, copy) + load `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy` via `../../data/f6/` + validate shapes (27,5)/(27,) + print stats (sample count, ranges, best value/index) per FR-001, FR-002

**Checkpoint**: Foundation ready — header and data available for all downstream cells

---

## Phase 3: User Story 1 — SFGP+NEI Submission Query (Priority: P1) 🎯 MVP

**Goal**: Train SFGP with Matérn-1.5 + aggressive noise floor, run NEI with feasibility-constrained bounds, select candidate via distance-based strategy, produce submission query

**Independent Test**: Execute cells 50–51 (setup) then 53–54, 57 (training, acquisition, submission) — produces a valid `x1-x2-x3-x4-x5` query with x4≥0.10

### Implementation for User Story 1

- [X] T006 [US1] Edit Cell 53 (code) in `functions/f6/f6.ipynb`: GP training — convert data to double tensors, 15-restart MLL loop (seed 0..14), each restart: construct SingleTaskGP with ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5)), GaussianLikelihood(noise_constraint=GreaterThan(1e-2)), default Standardize(m=1), init ℓ=0.5/noise=0.2/outputscale=1.0, fit_gpytorch_mll, score, deepcopy best. Print per-restart neg_MLL + fitted HPs (ℓ₁–ℓ₅, σ²_f, σ²_n). Per FR-003 through FR-008, RES-003
- [X] T007 [US1] Edit Cell 54 (code) in `functions/f6/f6.ipynb`: NEI acquisition — construct qLogNoisyExpectedImprovement(model, X_baseline, sampler=SobolQMCNormalSampler(512), prune_baseline=True), optimize_acqf with q=4, num_restarts=50, raw_samples=3000, feasibility-constrained bounds lower=[0.01,0.01,0.01,0.01,0.10] upper=[1,1,1,1,1], clamp to [0,0.999999]. Distance-based selection: posterior means (original space, auto-untransformed), filter mean≥median, select max min-distance to X_train. Print candidates, means, distances, selected point. Per FR-010 through FR-013, RES-005
- [X] T008 [US1] Edit Cell 57 (code) in `functions/f6/f6.ipynb`: format best_point as `x1-x2-x3-x4-x5` (6 decimal places, clamped [0,0.999999]), validate with assertions (5 parts, float-parseable, in range), print query prominently + summary table (surrogate, kernel, acquisition, q, strategy, fitted ℓ₁–ℓ₅, σ²_f, σ²_n, selected mean). Per FR-017, FR-018
- [X] T009 [US1] Execute cells 51, 53, 54, 57 in sequence in `functions/f6/f6.ipynb` and verify: fitted noise ≥ 1e-2, 5 distinct lengthscales, all candidate x4 values ≥ 0.10, submission format matches `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`, all values in [0,0.999999]. Per SC-001 through SC-005

**Checkpoint**: US1 complete — valid submission query produced with x4 ≥ 0.10

---

## Phase 4: User Story 2 — Visualise Surrogate and Convergence (Priority: P2)

**Goal**: 3-panel surrogate plot (mean/std/relevance) + convergence plot matching Week 6 layout

**Independent Test**: Execute cells 51, 53 (data+training) then 55, 56 — plots render without errors

### Implementation for User Story 2

- [X] T010 [P] [US2] Edit Cell 55 (code) in `functions/f6/f6.ipynb`: 3-panel surrogate visualisation — identify top-2 dims (shortest ℓ), build 80×80 grid fixing other 3 dims at best_point values, compute posterior mean+std (original space, auto-untransformed), Panel 1: mean contour + red observed points + magenta star proposed point + colourbar, Panel 2: std contour + points + colourbar, Panel 3: 1/ℓ normalised bar chart (5 bars x0–x4). 18×5 figure. Per FR-014, FR-015
- [X] T011 [P] [US2] Edit Cell 56 (code) in `functions/f6/f6.ipynb`: convergence plot — running_best = np.maximum.accumulate(y_raw), plot vs observation number, vertical dashed red line at x=26.5 (Week 6→7 boundary), legend, Week 7 title. Print running best at Week 6 end and Week 7. Per FR-016
- [X] T012 [US2] Execute cells 55 and 56 in `functions/f6/f6.ipynb` and verify: 3-panel figure renders with colorbars/labels/5 bars, convergence plot shows non-decreasing curve with boundary at 26.5. Per SC-007, SC-008

**Checkpoint**: US2 complete — all diagnostic plots match Week 6 layout

---

## Phase 5: User Story 3 — Hyperparameter Documentation (Priority: P3)

**Goal**: Markdown table documenting all 14+ hyperparameters with rationale before training cell

**Independent Test**: Cell 52 renders independently as formatted markdown

### Implementation for User Story 3

- [X] T013 [US3] Edit Cell 52 (markdown) in `functions/f6/f6.ipynb`: hyperparameter table with 14 entries — Kernel (Matérn-1.5), ARD (True, 5 ℓ), ℓ init (0.5), outputscale init (1.0), noise init (0.2), noise floor (1e-2), outcome transform (Standardize(m=1)), MLL restarts (15), acquisition (qLogNEI), q (4), raw_samples (3000), num_restarts (50), selection (distance-based, mean≥median + max distance), bounds (feasibility-constrained: x4≥0.10, others≥0.01). Each with plain-English rationale. Per FR-009

**Checkpoint**: US3 complete — all HPs documented with rationale

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all cells and commit

- [X] T014 Execute all Week 7 cells (50–57) top-to-bottom in `functions/f6/f6.ipynb` and verify full pipeline: no errors, submission query valid, plots render, noise ≥ 1e-2, x4 ≥ 0.10. Per SC-001 through SC-008
- [X] T015 Commit all changes to branch `012-f6-sfgp-nei` with message `fix(f6): feasibility-constrained bounds and exploration-promoting noise for week 7`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify environment first
- **Foundational (Phase 2)**: Depends on Setup — Cell 50 header + Cell 51 data loading BLOCK all user stories
- **US1 (Phase 3)**: Depends on Foundational — Cells 53, 54, 57 (training → acquisition → submission)
- **US2 (Phase 4)**: Depends on US1 Cell 53 (trained model) — Cells 55, 56 (visualization)
- **US3 (Phase 5)**: Depends only on Foundational — Cell 52 is markdown, no code dependencies
- **Polish (Phase 6)**: Depends on all user stories

### Within Each User Story

- US1: T006 (training) → T007 (acquisition, needs model) → T008 (submission, needs best_point) → T009 (validate)
- US2: T010 + T011 can run in parallel (different cells, both need model from T006) → T012 (validate)
- US3: T013 standalone (markdown only)

### Parallel Opportunities

- T010 and T011 can run in parallel (Cell 55 and Cell 56 are independent once model is trained)
- US3 (T013) can run in parallel with US1 (T006–T009) since it's a markdown cell
- T001, T002, T003 are verification steps that can be parallelized

---

## Parallel Example: User Story 2

```text
# After US1 T006 (training) completes, launch both visualisation cells together:
T010: "Edit Cell 55 — 3-panel surrogate visualisation in functions/f6/f6.ipynb"
T011: "Edit Cell 56 — convergence plot in functions/f6/f6.ipynb"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T005) — header + data loading
3. Complete Phase 3: US1 (T006–T009) — training + acquisition + submission
4. **STOP and VALIDATE**: Execute cells 51, 53, 54, 57 — verify submission query has x4 ≥ 0.10
5. Submission ready at this point

### Incremental Delivery

1. Setup + Foundational → Data loaded, header in place
2. US1 → Training + acquisition + submission query (MVP!)
3. US3 → HP documentation markdown (can be done alongside US1)
4. US2 → Visualization plots (requires trained model from US1)
5. Polish → Full top-to-bottom validation + commit
