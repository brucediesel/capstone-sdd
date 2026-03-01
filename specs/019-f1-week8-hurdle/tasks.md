# Tasks: F1 Week 8 — Hurdle Model Bayesian Optimisation Iteration

**Input**: Design documents from `/specs/019-f1-week8-hurdle/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not required (per constitution: "No unit tests are required").

**Organization**: Tasks are grouped by user story. All tasks target a single notebook file (`functions/f1/f1 - week 8.ipynb`), so parallelism within a story is limited to cells that can be written independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create the notebook file and establish the skeleton structure

- [x] T001 Create new notebook `functions/f1/f1 - week 8.ipynb` with title markdown cell introducing Week 8 iteration
- [x] T002 Add imports code cell with numpy, matplotlib.pyplot, sklearn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor) in `functions/f1/f1 - week 8.ipynb`
- [x] T003 Add hyperparameter documentation markdown cell with parameter table and rationale in `functions/f1/f1 - week 8.ipynb`
- [x] T004 Add hyperparameter constants code cell (C_STAGE1=1.0, N_ESTIMATORS=100, MAX_DEPTH=3, KAPPA=3.0, PENALTY_RADIUS=0.15, N_CANDIDATES=20000, GRID_RES=50, MIN_POSITIVE=3, STEEPNESS=0.1, FLOOR=0.01) in `functions/f1/f1 - week 8.ipynb`

**Checkpoint**: Notebook skeleton exists with all imports and hyperparameters defined

---

## Phase 2: User Story 1 — Load and Validate Week 8 Data (Priority: P1) 🎯 MVP

**Goal**: Load 18 observations from Week 8 .npy files, validate ranges, display in tabular format, identify best observation.

**Independent Test**: Run data-loading cells — 18 input rows (2 cols) and 18 output values displayed with no NaN or out-of-range values.

### Implementation for User Story 1

- [x] T005 [US1] Add data loading code cell: load `../../data/f1/updated_inputs - Week 8.npy` and `../../data/f1/updated_outputs - Week 8.npy`, validate shapes (18,2) and (18,), check input range [0.0,1.0], check for NaN in `functions/f1/f1 - week 8.ipynb`
- [x] T006 [US1] Add derived binary labels code: compute y_binary (y > 0), n_positive, X_pos, y_pos, y_pos_log (log1p), FALLBACK_MODE flag, print summary in `functions/f1/f1 - week 8.ipynb`
- [x] T007 [US1] Add data display markdown cell explaining the dataset and a code cell showing tabular display of all 18 inputs/outputs with best observation highlighted in `functions/f1/f1 - week 8.ipynb`

**Checkpoint**: Data loaded, validated, and displayed — fallback mode status determined

---

## Phase 3: User Story 2 — Fit Hurdle Model Surrogate (Priority: P1)

**Goal**: Train Stage 1 (calibrated logistic classifier) and Stage 2 (random forest regressor on log1p(y)) on Week 8 data, with fallback to pure exploration if fewer than 3 positive samples.

**Independent Test**: Classifier and regressor fitted without errors; per-point predictions queryable for arbitrary inputs.

### Implementation for User Story 2

- [x] T008 [US2] Add Stage 1 markdown explanation cell and code cell: fit LogisticRegression(C=C_STAGE1, max_iter=1000, class_weight='balanced') wrapped in CalibratedClassifierCV(cv=3, method='sigmoid') on X and y_binary in `functions/f1/f1 - week 8.ipynb`
- [x] T009 [US2] Add Stage 2 markdown explanation cell and code cell: if not FALLBACK_MODE, fit RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42) on X_pos and y_pos_log; else set stage2_rf=None and print fallback message in `functions/f1/f1 - week 8.ipynb`

**Checkpoint**: Surrogate model fitted (or fallback mode active and documented)

---

## Phase 4: User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

**Goal**: Maximise weighted UCB acquisition with local penalization and interior penalty over 20000 random candidates. Select best candidate, validate constraints, format submission string.

**Independent Test**: Notebook outputs a formatted string `0.xxxxxx-0.xxxxxx` with values in [0.0, 0.999999] and min distance ≥ 0.05 from existing points.

### Implementation for User Story 3

- [x] T010 [US3] Add acquisition function markdown cell explaining weighted UCB formula, local penalization, and interior penalty in `functions/f1/f1 - week 8.ipynb`
- [x] T011 [US3] Add acquisition code cell: generate N_CANDIDATES random candidates in [0,1]², compute p_cand from Stage 1, compute mu/sigma from Stage 2 (or fallback values), evaluate weighted UCB, apply local penalization over all 18 points, apply interior penalty, select best candidate in `functions/f1/f1 - week 8.ipynb`
- [x] T012 [US3] Add validation and submission code cell: validate proposed point coordinates in [0.0, 0.999999], validate min Euclidean distance ≥ 0.05 from all 18 existing points, clip and format as `0.xxxxxx-0.xxxxxx`, print submission query in `functions/f1/f1 - week 8.ipynb`

**Checkpoint**: Next sample point proposed, validated, and formatted for submission

---

## Phase 5: User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

**Goal**: Produce 3-panel contour plots (hurdle mean, uncertainty, penalised acquisition) and a convergence plot showing running maximum across all 18 observations.

**Independent Test**: Four plots generated — legible, correctly annotated with training points (colour-coded) and proposed candidate.

### Implementation for User Story 4

- [x] T013 [US4] Add 3-panel contour plot code cell: create 50×50 grid over [0, 0.999999]², compute hurdle mean (p·expm1(μ)), hurdle uncertainty (p·σ_RF), and penalised acquisition on grid; plot with contourf (viridis, YlOrRd, plasma); overlay positive points (red), non-positive points (blue), proposed point (yellow star) in `functions/f1/f1 - week 8.ipynb`
- [x] T014 [US4] Add convergence plot code cell: compute running maximum of observed outputs across all 18 points, plot as blue line with markers, add vertical line at observation 10.5 (initial/weekly boundary), annotate best value in `functions/f1/f1 - week 8.ipynb`

**Checkpoint**: All visualisations generated — notebook complete and ready for end-to-end execution

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup

- [x] T015 Verify notebook executes end-to-end without errors by running all cells in `functions/f1/f1 - week 8.ipynb`
- [x] T016 Verify original `functions/f1/f1.ipynb` has NOT been modified (no changes to existing files)
- [x] T017 Run quickstart.md validation: confirm submission query format, data table presence, all 4 plots generated in `functions/f1/f1 - week 8.ipynb`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 2)**: Depends on Setup (T001–T004)
- **US2 (Phase 3)**: Depends on US1 (T005–T007) — needs loaded data and binary labels
- **US3 (Phase 4)**: Depends on US2 (T008–T009) — needs fitted surrogate
- **US4 (Phase 5)**: Depends on US3 (T010–T012) — needs candidate selection for overlay
- **Polish (Phase 6)**: Depends on all user stories complete

### Within Each Phase

All tasks within each phase are **sequential** — they build cells in the same notebook file in order. No parallelism within a phase.

### Parallel Opportunities

Limited parallelism because all tasks target the same notebook file. However:

- T003 (hyperparameter markdown) and T004 (hyperparameter code) can be written together as a pair
- T008 (Stage 1) and T009 (Stage 2) are logically independent but Stage 2 depends on data derived in US1
- T013 (contour plots) and T014 (convergence plot) produce separate outputs but both need acquisition results from US3

**Recommended approach**: Execute sequentially in task order (T001 → T017) since all tasks target a single file.

---

## Implementation Strategy

### MVP First (User Stories 1–3)

1. Complete Setup (T001–T004) → Notebook skeleton
2. Complete US1 (T005–T007) → Data loaded and validated
3. Complete US2 (T008–T009) → Surrogate fitted
4. Complete US3 (T010–T012) → **Submission query produced — primary deliverable**
5. **STOP and VALIDATE**: Submission string is ready for use

### Full Delivery (Add US4)

6. Complete US4 (T013–T014) → Visualisations added
7. Complete Polish (T015–T017) → End-to-end validation

### Incremental Delivery

Each user story adds a complete slice of functionality:
- After US1: Data is loaded — can verify correctness
- After US2: Surrogate is fitted — can inspect model
- After US3: **Submission ready** — primary deliverable complete
- After US4: **Capstone-quality notebook** with all required visualisations

---

## Notes

- All 17 tasks target a single file: `functions/f1/f1 - week 8.ipynb`
- No unit tests (per constitution)
- No API contracts (notebook-only deliverable)
- Commit after each phase to preserve progress
- The fallback mode (FALLBACK_MODE) path must be handled in T009 and T011 — research.md confirms Week 8 data has output range [-0.003606, 0.000000], meaning fallback mode is likely active
