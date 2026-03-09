# Tasks: F1 Week 9 — log Transform (No Penalties)

**Input**: Design documents from `/specs/023-f1-week9-log/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Tests**: Not required (constitution: no unit tests)

**Organization**: Tasks are grouped by user story to enable incremental implementation. All tasks target a single file: `functions/f1/f1 - week 9.ipynb`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- All tasks target `functions/f1/f1 - week 9.ipynb` (single notebook deliverable)

---

## Phase 1: Setup

**Purpose**: Create the notebook file and establish imports and hyperparameter configuration.

- [X] T001 Create new Jupyter notebook at `functions/f1/f1 - week 9.ipynb` with a title markdown cell: "# F1 — Week 9: Hurdle Model with log Transform (No Penalties)"
- [X] T002 Add imports code cell: numpy, pandas, matplotlib.pyplot, sklearn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), scipy.spatial.distance (pdist, squareform) in `functions/f1/f1 - week 9.ipynb`
- [X] T003 Add hyperparameters markdown+code cell pair: document all constants with rationale (changes from Week 8: log1p → log, local penalization removed, interior penalty removed), then define N_INITIAL=10, N_TOTAL=19, MIN_POSITIVE=3, C_STAGE1=1.0, N_ESTIMATORS=100, MAX_DEPTH=3, KAPPA=3.0, N_CANDIDATES=20000, GRID_RES=50 in `functions/f1/f1 - week 9.ipynb`

---

## Phase 2: User Story 1 — Load and Validate Week 9 Data (Priority: P1) 🎯 MVP

**Goal**: Load all 19 F1 observations (10 initial + 9 submissions) from Week 9 .npy files, validate bounds and integrity, display in tabular form with current best identified.

**Independent Test**: Run the data-loading cells — they should display 19 samples in tabular format with no NaN or out-of-range values, and the current best observation identified.

### Implementation for User Story 1

- [X] T004 [US1] Add data loading code cell: load `data/f1/updated_inputs - Week 9.npy` and `data/f1/updated_outputs - Week 9.npy` using np.load, with a clear FileNotFoundError message if files are missing, in `functions/f1/f1 - week 9.ipynb`
- [X] T005 [US1] Add data validation code cell: assert shape is (19, 2) for inputs, (19,) for outputs; check all inputs in [0.0, 1.0]; check no NaN or Inf in outputs; print warning if sample count differs from N_TOTAL in `functions/f1/f1 - week 9.ipynb`
- [X] T006 [US1] Add tabular display code cell: show all 19 samples with columns (x1, x2, y) using pandas DataFrame; identify and print current best observation value and location in `functions/f1/f1 - week 9.ipynb`
- [X] T007 [US1] Add binary labels code cell: compute y_binary = y > 0; split X_initial = X[:N_INITIAL], X_submissions = X[N_INITIAL:]; print count of positive, zero, and negative outputs in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Data loaded, validated, and displayed — all subsequent stories can proceed.

---

## Phase 3: User Story 2 — Fit Hurdle Model with log Transform (Priority: P1)

**Goal**: Train the two-stage hurdle model with the key change: Stage 2 RF regressor trains on `np.log(y_pos)` instead of `np.log1p(y_pos)`, producing predictions in log-space (~-565 to ~-35).

**Independent Test**: After surrogate cells execute, the RF regressor is trained on log-transformed positive outputs and predictions can be queried.

### Implementation for User Story 2

- [X] T008 [US2] Add Stage 1 markdown header and code cell: train CalibratedClassifierCV(LogisticRegression(C=C_STAGE1, class_weight='balanced', max_iter=1000), cv=3, method='sigmoid') on X with y_binary targets; print training accuracy in `functions/f1/f1 - week 9.ipynb`
- [X] T009 [US2] Add Stage 2 markdown header and code cell: extract y_pos = y[y > 0] and X_pos = X[y > 0]; compute y_pos_log = np.log(y_pos) (NOT np.log1p); train RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42) on X_pos with y_pos_log targets; back-transformation uses np.exp(mu) (NOT np.expm1); print training R² and target range; include MIN_POSITIVE fallback check (if N_pos < MIN_POSITIVE: set FALLBACK_MODE=True, mu=0, sigma=1) in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Hurdle model fitted — acquisition and visualisation stories can proceed.

---

## Phase 4: User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

**Goal**: Evaluate simplified weighted UCB acquisition (no local penalization, no interior penalty) over 20,000 random candidates, select the best candidate, and output a formatted submission string.

**Independent Test**: The notebook outputs a formatted string in 0.xxxxxx-0.xxxxxx format with values clipped to [0.0, 0.999999].

### Implementation for User Story 3

- [X] T010 [US3] Add acquisition function markdown header and code cell: generate N_CANDIDATES random candidates in [0,1]²; compute p(x) from Stage 1 classifier; compute mu(x) and sigma_RF(x) from Stage 2 RF (std across individual tree predictions); compute weighted UCB a(x) = p(x)·mu(x) + KAPPA·p(x)·sigma_RF(x); select argmax candidate as next_x (no local penalization or interior penalty) in `functions/f1/f1 - week 9.ipynb`
- [X] T011 [US3] Add submission formatting code cell: clip next_x to [0.0, 0.999999]; format as "0.xxxxxx-0.xxxxxx" with 6 decimal places; compute minimum Euclidean distance to all existing observations and print warning if < 0.05; print submission string prominently in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Submission query produced — primary deliverable complete.

---

## Phase 5: User Story 4 — Visualise Surrogate with log Transform (Priority: P1)

**Goal**: Produce 3-panel contour visualisation in log-space and convergence plot, with training points colour-coded and the proposed next point marked.

**Independent Test**: Three-panel contour plots render with correct colour coding and log-space values (~-565 to ~-35); convergence plot shows running maximum.

### Implementation for User Story 4

- [X] T012 [US4] Add 3-panel contour plot code cell: create GRID_RES×GRID_RES meshgrid over [0,1]²; Panel 1: hurdle mean in log-space (p(x)·mu(x)); Panel 2: hurdle uncertainty in log-space (p(x)·sigma_RF(x)); Panel 3: raw weighted UCB acquisition surface titled "Acquisition (Weighted UCB)"; overlay initial samples (blue), weekly submissions (orange), and proposed next point (green star); add legend with three entries in `functions/f1/f1 - week 9.ipynb`
- [X] T013 [US4] Add convergence plot code cell: plot running maximum of y across all 19 observations; colour initial samples (indices 0-9) in blue, weekly submissions (indices 10-18) in orange; mark boundary between initial and submissions; add legend in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: All visualisations rendered — core functionality complete (US1-US4).

---

## Phase 6: User Story 5 — Performance Evaluation (Priority: P2)

**Goal**: Quantitatively assess convergence, exploration quality, and strategy effectiveness with actionable recommendations.

**Independent Test**: Code cells compute convergence metrics, stalling flag, and exploration spread; a final markdown cell interprets results.

### Implementation for User Story 5

- [X] T014 [US5] Add performance evaluation markdown header and convergence metrics code cell: compute best-value trajectory for submissions only; compute per-submission improvement deltas; compute stalling flag (True if no improvement in last 3 submissions) in `functions/f1/f1 - week 9.ipynb`
- [X] T015 [US5] Add exploration spread code cell: compute mean pairwise distance among submission points using scipy pdist; compute max nearest-neighbour distance; display both metrics in `functions/f1/f1 - week 9.ipynb`
- [X] T016 [US5] Add strategy recommendation markdown cell: interpret convergence metrics, stalling flag, and exploration spread; if stalling, propose at least one actionable strategy change for Week 10; if not stalling, confirm current strategy is performing well in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Full performance analysis complete — notebook is feature-complete.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all stories to ensure notebook integrity.

- [X] T017 Run full notebook end-to-end and verify all cells execute without errors in `functions/f1/f1 - week 9.ipynb`
- [X] T018 Validate against quickstart.md checklist: RF trains on log-space targets (~-565 to ~-35), contour shows meaningful variation, no local penalization or interior penalty in acquisition, Panel 3 titled "Acquisition (Weighted UCB)", submission formatted correctly, all values clipped in `functions/f1/f1 - week 9.ipynb`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 — Data Loading (Phase 2)**: Depends on Setup — BLOCKS all subsequent stories
- **US2 — Hurdle Model (Phase 3)**: Depends on US1 (needs loaded data and binary labels)
- **US3 — Acquisition (Phase 4)**: Depends on US2 (needs fitted classifier and RF)
- **US4 — Visualisation (Phase 5)**: Depends on US3 (needs acquisition results and proposed point)
- **US5 — Performance Eval (Phase 6)**: Depends on US1 only — independent of US2-US4
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

```
Setup (T001-T003)
  └──→ US1: Data Loading (T004-T007)
         ├──→ US2: Hurdle Model (T008-T009)
         │      └──→ US3: Acquisition (T010-T011)
         │             └──→ US4: Visualisation (T012-T013)
         └──→ US5: Performance Eval (T014-T016) [independent of US2-US4]
```

### Parallel Opportunities

- **US5 (T014-T016)** can be implemented in parallel with US2-US4 since it only depends on US1
- Within the notebook, all cells execute sequentially at runtime
- No [P] markers: all tasks target the same single notebook file

---

## Implementation Strategy

### MVP First (User Stories 1-3)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: US1 — Data Loading (T004-T007)
3. Complete Phase 3: US2 — Hurdle Model (T008-T009)
4. Complete Phase 4: US3 — Acquisition (T010-T011)
5. **STOP and VALIDATE**: Notebook produces a valid submission query

### Incremental Delivery

1. Setup + US1 → Data loads and validates ✓
2. + US2 → Hurdle model fits with log transform ✓
3. + US3 → Submission query produced (**MVP complete**) ✓
4. + US4 → Visualisations rendered ✓
5. + US5 → Performance evaluation complete ✓
6. Polish → End-to-end validation ✓

---

## Notes

- All tasks target a single file: `functions/f1/f1 - week 9.ipynb`
- No [P] markers because all tasks write to the same notebook
- Key changes: `np.log1p(y_pos)` → `np.log(y_pos)`, `np.expm1(mu)` → `np.exp(mu)`, local penalization and interior penalty removed
- Week 9 data files must exist in `data/f1/` before notebook execution
- Reference: `functions/f1/f1 - week 8.ipynb` (DO NOT MODIFY)
- Contour panels display in log-space per clarification (values ~-565 to ~-35)
- Panel 3 titled "Acquisition (Weighted UCB)" (no penalty mask)
