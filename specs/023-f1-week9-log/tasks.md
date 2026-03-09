# Tasks: F1 Week 9 — Hurdle Model with log Transform (No Penalties)

**Input**: Design documents from `/specs/023-f1-week9-log/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Tests**: Not required (Constitution Principle I — no unit tests)

**Organization**: Tasks are grouped by user story to enable incremental implementation. All tasks target a single file: `functions/f1/f1 - week 9.ipynb`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- All tasks target `functions/f1/f1 - week 9.ipynb` (single notebook deliverable)

---

## Phase 1: Setup

**Purpose**: Create the notebook file, establish imports, and configure all hyperparameters.

- [X] T001 Create new Jupyter notebook at `functions/f1/f1 - week 9.ipynb` with title markdown cell documenting strategy (hurdle model, log transform, no penalties, KAPPA=0.5 exploitation-focused), key changes from Week 8, and the weighted UCB formula $a(x) = p(x) \cdot \mu(x) + \kappa \cdot p(x) \cdot \sigma_{\text{RF}}(x)$
- [X] T002 Add imports code cell: numpy, matplotlib.pyplot, warnings, scipy.spatial.distance (pdist, squareform), sklearn.linear_model (LogisticRegression), sklearn.calibration (CalibratedClassifierCV), sklearn.ensemble (RandomForestRegressor) in `functions/f1/f1 - week 9.ipynb`
- [X] T003 Add hyperparameters markdown table and code cell: document all constants with rationale in markdown table (C_STAGE1=1.0, N_ESTIMATORS=100, MAX_DEPTH=3, KAPPA=0.5, N_CANDIDATES=20000, GRID_RES=50, MIN_POSITIVE=3); define as named constants in code cell including N_INITIAL=10, N_TOTAL=19, N_SUBMISSIONS=9, STALLING_CONSECUTIVE_THRESHOLD=3, STALLING_RELATIVE_THRESHOLD=0.05; print all values for verification in `functions/f1/f1 - week 9.ipynb`

---

## Phase 2: User Story 1 — Load and Validate Week 9 Data (Priority: P1) 🎯 MVP

**Goal**: Load all 19 F1 observations (10 initial + 9 submissions) from Week 9 .npy files, validate bounds and integrity, derive binary labels for the hurdle model, and display data in tabular form with current best identified.

**Independent Test**: Run the data-loading cells — they should display 19 samples in tabular format with no NaN or out-of-range values, and the current best observation identified.

### Implementation for User Story 1

- [X] T004 [US1] Add Step 1 markdown header and data loading code cell: load `../../data/f1/updated_inputs - Week 9.npy` and `../../data/f1/updated_outputs - Week 9.npy` via np.load; assert shape (N_TOTAL, N_DIMS) for inputs and (N_TOTAL,) for outputs; assert inputs in [0.0, 1.0] and no NaN/Inf in outputs; split into X_initial/y_initial (first N_INITIAL) and X_submissions/y_submissions (remaining); print summary with sample counts and ranges in `functions/f1/f1 - week 9.ipynb`
- [X] T005 [US1] Add binary labels code cell: compute y_binary = y > 0; count n_positive; extract X_pos = X[y_binary] and y_pos = y[y_binary]; compute y_pos_log = np.log(y_pos) (NOT np.log1p); set FALLBACK_MODE = n_positive < MIN_POSITIVE; print positive count and fallback status in `functions/f1/f1 - week 9.ipynb`
- [X] T006 [US1] Add data summary markdown header and tabular display code cell: loop through all observations printing index, source type (initial/wk#), x1, x2, y, binary label (POS/NEG/0), and star marker for best observation; print best value and location in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Data loaded, validated, and displayed — all subsequent stories can proceed.

---

## Phase 3: User Story 2 — Fit Hurdle Model with log Transform (Priority: P1)

**Goal**: Train the two-stage hurdle model with `np.log(y_pos)` for Stage 2 RF regressor, producing predictions in log-space (~-565 to ~-35). Per Research Tasks R1-R4: log(y) provides ~529 units of spread vs. near-zero for log1p; RF handles large-magnitude negative targets; existing y > 0 filter guards against log(0).

**Independent Test**: After surrogate cells execute, the RF regressor is trained on log-transformed positive outputs with R² reported, and can produce mean/uncertainty predictions.

### Implementation for User Story 2

- [X] T007 [US2] Add Stage 1 markdown header and code cell: train CalibratedClassifierCV(LogisticRegression(C=C_STAGE1, max_iter=1000, class_weight='balanced'), cv=3, method='sigmoid') on X with y_binary targets; compute and print training accuracy and probability estimates (min/max/mean) in `functions/f1/f1 - week 9.ipynb`
- [X] T008 [US2] Add Stage 2 markdown header and code cell: if not FALLBACK_MODE, train RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42) on X_pos with y_pos_log targets; print R² on log scale, tree count, depth, and sample count; else print fallback warning (mu=0, sigma=1 pure exploration) in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Hurdle model fitted — acquisition and visualisation stories can proceed.

---

## Phase 4: User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

**Goal**: Evaluate simplified weighted UCB acquisition (no local penalization per R5, no interior penalty per R6) over 20,000 random candidates with KAPPA=0.5 (exploitation-focused per R8), select the best candidate, and output a formatted submission string.

**Independent Test**: The notebook outputs a formatted string in 0.xxxxxx-0.xxxxxx format with values clipped to [0.0, 0.999999].

### Implementation for User Story 3

- [X] T009 [US3] Add acquisition function markdown header and code cell: seed np.random(42); generate N_CANDIDATES uniform random candidates in [0, 0.999999]²; compute p_cand from Stage 1 classifier; if not FALLBACK_MODE compute per-tree predictions, mu_log_cand (mean), sigma_rf_cand (std across trees) in log-space; else mu=0, sigma=1; compute acq_final = p_cand * mu_cand + KAPPA * p_cand * sigma_rf_cand; select argmax; clip next_x to [0.0, 0.999999]; compute min distance to existing data; print acquisition score, candidate coordinates, distance check in `functions/f1/f1 - week 9.ipynb`
- [X] T010 [US3] Add submission formatting code cell: clip next_x; assert shape (2,) and range [0, 0.999999]; format as "x1_str-x2_str" with 6 decimal places; print prominently with strategy summary (surrogate type, acquisition type, KAPPA value, FALLBACK_MODE status) in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Submission query produced — primary deliverable complete.

---

## Phase 5: User Story 4 — Visualise Surrogate with log Transform (Priority: P1)

**Goal**: Produce 3-panel contour visualisation in log-space and convergence plot with three-colour scheme (initial=blue, submissions=orange, proposed=green star).

**Independent Test**: Three-panel contour plots render with correct colour coding and log-space values (~-565 to ~-35); convergence plot shows running maximum with two-colour scheme.

### Implementation for User Story 4

- [X] T011 [US4] Add 3-panel contour plot markdown header and code cell: create GRID_RES×GRID_RES meshgrid over [0, 0.999999]; compute p_grid, mu_log_grid, sigma_rf_grid (or fallback); Panel 1 "Hurdle Mean (p·μ, log-space)" with viridis cmap; Panel 2 "Hurdle Uncertainty (p·σ_RF)" with YlOrRd cmap; Panel 3 "Acquisition (Weighted UCB)" with plasma cmap; overlay initial samples (tab:blue, s=40), weekly submissions (tab:orange, s=60), proposed next point (tab:green star, s=200); legend on Panel 1 only; colorbars on all panels; suptitle "Week 9 — Hurdle Model with log Transform (No Penalties)" in `functions/f1/f1 - week 9.ipynb`
- [X] T012 [US4] Add convergence plot code cell: compute running_max = np.maximum.accumulate(y); plot running max as black line; scatter initial samples (tab:blue) and weekly submissions (tab:orange); add vertical boundary line at N_INITIAL+0.5; print best observed value, location, and whether submissions improved over initial best in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: All visualisations rendered — core functionality complete (US1-US4).

---

## Phase 6: User Story 5 — Performance Evaluation (Priority: P2)

**Goal**: Quantitatively assess convergence, exploration spread, and LOO surrogate error with actionable strategy recommendations for Week 10.

**Independent Test**: Code cells compute convergence metrics, stalling flag, exploration spread, and LOO error; a final markdown cell interprets results and proposes strategy changes if stalling.

### Implementation for User Story 5

- [X] T013 [US5] Add performance evaluation section markdown header and convergence metrics code cell: compute best_initial, best_trajectory (running max after each submission), per-submission deltas, new_best_flags, trailing no-improvement streak (count backwards), relative improvement with zero-guard (if best_initial < 1e-10), stalling flag (consecutive >= 3 OR relative < 0.05); print summary table and per-submission detail in `functions/f1/f1 - week 9.ipynb`
- [X] T014 [US5] Add exploration spread markdown header and code cell: compute pairwise distances among submissions via scipy pdist; compute mean pairwise distance (compare to uniform 2D ≈ 0.52); compute nearest-neighbour distances (max and min); detect clustering if mean < 0.52 × 0.7; plot scatter of initial (blue) and submission (orange) points with W1-W9 annotations; equal aspect ratio in `functions/f1/f1 - week 9.ipynb`
- [X] T015 [US5] Add LOO surrogate error markdown header and code cell: 9-fold LOO over submission points; for each fold retrain both hurdle stages (with fallback if n_pos < MIN_POSITIVE); predict held-out point as p_held × exp(mu_log_held); compute MAE, RMSE, max/min error, fallback fold count; print per-fold detail table and summary in `functions/f1/f1 - week 9.ipynb`
- [X] T016 [US5] Add interpretation and strategy recommendations markdown cell: assess convergence (F1 stalling history, log transform improvement), explain clustering is intentional with KAPPA=0.5, interpret LOO accuracy; if stalling propose at least 4 actionable Week 10 changes (SingleTaskGP, KAPPA=0.1, re-introduce local penalization, LHS candidates); if not stalling confirm current strategy in `functions/f1/f1 - week 9.ipynb`

**Checkpoint**: Full performance analysis complete — notebook is feature-complete.

---

## Phase 7: User Story 1–5 Analysis — Why log, Why No Penalties, Why Exploit

**Goal**: Add a detailed analysis markdown cell explaining the mathematical rationale for all three key changes from Week 8, with concrete examples and code diffs.

- [X] T017 [US1] Add analysis markdown cell covering: (1) why log1p fails for F1 (Taylor expansion log1p(ε) ≈ ε for ultra-small values, table comparing raw/log1p/log for three sample values); (2) why log(y) works (maps to -565 to -35 range, prequential evaluation evidence); (3) why remove penalties (sparse 19-point coverage, simplicity per Constitution Principle I); (4) why KAPPA=0.5 (R² ≈ 0.90 surrogate accuracy, 4 budget submissions remaining); (5) code diffs showing all three changes from Week 8 in `functions/f1/f1 - week 9.ipynb`

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all stories to ensure notebook integrity and spec compliance.

- [X] T018 Run full notebook end-to-end and verify all cells execute without errors in `functions/f1/f1 - week 9.ipynb`
- [X] T019 Validate against quickstart.md checklist: data loads (19 samples, 2D); RF trains on log-space targets (~-565 to ~-35); contour shows meaningful variation in log-space; no local penalization or interior penalty in acquisition; KAPPA=0.5; Panel 3 titled "Acquisition (Weighted UCB)"; submission formatted as 0.xxxxxx-0.xxxxxx; all values clipped to [0.0, 0.999999] in `functions/f1/f1 - week 9.ipynb`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 — Data Loading (Phase 2)**: Depends on Setup — BLOCKS all subsequent stories
- **US2 — Hurdle Model (Phase 3)**: Depends on US1 (needs loaded data and binary labels)
- **US3 — Acquisition (Phase 4)**: Depends on US2 (needs fitted classifier and RF)
- **US4 — Visualisation (Phase 5)**: Depends on US3 (needs acquisition results and proposed point)
- **US5 — Performance Eval (Phase 6)**: Depends on US1 only — independent of US2-US4
- **Analysis (Phase 7)**: No code dependencies — markdown only, can be written anytime
- **Polish (Phase 8)**: Depends on all user stories complete

### User Story Dependencies

```
Setup (T001-T003)
  └──→ US1: Data Loading (T004-T006)
         ├──→ US2: Hurdle Model (T007-T008)
         │      └──→ US3: Acquisition (T009-T010)
         │             └──→ US4: Visualisation (T011-T012)
         ├──→ US5: Performance Eval (T013-T016) [independent of US2-US4]
         └──→ Analysis (T017) [markdown only, no code deps]
```

### Parallel Opportunities

- **US5 (T013-T016)** can be implemented in parallel with US2-US4 since it only depends on US1
- **T017 (Analysis)** is markdown-only and can be written at any time
- Within the notebook, all cells execute sequentially at runtime
- No [P] markers: all tasks target the same single notebook file

---

## Implementation Strategy

### MVP First (User Stories 1-3)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: US1 — Data Loading (T004-T006)
3. Complete Phase 3: US2 — Hurdle Model (T007-T008)
4. Complete Phase 4: US3 — Acquisition (T009-T010)
5. **STOP and VALIDATE**: Notebook produces a valid submission query

### Incremental Delivery

1. Setup + US1 → Data loads and validates
2. + US2 → Hurdle model fits with log transform
3. + US3 → Submission query produced (**MVP complete**)
4. + US4 → Visualisations rendered
5. + US5 → Performance evaluation complete
6. + Analysis → Mathematical rationale documented
7. Polish → End-to-end validation

---

## Notes

- All tasks target a single file: `functions/f1/f1 - week 9.ipynb`
- No [P] markers because all tasks write to the same notebook
- Key changes from Week 8: `np.log1p(y_pos)` → `np.log(y_pos)`, `np.expm1(mu)` → `np.exp(mu)`, local penalization and interior penalty removed, KAPPA 3.0 → 0.5
- Week 9 data files must exist in `data/f1/` before notebook execution
- Reference: `functions/f1/f1 - week 8.ipynb` (DO NOT MODIFY)
- Contour panels display in log-space per clarification (values ~-565 to ~-35)
- Panel 3 titled "Acquisition (Weighted UCB)" (no penalty mask)
- KAPPA=0.5 exploitation-focused — log transform gives surrogate meaningful signal, only 4 budget submissions remain
