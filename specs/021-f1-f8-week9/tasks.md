# Tasks: F1-F8 Week 9 — Bayesian Optimisation with Performance Evaluation

**Input**: Design documents from `/specs/021-f1-f8-week9/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/notebook-cells.md, quickstart.md

**Tests**: Not requested — no test tasks included.

**Organization**: Tasks are grouped by user story. Within each story, all 8 functions can be implemented in parallel (different files, no dependencies). Recommended sequential order follows quickstart.md: F2 → F1 → F3 → F5 → F4 → F6 → F7 → F8.

> **Note on US2 (Surrogate) and US3 (Acquisition)**: These user stories require zero code changes — Cells 5, 6, and 9 are carried from the Week 8 copy unchanged. They are fulfilled automatically by the Setup + US1 phases. No separate phase is needed.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US4, US5)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Create all 8 Week 9 notebook files by copying Week 8 notebooks.

- [ ] T001 Copy all 8 week 8 notebooks to week 9 filenames: cp functions/fX/fX - week 8.ipynb to functions/fX/fX - week 9.ipynb for X=1..8

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No shared infrastructure needed — all notebooks are self-contained per Constitution II. Foundational changes (imports, constants) are embedded in the US1 phase below since they modify the same per-notebook cells.

**⚠️ CRITICAL**: Setup (Phase 1) must complete before any user story work begins.

**Checkpoint**: All 8 notebook files exist and are identical copies of their Week 8 counterparts.

---

## Phase 3: US1 — Load and Validate Week 9 Data (Priority: P1)

**Goal**: Update each notebook's Cells 1-4 to load Week 9 data, add new imports (`scipy.spatial.distance`, `itertools.groupby`), add stalling-detection constants, and verify data shapes. Also carries US2 (Surrogate) and US3 (Acquisition) since those cells are unchanged from the copy.

**Independent Test**: Run cells 1-9 in each notebook — data loads with correct sample counts, surrogate fits, acquisition proposes a valid candidate, submission query prints.

### What to change in each notebook (Cells 1-4):

- **Cell 1**: Update title from "Week 8" to "Week 9"; add note about enhanced visualisation and performance evaluation
- **Cell 2**: Add `from scipy.spatial.distance import pdist, squareform` and `from itertools import groupby`
- **Cell 3**: Add `STALLING_CONSECUTIVE_THRESHOLD = 3` and `STALLING_RELATIVE_THRESHOLD = 0.05`; add `N_SUBMISSIONS = N_TOTAL - N_INITIAL`
- **Cell 4**: Change file paths from `Week 8` to `Week 9`; add `X_initial`, `y_initial`, `X_submissions`, `y_submissions` array slices

### Tasks

- [X] T002 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f2/f2 - week 9.ipynb — 19 samples, 2D, N_INITIAL=10
- [X] T003 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f1/f1 - week 9.ipynb — 19 samples, 2D, N_INITIAL=10
- [X] T004 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f3/f3 - week 9.ipynb — 24 samples, 3D, N_INITIAL=15
- [X] T005 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f5/f5 - week 9.ipynb — 29 samples, 4D, N_INITIAL=20
- [X] T006 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f4/f4 - week 9.ipynb — 39 samples, 4D, N_INITIAL=30
- [X] T007 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f6/f6 - week 9.ipynb — 29 samples, 5D, N_INITIAL=20
- [X] T008 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f7/f7 - week 9.ipynb — 39 samples, 6D, N_INITIAL=30
- [X] T009 [P] [US1] Update cells 1-4 (title, imports, constants, data paths) in functions/f8/f8 - week 9.ipynb — 49 samples, 8D, N_INITIAL=40

**Checkpoint**: All 8 notebooks execute cells 1-9 successfully. Each displays correct sample count, fits surrogate, proposes candidate, prints submission query.

---

## Phase 4: US4 — Visualise Surrogate, Convergence, and Submission Clustering (Priority: P1)

**Goal**: Update Cells 7-8 in each notebook to use the three-colour scheme: blue (initial samples), orange (weekly submissions), green star (proposed point). Add legends with three entries.

**Independent Test**: Each notebook generates plots where initial samples and weekly submissions are visually distinct, with a labelled proposed point and a legend.

### Colour scheme specification:

- Initial samples: `c='tab:blue'`, `s=40`, `edgecolors='white'`, `zorder=5`
- Weekly submissions: `c='tab:orange'`, `s=60`, `edgecolors='white'`, `zorder=5`
- Proposed point: `c='tab:green'`, `marker='*'`, `s=200`, `edgecolors='black'`, `zorder=6`
- Legend entries: "Initial samples", "Weekly submissions", "Proposed next point"

### Per-function visualisation type:

- F1, F2: 2D contour (3-panel for F1, standard for F2) — scatter X_initial vs X_submissions on all panels
- F3: 3D slice — top-2 dims by feature importance
- F4-F8: Higher-dim slice — top-2 dims by feature importance, remaining fixed at best point

### Tasks

- [X] T010 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f2/f2 - week 9.ipynb — 2D contour
- [X] T011 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f1/f1 - week 9.ipynb — 2D 3-panel contour
- [X] T012 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f3/f3 - week 9.ipynb — 3D slice
- [X] T013 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f5/f5 - week 9.ipynb — 4D slice
- [X] T014 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f4/f4 - week 9.ipynb — 4D slice
- [X] T015 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f6/f6 - week 9.ipynb — 5D slice
- [X] T016 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f7/f7 - week 9.ipynb — 6D slice
- [X] T017 [P] [US4] Update surrogate and convergence plot colour scheme (Cells 7-8) in functions/f8/f8 - week 9.ipynb — 8D slice

**Checkpoint**: All 8 notebooks show three-colour plots with legends. Initial samples in blue, weekly submissions in orange, proposed point as green star.

---

## Phase 5: US5 — Performance Evaluation and Strategy Recommendation (Priority: P1) 🎯

**Goal**: Add 4 new cells (10-13) to each notebook implementing convergence metrics, exploration spread, LOO surrogate error, and an interpretation markdown cell with strategy recommendations.

**Independent Test**: Each notebook ends with a performance evaluation section. Stalling flags are computed. LOO MAE/RMSE are displayed. If stalling, at least one specific strategy change is proposed.

### Cell 10 — Convergence Metrics (identical across all functions):

- `best_trajectory`: running max after each submission
- `per_submission_delta`: improvement per submission
- `new_best_flags`: boolean array
- `consecutive_no_improvement`: trailing streak of no-new-best counted from most recent submission backwards
- `relative_improvement`: `(best_final - best_initial) / abs(best_initial)` with zero-guard
- `stalling_flag`: True if consecutive >= 3 OR relative < 0.05
- Print summary table

### Cell 11 — Exploration Spread (identical across all functions):

- `mean_pairwise_distance = pdist(X_submissions).mean()`
- `max_nn_distance`, `min_nn_distance` from `squareform(pdist(X_submissions))`
- Print metrics
- For 2D/3D functions (F1, F2, F3): optional scatter plot of submission points

### Cell 12 — LOO Surrogate Error (VARIES by surrogate type):

| Function | Surrogate | LOO Implementation Notes |
|----------|-----------|--------------------------|
| F2, F6, F8 | SFGP | Retrain `SingleTaskGP(X_loo, Y_loo)` per fold. `Standardize(m=1)` auto-recomputes. |
| F3 | SFGP z-score | Recompute z-score stats (mean, std) per fold before fitting. Un-standardise predictions before error. |
| F5 | GP log1p+z-score | Recompute z-score per fold. Apply `expm1()` to invert `log1p` before error computation. |
| F4 | MFGP | Retrain `SingleTaskMultiFidelityGP`. Reduce MLL restarts to 5 for LOO. Append fidelity column. |
| F1 | Hurdle | Retrain both stages. If `n_positive < 3` for a fold, use fallback prediction = 0. |
| F7 | NN | Retrain 200 epochs per fold. `torch.manual_seed` for reproducibility. Single forward pass (no MC dropout). |

- Report per-point error, overall MAE, RMSE
- Note limited sample size (9 folds)

### Cell 13 — Interpretation & Strategy (Markdown, VARIES per function):

- Summarise convergence status
- Interpret exploration spread (reference expected pairwise distances from research.md R2)
- Assess LOO prediction accuracy
- If stalling: propose 1-3 strategy changes from research.md R4 table:

| Function | Rec 1 | Rec 2 | Rec 3 |
|----------|-------|-------|-------|
| F1 | Switch to SingleTaskGP | Raise KAPPA 3.0→5.0 | Use LHS candidates |
| F2 | Switch kernel to Matérn 2.5 | Increase raw_samples 512→2048 | Add random restart perturbation |
| F3 | Use BoTorch Standardize instead of manual z-score | Increase num_restarts 10→20 | Add interior penalty (S=0.5) |
| F4 | Switch to standard SingleTaskGP | Reduce q from 4 to 1 | Add interior penalty (S=0.5) |
| F5 | Relax STEEPNESS 1.0→0.3 | Increase Sobol 3000→5000 | Try Matérn 1.5 kernel |
| F6 | Switch kernel to Matérn 2.5 | Reduce noise floor 1e-2→1e-4 | Increase restarts 50→80 |
| F7 | Switch to SingleTaskGP Matérn 2.5 | Increase MC samples 50→200 | Widen network 6→10→10→1 |
| F8 | Switch from qEI to qLogNEI | Add interior penalty (S=0.5) | Increase raw_samples 4096→8192 |

### Tasks

- [X] T018 [US5] Add performance evaluation cells 10-13 to functions/f2/f2 - week 9.ipynb — SFGP Matérn 1.5 LOO (reference implementation)
- [X] T019 [P] [US5] Add performance evaluation cells 10-13 to functions/f1/f1 - week 9.ipynb — Hurdle model LOO with n_positive fallback
- [X] T020 [P] [US5] Add performance evaluation cells 10-13 to functions/f3/f3 - week 9.ipynb — SFGP LOO with z-score recomputation
- [X] T021 [P] [US5] Add performance evaluation cells 10-13 to functions/f5/f5 - week 9.ipynb — GP LOO with log1p+z-score inversion
- [X] T022 [P] [US5] Add performance evaluation cells 10-13 to functions/f4/f4 - week 9.ipynb — MFGP LOO with reduced restarts
- [X] T023 [P] [US5] Add performance evaluation cells 10-13 to functions/f6/f6 - week 9.ipynb — SFGP LOO (same retrain procedure as F2, Matern 1.5 kernel)
- [X] T024 [P] [US5] Add performance evaluation cells 10-13 to functions/f7/f7 - week 9.ipynb — NN LOO with manual seed and single forward pass
- [X] T025 [P] [US5] Add performance evaluation cells 10-13 to functions/f8/f8 - week 9.ipynb — SFGP LOO (same retrain procedure as F2, Matérn 2.5 kernel)

**Checkpoint**: All 8 notebooks end with performance evaluation section. Stalling flag, exploration metrics, and LOO error are computed and displayed. Interpretation markdown present with strategy recommendations where applicable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and final cleanup.

- [X] T026 Execute all 8 notebooks end-to-end and verify no errors per quickstart.md validation checklist
- [X] T027 Verify submission query format (0.xxxxxx-...-0.xxxxxx, all values in [0.0, 0.999999]) for all 8 functions
- [X] T028 Verify three-colour legend appears on all plots across all 8 notebooks
- [X] T029 Verify stalling detection logic against edge cases: zero initial best (F1), all-negative outputs, tied best values

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 3)**: Depends on Setup — updates cells 1-4 in copied notebooks
- **US4 (Phase 4)**: Depends on US1 — needs `X_initial`, `X_submissions` splits from Cell 4
- **US5 (Phase 5)**: Depends on US1 — needs data splits and surrogate configuration; T018 (F2) should complete first as reference
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (Data Loading)**: Can start after Setup — no dependencies on other stories
- **US2 (Surrogate)**: Fulfilled by Week 8 copy — no tasks required
- **US3 (Acquisition)**: Fulfilled by Week 8 copy — no tasks required
- **US4 (Visualisation)**: Depends on US1 for `X_initial`/`X_submissions` arrays
- **US5 (Performance Eval)**: Depends on US1 for data splits; LOO cell reuses surrogate config from Cell 5

### Within Each User Story

- All 8 functions are independent (different files) and can run in parallel
- For US5: F2 (T018) should be done first as reference implementation, then T019-T025 in parallel
- For sequential execution, follow quickstart order: F2 → F1 → F3 → F5 → F4 → F6 → F7 → F8

### Parallel Opportunities

- **Phase 3 (US1)**: All 8 tasks (T002-T009) are [P] — different notebook files
- **Phase 4 (US4)**: All 8 tasks (T010-T017) are [P] — different notebook files
- **Phase 5 (US5)**: T019-T025 are [P] after T018 completes — different files, T018 establishes pattern
- **Cross-phase**: US4 tasks for a given function can start as soon as that function's US1 task completes

---

## Parallel Example: F2 (Reference Implementation)

```bash
# Sequential within F2 (one notebook, cells must be in order):
T002: Update cells 1-4 in f2 - week 9.ipynb          # US1
T010: Update cells 7-8 colour scheme in f2            # US4
T018: Add cells 10-13 performance eval in f2          # US5

# Then all other functions in parallel:
T003 + T011 + T019  # F1 (all three phases)
T004 + T012 + T020  # F3
T005 + T013 + T021  # F5
T006 + T014 + T022  # F4
T007 + T015 + T023  # F6
T008 + T016 + T024  # F7
T009 + T017 + T025  # F8
```

---

## Implementation Strategy

### MVP First (F2 Only)

1. Complete Phase 1: Setup (copy all 8 notebooks)
2. Complete T002: US1 for F2 (update data loading)
3. Complete T010: US4 for F2 (update visualisation)
4. Complete T018: US5 for F2 (add performance evaluation)
5. **STOP and VALIDATE**: Execute F2 notebook end-to-end
6. F2 serves as the reference implementation for all other functions

### Incremental Delivery

1. F2 complete → validates full cell structure including performance eval
2. F1 complete → validates sklearn surrogate + hurdle LOO edge case
3. F3 complete → validates z-score LOO handling
4. F5 complete → validates double-transform (log1p + z-score) LOO
5. F4 complete → validates multi-fidelity LOO
6. F6 complete → validates higher-dim GP
7. F7 complete → validates neural network LOO
8. F8 complete → validates highest dimensionality (8D)
9. Polish phase → end-to-end validation of all 8

### LOO Complexity Grouping

Functions share LOO patterns — implement in this order to maximise reuse:

1. **F2** (simple SFGP LOO) → establishes base pattern
2. **F6, F8** (same SFGP LOO pattern as F2) → near-copy
3. **F3** (SFGP + z-score recompute) → adds one transform layer
4. **F5** (GP + log1p + z-score) → adds two transform layers
5. **F4** (MFGP + fidelity column) → different GP constructor
6. **F1** (Hurdle two-stage) → entirely different surrogate
7. **F7** (NN retrain) → entirely different surrogate

---

## Notes

- [P] tasks = different files, no dependencies on each other
- [US1/US4/US5] labels map tasks to user stories for traceability
- US2 (Surrogate) and US3 (Acquisition) require no code changes — carried from Week 8 copy
- Each notebook is fully self-contained — no shared code per Constitution II
- Existing notebooks (Week 8 and earlier) MUST NOT be modified per Constitution III
- Commit after completing each function (3 tasks per function = one logical commit)
- Stop at any checkpoint to validate independently
