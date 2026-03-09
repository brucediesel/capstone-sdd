# Tasks: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Input**: Design documents from `/specs/024-f3-week9-standardize/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

**Tests**: Not required (Constitution Principle I — no unit tests).

**Organization**: Tasks grouped by user story. All tasks target a single notebook: `functions/f3/f3 - week 9.ipynb`. US3 (interior penalty) was evaluated and removed per clarification — no phase for US3.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (independent cells, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US4, US5)
- All file paths are relative to repository root

---

## Phase 1: Setup

**Purpose**: Verify branch and data prerequisites before any code changes

- [X] T001 Verify branch `024-f3-week9-standardize` is checked out and up to date
- [X] T002 Verify week 9 data files exist: `data/f3/updated_inputs - Week 9.npy` (24×3) and `data/f3/updated_outputs - Week 9.npy` (24,)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Update shared cells (imports and hyperparameters) that ALL user stories depend on. Remove any interior penalty artefacts from previous implementation.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 [P] Add `from botorch.models.transforms.outcome import Standardize` to imports cell in `functions/f3/f3 - week 9.ipynb`
- [X] T004 [P] Update hyperparameters cell: rename `ACQ_RESTARTS = 10` to `NUM_RESTARTS_ACQ = 20` with rationale comment; remove `STEEPNESS` and `FLOOR` constants if present in `functions/f3/f3 - week 9.ipynb`

**Checkpoint**: Imports and hyperparameters ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Replace Manual z-score with BoTorch Standardize (Priority: P1) 🎯 MVP

**Goal**: Eliminate manual z-score standardisation by using BoTorch `Standardize(m=1)` outcome transform. The GP posterior will auto-untransform to original scale.

**Independent Test**: Run GP training cell — model trains without manual mean/std computation, posterior predictions are in original output scale.

### Implementation for User Story 1

- [X] T005 [US1] Remove manual z-score computation (`y_mean`, `y_std`, `y_std_safe`, `y_standardised`) from data loading/GP training cell in `functions/f3/f3 - week 9.ipynb`
- [X] T006 [US1] Change `Y_train` construction to use raw `y_raw` values instead of `y_standardised` in `functions/f3/f3 - week 9.ipynb`
- [X] T007 [US1] Add `outcome_transform=Standardize(m=1)` parameter to `SingleTaskGP` constructor in `functions/f3/f3 - week 9.ipynb`

**Checkpoint**: GP trains with Standardize — no manual z-score code remains in training cell

---

## Phase 4: User Story 2 — Increase Acquisition Restarts to 20 (Priority: P1)

**Goal**: Broaden multi-start acquisition search from 10 to 20 restarts for better candidate discovery. Acquisition uses plain qLogNEI (no penalty wrapper).

**Independent Test**: Run acquisition cell — `optimize_acqf` uses `num_restarts=20` and passes `nei` directly as `acq_function`.

### Implementation for User Story 2

- [X] T008 [US2] Update `optimize_acqf` call to use `num_restarts=NUM_RESTARTS_ACQ` (20) and `acq_function=nei` (plain qLogNEI, no wrapper) in `functions/f3/f3 - week 9.ipynb`
- [X] T009 [US2] Remove `PenalisedAcquisition` class definition and `penalised_nei` wrapping if present in acquisition cell in `functions/f3/f3 - week 9.ipynb`

**Checkpoint**: Acquisition cell runs with 20 restarts using plain qLogNEI — no penalty code remains

---

## Phase 5: User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

**Goal**: Update visualisation to work with Standardize (remove manual un-standardisation). Use 1×3 posterior mean layout matching Week 8.

**Independent Test**: Run visualisation cells — 1×3 contour plots render with three-colour scheme, convergence plot shows running maximum trajectory.

### Implementation for User Story 4

- [X] T010 [US4] Remove manual un-standardisation from contour visualisation cell (`mean_raw = mean_std * y_std_safe + y_mean`, `std_raw = std_std * y_std_safe`) — use `posterior.mean` and `posterior.variance.sqrt()` directly in `functions/f3/f3 - week 9.ipynb`
- [X] T011 [US4] Ensure 1×3 posterior mean layout with white uncertainty contours (remove penalty contour panel if present, revert from 2×3 to 1×3 grid) in `functions/f3/f3 - week 9.ipynb`
- [X] T012 [US4] Verify convergence plot shows running maximum with 15 initial samples (blue) and 9 submissions (orange) with vertical boundary line at observation 15 in `functions/f3/f3 - week 9.ipynb`

**Checkpoint**: All plots render correctly with Standardize-based predictions in 1×3 layout

---

## Phase 6: User Story 5 — Performance Evaluation and Strategy Recommendations (Priority: P2)

**Goal**: Update LOO and convergence metrics to work with Standardize. Add Week 10 strategy recommendations.

**Independent Test**: Run performance cells — LOO completes without manual z-score, convergence metrics print correctly.

### Implementation for User Story 5

- [X] T013 [US5] Update LOO cross-validation to construct each fold's GP with `outcome_transform=Standardize(m=1)` — remove per-fold manual z-score recomputation (`y_loo_mean`, `y_loo_std`, `y_loo_std_z`) in `functions/f3/f3 - week 9.ipynb`
- [X] T014 [US5] Remove manual prediction un-standardisation in LOO (`pred_raw = pred_std * y_loo_std + y_loo_mean`) — use `model.posterior(x_held).mean.item()` directly in `functions/f3/f3 - week 9.ipynb`
- [X] T015 [P] [US5] Verify convergence metrics cell reports stalling flag, per-submission deltas, and trailing no-improvement streak in `functions/f3/f3 - week 9.ipynb`
- [X] T016 [P] [US5] Add Week 10 strategy recommendations markdown cell at end of notebook in `functions/f3/f3 - week 9.ipynb`

**Checkpoint**: LOO and convergence metrics work without manual z-score — strategy section present

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [X] T017 Update hyperparameter documentation markdown table with rationale for two changes (Standardize, restarts) — note that interior penalty was evaluated and removed in `functions/f3/f3 - week 9.ipynb`
- [X] T018 Execute all notebook cells end-to-end and verify submission query format `0.xxxxxx-0.xxxxxx-0.xxxxxx` with all coordinates in [0.0, 0.999999] in `functions/f3/f3 - week 9.ipynb`
- [X] T019 Validate against quickstart.md verification checklist (9 items) in `specs/024-f3-week9-standardize/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (T003 import, T004 hyperparams)
- **US2 (Phase 4)**: Depends on Foundational (T004 hyperparams) — independent of US1
- **US4 (Phase 5)**: Depends on US1 (Standardize changes posterior output)
- **US5 (Phase 6)**: Depends on US1 (LOO requires Standardize)
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)** 🎯: After Foundational — no dependencies on other stories
- **US2 (P1)**: After Foundational — no dependencies on other stories
- **US4 (P2)**: After US1 — needs Standardize posterior for correct contour values
- **US5 (P2)**: After US1 — needs Standardize for LOO simplification

### Within Each User Story

- Single notebook file — tasks within a story execute sequentially
- Remove old code before adding new code (e.g., remove z-score before adding Standardize)

### Parallel Opportunities

- T003 and T004 (Foundational): different cells, can edit in parallel
- US1 (Phase 3) and US2 (Phase 4): independent P1 stories — can execute in any order after Foundational
- T015 and T016 (US5): independent cells, can execute in parallel

---

## Parallel Example: P1 Stories After Foundational

```
# After T003 + T004 complete, these two story groups are independent:

# US1: Standardize (modifies GP training cell + data loading cell)
T005 → T006 → T007

# US2: Restarts + penalty cleanup (modifies acquisition cell)
T008 → T009
```

> **Note**: US1 and US2 modify different notebook cells (GP training vs acquisition), so they can safely execute in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (verify branch + data)
2. Complete Phase 2: Foundational (imports + hyperparams)
3. Complete Phase 3: User Story 1 (Standardize)
4. **STOP and VALIDATE**: Run GP training cell — confirms no manual z-score, model trains correctly
5. This alone produces a working notebook (just without increased restarts)

### Incremental Delivery

1. Setup + Foundational → Infrastructure ready
2. US1 (Standardize) → GP trains correctly, simplifies pipeline (MVP!)
3. US2 (Restarts + penalty cleanup) → Better acquisition search, clean acquisition cell
4. US4 (Visualisation) → Updated 1×3 plots confirm changes visually
5. US5 (Performance) → LOO + strategy recommendations complete
6. Polish → End-to-end validation, ready for submission

### Recommended Execution Order

Given single-file constraint and cell dependencies:

```
T001 → T002 → T003+T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016+T017 → T018 → T019
```

---

## Notes

- All 19 tasks target the same notebook file: `functions/f3/f3 - week 9.ipynb`
- No tests required (Constitution Principle I)
- No new files created — existing notebook is modified in-place
- Commit after each phase completion for clean history
- T018 (end-to-end execution) is the critical validation gate before submission
