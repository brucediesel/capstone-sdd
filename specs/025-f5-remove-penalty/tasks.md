# Tasks: F5 Week 9 — Remove Interior Penalty

**Input**: Design documents from `/specs/025-f5-remove-penalty/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

**Tests**: Not requested — no test tasks included (Constitution Principle I: Simplicity).

**Organization**: Tasks grouped by user story. All edits target a single file: `functions/f5/f5 - week 9.ipynb`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies between tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Cell references from research.md R4

---

## Phase 1: Setup

**Purpose**: Verify branch and prerequisites before making changes

- [X] T001 Verify active branch is `025-f5-remove-penalty` and notebook `functions/f5/f5 - week 9.ipynb` exists
- [X] T002 Verify week 9 data files exist: `data/f5/updated_inputs - Week 9.npy` and `data/f5/updated_outputs - Week 9.npy`

---

## Phase 2: User Story 1 — Remove Interior Penalty from Acquisition (Priority: P1) 🎯 MVP

**Goal**: Remove all penalty code so the notebook uses plain qLogNEI acquisition without the interior penalty wrapper

**Independent Test**: Run the acquisition cell — `optimize_acqf` passes `nei` directly as `acq_function`. No `STEEPNESS`, `FLOOR`, `EPS_BOUND`, `PenalisedAcquisition`, `penalised_nei`, or `BOUNDS_IP` variables exist in the notebook.

### Implementation for User Story 1

- [X] T003 [P] [US1] Remove constants `STEEPNESS`, `FLOOR`, `EPS_BOUND` from the hyperparameters code cell (cell 4) in `functions/f5/f5 - week 9.ipynb`
- [X] T004 [P] [US1] Remove Step 4 interior penalty explanation markdown cell entirely (cell 11) from `functions/f5/f5 - week 9.ipynb`
- [X] T005 [P] [US1] Remove Step 4 interior penalty code cell entirely (cell 12) — contains `PenalisedAcquisition` class, `penalised_nei`, `BOUNDS_IP`, and penalised `optimize_acqf` call — from `functions/f5/f5 - week 9.ipynb`

**Checkpoint**: All penalty acquisition code removed. Base NEI (Step 3) is now the only acquisition path.

---

## Phase 3: User Story 2 — Remove Penalty Visualisation (Priority: P1)

**Goal**: Remove penalty-specific visualisation cells and update surrogate plot to use base NEI selected point

**Independent Test**: No penalty contour panels exist. Step 5 surrogate viz uses `best_point` (base NEI). Step 6 section is gone entirely.

### Implementation for User Story 2

- [X] T006 [P] [US2] Remove Step 6 penalty visualisation markdown header cell entirely (cell 15) from `functions/f5/f5 - week 9.ipynb`
- [X] T007 [P] [US2] Remove Step 6 penalty visualisation code cell entirely (cell 16) from `functions/f5/f5 - week 9.ipynb`
- [X] T008 [P] [US2] Update Step 5 surrogate visualisation code cell (cell 14) in `functions/f5/f5 - week 9.ipynb`: replace `next_x_ip` with `best_point` in scatter calls, remove `+ IP` from suptitle

**Checkpoint**: All penalty visualisation removed. Surrogate plots reference base NEI point only.

---

## Phase 4: User Story 3 — Update Documentation and Submission (Priority: P2)

**Goal**: Update title, hyperparameter table, submission cell, and strategy to remove all penalty references

**Independent Test**: Title says "GP Matérn-5/2 + qLogNEI (4D)" without "Interior Penalty". Hyperparameter table has no IP rows. Submission shows only base NEI. Strategy notes penalty was removed.

### Implementation for User Story 3

- [X] T009 [P] [US3] Update title markdown cell (cell 1) in `functions/f5/f5 - week 9.ipynb`: remove "Interior Penalty" from heading and description
- [X] T010 [P] [US3] Update hyperparameters markdown table cell (cell 3) in `functions/f5/f5 - week 9.ipynb`: remove rows 16 (IP STEEPNESS) and 17 (IP FLOOR)
- [X] T011 [P] [US3] Update Step 8 submission code cell (cell 20) in `functions/f5/f5 - week 9.ipynb`: remove IP submission block and penalty parameter print statements
- [X] T012 [P] [US3] Update strategy markdown cell (cell 27) in `functions/f5/f5 - week 9.ipynb`: note penalty was evaluated and removed, remove STEEPNESS recommendations

**Checkpoint**: All documentation consistent — no penalty references remain anywhere in the notebook.

---

## Phase 5: Polish & Validation

**Purpose**: End-to-end execution and final verification

- [X] T013 Execute all cells in `functions/f5/f5 - week 9.ipynb` end-to-end and confirm no errors
- [X] T014 Validate against quickstart.md 12-item verification checklist (data loads, GP trains, NEI runs, submission format, visualisations render, no penalty references)
- [X] T015 Search entire notebook for zero remaining references to `PenalisedAcquisition`, `penalised_nei`, `STEEPNESS`, `FLOOR`, `EPS_BOUND`, `BOUNDS_IP`, or `next_x_ip`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 2)**: Depends on Setup — removes penalty acquisition code
- **US2 (Phase 3)**: Depends on Setup — removes penalty visualisation (independent of US1)
- **US3 (Phase 4)**: Depends on Setup — updates documentation (independent of US1 and US2)
- **Polish (Phase 5)**: Depends on ALL user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Setup — no dependencies on other stories
- **US2 (P1)**: Can start after Setup — no dependencies on US1 (operates on different cells)
- **US3 (P2)**: Can start after Setup — no dependencies on US1 or US2 (operates on different cells)

### Within Each User Story

- All tasks within US1 are independent (different cells) — marked [P]
- US2: T006 and T007 are independent removals [P]; T008 edits a separate cell
- All tasks within US3 are independent (different cells) — marked [P]

### Parallel Opportunities

- US1, US2, and US3 operate on entirely different cells and can proceed in parallel
- Within each story, [P]-marked tasks can run in parallel
- All single-notebook edits are sequential in practice but logically independent

---

## Parallel Example: All User Stories

```text
# All three user stories can proceed simultaneously (different cells):
Phase 2 (US1): T003, T004, T005 — penalty constants + Step 4 cells
Phase 3 (US2): T006, T007, T008 — Step 6 cells + Step 5 viz update
Phase 4 (US3): T009, T010, T011, T012 — title, table, submission, strategy
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: US1 — Remove penalty acquisition code (T003-T005)
3. **STOP and VALIDATE**: Notebook should execute through Step 3 without penalty references
4. Proceed to US2 and US3

### Incremental Delivery

1. Setup → Verify prerequisites
2. US1 → Remove penalty code → Core deliverable complete
3. US2 → Remove penalty visualisation → Clean plots
4. US3 → Update documentation → Fully consistent notebook
5. Polish → End-to-end validation → Ready for submission

---

## Notes

- All 15 tasks target a single file: `functions/f5/f5 - week 9.ipynb`
- 4 cells removed entirely (Step 4 md + code, Step 6 md + code)
- 6 cells edited (title, hyperparams table, constants, Step 5 viz, submission, strategy)
- 17 cells unchanged
- No new files created
- FR-011: No other changes — surrogate model, kernel, transforms, acquisition type, MC samples, q value, restarts, raw samples, distance selection, MLL restarts all remain unchanged
