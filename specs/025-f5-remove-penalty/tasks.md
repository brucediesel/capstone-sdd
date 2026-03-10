# Tasks: F5 Week 9 — Kernel, Standardize & Raw Samples

**Input**: Design documents from `/specs/025-f5-remove-penalty/` (Clarifications session)
**Prerequisites**: plan.md ✅, spec.md ✅ (FR-012/013/014), research.md ✅, data-model.md ✅, quickstart.md ✅

**Tests**: Not requested — no test tasks included (Constitution Principle I: Simplicity).

**Organization**: Tasks grouped by user story. All edits target a single file: `functions/f5/f5 - week 9.ipynb` (23 cells).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies between tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Cell references from research.md R4

---

## Phase 1: Setup

**Purpose**: Create branch and verify prerequisites

- [X] T001 Create branch `026-f5-kernel-standardize` from `025-f5-remove-penalty` and verify notebook `functions/f5/f5 - week 9.ipynb` has 23 cells
- [X] T002 Verify week 9 data files exist: `data/f5/updated_inputs - Week 9.npy` and `data/f5/updated_outputs - Week 9.npy`

---

## Phase 2: User Story 1 — Core Pipeline Changes (Priority: P1) 🎯 MVP

**Goal**: Apply Matérn-1.5 kernel, BoTorch Standardize(m=1), and raw_samples=5000 to the GP training and acquisition cells

**Independent Test**: Run cells 2→4→8→10 — GP trains with `nu=1.5` and `outcome_transform=Standardize(m=1)`, acquisition uses `raw_samples=5000`, no `y_mean`/`y_std_val`/`y_std` variables exist, inverse transform is `expm1(posterior.mean)`.

### Implementation for User Story 1

- [X] T003 [P] [US1] Add `from botorch.models.transforms.outcome import Standardize` to imports in cell 2 of `functions/f5/f5 - week 9.ipynb`
- [X] T004 [P] [US1] Simplify transform code in cell 4 of `functions/f5/f5 - week 9.ipynb`: remove manual z-score computation (`y_mean`, `y_std_val`, `y_std`), set `Y_train = torch.tensor(y_log, ...).unsqueeze(-1)` directly from `y_log`
- [X] T005 [US1] Update GP training in cell 8 of `functions/f5/f5 - week 9.ipynb`: change `MaternKernel(nu=2.5, ...)` to `MaternKernel(nu=1.5, ...)`, change `outcome_transform=None` to `outcome_transform=Standardize(m=1)`, update print statements to reflect new kernel/transform
- [X] T006 [US1] Update acquisition in cell 10 of `functions/f5/f5 - week 9.ipynb`: change `raw_samples=3000` to `raw_samples=5000`, simplify inverse transform from `expm1(pred * y_std_val + y_mean)` to `expm1(posterior.mean)`

**Checkpoint**: Core pipeline changed — GP uses Matérn-1.5 + Standardize(m=1), acquisition uses 5000 raw samples. MVP complete.

---

## Phase 3: User Story 2 — Downstream Cell Updates (Priority: P1)

**Goal**: Update visualisation, submission, and LOO cells to match the new pipeline (simplified inverse, new kernel name)

**Independent Test**: Run cells 12, 16, 22 — viz uses `expm1(grid_mu)` inverse and suptitle says "Matérn-1.5", submission prints "Standardize(m=1)", LOO uses `nu=1.5` with `Standardize(m=1)` and simplified inverse per fold.

### Implementation for User Story 2

- [X] T007 [P] [US2] Update visualisation in cell 12 of `functions/f5/f5 - week 9.ipynb`: simplify grid inverse transform (remove manual z-score inverse), update suptitle from "Matérn-5/2" to "Matérn-1.5"
- [X] T008 [P] [US2] Update submission print in cell 16 of `functions/f5/f5 - week 9.ipynb`: change surrogate description from "GP Matérn-5/2 ARD (outcome_transform=None)" to "GP Matérn-1.5 ARD (outcome_transform=Standardize(m=1))"
- [X] T009 [US2] Update LOO cross-validation in cell 22 of `functions/f5/f5 - week 9.ipynb`: change `nu=2.5` to `nu=1.5`, add `outcome_transform=Standardize(m=1)` to each fold GP, remove manual z-score per fold, simplify inverse from `expm1(pred * std + mean)` to `expm1(pred)`

**Checkpoint**: All downstream cells consistent with new pipeline. No manual z-score inverse remains.

---

## Phase 4: User Story 3 — Documentation Updates (Priority: P2)

**Goal**: Update title, hyperparameters table, and strategy markdown cells to reflect kernel/transform/raw_samples changes

**Independent Test**: Title says "Matérn-1.5", hyperparams table shows `nu=1.5`, `raw_samples=5000`, `Standardize(m=1)`, strategy documents the changes made.

### Implementation for User Story 3

- [X] T010 [P] [US3] Update title markdown in cell 1 of `functions/f5/f5 - week 9.ipynb`: change "Matérn-5/2" to "Matérn-1.5"
- [X] T011 [P] [US3] Update hyperparameters table in cell 3 of `functions/f5/f5 - week 9.ipynb`: change kernel nu from 2.5 to 1.5, raw_samples from 3000 to 5000, outcome_transform from None to Standardize(m=1)
- [X] T012 [P] [US3] Update strategy markdown in cell 23 of `functions/f5/f5 - week 9.ipynb`: document that Matérn-1.5, Standardize(m=1), and raw_samples=5000 have been applied; update recommendations for future iterations

**Checkpoint**: All documentation consistent with new pipeline configuration.

---

## Phase 5: Polish & Validation

**Purpose**: End-to-end execution, verification, and results review

- [X] T013 Execute all 23 cells in `functions/f5/f5 - week 9.ipynb` end-to-end and confirm no errors
- [X] T014 Validate against quickstart.md 14-item verification checklist (branch, cell count, imports, no z-score vars, kernel nu, Standardize, raw_samples, inverse transforms, execution, submission format, LOO, suptitle/prints, zero z-score refs, results review)
- [X] T015 Search entire notebook for zero remaining references to manual z-score variables (`y_mean`, `y_std_val`, `y_std`), `nu=2.5`, `outcome_transform=None`, and `raw_samples=3000`
- [X] T016 Review results: examine proposed candidates for boundary-sticking, compare LOO MAE, and suggest further improvements

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 2)**: Depends on Setup — applies core kernel/transform/acquisition changes
- **US2 (Phase 3)**: Depends on US1 — downstream cells must match the new pipeline from Phase 2
- **US3 (Phase 4)**: Depends on Setup — documentation updates (independent of US1 and US2, operates on markdown cells)
- **Polish (Phase 5)**: Depends on ALL user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Setup — no dependencies on other stories
- **US2 (P1)**: Depends on US1 — viz/submission/LOO inverse transforms must match the pipeline changes in US1
- **US3 (P2)**: Can start after Setup — operates on markdown-only cells, independent of US1 and US2

### Within Each User Story

- US1: Import (T003) and transform (T004) can run in parallel [P]; GP training (T005) depends on both; acquisition (T006) depends on T005
- US2: Viz (T007) and submission (T008) can run in parallel [P]; LOO (T009) is independent but complex
- US3: All tasks can run in parallel [P] (different markdown cells)

### Parallel Opportunities

```text
Phase 1:  T001 ─── T002                              (sequential)
Phase 2:  T003 ─┬─ T005 ─── T006                     (T003+T004 parallel, then sequential)
          T004 ─┘
Phase 3:  T007 ─┬─ (all parallel, after Phase 2)
          T008 ─┤
          T009 ─┘
Phase 4:  T010 ─┬─ (all parallel, after Phase 1)
          T011 ─┤
          T012 ─┘
Phase 5:  T013 ─── T014 ─── T015 ─── T016            (sequential, after all phases)
```

---

## Implementation Strategy

**MVP**: Phase 1 + Phase 2 (Setup + Core Pipeline Changes) — the GP trains with the new kernel/transform and acquisition runs with more raw samples. This alone addresses the boundary-stuck issue.

**Incremental Delivery**:
1. Setup → Core changes (MVP)
2. Downstream updates (viz, submission, LOO consistent)
3. Documentation (markdown cells)
4. Validation & results review

**Total Tasks**: 16 (2 setup, 4 core, 3 downstream, 3 documentation, 4 validation)

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
