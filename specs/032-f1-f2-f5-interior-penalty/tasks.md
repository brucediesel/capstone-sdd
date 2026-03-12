# Tasks: F1, F2 & F5 Interior Penalty

**Input**: Design documents from `specs/032-f1-f2-f5-interior-penalty/`
**Prerequisites**: plan.md (✅), spec.md (✅), research.md (✅), data-model.md (✅), quickstart.md (✅)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks grouped by user story. All 3 user stories are P1 and independent of each other — they can execute in parallel after the (empty) setup phase.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No setup needed — all infrastructure (notebooks, data files, BoTorch stack) already exists. The interior penalty uses only NumPy which is already imported in all 3 notebooks.

**Checkpoint**: Setup is a no-op; proceed directly to user stories.

---

## Phase 2: User Story 1 — F1 Interior Penalty (Priority: P1) 🎯 MVP

**Goal**: Add a very shallow interior penalty (STEEPNESS=0.02, FLOOR=0.01) to the F1 week 10 notebook that re-scores distance-filter survivors by penalised acquisition value.

**Independent Test**: Run `functions/f1/f1 - week 10.ipynb` end-to-end; verify penalty weights printed, all in [0.01, 1.0], valid 2D submission in [0, 0.999999] format.

### Implementation for User Story 1

- [ ] T001 [US1] Add `STEEPNESS = 0.02` and `FLOOR = 0.01` constants to the configuration cell in `functions/f1/f1 - week 10.ipynb`
- [ ] T002 [US1] Add interior penalty re-scoring cell after the existing distance-based selection cell in `functions/f1/f1 - week 10.ipynb` — compute `interior_weight` for each distance-filter survivor using `w(x) = FLOOR + (1 - FLOOR) * np.prod(np.sin(np.pi * x) ** (2 * STEEPNESS), axis=1)`, evaluate per-candidate acquisition values via `acqf(survivors[i:i+1].unsqueeze(0))`, compute `penalised_acq = acq_values * interior_weight`, select `argmax(penalised_acq)`, print penalty weights and whether selection changed, update `x_new` with `np.clip(..., 0.0, 0.999999)`
- [ ] T003 [US1] Update the submission formatting cell in `functions/f1/f1 - week 10.ipynb` to use the penalised-selected `x_new` (verify 2D format `0.XXXXXX-0.XXXXXX` and not a duplicate of existing observations)

**Checkpoint**: F1 notebook executes end-to-end with interior penalty applied and valid submission produced.

---

## Phase 3: User Story 2 — F2 Interior Penalty (Priority: P1)

**Goal**: Add the same very shallow interior penalty to the F2 week 10 notebook. F2 shares the same 2D structure and selection pattern as F1.

**Independent Test**: Run `functions/f2/f2 - week 10.ipynb` end-to-end; verify penalty weights printed, all in [0.01, 1.0], valid 2D submission.

### Implementation for User Story 2

- [ ] T004 [P] [US2] Add `STEEPNESS = 0.02` and `FLOOR = 0.01` constants to the configuration cell in `functions/f2/f2 - week 10.ipynb`
- [ ] T005 [P] [US2] Add interior penalty re-scoring cell after the existing distance-based selection cell in `functions/f2/f2 - week 10.ipynb` — same penalty formula and pattern as F1 (T002), adapted to F2's variable names (`qualified_idx`, `candidates`, `acqf`, `x_new`), print penalty weights and whether selection changed
- [ ] T006 [P] [US2] Update the submission formatting cell in `functions/f2/f2 - week 10.ipynb` to use the penalised-selected `x_new` (verify 2D format and not a duplicate)

**Checkpoint**: F2 notebook executes end-to-end with interior penalty applied and valid submission produced.

---

## Phase 4: User Story 3 — F5 Interior Penalty (Priority: P1)

**Goal**: Add the same very shallow interior penalty to the F5 week 10 notebook. F5 is 4D and uses numpy-based selection with 25th percentile gate (different variable names from F1/F2).

**Independent Test**: Run `functions/f5/f5 - week 10.ipynb` end-to-end; verify penalty weights printed across all 4 dimensions, all in [0.01, 1.0], valid 4D submission.

### Implementation for User Story 3

- [ ] T007 [P] [US3] Add `STEEPNESS = 0.02` and `FLOOR = 0.01` constants to the configuration cell in `functions/f5/f5 - week 10.ipynb`
- [ ] T008 [P] [US3] Add interior penalty re-scoring cell after the existing distance-based selection cell in `functions/f5/f5 - week 10.ipynb` — same penalty formula as F1/F2 but adapted to F5's numpy-based variables (`qualified_indices`, `qualified_mask`, `candidates`, `acqf`, `x_new`), the penalty product is over all 4 input dimensions, print penalty weights and whether selection changed
- [ ] T009 [P] [US3] Update the submission formatting cell in `functions/f5/f5 - week 10.ipynb` to use the penalised-selected `x_new` (verify 4D format `0.XXXXXX-0.XXXXXX-0.XXXXXX-0.XXXXXX` and not a duplicate)

**Checkpoint**: F5 notebook executes end-to-end with interior penalty applied and valid 4D submission produced.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all 3 notebooks.

- [ ] T010 Verify STEEPNESS = 0.02 and FLOOR = 0.01 in all 3 notebooks (SC-005)
- [ ] T011 Run quickstart.md validation — confirm all 3 notebooks produce valid submissions with penalty weights in [0.01, 1.0]

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No-op — already complete
- **User Story 1 (Phase 2)**: Can start immediately — no dependencies
- **User Story 2 (Phase 3)**: Can start immediately — independent of US1
- **User Story 3 (Phase 4)**: Can start immediately — independent of US1/US2
- **Polish (Phase 5)**: Depends on all 3 user stories being complete

### User Story Dependencies

- **US1 (F1)**: No dependencies on other stories. T001 → T002 → T003 (sequential within story).
- **US2 (F2)**: No dependencies on other stories. T004 → T005 → T006 (sequential within story). Marked [P] — can run in parallel with US1 and US3.
- **US3 (F5)**: No dependencies on other stories. T007 → T008 → T009 (sequential within story). Marked [P] — can run in parallel with US1 and US2.

### Within Each User Story

1. Constants cell (T001/T004/T007) — must be added first so penalty cell can reference STEEPNESS/FLOOR
2. Penalty re-scoring cell (T002/T005/T008) — core implementation
3. Submission update (T003/T006/T009) — verify output format uses penalised selection

### Parallel Opportunities

All 3 user stories operate on different notebook files and can execute fully in parallel:

```text
Phase 2 (US1: F1)  ──►  T001 → T002 → T003  ─┐
Phase 3 (US2: F2)  ──►  T004 → T005 → T006  ─┤──► Phase 5: Polish (T010 → T011)
Phase 4 (US3: F5)  ──►  T007 → T008 → T009  ─┘
```

---

## Implementation Strategy

**MVP**: User Story 1 (F1) alone — validates the interior penalty pattern on the simplest 2D function.

**Incremental delivery**:
1. Complete US1 (F1) — validate the pattern works
2. US2 (F2) and US3 (F5) can proceed in parallel — same pattern, adapted variable names
3. Polish — cross-cutting validation across all 3 notebooks

**Estimated scope**: 11 tasks, 3 notebooks, ~10 lines of new code per notebook.
