# Tasks: F1 Interior Penalty on Acquisition Function

**Input**: Design documents from `/specs/014-f1-interior-penalty/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cells.md, quickstart.md  
**Tests**: Not required (per CONSTITUTION — no unit tests)

**Organization**: All 6 new cells are appended to a single notebook (`functions/f1/f1.ipynb`). User Story 1 (P1) is the core implementation and delivers all 6 cells. User Story 2 (Hyperparameter Documentation, P2) is satisfied by the markdown header cell (T002) and constants cell (T003). User Story 3 (Visualisation, P2) is satisfied by the 3-panel viz cell (T005) and convergence plot cell (T006).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single notebook**: `functions/f1/f1.ipynb` (all new cells appended after existing Cell 65)
- **Spec docs**: `specs/014-f1-interior-penalty/`
- **Data**: `data/f1/`

---

## Phase 1: Setup

**Purpose**: Verify branch and prerequisites before any cell edits

- [X] T001 Verify branch is `014-f1-interior-penalty`, Week 7 data files exist in data/f1/, and existing notebook has 65 cells with W7 variables defined

---

## Phase 2: Foundational

**Purpose**: No foundational tasks required — all dependencies are existing Week 7 cells already present in `functions/f1/f1.ipynb`

**⚠️ NOTE**: The following kernel variables MUST be available from running cells 1–65 before the new section can execute: `X_w7`, `y_w7`, `y_binary`, `stage1_clf`, `stage2_rf`, `KAPPA`, `PENALTY_RADIUS`, `N_CANDIDATES`, `GRID_RES`, `FALLBACK_MODE`, `X_cand`, `acq_penalized`

---

## Phase 3: User Story 1 — Add Interior Penalty to F1 Acquisition (Priority: P1) 🎯 MVP

**Goal**: Append 6 new cells to the F1 notebook that apply a sin²-based interior penalty to the existing penalised acquisition function, discouraging boundary-hugging candidates and proposing an interior sample point.

**Delivers**: US1 (core penalty), US2 (hyperparameter documentation via T002+T003), US3 (visualisation via T005+T006)

**Independent Test**: Run notebook cells 1–71 end-to-end. The proposed point should have both coordinates in [0.05, 0.95], the 3-panel viz should show boundary suppression, and the submission query should be valid `0.xxxxxx-0.xxxxxx` format.

### Implementation

- [X] T002 [US1] Append markdown header cell (Cell 66) with section title "Week 7 — Interior Penalty", motivation paragraph, penalty formula w(x) = FLOOR + (1-FLOOR)·∏sin(πxᵢ)^(2·STEEPNESS), and hyperparameter table documenting STEEPNESS, FLOOR, KAPPA, PENALTY_RADIUS with rationale in functions/f1/f1.ipynb
- [X] T003 [US1] Append hyperparameter constants cell (Cell 67) defining STEEPNESS=2.0 and FLOOR=0.01 as named constants, printing both values for audit trail in functions/f1/f1.ipynb
- [X] T004 [US1] Append interior penalty computation and candidate selection cell (Cell 68) that computes interior_weight via sin²-product formula on X_cand, multiplies with acq_penalized to get acq_with_interior, selects argmax as next_x_ip clipped to [0, 0.999999], validates min distance ≥ 0.05 from X_w7, and prints selected point coordinates with acquisition value in functions/f1/f1.ipynb
- [X] T005 [P] [US1] Append 3-panel surrogate and acquisition visualisation cell (Cell 69) showing (1) hurdle mean p(x)·μ(x), (2) hurdle uncertainty p(x)·σ_RF(x), (3) penalised acquisition with interior penalty on GRID_RES×GRID_RES grid, with scatter for training points and yellow star for next_x_ip, title including STEEPNESS and FLOOR values in functions/f1/f1.ipynb
- [X] T006 [P] [US1] Append convergence plot cell (Cell 70) showing np.maximum.accumulate(y_w7) running best line, individual observation scatter, week boundary vertical dashed lines, and best observed value annotation in functions/f1/f1.ipynb
- [X] T007 [US1] Append submission query cell (Cell 71) formatting next_x_ip as "0.xxxxxx-0.xxxxxx" with 6 decimal places, validating 2 dimensions in [0.0, 0.999999], and printing summary of surrogate type, acquisition type, and interior penalty parameters in functions/f1/f1.ipynb

**Checkpoint**: All 6 cells appended. Running cells 66–71 after existing W7 section should produce: a selected interior point, 3-panel visualisation with boundary suppression visible, convergence plot, and valid submission string. US1, US2, and US3 acceptance scenarios all satisfied.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and success criteria verification

- [X] T008 Run full notebook end-to-end (cells 1–71) and verify: SC-001 (point in [0.05, 0.95]²), SC-002 (≥50% acquisition reduction near boundaries), SC-003 (no errors), SC-004 (valid submission format), SC-005 (existing cells unchanged) in functions/f1/f1.ipynb
- [X] T009 Execute quickstart.md verification checklist — confirm no error output, selected point printed, min distance ≥ 0.05, boundary suppression visible in Panel 3, submission query formatted correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify prerequisites first
- **Foundational (Phase 2)**: N/A — existing notebook cells provide all dependencies
- **US1 (Phase 3)**: Depends on Setup verification. All cells appended to single file in order
- **Polish (Phase 4)**: Depends on all Phase 3 tasks complete

### Within Phase 3 (US1)

```
T002 (markdown) → T003 (constants) → T004 (computation)
                                          ├── T005 [P] (3-panel viz)
                                          ├── T006 [P] (convergence)
                                          └── T007 (submission query)
```

- T002 → T003 → T004: Strict sequence (constants before computation)
- T005, T006: Parallel — independent visualisation cells, both depend on T004
- T007: Depends on T004 (uses `next_x_ip`); must be final cell in notebook

### User Story Mapping

| Story | Tasks | Status |
|-------|-------|--------|
| US1 — Interior Penalty (P1) | T002, T003, T004, T005, T006, T007 | Core implementation |
| US2 — Documentation (P2) | T002 (table), T003 (printed constants) | Co-delivered by US1 cells |
| US3 — Visualisation (P2) | T005 (3-panel), T006 (convergence) | Co-delivered by US1 cells |

---

## Parallel Example: Phase 3

```text
# Sequential: Markdown → Constants → Computation
T002: Append markdown header cell in functions/f1/f1.ipynb
T003: Append constants cell in functions/f1/f1.ipynb
T004: Append computation cell in functions/f1/f1.ipynb

# Parallel after T004 completes:
T005: Append 3-panel viz cell in functions/f1/f1.ipynb     ← [P]
T006: Append convergence plot cell in functions/f1/f1.ipynb ← [P]

# Sequential: Final cell
T007: Append submission query cell in functions/f1/f1.ipynb
```

---

## Implementation Strategy

### MVP First (All Stories in One Pass)

1. Complete Phase 1: Verify prerequisites (T001)
2. Complete Phase 3: Append all 6 cells in notebook order (T002–T007)
3. **STOP and VALIDATE**: Run notebook end-to-end (T008)
4. Verify quickstart checklist (T009)

All three user stories are delivered by the same 6 cells — there is no incremental delivery path within this feature. The MVP is the complete feature.

### Cell-to-Contract Mapping

| Task | Cell # | Contract Reference |
|------|--------|-------------------|
| T002 | 66 | contracts/cells.md — Cell 1: Section Header |
| T003 | 67 | contracts/cells.md — Cell 2: Hyperparameter Constants |
| T004 | 68 | contracts/cells.md — Cell 3: Computation + Selection |
| T005 | 69 | contracts/cells.md — Cell 4: 3-Panel Visualisation |
| T006 | 70 | contracts/cells.md — Cell 5: Convergence Plot |
| T007 | 71 | contracts/cells.md — Cell 6: Submission Query |

---

## Notes

- All 6 cells are additive — no existing cells modified (FR-001, SC-005)
- [P] tasks = independent outputs, can be written simultaneously but must maintain notebook cell order
- Each task maps 1:1 to a cell contract in `contracts/cells.md`
- Run `Kernel → Restart & Run All` for end-to-end validation
- Commit after Phase 3 completion (all 6 cells appended)
