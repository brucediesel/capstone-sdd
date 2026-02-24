# Tasks: F5 Interior Penalty on Acquisition Function

**Input**: Design documents from `/specs/015-f5-interior-penalty/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cells.md, quickstart.md

**Tests**: No test tasks — constitution states no unit tests are required. Validation is via inline assertions and manual execution.

**Organization**: Tasks grouped by user story. US1 (core penalty) is MVP. US2 (documentation) and US3 (visualisation) build on it.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies)
- **[Story]**: Which user story (US1, US2, US3)
- All cells appended to `functions/f5/f5.ipynb`

---

## Phase 1: Setup

**Purpose**: No new project setup needed — all code appends to existing notebook.

_(No tasks — the notebook and branch already exist.)_

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Markdown section header and hyperparameter constants that ALL subsequent cells depend on.

- [X] T001 [US2] Append markdown cell (Cell 58, ID: IP-00) with section title "Week 7 — Interior Penalty on Acquisition Function", motivation paragraph, hyperparameter table (STEEPNESS, FLOOR), and penalty formula in `functions/f5/f5.ipynb`
- [X] T002 [US2] Append code cell (Cell 59, ID: IP-01) defining `STEEPNESS = 1.0` and `FLOOR = 0.01` as named constants, printing both values, in `functions/f5/f5.ipynb`

**Checkpoint**: Hyperparameters defined — US1 implementation can begin.

---

## Phase 3: User Story 1 — Add Interior Penalty to F5 Acquisition (Priority: P1) MVP

**Goal**: Compute the 4D interior penalty, re-score the 4 candidates from `optimize_acqf` using penalty-weighted posterior means, and select the best interior point via the existing median-filter + distance pipeline.

**Independent Test**: Run cells 58–60; the selected point (`next_x_ip`) should have all four coordinates in [0.05, 0.95] and min distance ≥ 0.05 from training data.

### Implementation for User Story 1

- [X] T003 [US1] Append code cell (Cell 60, ID: IP-02) implementing interior penalty computation, candidate re-scoring (`weighted_means = pred_means_orig * interior_weight`), median filter on weighted means, distance-based selection, clipping to [0, 0.999999], diagnostic table, and min-distance warning in `functions/f5/f5.ipynb`

**Checkpoint**: Core penalty logic complete. `next_x_ip`, `interior_weight`, `weighted_means`, `min_dist_ip` available in kernel. US1 is independently testable.

---

## Phase 4: User Story 3 — Visualise Interior Penalty Effect (Priority: P2)

**Goal**: 3-panel surrogate visualisation (GP mean, GP std, penalised mean) on a 2D slice through the top-2 important dimensions, plus a convergence plot.

**Independent Test**: Panel 3 shows near-zero penalised mean values along all four edges of the 2D slice; convergence plot matches Week 7 style.

### Implementation for User Story 3

- [X] T004 [US3] Append code cell (Cell 61, ID: IP-03) with 3-panel visualisation: Panel 1 GP posterior mean (viridis), Panel 2 GP posterior std (magma), Panel 3 penalised mean `mean_orig * w(x)` on 80x80 grid (plasma), red scatter + magenta star for `next_x_ip`, suptitle with STEEPNESS/FLOOR values, in `functions/f5/f5.ipynb`
- [X] T005 [US3] Append code cell (Cell 62, ID: IP-04) with convergence plot: `np.maximum.accumulate(y_raw)` running best, individual observations, Week 6→7 boundary at x=26.5, print best observed value and index, in `functions/f5/f5.ipynb`

**Checkpoint**: Visualisation complete. Penalised acquisition surface confirms boundary suppression.

---

## Phase 5: User Story 1 (cont.) — Submission Query (Priority: P1)

**Goal**: Format and validate the selected interior-penalised point as a submission string.

- [X] T006 [US1] Append code cell (Cell 63, ID: IP-05) with submission query: clip `next_x_ip` to [0, 0.999999], format as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`, validate 4 dimensions in range, print summary with surrogate type, acquisition type, interior penalty parameters, and fitted lengthscales, in `functions/f5/f5.ipynb`

**Checkpoint**: Complete feature — all 6 cells appended, notebook produces penalised submission.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all cells.

- [X] T007 Validate syntax of all 6 new cells by parsing notebook JSON and compiling each code cell with `py_compile` or `ast.parse`
- [X] T008 Verify all contract variables (`STEEPNESS`, `FLOOR`, `interior_weight`, `weighted_means`, `next_x_ip`, `min_dist_ip`) are produced by the correct cells per contracts/cells.md
- [X] T009 Run quickstart.md verification checklist: no errors, coordinates in [0.05, 0.95], min distance ≥ 0.05, 3-panel plot boundary suppression, submission format valid

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: T001, T002 — must complete first; define hyperparameters all cells need
- **US1 Core (Phase 3)**: T003 — depends on T002 (needs `STEEPNESS`, `FLOOR`)
- **US3 Visualisation (Phase 4)**: T004, T005 — depend on T003 (need `next_x_ip`)
- **US1 Submission (Phase 5)**: T006 — depends on T003 (needs `next_x_ip`)
- **Polish (Phase 6)**: T007–T009 — depend on all implementation tasks

### User Story Dependencies

- **US1 (P1)**: T002 → T003 → T006 (sequential — same kernel state)
- **US2 (P2)**: T001, T002 (foundational — no further tasks)
- **US3 (P2)**: T003 → T004, T005 (viz depends on penalty output)

### Within Each Phase

- T001 and T002 are sequential (markdown before constants, both foundational)
- T004 and T005 are sequential (cell ordering in notebook matters)
- T003 → T006 sequential (selection before submission)

### Parallel Opportunities

- T004 and T005 content can be _developed_ in parallel (different cell logic) but must be _appended_ in order
- T007 and T008 can run in parallel (syntax check vs. variable audit)

---

## Parallel Example: US3 Visualisation

```bash
# After T003 completes, both visualisation cells can be developed simultaneously:
Task T004: "3-panel surrogate viz in Cell 61"
Task T005: "Convergence plot in Cell 62"
# But append T004 before T005 to maintain cell order
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T001 + T002: Foundational markdown + constants
2. T003: Core penalty + selection
3. T006: Submission query
4. **VALIDATE**: Run cells 58–60, 63 — verify `next_x_ip` has interior coordinates

### Incremental Delivery

1. T001–T003 → Core penalty works, candidate selected (MVP)
2. T004–T005 → Visualisation confirms boundary suppression
3. T006 → Submission string ready
4. T007–T009 → Polish and validation
5. Each increment adds value without breaking previous cells

---

## Notes

- All 6 cells appended to `functions/f5/f5.ipynb` after cell 57
- STEEPNESS defaults to 1.0 (not 2.0 as in F1) — research.md §1 explains the 4D reduction
- Penalty applied to `pred_means_orig` (posterior mean), not NEI acq value — research.md §2
- No new files created; no existing cells modified
- Total: 9 tasks (2 foundational, 4 implementation, 3 polish)
