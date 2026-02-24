# Tasks: F6 Interior Penalty on Acquisition Function

**Input**: Design documents from `/specs/016-f6-interior-penalty/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cells.md, quickstart.md

**Tests**: No tests (per constitution — no unit tests required; manual execution only).

**Organization**: Tasks grouped by user story. All tasks target a single file: `functions/f6/f6.ipynb` (append cells 60–65 after existing cell 59).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1, US2, US3)
- All paths relative to repository root

---

## Phase 1: Setup

**Purpose**: Verify environment and existing notebook state before adding new cells.

- [ ] T001 Verify conda environment `sdd-dev` is active, Python 3.11 with torch/botorch/gpytorch/numpy/matplotlib available
- [ ] T002 Verify `functions/f6/f6.ipynb` has 60 cells (0–59) and existing Week 7 cells execute without errors

**Checkpoint**: Notebook runs end-to-end; all Week 7 variables (`candidates`, `pred_means`, `dists`, `best_model`, `X_train`, `y_raw`, `ls`) are populated.

---

## Phase 2: User Story 1 — Add Interior Penalty to F6 Acquisition (Priority: P1) 🎯 MVP

**Goal**: Apply a rank-based interior penalty to the 4 NEI candidates, selecting the best point that balances posterior mean quality with interior location — correctly handling all-negative outputs.

**Independent Test**: Run all cells top-to-bottom. The selected `best_point` should have all coordinates away from boundaries (x0–x3 ≥ 0.05, x4 ≥ 0.15). The comparison table should show that boundary candidates are demoted relative to interior candidates.

### Implementation for User Story 1

- [ ] T003 [US1] Append markdown cell 60 — section header "Week 7 — Interior Penalty Re-Scoring" with hyperparameter table documenting STEEPNESS=1.0 and FLOOR=0.01 in `functions/f6/f6.ipynb`
- [ ] T004 [US1] Append code cell 61 — compute `interior_weight` (4,) from `candidates` (4,5) using `w(x) = FLOOR + (1-FLOOR) · ∏ sin(πxᵢ)^(2·STEEPNESS)` with assertions and boundary/interior labels in `functions/f6/f6.ipynb`
- [ ] T005 [US1] Append code cell 62 — rank-based re-scoring (`rank_mean`, `rank_weight`, `combined_score`), median filter on `combined_score`, distance-based selection among above-median, fallback logic, comparison table with `◄` marker, feasibility assertions in `functions/f6/f6.ipynb`
- [ ] T006 [US1] Append code cell 65 — submission query printing `best_point` to 4 decimal places, feasibility validation (all ≥ 0.01, x4 ≥ 0.10), penalty metadata (STEEPNESS, FLOOR, weight, score), raw-vs-penalty comparison in `functions/f6/f6.ipynb`

**Checkpoint**: Cells 60–62 + 65 execute without errors. `best_point` shape is (5,), feasibility bounds satisfied, comparison table printed.

---

## Phase 3: User Story 2 — Explicit Hyperparameter Documentation (Priority: P2)

**Goal**: Ensure STEEPNESS and FLOOR are documented with rationale in the markdown cell, satisfying capstone examiner requirements.

**Independent Test**: Read cell 60 markdown — hyperparameter table present with values and rationales. Cell 61 prints the constants.

> **Note**: User Story 2 is already satisfied by T003 (markdown cell) and T004 (constant definitions with print). No additional tasks needed — this story is delivered as part of US1 implementation.

**Checkpoint**: Hyperparameters documented in markdown (cell 60) and defined as named constants (cell 61).

---

## Phase 4: User Story 3 — Visualise Interior Penalty Effect (Priority: P2)

**Goal**: Replace the dimension-relevance bar chart with a 3-panel visualisation showing posterior mean, posterior std, and interior penalty heatmap on the top-2 ARD dimensions.

**Independent Test**: Three panels displayed. Panel 3 shows penalty weight colour-mapped from ~0.01 (red, boundary) to ~1.0 (green, interior) with `best_point` star marker.

### Implementation for User Story 3

- [ ] T007 [US3] Append code cell 63 — 3-panel visualisation: Panel 1 (posterior mean heatmap on top-2 ARD dims, 80×80 grid), Panel 2 (posterior std heatmap), Panel 3 (interior penalty `w(x)` heatmap, RdYlGn cmap, candidate scatter by combined_score, best_point star) in `functions/f6/f6.ipynb`
- [ ] T008 [US3] Append code cell 64 — convergence plot showing `y_raw` history, IP-selected predicted mean line, best raw candidate mean dashed line for comparison in `functions/f6/f6.ipynb`

**Checkpoint**: Cells 63–64 execute without errors. 3-panel figure shows penalty suppression at edges. Convergence plot shows both IP-selected and raw-best lines.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and commit.

- [ ] T009 Run all notebook cells top-to-bottom (cells 0–65) and verify no errors, all assertions pass, all plots render
- [ ] T010 Verify cells 0–59 are unmodified (diff check against master)
- [ ] T011 Run quickstart.md verification checklist (11 items) against notebook output

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify environment first
- **US1 (Phase 2)**: Depends on Setup — core penalty logic
- **US2 (Phase 3)**: Satisfied by US1 tasks — no additional work needed
- **US3 (Phase 4)**: Depends on US1 (needs `best_point`, `combined_score`, `interior_weight` from cells 61–62)
- **Polish (Phase 5)**: Depends on all user stories complete

### Task Dependencies

```text
T001 ──► T002 ──► T003 ──► T004 ──► T005 ──► T006
                                       │
                                       ▼
                                T007 ──► T008
                                           │
                                           ▼
                                T009 ──► T010 ──► T011
```

### Cell Ordering Constraints

All cells are appended sequentially to one notebook — no parallelism within the notebook file. Tasks are sequential:
- Cell 60 (T003) → Cell 61 (T004) → Cell 62 (T005) → Cell 63 (T007) → Cell 64 (T008) → Cell 65 (T006)
- Note: T006 (submission cell 65) is listed in Phase 2 for user story grouping, but is appended last in notebook order.

---

## Parallel Example: User Story 1

```text
# No parallel tasks — all cells append to one file sequentially.
# Execute in order: T003 → T004 → T005 → T006
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup verification (T001–T002)
2. Complete Phase 2: US1 core penalty (T003–T006) — cells 60, 61, 62, 65
3. **STOP and VALIDATE**: `best_point` produced, feasibility passes, comparison table shows penalty effect
4. This is a functional submission-ready notebook

### Incremental Delivery

1. Setup → verify environment
2. US1 (cells 60–62, 65) → core penalty logic, submission-ready
3. US3 (cells 63–64) → visualisation and convergence, examiner-ready
4. Polish → final validation against quickstart checklist

---

## Notes

- All tasks modify a single file: `functions/f6/f6.ipynb`
- No new files created, no new dependencies added
- Cell ordering in notebook: 60 (md), 61 (code), 62 (code), 63 (code), 64 (code), 65 (code)
- US2 has no dedicated tasks — it's satisfied by the documentation in T003 and T004
- Total: 11 tasks across 5 phases
- The rank-based re-scoring (T005) is the critical task — it solves the all-negative output problem unique to F6
