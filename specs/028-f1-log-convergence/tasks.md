# Tasks: F1 Log-Scale Convergence Plot

**Input**: Design documents from `/specs/028-f1-log-convergence/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: No tests required (per constitution).

**Organization**: Single user story — all tasks in sequence.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify branch and environment

- [x] T001 Verify branch `028-f1-log-convergence` is checked out and data files exist in `data/f1/`

---

## Phase 2: User Story 1 — View F1 convergence on log scale (Priority: P1) 🎯 MVP

**Goal**: Change the F1 convergence subplot to clip negative outputs to zero and use a plain log y-axis scale, so output values spanning ~230 orders of magnitude are visually distinguishable.

**Independent Test**: Run all cells of `functions/results/process_results.ipynb`, verify F1 subplot has log-scale y-axis while F2–F8 remain linear.

### Implementation

- [x] T002 [US1] Add `np.maximum(out, 0)` clipping for F1 outputs before plotting in convergence graph cell (Cell 13) of `functions/results/process_results.ipynb`
- [x] T003 [US1] Add `ax.set_yscale('log')` and update ylabel to `'Output Value (log)'` for F1 subplot in Cell 13 of `functions/results/process_results.ipynb`

**Checkpoint**: F1 subplot renders with logarithmic y-axis; zero/negative points omitted; F2–F8 unchanged.

---

## Phase 3: Validation

**Purpose**: End-to-end verification against success criteria

- [x] T004 Run all cells of `functions/results/process_results.ipynb` and verify SC-001 (F1 log ticks), SC-002 (no errors), SC-003 (F2–F8 unchanged)

---

## Dependencies

```text
T001 → T002 → T003 → T004
```

All tasks are sequential (single file, single cell).

## Parallel Execution

No parallel opportunities — all tasks modify the same cell in the same file.

## Implementation Strategy

- **MVP**: T001–T004 (the entire feature is the MVP)
- **Total tasks**: 4
- **Files modified**: 1 (`functions/results/process_results.ipynb`, Cell 13)
- **Key decision**: Clip negative F1 outputs to zero via `np.maximum(out, 0)`, then use `ax.set_yscale('log')` (from research.md R1 — user directed, no symlog)
