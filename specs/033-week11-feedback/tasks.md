# Tasks: Week 11 Best-Marker Visibility Fix (Green Star, s=500)

**Input**: Design documents from `/specs/033-week11-feedback/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not requested — no test tasks included.

**Scope**: This is the **second incremental fix** to the already-implemented 033-week11-feedback feature. The original full implementation (41 tasks) and the first marker-size fix (s=200→s=350, 12 tasks) are both complete. These tasks address continued user feedback: the red star remains invisible on pair plots. Two changes per notebook: (1) colour `c='red'` → `c='green'`, and (2) size `s=350` → `s=500`. The legend `markerfacecolor` and the code comment also update to match.

**Organization**: Only User Story 3 (Best Output Location on Pair Plots) is affected. Three lines change per notebook: the comment (line 36), the scatter call (line 38), and the legend element (line 52), all in cell index 7.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US3)
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: Verify current notebook state before applying the fix

- [X] T001 Verify all 8 notebooks contain `c='red'` and `s=350` in cell index 7, line 38, and `markerfacecolor='red'` in line 52 across functions/f1/ through functions/f8/

**Checkpoint**: All 8 notebooks confirmed at old values — ready for update

---

## Phase 2: User Story 3 — Green Star Marker Visibility (Priority: P1) 🎯 MVP

**Goal**: Change the best-output marker from red/s=350 to green/s=500 in all 8 notebooks so the marker is clearly visible on every pair plot subplot, including F7 (15 subplots) and F8 (28 subplots)

**Independent Test**: Open any notebook (especially F7 or F8), run all cells, verify the green star marker is clearly visible on every pair plot subplot

### Implementation

- [X] T002 [P] [US3] Update best marker to green/s=500 in functions/f1/f1 - week 11.ipynb cell[7]: comment line 36 (red→green), scatter line 38 (c='red',s=350 → c='green',s=500), legend line 52 (markerfacecolor='red' → markerfacecolor='green')
- [X] T003 [P] [US3] Update best marker to green/s=500 in functions/f2/f2 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T004 [P] [US3] Update best marker to green/s=500 in functions/f3/f3 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T005 [P] [US3] Update best marker to green/s=500 in functions/f4/f4 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T006 [P] [US3] Update best marker to green/s=500 in functions/f5/f5 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T007 [P] [US3] Update best marker to green/s=500 in functions/f6/f6 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T008 [P] [US3] Update best marker to green/s=500 in functions/f7/f7 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52
- [X] T009 [P] [US3] Update best marker to green/s=500 in functions/f8/f8 - week 11.ipynb cell[7]: comment line 36, scatter line 38, legend line 52

**Checkpoint**: All 8 notebooks updated — green star marker fix applied

---

## Phase 3: Polish & Cross-Cutting Concerns

**Purpose**: Re-run and validate all notebooks after the change

- [X] T010 Run all 8 notebooks top-to-bottom and verify zero errors (SC-001)
- [X] T011 Verify green star marker is clearly visible on all pair plot subplots across all 8 notebooks, especially F7 (15 subplots) and F8 (28 subplots) (SC-003)
- [X] T012 Run quickstart.md validation against specs/033-week11-feedback/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US3 Fix (Phase 2)**: Depends on Phase 1 verification
- **Polish (Phase 3)**: Depends on all Phase 2 tasks complete

### Parallel Opportunities

- All 8 notebook update tasks (T002–T009) can run in parallel — each modifies a different file with no cross-dependencies
- Validation tasks T010–T012 must run after all updates are applied

---

## Parallel Example: All Updates

```text
T001 (verify red/s=350 in all 8 notebooks)
  ↓
T002 (F1) ─┐
T003 (F2) ─┤
T004 (F3) ─┤
T005 (F4) ─┤  All [P] — run in parallel
T006 (F5) ─┤
T007 (F6) ─┤
T008 (F7) ─┤
T009 (F8) ─┘
  ↓
T010 (run all notebooks)
T011 (verify green star visible)
T012 (quickstart validation)
```

---

## Implementation Strategy

### MVP Scope

All tasks are part of the MVP — the entire change is a colour + size update in 8 files. No incremental delivery needed; the fix is atomic.

### Per-Notebook Change Summary

Each notebook has 3 edits in cell[7]:

| Line | Old | New |
|------|-----|-----|
| 36 | `# Best output — red star` | `# Best output — green star` |
| 38 | `c='red', marker='*', s=350, zorder=5, label='Best'` | `c='green', marker='*', s=500, zorder=5, label='Best'` |
| 52 | `markerfacecolor='red'` | `markerfacecolor='green'` |
