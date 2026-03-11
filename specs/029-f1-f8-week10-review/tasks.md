# Tasks: Week 10 Performance Review & Visualisation

**Input**: Design documents from `/specs/029-f1-f8-week10-review/`
**Prerequisites**: plan.md (loaded), spec.md (loaded), research.md (loaded), data-model.md (loaded), quickstart.md (loaded)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify data availability and confirm branch is ready

- [ ] T001 Verify Week 10 data files exist in ./data/f1/ through ./data/f8/ (updated_inputs - Week 10.npy & updated_outputs - Week 10.npy)

---

## Phase 2: Foundational — Create Notebook Template Structure

**Purpose**: Establish the common notebook pattern that all 8 function notebooks share. Since all notebooks follow the same structure (data loading → convergence plot → pair plots → evaluation markdown → improvement markdown), create the first notebook as the reference implementation.

**⚠️ CRITICAL**: T002 establishes the pattern. All subsequent [P] notebooks in Phases 3–4 follow this pattern with function-specific constants and strategy text.

- [ ] T002 [US1] Create reference notebook functions/f1/f1 - week 10.ipynb with all cells: imports (numpy, matplotlib, itertools), FunctionConfig constants (FUNC_NUM=1, N_DIMS=2, N_INITIAL=10, USE_LOG_SCALE=True, DATA_DIR), data loading from ../../data/f1/updated_inputs - Week 10.npy and updated_outputs - Week 10.npy, data summary display, convergence plot with log y-axis (np.maximum(outputs, 0) before log, blue initial / orange submissions, vertical dashed line at sample 10), single 2D pair plot (only 1 pair for 2D) with blue initial unmarked / orange submissions numbered by week (3–10), performance evaluation markdown summarising week 9 Hurdle Model + Weighted UCB strategy and evaluating 0/9 improvements as stalling, and improvement suggestions markdown proposing specific changes to the F1 strategy

**Checkpoint**: F1 notebook is complete and serves as the template for F2–F8

---

## Phase 3: User Story 1 — Review Week 10 Data Across All Functions (Priority: P1) 🎯 MVP

**Goal**: Create all 8 notebooks with data loading, convergence plots, and 2D pair plots

**Independent Test**: Open any notebook, run all cells, verify convergence and pair plots render correctly

### Implementation for User Story 1

- [ ] T003 [P] [US1] Create notebook functions/f2/f2 - week 10.ipynb — adapt from F1 pattern: FUNC_NUM=2, N_DIMS=2, N_INITIAL=10, USE_LOG_SCALE=False, DATA_DIR=../../data/f2/, linear y-axis convergence plot, 1 pair plot subplot
- [ ] T004 [P] [US1] Create notebook functions/f3/f3 - week 10.ipynb — FUNC_NUM=3, N_DIMS=3, N_INITIAL=15, USE_LOG_SCALE=False, DATA_DIR=../../data/f3/, linear convergence plot, 3 pair plot subplots in 1×3 grid
- [ ] T005 [P] [US1] Create notebook functions/f4/f4 - week 10.ipynb — FUNC_NUM=4, N_DIMS=4, N_INITIAL=30, USE_LOG_SCALE=False, DATA_DIR=../../data/f4/, linear convergence plot, 6 pair plot subplots in 2×3 grid
- [ ] T006 [P] [US1] Create notebook functions/f5/f5 - week 10.ipynb — FUNC_NUM=5, N_DIMS=4, N_INITIAL=20, USE_LOG_SCALE=False, DATA_DIR=../../data/f5/, linear convergence plot, 6 pair plot subplots in 2×3 grid
- [ ] T007 [P] [US1] Create notebook functions/f6/f6 - week 10.ipynb — FUNC_NUM=6, N_DIMS=5, N_INITIAL=20, USE_LOG_SCALE=False, DATA_DIR=../../data/f6/, linear convergence plot, 10 pair plot subplots in 2×5 grid
- [ ] T008 [P] [US1] Create notebook functions/f7/f7 - week 10.ipynb — FUNC_NUM=7, N_DIMS=6, N_INITIAL=30, USE_LOG_SCALE=False, DATA_DIR=../../data/f7/, linear convergence plot, 15 pair plot subplots in 3×5 grid
- [ ] T009 [P] [US1] Create notebook functions/f8/f8 - week 10.ipynb — FUNC_NUM=8, N_DIMS=8, N_INITIAL=40, USE_LOG_SCALE=False, DATA_DIR=../../data/f8/, linear convergence plot, 28 pair plot subplots in 4×7 grid

**Checkpoint**: All 8 notebooks load data and display convergence + pair plots correctly

---

## Phase 4: User Story 2 — Identify Performance Issues & Propose Improvements (Priority: P2)

**Goal**: Add performance evaluation and strategy improvement markdown to each notebook

**Independent Test**: After running any notebook, verify markdown sections at the end evaluate performance and propose specific strategy improvements relative to week 9 configuration

### Implementation for User Story 2

> Note: T002 (F1) already includes both US1 and US2 content. The tasks below add the US2 markdown content to F2–F8.

- [ ] T010 [P] [US2] Add performance evaluation markdown to functions/f2/f2 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-1.5 ARD, qLogNEI q=4), evaluate 1/9 improvements, flag stalling and local-optimum trapping
- [ ] T011 [P] [US2] Add improvement suggestions markdown to functions/f2/f2 - week 10.ipynb — propose specific changes relative to SFGP Matérn-1.5 (e.g., increase exploration, change kernel, add restart mechanism)
- [ ] T012 [P] [US2] Add performance evaluation markdown to functions/f3/f3 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-2.5 ARD, qLogNEI q=1), evaluate 2/9 improvements, note slow but steady progress
- [ ] T013 [P] [US2] Add improvement suggestions markdown to functions/f3/f3 - week 10.ipynb — propose specific changes relative to SFGP Matérn-2.5 (e.g., increase q, adjust restarts, consider different acquisition)
- [ ] T014 [P] [US2] Add performance evaluation markdown to functions/f4/f4 - week 10.ipynb — summarise week 9 strategy (MFGP Matérn-2.5 + LinearTruncated, MF-qNEI q=4), evaluate 5/9 improvements but recent stalling, flag MFGP on single-fidelity data concern
- [ ] T015 [P] [US2] Add improvement suggestions markdown to functions/f4/f4 - week 10.ipynb — propose specific changes relative to MFGP (e.g., switch to SFGP since only single-fidelity data available, adjust noise bounds)
- [ ] T016 [P] [US2] Add performance evaluation markdown to functions/f5/f5 - week 10.ipynb — summarise week 9 strategy (GP Matérn-1.5 ARD, qLogNEI q=4, log1p + Standardize), evaluate 9/9 improvements, note strong performance
- [ ] T017 [P] [US2] Add improvement suggestions markdown to functions/f5/f5 - week 10.ipynb — propose specific refinements for continued improvement (e.g., fine-tune exploration/exploitation balance, review interior penalty parameters)
- [ ] T018 [P] [US2] Add performance evaluation markdown to functions/f6/f6 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-1.5 ARD, qLogNEI q=4, rank-based IP), evaluate 9/9 improvements, note all-negative but trending towards zero
- [ ] T019 [P] [US2] Add improvement suggestions markdown to functions/f6/f6 - week 10.ipynb — propose specific refinements (e.g., adjust milk constraint, review IP steepness, consider output transformation)
- [ ] T020 [P] [US2] Add performance evaluation markdown to functions/f7/f7 - week 10.ipynb — summarise week 9 strategy (NN 6→5→5→1, 70% mean + 30% EI, interior penalty), evaluate 4/9 improvements, flag stalling in recent weeks
- [ ] T021 [P] [US2] Add improvement suggestions markdown to functions/f7/f7 - week 10.ipynb — propose specific changes relative to NN surrogate (e.g., increase exploration weight, change network architecture, consider switch to GP-based surrogate)
- [ ] T022 [P] [US2] Add performance evaluation markdown to functions/f8/f8 - week 10.ipynb — summarise week 9 strategy (SFGP Matérn-2.5 ARD, qEI XI=0.01), evaluate 8/9 near-continuous improvement, note positive trajectory
- [ ] T023 [P] [US2] Add improvement suggestions markdown to functions/f8/f8 - week 10.ipynb — propose specific refinements for final rounds (e.g., fine-tune XI, increase MC samples if stable, review noise floor)

**Checkpoint**: All 8 notebooks have performance evaluation and improvement suggestion markdown sections

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all notebooks

- [ ] T024 Run all 8 notebooks top-to-bottom and verify no execution errors
- [ ] T025 Verify convergence plot styling consistency across all notebooks (blue/orange colours, dashed separator line, axis labels)
- [ ] T026 Verify pair plot week numbering is correct (3–10) across all notebooks and initial points are unmarked
- [ ] T027 Commit all 8 notebooks to branch 029-f1-f8-week10-review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify data exists
- **Foundational (Phase 2)**: Depends on Phase 1 — create F1 reference notebook
- **US1 (Phase 3)**: Depends on Phase 2 (F1 pattern established) — create F2–F8 notebooks
- **US2 (Phase 4)**: Depends on Phase 3 — add evaluation/improvement markdown to F2–F8
- **Polish (Phase 5)**: Depends on Phases 3 and 4 — validate all notebooks

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational (Phase 2). F2–F8 notebooks can all be created in parallel.
- **User Story 2 (P2)**: F1 evaluation is already in T002. F2–F8 evaluation tasks (T010–T023) can run in parallel once their respective notebooks exist.

> **Note**: In practice, each notebook (T003–T009) will be created with all content (US1 + US2) in a single pass, since the evaluation/improvement markdown cells are part of the notebook structure. The task separation is for tracking purposes — all content for a given function can be implemented together.

### Within Each User Story

- Data loading and convergence plot before pair plots (data must be loaded first)
- Pair plots before evaluation markdown (evaluation references plot observations)
- Evaluation before improvement suggestions (improvements reference evaluation findings)

### Parallel Opportunities

**Phase 3 (US1)**: T003–T009 can all run in parallel (7 independent notebooks, different files)

**Phase 4 (US2)**: T010–T023 can all run in parallel (each modifies a different notebook)

**Combined**: Since each notebook is a separate file, a practical approach creates each notebook completely (US1 + US2 content) in a single pass:
- T002 (F1) first as reference
- Then T003+T010+T011 (F2), T004+T012+T013 (F3), T005+T014+T015 (F4), T006+T016+T017 (F5), T007+T018+T019 (F6), T008+T020+T021 (F7), T009+T022+T023 (F8) — all 7 in parallel

---

## Implementation Strategy

### MVP Scope (Recommended)

**MVP = Phase 1 + Phase 2 + Phase 3 (User Story 1 only)**

This delivers all 8 notebooks with data loading and visualisations. The student can visually assess performance before the evaluation markdown is added.

### Incremental Delivery

1. **Increment 1**: F1 notebook complete (T001 + T002) — validates the pattern
2. **Increment 2**: F2–F8 notebooks with visualisations (T003–T009) — all functions visible
3. **Increment 3**: Performance evaluation + improvements for all functions (T010–T023) — strategy decisions enabled
4. **Increment 4**: Validation pass (T024–T027) — quality assurance

### Practical Note

Given that each notebook is self-contained and the content for US1 and US2 is tightly coupled (evaluation references the plots), the most efficient implementation creates each notebook in full (all cells) in a single pass per function. The task breakdown above separates concerns for tracking, but implementation should proceed function-by-function after the F1 reference is established.
