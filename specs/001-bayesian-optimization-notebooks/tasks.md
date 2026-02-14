# Tasks: Bayesian Optimization Notebooks for 8 Black Box Problems

**Feature**: `001-bayesian-optimization-notebooks`
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)
**Generated**: 2026-02-14

**Tests**: No unit tests (per CONSTITUTION). Validation is manual notebook execution.

**Organization**: Tasks grouped by user story. US1/US3/US5 are P1 (MVP); US2/US4 are P2.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US5)
- Exact file paths included in descriptions

## Path Conventions

- **Notebooks**: `functions/f{1-8}/f{N}.ipynb`, `functions/results/process_results.ipynb`
- **Data**: `data/f{1-8}/`, `data/results/`
- **Specs**: `specs/001-bayesian-optimization-notebooks/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify environment and data integrity before notebook work begins

- [X] T001 Verify sdd-dev Python environment has all dependencies (botorch, gpytorch, torch, numpy, matplotlib, pandas) installed
- [X] T002 Verify all initial data files exist and have expected shapes across data/f1/ through data/f8/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Apply kernel fixes needed across multiple notebooks before any story-specific work

**⚠️ CRITICAL**: Notebooks f6, f7, f8 will crash without the base_kernel fix

- [ ] T003 Apply conditional `hasattr(gp_model.covar_module, 'base_kernel')` kernel fix to all hyperparameter display and visualization cells in functions/f6/f6.ipynb
- [ ] T004 [P] Apply conditional `hasattr(gp_model.covar_module, 'base_kernel')` kernel fix to all hyperparameter display and visualization cells in functions/f7/f7.ipynb
- [ ] T005 [P] Apply conditional `hasattr(gp_model.covar_module, 'base_kernel')` kernel fix to all hyperparameter display and visualization cells in functions/f8/f8.ipynb

**Checkpoint**: Foundation ready — all 8 notebooks can now be executed without kernel AttributeErrors

---

## Phase 3: User Story 1 — Initial BO Implementation & Validation (Priority: P1) 🎯 MVP

**Goal**: All 8 notebooks execute end-to-end on initial data, producing valid next-point proposals within [0, 0.999999] bounds

**Independent Test**: Execute each notebook (f1–f8) cell-by-cell with initial/updated data, verify GP trains, acquisition optimizes, next point is within bounds, no errors occur

### Implementation for User Story 1

- [ ] T006 [US1] Execute and validate functions/f3/f3.ipynb end-to-end (3D drug discovery, 15 initial samples) — verify GP trains, next_point within bounds
- [ ] T007 [P] [US1] Execute and validate functions/f5/f5.ipynb end-to-end (4D chemical process, 20 initial samples) — verify GP trains, next_point within bounds
- [ ] T008 [P] [US1] Execute and validate functions/f6/f6.ipynb end-to-end (5D cake recipe, 20 initial samples) — verify GP trains, next_point within bounds
- [ ] T009 [P] [US1] Execute and validate functions/f7/f7.ipynb end-to-end (6D ML model, 30 initial samples) — verify GP trains, next_point within bounds
- [ ] T010 [P] [US1] Execute and validate functions/f8/f8.ipynb end-to-end (8D hyperparameter tuning, 40 initial samples) — verify GP trains, next_point within bounds
- [ ] T011 [US1] Re-validate functions/f1/f1.ipynb, functions/f2/f2.ipynb, functions/f4/f4.ipynb still execute cleanly (2D, 2D, 4D — previously tested, confirm no regressions)
- [ ] T012 [US1] Extract all 8 next_point proposals and format as x1-x2-...-xn (6 decimal places, each starting with 0) per FR-009

**Checkpoint**: All 8 notebooks produce valid next-point submissions — US1 is complete

---

## Phase 4: User Story 3 — Hyperparameter Documentation & Justification (Priority: P1)

**Goal**: Each notebook explicitly documents and justifies hyperparameter choices (NUM_RESTARTS, RAW_SAMPLES, kernel, bounds) per FR-003/FR-005

**Independent Test**: Read each notebook's markdown and code cells; verify hyperparameters are printed with text explanations of selection rationale

### Implementation for User Story 3

- [ ] T013 [P] [US3] Verify hyperparameter documentation in functions/f1/f1.ipynb — NUM_RESTARTS, RAW_SAMPLES, kernel choice, bounds, learned noise/lengthscales with interpretation
- [ ] T014 [P] [US3] Verify hyperparameter documentation in functions/f2/f2.ipynb — same checks
- [ ] T015 [P] [US3] Verify hyperparameter documentation in functions/f3/f3.ipynb — same checks
- [ ] T016 [P] [US3] Verify hyperparameter documentation in functions/f4/f4.ipynb — same checks
- [ ] T017 [P] [US3] Verify hyperparameter documentation in functions/f5/f5.ipynb — same checks
- [ ] T018 [P] [US3] Verify hyperparameter documentation in functions/f6/f6.ipynb — same checks
- [ ] T019 [P] [US3] Verify hyperparameter documentation in functions/f7/f7.ipynb — same checks
- [ ] T020 [P] [US3] Verify hyperparameter documentation in functions/f8/f8.ipynb — same checks

**Checkpoint**: All 8 notebooks have documented hyperparameter justifications — US3 is complete

---

## Phase 5: User Story 5 — Weekly Results Processing & Data Pipeline (Priority: P1)

**Goal**: The results processing notebook correctly parses weekly text files, creates cumulative .npy files, displays tabular data, and shows convergence plots with running maximum

**Independent Test**: Run process_results.ipynb with week=5 using existing data/results/ text files; verify 5 records parsed, .npy files saved with correct shapes, tables show "Initial"/"Week N" labels, convergence plot shows running max only

### Implementation for User Story 5

- [X] T021 [US5] Fix multi-line array parsing in functions/results/process_results.ipynb — replace line-by-line readlines()+eval() with bracket-depth grouping that accumulates physical lines until bracket count balances, handling f8's 8D arrays that wrap across 2 physical lines (FR-013)
- [X] T022 [US5] Add fail-fast error handling around eval() in parse_inputs_file() and parse_outputs_file() in functions/results/process_results.ipynb — wrap each record's eval() in try/except, raise clear ValueError with record number and content on failure (FR-013)
- [X] T023 [US5] Add out-of-range input validation after parsing in functions/results/process_results.ipynb — scan all parsed input values for <=0.0 or >=1.0, print warning with function name and row index, do not block processing (edge case)
- [X] T024 [US5] Add overwrite guard before saving .npy files in functions/results/process_results.ipynb — check if updated files for Week X already exist across all 8 function folders, warn and prompt user for confirmation via input(), skip save if not confirmed (FR-014)
- [X] T025 [US5] Fix convergence plot in functions/results/process_results.ipynb — remove running_min line and np.minimum.accumulate, keep only running_max as "Best Found" (green solid line), remove "assumes minimisation" comment (FR-016, CONSTITUTION maximisation)
- [X] T026 [US5] Fix summary stats in functions/results/process_results.ipynb — change header from "Best Input (at Min)" to "Best Input (at Max)", change best_input to use argmax instead of argmin (CONSTITUTION maximisation)
- [X] T027 [US5] Remove unused `import ast` from functions/results/process_results.ipynb (code cleanup)
- [X] T028 [US5] Validate process_results.ipynb end-to-end with week=5 using data/results/ text files — confirm 5 records parsed per function, .npy shapes correct (f1: (15,2) inputs, (15,) outputs; f8: (45,8) inputs, (45,) outputs), convergence plot shows running max, tables show "Initial"/"Week N" labels

**Checkpoint**: Results processing pipeline works end-to-end — US5 is complete

---

## Phase 6: User Story 2 — Module-by-Module Iterative Updates (Priority: P2)

**Goal**: Demonstrate weekly update workflow — load updated data, add new notebook section, retrain GP, propose new next_point without modifying old cells

**Independent Test**: Add updated_inputs/outputs Week 5 files to data/f1/, create new "Week 5" section in f1.ipynb, re-execute BO workflow, verify new next_point differs from previous

### Implementation for User Story 2

- [ ] T029 [US2] Demonstrate weekly iteration workflow in functions/f1/f1.ipynb — add a new "## Week 5 Update" section with cells to load updated data (Week 5 .npy files), retrain GP on expanded dataset, propose new next_point, without modifying any existing cells (FR-010, FR-011)
- [ ] T030 [US2] Verify convergence plot in the new Week 5 section of functions/f1/f1.ipynb shows improvement trend across all iterations including new observations (FR-008)

**Checkpoint**: Weekly iteration pattern demonstrated and validated — US2 is complete

---

## Phase 7: User Story 4 — Problem-Specific Visualizations (Priority: P2)

**Goal**: Surrogate, acquisition, and convergence visualizations render correctly across all dimensionalities (2D–8D)

**Independent Test**: Execute visualization cells for 2D problems (f1–f2) to verify contour plots, and for higher-D problems (f3–f8) to verify lengthscale-guided 2D slice visualizations

### Implementation for User Story 4

- [ ] T031 [P] [US4] Verify 2D contour plot visualizations in functions/f1/f1.ipynb — GP mean, uncertainty, acquisition heatmap with observed points and next_point marked (FR-006, FR-007)
- [ ] T032 [P] [US4] Verify 2D contour plot visualizations in functions/f2/f2.ipynb — same checks
- [ ] T033 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f3/f3.ipynb — identify 2 most important dimensions (smallest lengthscales), create contour plot fixing others at best observed values (FR-006)
- [ ] T034 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f4/f4.ipynb — same approach
- [ ] T035 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f5/f5.ipynb — same approach
- [ ] T036 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f6/f6.ipynb — same approach
- [ ] T037 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f7/f7.ipynb — same approach
- [ ] T038 [P] [US4] Verify or implement lengthscale-guided 2D slice visualization in functions/f8/f8.ipynb — same approach

**Checkpoint**: All 8 notebooks have appropriate dimensionality-aware visualizations — US4 is complete

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and consistency checks across all notebooks

- [ ] T039 Verify all 8 function notebooks use consistent cell structure: data loading → hyperparameters → GP training → acquisition optimization → visualization → progress tracking (NFR-005)
- [ ] T040 Verify all notebooks use float64 precision for GP computations (NFR-006)
- [ ] T041 Final end-to-end validation: execute all 8 function notebooks plus process_results.ipynb, confirm no errors and all outputs are valid

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS f6/f7/f8 execution
- **US1 (Phase 3)**: Depends on Foundational — core BO validation
- **US3 (Phase 4)**: Depends on US1 — hyperparameters must exist to document
- **US5 (Phase 5)**: Independent of US1 — only touches process_results.ipynb
- **US2 (Phase 6)**: Depends on US1 + US5 — needs working notebooks AND .npy pipeline
- **US4 (Phase 7)**: Depends on US1 — needs working GP models to visualize
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational (Phase 2) — no dependencies on other stories
- **US3 (P1)**: Can start after US1 — needs trained models to document hyperparameters
- **US5 (P1)**: Can start after Setup (Phase 1) — independent of US1, touches separate notebook
- **US2 (P2)**: Depends on US1 (working notebooks) and US5 (data pipeline to produce .npy files)
- **US4 (P2)**: Can start after US1 — needs working GP models for visualization

### Parallel Opportunities

- T003, T004, T005 (kernel fixes for f6, f7, f8) can run in parallel
- T006–T010 (notebook validation for f3, f5–f8) can run in parallel after foundational fixes
- T013–T020 (hyperparameter verification) are all parallelizable
- T021–T027 (process_results.ipynb fixes) are sequential within the same file
- T031–T038 (visualization verification) are all parallelizable
- US1 and US5 can be worked on in parallel (separate files)

---

## Parallel Example: User Story 1

```
After Foundational phase completes, validate all pending notebooks in parallel:
Task T006: Execute and validate functions/f3/f3.ipynb
Task T007: Execute and validate functions/f5/f5.ipynb
Task T008: Execute and validate functions/f6/f6.ipynb
Task T009: Execute and validate functions/f7/f7.ipynb
Task T010: Execute and validate functions/f8/f8.ipynb
```

## Parallel Example: User Story 5

```
US5 tasks are sequential (same file), but US5 as a whole runs parallel with US1:
Developer A works on US1 (function notebooks)
Developer B works on US5 (process_results.ipynb)
```

---

## Implementation Strategy

### MVP First (US1 + US3 + US5)

1. Complete Phase 1: Setup — verify environment
2. Complete Phase 2: Foundational — apply kernel fixes to f6/f7/f8
3. Complete Phase 3: US1 — validate all 8 notebooks, extract next_points
4. Complete Phase 4: US3 — verify hyperparameter documentation
5. Complete Phase 5: US5 — fix and validate results processing notebook
6. **STOP and VALIDATE**: All P1 stories independently testable

### Incremental Delivery

1. Setup + Foundational → Environment ready
2. US1 → All 8 notebooks produce valid submissions (MVP!)
3. US3 → Documentation meets reviewer requirements
4. US5 → Data pipeline automates weekly result processing
5. US2 → Weekly iteration workflow demonstrated
6. US4 → Dimensionality-aware visualizations complete
7. Polish → Final consistency and validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- No unit tests per CONSTITUTION — validation is manual notebook execution
- All problems are maximisation tasks — convergence should track running maximum
- Initial sample counts vary: f1=10, f2=10, f3=15, f4=30, f5=20, f6=20, f7=30, f8=40
- Input dimensions: f1=2D, f2=2D, f3=3D, f4=4D, f5=4D, f6=5D, f7=6D, f8=8D
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
