# Tasks: Bayesian Optimization Notebooks for 8 Black Box Problems

**Feature**: `001-bayesian-optimization-notebooks`  
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)  
**Generated**: 2026-02-07

## Overview

This task list implements Bayesian Optimization solutions for 8 black box problems (f1-f8) using BoTorch. Tasks are organized by user story to enable independent implementation and testing. Each phase represents a deliverable increment.

**Tests**: Not required per CONSTITUTION ("No unit tests are required"). Validation through manual notebook execution and visual inspection.

## Task Format

```
- [ ] [TaskID] [P?] [Story?] Description with file path
```

- **[P]**: Parallelizable task (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label [US1], [US2], [US3], [US4] (omit for Setup/Foundational/Polish phases)
- **TaskID**: Sequential execution order (T001, T002, T003...)

---

## Phase 1: Setup

**Purpose**: Initialize feature branch and specification documents

**Status**: ✅ COMPLETE

- [x] T001 Create feature branch `001-bayesian-optimization-notebooks`
- [x] T002 Generate feature specification in specs/001-bayesian-optimization-notebooks/spec.md
- [x] T003 Run clarification workflow and document 5 Q&A decisions
- [x] T004 Generate implementation plan in specs/001-bayesian-optimization-notebooks/plan.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Apply kernel structure fixes to ALL notebooks before execution/validation

**⚠️ CRITICAL**: These fixes must be complete before notebooks can be executed without errors. Fixes address BoTorch SingleTaskGP kernel structure variations (ScaleKernel wrapper may or may not be present).

### Kernel Attribute Access Fixes

- [x] T005 Apply outputscale conditional check to functions/f2/f2.ipynb hyperparameter display cell
- [x] T006 Apply outputscale conditional check to functions/f3/f3.ipynb hyperparameter display cell
- [x] T007 Apply base_kernel conditional check to functions/f4/f4.ipynb GP training cell (lengthscale access)
- [x] T008 Apply base_kernel conditional check to functions/f4/f4.ipynb visualization cell (lengthscale bar chart)
- [x] T009 [P] Apply base_kernel conditional check to functions/f6/f6.ipynb visualization cell at line ~155
- [x] T010 [P] Apply base_kernel conditional check to functions/f7/f7.ipynb visualization cell at line ~154
- [x] T011 [P] Apply base_kernel conditional check to functions/f8/f8.ipynb visualization cell at line ~170

**Fix Pattern**:
```python
# For lengthscale access:
if hasattr(gp_model.covar_module, 'base_kernel'):
    ls = gp_model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
else:
    ls = gp_model.covar_module.lengthscale.detach().numpy()[0]
```

**Checkpoint**: All kernel access is now defensive - notebooks ready for execution

---

## Phase 3: User Story 1 - Initial BO Implementation & Validation (Priority: P1) 🎯 MVP

**Goal**: Execute and validate all 8 notebooks with initial data, generate Week 1 submission coordinates

**Independent Test**: Each notebook executes without errors, produces valid next_point within bounds, displays GP hyperparameters, renders visualizations

**Why P1**: This is the foundational MVP - without working implementations, no submissions can be made and no iterative optimization can occur. Delivers immediate value by enabling Week 1 submissions for the 15-week campaign.

### Execute & Validate f1-f2 (2D Problems)

**Status**: ✅ f1 and f2 complete and validated

- [x] T012 [US1] Execute all cells in functions/f1/f1.ipynb and verify outputs
- [x] T013 [US1] Extract next_point from f1: expected format [x1, x2] ∈ [0,1]²
- [x] T014 [US1] Execute all cells in functions/f2/f2.ipynb and verify outputs
- [x] T015 [US1] Extract next_point from f2: expected format [x1, x2] ∈ [0,1]²

### Execute & Validate f3 (3D Problem)

- [x] T016 [US1] Start Python kernel (sdd-dev) for functions/f3/f3.ipynb
- [x] T017 [US1] Execute imports cell in functions/f3/f3.ipynb
- [x] T018 [US1] Execute data loading cells and verify 3D input shape
- [x] T019 [US1] Execute hyperparameters cell and verify data-driven bounds calculation
- [x] T020 [US1] Execute GP training cell and verify model trains successfully with outputscale fix
- [x] T021 [US1] Execute acquisition optimization cell and verify next_point proposed
- [x] T022 [US1] Execute visualization cells and verify 3D-appropriate plots render
- [x] T023 [US1] Execute progress tracking cell and verify convergence plot displays
- [x] T024 [US1] Extract next_point from f3: expected format [x1, x2, x3] within data-driven bounds

### Execute & Validate f4 (4D Problem)

**Status**: ✅ f4 complete and validated

- [x] T025 [US1] Execute all cells in functions/f4/f4.ipynb and verify outputs
- [x] T026 [US1] Extract next_point from f4: expected format [x1, x2, x3, x4]

### Execute & Validate f5 (5D Problem)

- [x] T027 [US1] Start Python kernel (sdd-dev) for functions/f5/f5.ipynb
- [x] T028 [US1] Execute all cells sequentially in functions/f5/f5.ipynb
- [x] T029 [US1] Verify GP trains on 5D inputs without errors
- [x] T030 [US1] Verify lengthscale bar chart or feature importance visualization renders
- [x] T031 [US1] Extract next_point from f5: expected format [x1, ..., x5]

### Execute & Validate f6 (6D Problem)

- [x] T032 [US1] Start Python kernel (sdd-dev) for functions/f6/f6.ipynb
- [x] T033 [US1] Execute all cells sequentially in functions/f6/f6.ipynb with base_kernel fix applied
- [x] T034 [US1] Verify GP trains on 6D inputs and lengthscales are displayed correctly
- [x] T035 [US1] Verify visualization cell executes without AttributeError (fix validates)
- [x] T036 [US1] Extract next_point from f6: expected format [x1, ..., x6]

### Execute & Validate f7 (7D Problem)

- [x] T037 [US1] Start Python kernel (sdd-dev) for functions/f7/f7.ipynb
- [x] T038 [US1] Execute all cells sequentially in functions/f7/f7.ipynb with base_kernel fix applied
- [x] T039 [US1] Verify GP trains on 7D inputs and lengthscales are displayed correctly
- [x] T040 [US1] Verify visualization cell executes without AttributeError (fix validates)
- [x] T041 [US1] Extract next_point from f7: expected format [x1, ..., x7]

### Execute & Validate f8 (8D Problem)

- [ ] T042 [US1] Start Python kernel (sdd-dev) for functions/f8/f8.ipynb
- [ ] T043 [US1] Execute all cells sequentially in functions/f8/f8.ipynb with base_kernel fix applied
- [ ] T044 [US1] Verify GP trains on 8D inputs with high-dimensional hyperparameters (30 restarts, 4096 raw samples)
- [ ] T045 [US1] Verify parallel coordinates plot and lengthscale bar chart render correctly
- [ ] T046 [US1] Extract next_point from f8: expected format [x1, ..., x8]

### Validation Summary

- [ ] T047 [US1] Create validation summary table with all 8 next_point values
- [ ] T048 [US1] Verify all next_point coordinates fall within their respective problem bounds
- [ ] T049 [US1] Verify EI values are printed with interpretation for all problems
- [ ] T050 [US1] Verify GP hyperparameters (noise variance, lengthscales) are learned and displayed for all problems
- [ ] T051 [US1] Verify all visualizations rendered correctly (surrogate, acquisition, convergence)

**Deliverable**: Week 1 Submission Table with 8 next_point proposals ready for submission

**Checkpoint**: All 8 notebooks execute successfully, Week 1 deliverable complete

---

## Phase 4: User Story 3 - Hyperparameter Documentation & Justification (Priority: P1)

**Goal**: Verify hyperparameters are explicitly documented with justifications in all notebooks

**Independent Test**: Read each notebook markdown and code cells to confirm NUM_RESTARTS, RAW_SAMPLES, kernel choice, acquisition function are printed with text explanations

**Why P1**: Mandatory project requirement per CONSTITUTION - must be included from Week 1 submission

**Status**: Partially complete - f1, f2, f4 have hyperparameter documentation

### Verify Existing Documentation

- [x] T052 [US3] Verify functions/f1/f1.ipynb has hyperparameter section with justifications
- [x] T053 [US3] Verify functions/f2/f2.ipynb has hyperparameter section with justifications
- [x] T054 [US3] Verify functions/f4/f4.ipynb has hyperparameter section with justifications

### Enhance Documentation for f3, f5-f8

- [ ] T055 [P] [US3] Review functions/f3/f3.ipynb markdown cells and verify hyperparameter justifications are clear
- [ ] T056 [P] [US3] Review functions/f5/f5.ipynb markdown cells and add/enhance hyperparameter justifications if missing
- [ ] T057 [P] [US3] Review functions/f6/f6.ipynb markdown cells and add/enhance hyperparameter justifications if missing
- [ ] T058 [P] [US3] Review functions/f7/f7.ipynb markdown cells and add/enhance hyperparameter justifications if missing
- [ ] T059 [P] [US3] Review functions/f8/f8.ipynb markdown cells and verify extensive 8D hyperparameter justifications

### Add Learned Hyperparameter Interpretation

- [ ] T060 [P] [US3] Verify functions/f1/f1.ipynb displays learned noise variance and lengthscales with interpretation
- [ ] T061 [P] [US3] Verify functions/f2/f2.ipynb displays learned noise variance and lengthscales with interpretation
- [ ] T062 [P] [US3] Add interpretation text to functions/f3/f3.ipynb for learned lengthscales (which dimensions matter most)
- [ ] T063 [P] [US3] Add interpretation text to functions/f4/f4.ipynb for learned lengthscales (feature sensitivity)
- [ ] T064 [P] [US3] Add interpretation text to functions/f5-f8 notebooks for learned lengthscales (feature importance)

### Add EI Interpretation

- [ ] T065 [P] [US3] Add EI value interpretation to all 8 notebooks: "High EI indicates potential improvement" or "Low EI suggests convergence"
- [ ] T066 [US3] Verify acceptance scenario 3: acquisition optimization result prints EI value with interpretation

**Checkpoint**: All notebooks have complete hyperparameter documentation meeting CONSTITUTION requirements

---

## Phase 5: User Story 4 - Problem-Specific Visualizations (Priority: P2)

**Goal**: Implement lengthscale-guided slicing visualization for high-dimensional problems (f3-f8)

**Independent Test**: Execute visualization cells for f3-f8 and verify 2D contour slices show the 2 most important dimensions (smallest lengthscales) with labels indicating which dimensions are displayed

**Why P2**: Required by CONSTITUTION for all problems but secondary to working optimization (US1 must complete first)

**Implementation Strategy** (per spec clarification Q1):
- Identify 2 dimensions with smallest GP lengthscales (most important)
- Create 2D contour plot slicing along those dimensions
- Fix other dimensions at best observed values
- Label which dimensions are shown in plot title

### Current State Assessment

- [x] T067 [US4] Verify functions/f1/f1.ipynb uses full 2D contour plots (already optimal for 2D)
- [x] T068 [US4] Verify functions/f2/f2.ipynb uses full 2D contour plots (already optimal for 2D)
- [ ] T069 [US4] Check functions/f3/f3.ipynb current visualization approach (pair plots or other)
- [ ] T070 [US4] Check functions/f4/f4.ipynb current visualization approach (pair plots or lengthscale bars)
- [ ] T071 [US4] Check functions/f5-f8 current visualization (parallel coordinates, lengthscale bars, etc.)

### Implement Lengthscale-Guided Slicing

**Note**: Only implement if current visualization doesn't already meet spec requirements

- [ ] T072 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f3/f3.ipynb surrogate visualization
- [ ] T073 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f4/f4.ipynb surrogate visualization (may keep pair plots as alternative)
- [ ] T074 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f5/f5.ipynb surrogate visualization
- [ ] T075 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f6/f6.ipynb surrogate visualization
- [ ] T076 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f7/f7.ipynb surrogate visualization
- [ ] T077 [P] [US4] If needed, implement lengthscale-guided 2D slice for functions/f8/f8.ipynb surrogate visualization (may keep parallel coordinates as alternative)

**Implementation Pattern**:
```python
# After GP training, identify most important dimensions
lengthscales = ... # extract from GP
dim_order = np.argsort(lengthscales)  # smallest first
important_dims = dim_order[:2]

# Create 2D grid for those dimensions
# Fix other dimensions at best observed values
# Evaluate GP posterior
# Plot 2D contour with labels
```

### Validation

- [ ] T078 [US4] Execute visualization cells for f3-f8 and verify plots show dimension labels
- [ ] T079 [US4] Verify spec acceptance scenario 2: higher-D notebooks create 2D slices along dimensions with smallest lengthscales

**Checkpoint**: All problems have appropriate dimension-specific visualizations

---

## Phase 6: User Story 2 - Weekly Iterative Updates (Priority: P2)

**Goal**: Demonstrate weekly update workflow by adding Week N section to one notebook

**Independent Test**: Load updated data, add new Week N section, re-run optimization, verify old cells unchanged, generate new next_point

**Why P2**: Enables core capstone workflow but depends on US1 working first

**Choice**: Demonstrate on functions/f1/f1.ipynb or functions/f2/f2.ipynb (simplest 2D cases)

### Demonstrate Weekly Iteration on f1

- [ ] T080 [US2] Load updated data for f1: combine initial + updated - Week 3 or Week 4 arrays
- [ ] T081 [US2] Add new markdown cell in functions/f1/f1.ipynb: "## Week 2 Update" (or appropriate week number)
- [ ] T082 [US2] Add markdown cell explaining new data: "Loaded N additional observations from Week X submission"
- [ ] T083 [US2] Add code cell to load and combine data: `X_combined = np.concatenate([X_init, X_week2])`
- [ ] T084 [US2] Add code cell to retrain GP on expanded dataset
- [ ] T085 [US2] Add code cell to optimize acquisition and propose new next_point
- [ ] T086 [US2] Add visualization cell to show updated surrogate and convergence Plot
- [ ] T087 [US2] Verify original Week 1 cells remain unchanged (append-only pattern)
- [ ] T088 [US2] Execute new Week 2 section and verify new next_point differs from Week 1
- [ ] T089 [US2] Verify convergence plot shows improvement (new observations added to curve)

### Validation

- [ ] T090 [US2] Verify spec acceptance scenario 1: old cells unchanged, new cells load combined data
- [ ] T091 [US2] Verify spec acceptance scenario 2: convergence plot shows improvement across iterations

**Deliverable**: Demonstrated weekly iteration pattern that can be replicated for all problems over 15-week campaign

**Checkpoint**: Weekly workflow validated, project ready for iterative optimization

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements, documentation, and project cleanup

### Documentation

- [ ] T092 [P] Update CONSTITUTION.md with any learnings from implementation (optional)
- [ ] T093 [P] Create submission summary document with all 8 Week 1 next_point values
- [ ] T094 [P] Document known issues and workarounds in plan.md or separate notes.md

### Code Quality

- [ ] T095 [P] Review all notebooks for code simplicity per CONSTITUTION
- [ ] T096 [P] Verify all markdown cells have clear step-by-step explanations
- [ ] T097 [P] Check execution time for all notebooks <5 minutes per spec NFR-001
- [ ] T097-A [P] Verify all notebooks follow NFR-005 consistent structure: data loading → hyperparameters → GP training → acquisition optimization → visualization → progress tracking
- [ ] T097-B [P] Verify all PyTorch tensors use torch.float64 dtype per NFR-006 for numerical stability in GP computations

### Final Validation

- [ ] T098 Run all 8 notebooks end-to-end one final time to confirm no regressions
- [ ] T099 Verify all success criteria from plan.md are met
- [ ] T100 Prepare Week 1 submission with 8 next_point values

**Checkpoint**: Project complete, ready for Week 1 submission and ongoing weekly iterations

---

## Dependencies & Execution Order

### Critical Path

1. **Phase 2 (Foundational)** → Blocks all notebook execution
2. **Phase 3 (US1)** → Generates Week 1 submission, unblocks US2
3. **Phase 4 (US3)** → Can run in parallel with Phase 3 after foundational fixes
4. **Phase 5 (US4)** → Depends on Phase 3 (notebooks must execute successfully)
5. **Phase 6 (US2)** → Depends on Phase 3 (at least one notebook validated)
6. **Phase 7 (Polish)** → Depends on all previous phases

### Parallelization Opportunities

Within Phase 3 (after T011 foundational fixes complete):
- f3, f5 execution can run in parallel (independent 3D and 5D problems)
- f6, f7, f8 execution can run in parallel (all need same base_kernel fix pattern)

Within Phase 4:
- All hyperparameter documentation review tasks (T055-T064) are parallelizable

Within Phase 5:
- All visualization implementation tasks (T072-T077) are parallelizable

---

## Implementation Strategy

**MVP Scope** (Week 1 Deliverable):
- Phase 1: Setup ✅
- Phase 2: Foundational (kernel fixes)
- Phase 3: US1 (execute & validate all 8 notebooks)
- Phase 4: US3 (verify hyperparameter documentation)
- Submit: 8 next_point values

**Incremental Delivery** (Future Weeks):
- Phase 5: US4 (enhanced visualizations) - Week 2-3
- Phase 6: US2 (weekly iteration demo) - Week 2-4
- Phase 7: Polish - Week 4-5

**Testing Strategy**:
- Manual execution: Run each notebook cell-by-cell
- Visual validation: Check plots render correctly
- Output verification: Confirm next_point within bounds, GP hyperparameters reasonable
- No automated tests per CONSTITUTION

---

## Success Criteria Mapping

From plan.md success metrics:

| Criterion | Tasks | Status |
|-----------|-------|--------|
| All 8 notebooks execute without errors | T012-T046 | 3 of 8 complete |
| Each produces valid next_point within bounds | T013, T015, T024, T026, T031, T036, T041, T046 | 3 of 8 complete |
| GP hyperparameters learned and displayed | T020, T029, T034, T039, T044 + T052-T064 | 3 of 8 validated |
| Visualizations render inline | T022, T030, T035, T040, T045 | 3 of 8 confirmed |
| Next submission coordinates extracted | T047-T051 | Pending |
| Weekly iteration demonstrated | T080-T091 | Not started |
| Lengthscale visualization for f3-f8 | T072-T079 | Not started |

---

**Tasks generated**: 2026-02-07  
**Total task count**: 100 tasks  
**Task count per user story**: US1: 40 tasks, US2: 12 tasks, US3: 15 tasks, US4: 11 tasks  
**Parallel opportunities**: 22 tasks marked [P] can run concurrently  
**MVP scope**: Tasks T001-T051, T052-T064, T092-T100 (67 tasks)
