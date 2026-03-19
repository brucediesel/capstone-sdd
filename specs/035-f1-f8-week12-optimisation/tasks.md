# Tasks: Week 12 Bayesian Optimisation Loop (F1–F8)

**Input**: Design documents from `/specs/035-f1-f8-week12-optimisation/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Not requested — verification is manual notebook execution per constitution.

**Organization**: US1 (generate submission candidates) and US2 (performance visualisations) are co-delivered in each notebook since every notebook contains both the BO loop and all visualisations. US1 covers notebook creation; US2 covers execution and visual verification.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 or US2)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Verify environment prerequisites

- [X] T001 Verify Python environment (pyenv sdd-dev, Python 3.14.2) and dependencies (BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch 2.10.0, NumPy 2.4.1, Matplotlib 3.10.8)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Verify data and source templates exist before notebook creation

**⚠️ CRITICAL**: No notebook creation can begin until data and source templates are confirmed

- [X] T002 [P] Verify week 11 data files exist for all 8 functions: data/f1/updated_inputs - Week 11.npy and data/f1/updated_outputs - Week 11.npy through data/f8/
- [X] T003 [P] Verify all 8 week 10 source notebooks exist: functions/f1/f1 - week 10.ipynb through functions/f8/f8 - week 10.ipynb

**Checkpoint**: Environment, data, and source templates verified — notebook creation can begin

---

## Phase 3: User Story 1 — Generate Week 12 Submission Candidates (Priority: P1) 🎯 MVP

**Goal**: Create 8 new notebooks that clone week 10 optimisation strategies with week 11 data to propose week 12 candidates

**Independent Test**: Open any notebook, run all cells, verify it loads week 11 data, fits the surrogate, and outputs a submission query with coordinates in [0, 0.999999]

**Implementation approach**: Clone each week 10 notebook → update WEEK=10 to WEEK=11 (data source) → update titles/labels from "Week 10" to "Week 12" (submission target). No strategy or hyperparameter changes. All hyperparameters must match research.md values exactly.

### GP-Based Notebooks (F1–F6, F8)

- [X] T004 [P] [US1] Clone functions/f1/f1 - week 10.ipynb to functions/f1/f1 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-2.5 + log(max(y,1e-300)) + qLogNEI q=4 + IP (S=0.5, F=0.01). 2D, 21 samples.
- [X] T005 [P] [US1] Clone functions/f2/f2 - week 10.ipynb to functions/f2/f2 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-2.5 + Standardize(m=1) + qLogNEI q=4 + IP (S=0.02, F=0.01). 2D, 21 samples.
- [X] T006 [P] [US1] Clone functions/f3/f3 - week 10.ipynb to functions/f3/f3 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-2.5 + shift(y−y_min) + qLogNEI q=3. No IP. 3D, 26 samples.
- [X] T007 [P] [US1] Clone functions/f4/f4 - week 10.ipynb to functions/f4/f4 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-2.5 + Standardize(m=1) + qLogNEI q=4. No IP. 4D, 41 samples.
- [X] T008 [P] [US1] Clone functions/f5/f5 - week 10.ipynb to functions/f5/f5 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-1.5 + log + Standardize(m=1) + qLogNEI q=4 + IP (S=0.02, F=0.01). 4D, 31 samples.
- [X] T009 [P] [US1] Clone functions/f6/f6 - week 10.ipynb to functions/f6/f6 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-1.5 + Standardize(m=1) + rank-based IP (S=1.0, F=0.01) + qLogNEI q=4 + milk≥0.12. 5D, 31 samples.
- [X] T010 [P] [US1] Clone functions/f8/f8 - week 10.ipynb to functions/f8/f8 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: SFGP Matérn-2.5 + Standardize(m=1) + qLogNEI q=1. No IP. Cholesky check. 8D, 51 samples.

### NN-Based Notebook (F7)

- [X] T011 [P] [US1] Clone functions/f7/f7 - week 10.ipynb to functions/f7/f7 - week 12.ipynb — update WEEK=11, title "Week 12", labels "WEEK 12 SUBMISSION". Strategy: NN 6→5→5→1 + MC dropout(0.05) + blended 50% mean + 50% EI + IP (S=0.02, F=0.02). 6D, 41 samples.

**Checkpoint**: All 8 notebooks created — each is an independent, self-contained file

---

## Phase 4: User Story 2 — Verify Performance Visualisations (Priority: P1)

**Goal**: Execute each notebook end-to-end and verify all visualisations render correctly per spec FR-005 through FR-010

**Independent Test**: Run any notebook and verify: convergence plot, pair plots with green star on best, performance evaluation metrics, surrogate contour (2D functions), submission query with valid coordinates, updated convergence with proposed point

### Execution & Verification

- [X] T012 [P] [US2] Execute and verify functions/f1/f1 - week 12.ipynb — log-scale convergence, C(2,2)=1 pair plot, 3-panel contour viz, performance eval, submission query in [0, 0.999999]², updated convergence
- [X] T013 [P] [US2] Execute and verify functions/f2/f2 - week 12.ipynb — convergence, C(2,2)=1 pair plot, 3-panel contour viz, performance eval, submission query in [0, 0.999999]², updated convergence
- [X] T014 [P] [US2] Execute and verify functions/f3/f3 - week 12.ipynb — convergence, C(3,2)=3 pair plots, performance eval, submission query in [0, 0.999999]³, updated convergence
- [X] T015 [P] [US2] Execute and verify functions/f4/f4 - week 12.ipynb — convergence, C(4,2)=6 pair plots, performance eval, submission query in [0, 0.999999]⁴, updated convergence
- [X] T016 [P] [US2] Execute and verify functions/f5/f5 - week 12.ipynb — convergence, C(4,2)=6 pair plots, performance eval, IP re-scoring, submission query in [0, 0.999999]⁴, updated convergence
- [X] T017 [P] [US2] Execute and verify functions/f6/f6 - week 12.ipynb — convergence, C(5,2)=10 pair plots, performance eval, rank IP + milk constraint, submission query in [0, 0.999999]⁵, updated convergence
- [X] T018 [P] [US2] Execute and verify functions/f7/f7 - week 12.ipynb — convergence, C(6,2)=15 pair plots, NN training loss curve, performance eval, gradient importance, submission query in [0, 0.999999]⁶
- [X] T019 [P] [US2] Execute and verify functions/f8/f8 - week 12.ipynb — convergence, C(8,2)=28 pair plots, performance eval, Cholesky stability check, submission query in [0, 0.999999]⁸, updated convergence

**Checkpoint**: All 8 notebooks execute without errors, all visualisations render, all submission queries output valid non-duplicate coordinates

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and version control

- [X] T020 Run quickstart.md verification checklist against all 8 notebooks in specs/035-f1-f8-week12-optimisation/quickstart.md
- [X] T021 Git add and commit all 8 new notebooks on branch 035-f1-f8-week12-optimisation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Can run in parallel with Setup (file checks only)
- **US1 (Phase 3)**: Depends on Phase 1 + Phase 2 completion — creates all 8 notebooks
- **US2 (Phase 4)**: Depends on Phase 3 — executes and verifies created notebooks
- **Polish (Phase 5)**: Depends on Phase 4 — final commit after all notebooks verified

### User Story Dependencies

- **US1 (Create notebooks)**: All 8 notebooks are independent files in separate directories — all [P] parallel
- **US2 (Verify visualisations)**: Each notebook is executed independently — all [P] parallel
- US2 depends on US1 (must create before executing), but within each story all function tasks are parallel

### Within Each Phase

- Phase 3: All creation tasks (T004–T011) are independent and parallelisable
- Phase 4: All execution tasks (T012–T019) are independent and parallelisable
- No intra-phase ordering constraints exist

---

## Parallel Example: US1 — Create Notebooks

```bash
# All 8 notebooks can be created simultaneously (different directories):
Task T004: "Clone f1 - week 10 → f1 - week 12 in functions/f1/"
Task T005: "Clone f2 - week 10 → f2 - week 12 in functions/f2/"
Task T006: "Clone f3 - week 10 → f3 - week 12 in functions/f3/"
Task T007: "Clone f4 - week 10 → f4 - week 12 in functions/f4/"
Task T008: "Clone f5 - week 10 → f5 - week 12 in functions/f5/"
Task T009: "Clone f6 - week 10 → f6 - week 12 in functions/f6/"
Task T010: "Clone f8 - week 10 → f8 - week 12 in functions/f8/"
Task T011: "Clone f7 - week 10 → f7 - week 12 in functions/f7/"
```

## Parallel Example: US2 — Verify Notebooks

```bash
# Suggested order by runtime (fastest first):
Task T012: "Execute f1 - week 12" (2D, 15 MLL restarts — fast)
Task T013: "Execute f2 - week 12" (2D, 50 MLL restarts — fast)
Task T014: "Execute f3 - week 12" (3D, 40 restarts — moderate)
Task T015: "Execute f4 - week 12" (4D, 30 restarts — moderate)
Task T016: "Execute f5 - week 12" (4D, 15 restarts + IP — moderate)
Task T017: "Execute f6 - week 12" (5D, 15 restarts + rank IP — moderate)
Task T019: "Execute f8 - week 12" (8D, 30 restarts — heavy)
Task T018: "Execute f7 - week 12" (6D, 200 NN epochs + 50k candidates — heavy)
```

---

## Implementation Strategy

### MVP First (F1 Only)

1. Complete Phase 1 + Phase 2: Verify prerequisites
2. Create F1 notebook (T004)
3. Execute F1 notebook (T012)
4. **STOP and VALIDATE**: Verify clone-and-update approach works — all cells pass, visualisations render, submission query valid
5. Proceed with remaining 7 functions

### Incremental Delivery

1. Setup + Foundational → Prerequisites confirmed
2. F1 → Execute → Verify (MVP — validates the approach)
3. F2 → Execute → Verify (confirms pattern for second 2D function)
4. F3–F6, F8 → Execute → Verify (remaining GP functions)
5. F7 → Execute → Verify (NN variant — different cell structure)
6. Polish → Commit

---

## Notes

- All hyperparameters must exactly match week 10 values documented in research.md
- WEEK constant in notebooks is 11 (data source week); titles say "Week 12" (submission target)
- F1 uses manual `log(max(y, 1e-300))` — NOT Standardize(m=1) for the optimisation GP
- F3 shift transform `(y − y_min)` is computed at runtime — not hardcoded
- F5 uses `np.log(outputs)` + Standardize(m=1) — requires strictly positive outputs (confirmed)
- F6 milk constraint: candidates must have milk dimension (index 4) ≥ 0.12; fallback to 0.10
- F7 is the only NN-based function — different cell structure per contracts/notebook-cell-contract.md
- F8 includes Cholesky stability check post-fit
- No strategy changes from week 10 — hyperparameters are frozen per spec FR-004
