# Tasks: Datasheets & Model Cards

**Input**: Design documents from `/specs/022-datasheets-modelcards/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Not requested — no test tasks included.

**Organization**: Tasks are grouped by user story. US1 (modelcards.md) and US2 (datasheets.md) target different files and can run in parallel at the phase level.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 or US2)
- File paths are relative to the repository root

---

## Phase 1: Setup

**Purpose**: Verify prerequisites before writing documentation

- [X] T001 Verify branch `022-datasheets-modelcards` is checked out and all data files exist in `data/f1/` through `data/f8/`

---

## Phase 2: User Story 1 — Model Cards Document (Priority: P1) 🎯 MVP

**Goal**: Create `modelcards.md` in the project root containing 8 model cards (6 subsections each) plus a summary comparison table, documenting the final Week 9 surrogate model for each function.

**Independent Test**: Open `modelcards.md` in a markdown viewer and verify 8 function sections × 6 subsections are present with substantive, data-grounded content. No placeholders remain. Summary table has 8 rows.

**Acceptance criteria**: FR-001 through FR-008, FR-016 through FR-019, SC-001 through SC-003, SC-005 through SC-007

### Implementation for User Story 1

- [X] T002 [US1] Create `modelcards.md` with document title, 1–2 paragraph introduction explaining model cards and the capstone project context, and 8 empty section headers (## F1–F8 with domain names) per `contracts/modelcard-template.md`
- [X] T003 [US1] Write F1 — Radiation Source Detection model card (Overview, Intended Use, Details, Performance, Assumptions & Limitations, Ethical Considerations) in `modelcards.md` — Details: Hurdle model (LR classifier + RF regressor on log1p), Weighted UCB κ=3, interior penalty S=0.1 F=0.01, local penalisation; Edge case: zero-inflated outputs, FALLBACK_MODE
- [X] T004 [US1] Write F2 — Noisy Log-Likelihood model card (all 6 subsections) in `modelcards.md` — Details: SFGP Matérn-1.5 ARD, qLogNEI q=4, no interior penalty; Special: LS bounds [0.01, 2.0], distance-based candidate selection
- [X] T005 [US1] Write F3 — Drug Discovery model card (all 6 subsections) in `modelcards.md` — Details: SFGP Matérn-2.5 ARD, qLogNEI, no interior penalty; Edge case: all-negative outputs, manual z-score
- [X] T006 [US1] Write F4 — Warehouse Product Placement model card (all 6 subsections) in `modelcards.md` — Details: MFGP Matérn-5/2 + LinearTruncatedFidelityKernel, qLogNEI q=4, no interior penalty; Edge case: multi-fidelity GP as regulariser, constant fidelity column
- [X] T007 [US1] Write F5 — Chemical Process Yield model card (all 6 subsections) in `modelcards.md` — Details: SFGP Matérn-5/2 ARD, qLogNEI q=4 + in-loop PenalisedAcquisition, interior penalty S=1.0 F=0.01; Special: log1p→z-score transform, expm1 inverse
- [X] T008 [US1] Write F6 — Cake Recipe Optimisation model card (all 6 subsections) in `modelcards.md` — Details: SFGP Matérn-1.5 ARD, qLogNEI q=4, rank-based interior penalty S=1.0; Edge case: all-negative outputs, feasibility constraints
- [X] T009 [US1] Write F7 — ML Hyperparameter Tuning model card (all 6 subsections) in `modelcards.md` — Details: NN SurrogateNN(6→5→5→1) 71 params, MC Dropout EI (50 passes), interior penalty S=0.1; Edge case: non-GP surrogate, poorly calibrated uncertainty
- [X] T010 [US1] Write F8 — 8D ML Hyperparameters model card (all 6 subsections) in `modelcards.md` — Details: SFGP Matérn-2.5 ARD, qEI q=1, no interior penalty; Edge case: single-query (q=1), Sobol fallback when qEI=0
- [X] T011 [US1] Add summary comparison table to `modelcards.md` per FR-019 — columns: Function, Domain, Dims, Final Surrogate, Final Acquisition, Interior Penalty — 8 rows (F1–F8) using data from `research.md` summary table

**Checkpoint**: `modelcards.md` is complete and independently verifiable. All 48 subsections + summary table populated with real data.

---

## Phase 3: User Story 2 — Datasheets Document (Priority: P1)

**Goal**: Create `datasheets.md` in the project root containing 8 datasheets (5 subsections each) plus a summary comparison table, documenting the data used for each function's optimisation.

**Independent Test**: Open `datasheets.md` in a markdown viewer and verify 8 function sections × 5 subsections are present with substantive, data-grounded content. No placeholders remain. Summary table has 8 rows. Verify dimensions and sizes match `.npy` files.

**Acceptance criteria**: FR-009 through FR-018, FR-020, SC-001, SC-002, SC-004 through SC-007

### Implementation for User Story 2

- [X] T012 [US2] Create `datasheets.md` with document title, 1–2 paragraph introduction explaining datasheets and the capstone project context, and 8 empty section headers (## F1–F8 with domain names) per `contracts/datasheet-template.md`
- [X] T013 [US2] Write F1 — Radiation Source Detection datasheet (Motivation, Composition, Collection Process, Preprocessing & Uses, Distribution & Maintenance) in `datasheets.md` — Composition: 2D, 10→19 pts, zero-inflated [−3.6e-3, 0]; Preprocessing: log1p on positive outputs; Edge case: zero-inflated output distribution
- [X] T014 [US2] Write F2 — Noisy Log-Likelihood datasheet (all 5 subsections) in `datasheets.md` — Composition: 2D, 10→19 pts, mixed [−0.07, 0.67]; Preprocessing: BoTorch Normalize
- [X] T015 [US2] Write F3 — Drug Discovery datasheet (all 5 subsections) in `datasheets.md` — Composition: 3D, 15→24 pts, all negative [−0.40, −0.03]; Preprocessing: manual z-score; Edge case: all-negative outputs
- [X] T016 [US2] Write F4 — Warehouse Product Placement datasheet (all 5 subsections) in `datasheets.md` — Composition: 4D, 30→39 pts, mostly negative [−32.6, 0.53]; Preprocessing: manual z-score; Edge case: constant fidelity column appended to inputs
- [X] T017 [US2] Write F5 — Chemical Process Yield datasheet (all 5 subsections) in `datasheets.md` — Composition: 4D, 20→29 pts, all positive [0.11, 3395]; Preprocessing: log1p → z-score; Edge case: heavy-tailed distribution
- [X] T018 [US2] Write F6 — Cake Recipe Optimisation datasheet (all 5 subsections) in `datasheets.md` — Composition: 5D, 20→29 pts, all negative [−2.57, −0.11]; Preprocessing: Standardize(m=1); Edge case: all-negative outputs
- [X] T019 [US2] Write F7 — ML Hyperparameter Tuning datasheet (all 5 subsections) in `datasheets.md` — Composition: 6D, 30→39 pts, all positive [0.003, 2.30]; Preprocessing: manual z-score on X and Y
- [X] T020 [US2] Write F8 — 8D ML Hyperparameters datasheet (all 5 subsections) in `datasheets.md` — Composition: 8D, 40→49 pts, all positive [5.59, 9.98]; Preprocessing: Standardize(m=1)
- [X] T021 [US2] Add summary comparison table to `datasheets.md` per FR-020 — columns: Function, Domain, Dims, Initial Size, Final Size, Output Sign — 8 rows (F1–F8) using data from `research.md` R3 table

**Checkpoint**: `datasheets.md` is complete and independently verifiable. All 40 subsections + summary table populated with real data.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Validate consistency, accuracy, and completeness across both documents

- [X] T022 [P] Validate `modelcards.md` formatting consistency — all 8 sections use identical heading hierarchy and subsection order per FR-016
- [X] T023 [P] Validate `datasheets.md` formatting consistency — all 8 sections use identical heading hierarchy and subsection order per FR-016
- [X] T024 Verify data accuracy in both documents against `.npy` files — spot-check dimensions, sizes, and output ranges for at least F1, F4, and F8 per SC-004
- [X] T025 Confirm no placeholder text remains in either document — search for `{`, `TODO`, `TBD`, `PLACEHOLDER`, `XXX` per SC-007
- [X] T026 Run `quickstart.md` validation checklist — verify section counts (48 model card subsections, 40 datasheet subsections), summary table completeness, and markdown rendering

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 Model Cards (Phase 2)**: Depends on Setup (Phase 1) completion
- **US2 Datasheets (Phase 3)**: Depends on Setup (Phase 1) completion — can run in parallel with Phase 2 (different file)
- **Polish (Phase 4)**: Depends on both Phase 2 and Phase 3 completion

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Setup — no dependency on US2
- **User Story 2 (P1)**: Can start after Setup — no dependency on US1
- **Independence**: US1 and US2 produce separate files (`modelcards.md` vs `datasheets.md`) and can be implemented and verified independently

### Within Each User Story

- Scaffold (T002/T012) must complete before function sections
- Function sections (T003–T010 / T013–T020) are sequential within the same file
- Summary table (T011/T021) must be last (needs all sections present for consistency)

### Parallel Opportunities

- **Phase-level parallelism**: Phase 2 (US1) and Phase 3 (US2) can run simultaneously — different output files
- **Polish parallelism**: T022 and T023 can run in parallel (read-only validation on different files)
- **Within-phase**: Tasks are sequential (all write to the same file)

---

## Parallel Example: US1 and US2 Simultaneously

```text
# If two agents are available:

Agent A (modelcards.md):             Agent B (datasheets.md):
T002 Create scaffold                 T012 Create scaffold
T003 Write F1 model card             T013 Write F1 datasheet
T004 Write F2 model card             T014 Write F2 datasheet
...                                  ...
T010 Write F8 model card             T020 Write F8 datasheet
T011 Add summary table               T021 Add summary table

# Then both agents complete → Polish phase (T022–T026)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: User Story 1 — `modelcards.md` (T002–T011)
3. **STOP and VALIDATE**: Open `modelcards.md`, verify 8 sections × 6 subsections, check summary table
4. Commit MVP: `feat(022): add model cards for F1–F8`

### Incremental Delivery

1. Setup (T001) → Ready
2. User Story 1 (T002–T011) → `modelcards.md` complete → Commit
3. User Story 2 (T012–T021) → `datasheets.md` complete → Commit
4. Polish (T022–T026) → Both documents validated → Final commit

### Content Sources Per Task

All content is drawn from existing project artefacts:

| Source | Used For |
|--------|----------|
| `specs/022-datasheets-modelcards/research.md` | Per-function model details, data shapes, output ranges, preprocessing |
| `specs/022-datasheets-modelcards/contracts/modelcard-template.md` | Model card section structure and placeholder patterns |
| `specs/022-datasheets-modelcards/contracts/datasheet-template.md` | Datasheet section structure and placeholder patterns |
| `specs/022-datasheets-modelcards/quickstart.md` | Reference data table and validation checklist |
| `data/f1/` through `data/f8/` | Ground-truth verification of dimensions, sizes, ranges |

---

## Notes

- All tasks produce or validate markdown — no code execution required
- Each function's section is self-contained: use `research.md` R4 as the primary data source
- FR-005 constraint: Details section documents Week 9 final model only, NOT weekly evolution — include one-sentence selection note
- Edge cases (zero-inflated F1, all-negative F3/F6, multi-fidelity F4, NN F7, q=1 F8, interior penalty) are handled within the relevant per-function tasks
- Commit after each story phase completes for incremental delivery
