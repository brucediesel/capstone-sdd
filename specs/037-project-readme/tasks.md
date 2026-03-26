# Tasks: Project README & Documentation Update

**Input**: Design documents from `/specs/037-project-readme/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓

**Tests**: Not requested — no test tasks included.

**Organization**: Tasks grouped by user story. US1–US3 (P1) deliver the README; US4–US5 (P2) add critical evaluation and lessons learnt; modelcards/datasheets updates are foundational since they must be accurate before the README references them.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1–US5) from spec.md
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: Verify branch and confirm source data availability

- [X] T001 Verify branch `037-project-readme` is checked out and working directory is clean
- [X] T002 Verify `specs/037-project-readme/research.md` is accessible and contains F1–F8 strategy timelines and dataset sizes

---

## Phase 2: Foundational (Documentation Updates — Blocking)

**Purpose**: Update modelcards.md and datasheets.md to Week 13 before the README references them. These files are prerequisites for US1 (FR-010, SC-006).

**⚠️ CRITICAL**: README references modelcards.md and datasheets.md — they must be current before README is written.

- [X] T003 Update the introduction paragraph in `modelcards.md` from "Weeks 3–9" to "Weeks 3–13" temporal scope, updating submission round count and final week reference
- [X] T004 [P] Update F1 model card in `modelcards.md`: add strategy evolution timeline (Weeks 3–13) to Details section, update final surrogate to SFGP Matérn-2.5 + qLogNEI, update Performance to Week 13 dataset size (23 samples) and best output (7.71 × 10⁻¹⁶)
- [X] T005 [P] Update F2 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to SFGP Matérn-2.5 + qLogNEI, update Performance to Week 13 dataset size (23 samples) and best output (0.674)
- [X] T006 [P] Update F3 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to SFGP Matérn-2.5 + qLogNEI, update Performance to Week 13 dataset size (28 samples) and best output (−0.0114)
- [X] T007 [P] Update F4 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate from MFGP to SFGP Matérn-2.5 + qLogNEI, update Performance to Week 13 dataset size (43 samples) and best output (0.532)
- [X] T008 [P] Update F5 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to SFGP Matérn-1.5 + qLogNEI, update Performance to Week 13 dataset size (33 samples) and best output (8,662.4)
- [X] T009 [P] Update F6 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to SFGP Matérn-1.5 + qLogNEI with rank-based IP, update Performance to Week 13 dataset size (33 samples) and best output (−0.111)
- [X] T010 [P] Update F7 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to Compact NN (6→5→5→1) with blended acquisition, update Performance to Week 13 dataset size (43 samples) and best output (2.305)
- [X] T011 [P] Update F8 model card in `modelcards.md`: add strategy evolution timeline to Details section, update final surrogate to SFGP Matérn-2.5 + qEI, update Performance to Week 13 dataset size (53 samples) and best output (9.982)
- [X] T012 Update the introduction paragraph in `datasheets.md` from "Weeks 3–9" to "Weeks 3–13" temporal scope, updating submission round count and final week reference
- [X] T013 [P] Update F1 datasheet in `datasheets.md`: Composition (final size 23, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T014 [P] Update F2 datasheet in `datasheets.md`: Composition (final size 23, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T015 [P] Update F3 datasheet in `datasheets.md`: Composition (final size 28, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T016 [P] Update F4 datasheet in `datasheets.md`: Composition (final size 43, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T017 [P] Update F5 datasheet in `datasheets.md`: Composition (final size 33, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T018 [P] Update F6 datasheet in `datasheets.md`: Composition (final size 33, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T019 [P] Update F7 datasheet in `datasheets.md`: Composition (final size 43, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)
- [X] T020 [P] Update F8 datasheet in `datasheets.md`: Composition (final size 53, 13 points added), Collection Process (Weeks 3–13, 12 rounds), Distribution (add Week 10–13 .npy file entries)

**Checkpoint**: modelcards.md and datasheets.md are fully updated to Week 13 — README can now safely reference them.

---

## Phase 3: User Story 1 — Read Project Overview (Priority: P1) 🎯 MVP

**Goal**: A reader opens the README and understands the project purpose, 8 functions, BO approach, and 13-week timeline.

**Independent Test**: Open README.md in isolation; verify it conveys project purpose, all 8 functions, overall approach, and timeline.

### Implementation for User Story 1

- [X] T021 [US1] Create `README.md` in project root with title, project overview section (FR-001, FR-002): black-box optimisation challenge, 8 functions with input dimensionalities, [0, 0.999999] bounds, maximisation objective, 13-week iterative structure, Bayesian Optimisation with tailored surrogates
- [X] T022 [US1] Add table of contents to `README.md` with links to all major sections: overview, per-function strategy & results, results summary, convergence plots, critical evaluation, lessons learnt, project structure, references
- [X] T023 [US1] Add project structure section to `README.md` (FR-008): data/, functions/, specs/, research/ folder descriptions with purpose of each
- [X] T024 [US1] Add references section to `README.md` (FR-010): links to `modelcards.md`, `datasheets.md`, `functions/results/process_results.ipynb`

**Checkpoint**: README has a complete structural skeleton with overview, TOC, project structure, and references.

---

## Phase 4: User Story 2 — Understand Per-Function Strategy (Priority: P1)

**Goal**: A reader finds per-function strategy summaries covering surrogate, acquisition, transforms, and key changes across weeks.

**Independent Test**: For each function, verify README states surrogate model type, kernel/architecture, acquisition function, output transform, and notable strategy changes.

### Implementation for User Story 2

- [X] T025 [P] [US2] Write F1 strategy & results subsection in `README.md` (FR-003, FR-004): Hurdle Model → SFGP transition, zero-inflated challenge, best output 7.71 × 10⁻¹⁶, convergence behaviour
- [X] T026 [P] [US2] Write F2 strategy & results subsection in `README.md` (FR-003, FR-004): SFGP Matérn-1.5 → 2.5 transition, qLogNEI, best output 0.674, convergence behaviour
- [X] T027 [P] [US2] Write F3 strategy & results subsection in `README.md` (FR-003, FR-004): SFGP Matérn-2.5, batch q=1→3, negative output handling, best output −0.0114, convergence behaviour
- [X] T028 [P] [US2] Write F4 strategy & results subsection in `README.md` (FR-003, FR-004): MFGP → SFGP transition, multi-fidelity rationale, best output 0.532, convergence behaviour
- [X] T029 [P] [US2] Write F5 strategy & results subsection in `README.md` (FR-003, FR-004): GBT → GP Matérn-5/2 → Matérn-1.5 evolution, log transform, interior penalty, best output 8,662.4, convergence behaviour
- [X] T030 [P] [US2] Write F6 strategy & results subsection in `README.md` (FR-003, FR-004): NN → SFGP Matérn-1.5, rank-based interior penalty for all-negative outputs, milk constraint, best output −0.111, convergence behaviour
- [X] T031 [P] [US2] Write F7 strategy & results subsection in `README.md` (FR-003, FR-004): NN surrogates throughout, compact architecture, blended acquisition 0.7×mean + 0.3×EI, best output 2.305, convergence behaviour
- [X] T032 [P] [US2] Write F8 strategy & results subsection in `README.md` (FR-003, FR-004): NN → SFGP Matérn-2.5 return, qEI with fallback, 8D challenge, best output 9.982, convergence behaviour

**Checkpoint**: All 8 functions have strategy & results subsections in README.

---

## Phase 5: User Story 3 — Review Optimisation Results (Priority: P1)

**Goal**: A reader sees a cross-function results summary table and convergence plot references.

**Independent Test**: Verify README includes a results summary table with quantitative data for all 8 functions, and a convergence plot reference to process_results.ipynb.

### Implementation for User Story 3

- [X] T033 [US3] Add results summary table to `README.md` (FR-004, SC-003): all 8 functions with dimensionality, initial samples, final samples, best output, improvement factor, convergence trend
- [X] T034 [US3] Add convergence plots section to `README.md` (FR-005): describe the 2×4 grid in `functions/results/process_results.ipynb`, explain what each plot shows, how to generate/view them

**Checkpoint**: README has quantitative results for every function and convergence plot references.

---

## Phase 6: User Story 4 — Critical Evaluation (Priority: P2)

**Goal**: A reader finds an honest per-function assessment of what worked, what did not, and why.

**Independent Test**: Verify README includes per-function evaluation with at least one strength and one weakness per function (SC-004).

### Implementation for User Story 4

- [X] T035 [P] [US4] Write critical evaluation for F1–F4 in `README.md` (FR-006, SC-004): F1 zero-inflated challenge and source not located; F2 steady improvement on smooth surface; F3 negative output handling; F4 MFGP to simpler GP transition success
- [X] T036 [P] [US4] Write critical evaluation for F5–F8 in `README.md` (FR-006, SC-004): F5 dramatic improvement via output transforms; F6 rank-based IP innovation for negative outputs; F7 NN limitations in 6D; F8 GP resilience in 8D

**Checkpoint**: All 8 functions have critical evaluation with strengths and weaknesses.

---

## Phase 7: User Story 5 — Lessons Learnt (Priority: P2)

**Goal**: A reader finds ≥5 generalisable, actionable insights from the 13-week campaign.

**Independent Test**: Verify ≥5 distinct insights covering surrogates, acquisition, transforms, dimensionality, and adaptation.

### Implementation for User Story 5

- [X] T037 [US5] Write lessons learnt section in `README.md` (FR-007, SC-005): at least 5 insights covering surrogate selection (prequential evaluation), acquisition function tuning (exploitation vs exploration), output transforms (log, standardize, rank-based), dimensionality challenges (GP vs NN trade-off), adaptive strategy changes (interior penalties, batch sizing), and late-stage exploitation

**Checkpoint**: README is feature-complete — all user stories satisfied.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Ensure cross-document consistency and final quality

- [X] T038 Review `README.md` for academic tone and clarity (FR-009): ensure professional prose accessible to a reader with no prior knowledge (SC-001)
- [X] T039 Validate cross-document consistency: best output values in README match modelcards.md Performance sections; dataset sizes in README match datasheets.md Composition sections; temporal scope "Weeks 3–13" consistent across all three files
- [X] T040 Run `specs/037-project-readme/quickstart.md` validation checklist: confirm all 9 checks pass (README exists, all 8 functions documented, quantitative results, strengths/weaknesses, ≥5 lessons, correct references, modelcards updated, datasheets updated, cross-document consistency)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all README work
  - T003 (modelcards intro) must precede T004–T011 (per-function model cards)
  - T012 (datasheets intro) must precede T013–T020 (per-function datasheets)
  - T004–T011 and T013–T020 are all [P] — can run in parallel within their file
- **US1 (Phase 3)**: Depends on Phase 2 — creates README skeleton
- **US2 (Phase 4)**: Depends on Phase 3 (T021 for file creation) — per-function sections
  - T025–T032 are all [P] — 8 function sections can be written in parallel
- **US3 (Phase 5)**: Depends on Phase 4 — summary table references per-function data
- **US4 (Phase 6)**: Depends on Phase 4 — evaluation references strategy and results
  - T035–T036 are [P] — F1–F4 and F5–F8 evaluations can run in parallel
  - US4 can run in parallel with US3
- **US5 (Phase 7)**: Depends on Phase 4 — lessons synthesise per-function experience
  - US5 can run in parallel with US3 and US4
- **Polish (Phase 8)**: Depends on all previous phases

### User Story Dependencies

- **US1 (P1)**: Creates README.md — BLOCKS US2, US3, US4, US5
- **US2 (P1)**: Populates per-function content — BLOCKS US3
- **US3 (P1)**: Results summary table — independent of US4, US5
- **US4 (P2)**: Critical evaluation — can run in parallel with US3, US5
- **US5 (P2)**: Lessons learnt — can run in parallel with US3, US4

### Parallel Opportunities

- **Phase 2**: T004–T011 (8 model cards) all [P]; T013–T020 (8 datasheets) all [P]
- **Phase 4**: T025–T032 (8 function strategy sections) all [P]
- **Phase 6**: T035–T036 (F1–F4 and F5–F8 evaluations) are [P]
- **Phases 5, 6, 7**: US3, US4, US5 can run in parallel after US2 completes

---

## Parallel Example: Phase 2 (Foundational)

```text
# After T003 (modelcards intro update):
T004: Update F1 model card in modelcards.md
T005: Update F2 model card in modelcards.md
T006: Update F3 model card in modelcards.md
T007: Update F4 model card in modelcards.md
T008: Update F5 model card in modelcards.md
T009: Update F6 model card in modelcards.md
T010: Update F7 model card in modelcards.md
T011: Update F8 model card in modelcards.md

# After T012 (datasheets intro update):
T013: Update F1 datasheet in datasheets.md
T014: Update F2 datasheet in datasheets.md
...through T020
```

## Parallel Example: Phase 4 (User Story 2)

```text
# After T021–T024 (README skeleton):
T025: Write F1 strategy & results in README.md
T026: Write F2 strategy & results in README.md
T027: Write F3 strategy & results in README.md
T028: Write F4 strategy & results in README.md
T029: Write F5 strategy & results in README.md
T030: Write F6 strategy & results in README.md
T031: Write F7 strategy & results in README.md
T032: Write F8 strategy & results in README.md
```

---

## Implementation Strategy

### MVP First (User Stories 1–3)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (modelcards + datasheets updates)
3. Complete Phase 3: US1 — README skeleton with overview, TOC, structure, references
4. Complete Phase 4: US2 — Per-function strategy & results sections
5. Complete Phase 5: US3 — Results summary table + convergence plot references
6. **STOP and VALIDATE**: README has overview, all 8 function strategies, results, and references

### Full Delivery (Add P2 Stories)

7. Complete Phase 6: US4 — Critical evaluation per function
8. Complete Phase 7: US5 — Lessons learnt section
9. Complete Phase 8: Polish — cross-document validation
10. **FINAL VALIDATION**: Run quickstart.md checklist

### Incremental Value

- After Phase 2: modelcards.md and datasheets.md are standalone deliverables
- After Phase 5: README is informative (strategies + results) — publishable MVP
- After Phase 7: README is complete with evaluation and lessons — full deliverable
- After Phase 8: All three documents are cross-validated and consistent

---

## Notes

- All tasks target Markdown files — no code execution or notebook modification
- Source data for all content: `specs/037-project-readme/research.md` (R1: strategies, R2: dataset sizes)
- Cross-document consistency (T039) ensures README, modelcards.md, and datasheets.md agree on all shared data points
- [P] tasks within the same file are conceptually parallel but may require sequential file edits — treat as independently writable sections
