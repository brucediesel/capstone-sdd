# Tasks: ARD Relevance Visualisation

**Input**: Design documents from `/specs/034-ard-relevance-visualisation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: No tests requested — manual notebook execution per Constitution.

**Organization**: Tasks are grouped by user story. US1 tasks (F1–F8) are all parallelizable since each modifies a different notebook file.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify environment and data prerequisites before implementation

- [X] T001 Verify Python environment `sdd-dev` has botorch, gpytorch, torch, numpy, matplotlib installed and week 11 data files exist in data/f1/ through data/f8/

---

## Phase 2: User Story 1 — ARD Relevance Calculation and Bar Chart (Priority: P1) 🎯 MVP

**Goal**: Each week 11 notebook (F1–F8) gets 3 new cells appended: a markdown section header, a code cell that fits a SingleTaskGP and prints a raw lengthscale table, and a code cell that computes normalised relevance and displays a horizontal bar chart.

**Independent Test**: Run each notebook end-to-end and verify the ARD bar chart renders with the correct number of bars matching the function's dimensionality.

**References**: Cell interface in contracts/notebook-cell-contract.md; per-function config in research.md R1; extraction pattern in research.md R2; relevance formula in research.md R3; visualisation spec in research.md R5.

### Implementation for User Story 1

> All 8 tasks below are [P] — they modify different notebook files and have no cross-dependencies. Each task appends 3 cells (Cell 12: markdown header, Cell 13: GP fit + raw table, Cell 14: normalised relevance bar chart) after the existing Cell 11.

- [X] T002 [P] [US1] Append ARD section (3 cells) to functions/f1/f1 - week 11.ipynb — Config: 2 dims [x1, x2], Matérn-2.5, output_transform=log(y + 1e-300), noise_lb=1e-4
- [X] T003 [P] [US1] Append ARD section (3 cells) to functions/f2/f2 - week 11.ipynb — Config: 2 dims [x1, x2], Matérn-2.5, output_transform=Standardize(m=1), noise_lb=1e-4
- [X] T004 [P] [US1] Append ARD section (3 cells) to functions/f3/f3 - week 11.ipynb — Config: 3 dims [x1, x2, x3], Matérn-2.5, output_transform=shift (y - y_min), noise_lb=1e-4
- [X] T005 [P] [US1] Append ARD section (3 cells) to functions/f4/f4 - week 11.ipynb — Config: 4 dims [x1, x2, x3, x4], Matérn-2.5, output_transform=Standardize(m=1), noise_lb=1e-3
- [X] T006 [P] [US1] Append ARD section (3 cells) to functions/f5/f5 - week 11.ipynb — Config: 4 dims [x1, x2, x3, x4], Matérn-1.5, output_transform=log(y) then Standardize(m=1), noise_lb=1e-6
- [X] T007 [P] [US1] Append ARD section (3 cells) to functions/f6/f6 - week 11.ipynb — Config: 5 dims [flour, sugar, eggs, butter, milk], Matérn-1.5, output_transform=Standardize(m=1), noise_lb=1e-3
- [X] T008 [P] [US1] Append ARD section (3 cells) to functions/f7/f7 - week 11.ipynb — Config: 6 dims [learning_rate, reg_strength, n_layers, dropout, batch_size, optimizer], Matérn-2.5 DIAGNOSTIC GP (production is NN), output_transform=Standardize(m=1), noise_lb=1e-4; markdown cell MUST note this is a diagnostic GP analysis (not the production NN surrogate)
- [X] T009 [P] [US1] Append ARD section (3 cells) to functions/f8/f8 - week 11.ipynb — Config: 8 dims [x1..x8], Matérn-2.5, output_transform=Standardize(m=1), noise_lb=1e-7

### Cell Implementation Details (applies to all T002–T009)

**Cell 12 (markdown):**
- Title: `## ARD Feature Relevance Analysis`
- Brief explanation of what ARD lengthscales mean (smaller = more relevant)
- For F7 only: note that this is a diagnostic GP analysis since the production surrogate is a neural network

**Cell 13 (code — GP fit + raw table):**
1. Import: torch, SingleTaskGP, fit_gpytorch_mll, ExactMarginalLogLikelihood, ScaleKernel, MaternKernel, GaussianLikelihood, GreaterThan, Standardize (where needed)
2. Convert existing `inputs`/`outputs` variables (from Cell 4) to tensors
3. Apply function-specific output transform to Y_train (see per-task config above)
4. Construct `SingleTaskGP(X_train, Y_train, covar_module=ScaleKernel(MaternKernel(nu=NU, ard_num_dims=N_DIMS)), likelihood=GaussianLikelihood(noise_constraint=GreaterThan(NOISE_LB)), outcome_transform=...)` — use `outcome_transform=Standardize(m=1)` where config says Standardize; apply manual log/shift before tensor conversion for log/shift transforms
5. Fit via `fit_gpytorch_mll(mll)`
6. Extract: `lengthscales = model.covar_module.base_kernel.lengthscale.detach().squeeze().numpy()`
7. Print formatted raw lengthscale table: `Dimension | Lengthscale` with dimension names from config

**Cell 14 (code — normalised relevance bar chart):**
1. Compute: `relevance = (1/lengthscales) / np.sum(1/lengthscales)`
2. Create: `plt.barh(dim_names, relevance, color='steelblue')`
3. Annotate each bar with its percentage value
4. Title: `"FX: ARD Feature Relevance (Matérn-ν kernel)"` (substitute function number and kernel nu)
5. X-axis label: `"Normalised Relevance"`
6. Y-axis label: `"Input Dimension"`
7. `plt.tight_layout(); plt.show()`

**Checkpoint**: After completing all T002–T009, each notebook should execute end-to-end showing the ARD bar chart with the correct number of bars.

---

## Phase 3: User Story 2 — Consistent Presentation Across All 8 Notebooks (Priority: P2)

**Goal**: Ensure all 8 ARD sections follow identical formatting conventions so the researcher can compare relevance patterns across functions.

**Independent Test**: Open all 8 notebooks side-by-side and confirm consistent chart style, labelling, colour, title format, and section placement.

### Implementation for User Story 2

- [X] T010 [US2] Review all 8 notebooks for consistency: verify identical chart type (horizontal bar), colour (steelblue), axis labels ("Normalised Relevance" / "Input Dimension"), title format ("FX: ARD Feature Relevance (Matérn-ν kernel)"), bar annotation format (percentage), and section position (after existing Cell 11) in functions/f1/ through functions/f8/ week 11 notebooks

**Checkpoint**: All 8 notebooks should be visually consistent when compared side-by-side.

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all notebooks

- [X] T011 Run quickstart.md validation: execute all 8 week 11 notebooks end-to-end per specs/034-ard-relevance-visualisation/quickstart.md and verify each produces (a) no errors, (b) a raw lengthscale table with correct dimension count, (c) a bar chart with correct bar count matching function dimensionality (F1:2, F2:2, F3:3, F4:4, F5:4, F6:5, F7:6, F8:8)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 2)**: Depends on Setup (T001)
- **US2 (Phase 3)**: Depends on ALL US1 tasks (T002–T009) being complete
- **Polish (Phase 4)**: Depends on US2 (T010) being complete

### Within Phase 2 (US1)

All 8 tasks (T002–T009) are fully independent and parallel:
- Each modifies a different notebook file
- No shared code or imports between notebooks
- No data dependencies between functions

### Parallel Opportunities

- **T002–T009**: All 8 can execute in parallel (different files, zero dependencies)
- **T010**: Sequential — requires reviewing output from all US1 tasks
- **T011**: Sequential — requires all previous tasks complete

---

## Parallel Example: User Story 1

```text
# All 8 notebooks can be implemented simultaneously:
T002: F1 ARD section → functions/f1/f1 - week 11.ipynb
T003: F2 ARD section → functions/f2/f2 - week 11.ipynb
T004: F3 ARD section → functions/f3/f3 - week 11.ipynb
T005: F4 ARD section → functions/f4/f4 - week 11.ipynb
T006: F5 ARD section → functions/f5/f5 - week 11.ipynb
T007: F6 ARD section → functions/f6/f6 - week 11.ipynb
T008: F7 ARD section → functions/f7/f7 - week 11.ipynb  (diagnostic GP)
T009: F8 ARD section → functions/f8/f8 - week 11.ipynb
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Verify prerequisites (T001)
2. Complete Phase 2: Implement all 8 ARD sections in parallel (T002–T009)
3. **STOP and VALIDATE**: Run each notebook end-to-end — verify bar charts render
4. This alone delivers full P1 value

### Incremental Delivery

1. Setup → Prerequisites verified
2. US1 (T002–T009) → All 8 notebooks have ARD bar charts (MVP!)
3. US2 (T010) → Consistency verified and any formatting adjustments made
4. Polish (T011) → Full quickstart validation passed

---

## Notes

- All [P] tasks modify different notebook files — safe to parallelize
- Each task appends 3 cells; no existing cells are modified
- F7 is the only special case (diagnostic GP with NN annotation)
- Imports are self-contained within Cell 13 (not added to existing import cell)
- `inputs` and `outputs` variables from existing Cell 4 are reused — no re-loading needed
