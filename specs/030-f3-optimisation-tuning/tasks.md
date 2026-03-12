# Tasks: F3 Week 10 — Optimisation Tuning

**Input**: Design documents from `/specs/030-f3-optimisation-tuning/`
**Prerequisites**: plan.md (loaded), spec.md (loaded), research.md §R1–R7 (loaded), data-model.md (loaded), contracts/f3-optimisation-pipeline.md (loaded), quickstart.md (loaded)

**Tests**: Not required (per constitution — no unit tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. All tasks append cells to the existing `functions/f3/f3 - week 10.ipynb` notebook (cells 1–12 already exist).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify existing notebook and data availability

- [X] T001 Verify existing F3 week 10 notebook has 12 cells and variables `inputs` (25,3), `outputs` (25,), `N_INITIAL`=15, `N_DIMS`=3, `n_total`=25, `running_best`, `stalling`=True are in scope after execution in functions/f3/f3 - week 10.ipynb

---

## Phase 2: Foundational — Markdown Separator & Configuration Cell

**Purpose**: Add the optimisation section header and all imports/constants before any computation cells. These cells MUST exist before any US1/US2/US3 cells can be appended.

**⚠️ CRITICAL**: No user story implementation can begin until these cells are appended.

- [X] T002 Append markdown cell to functions/f3/f3 - week 10.ipynb — section header "## Step 6 — Week 10 Optimisation Run" with brief rationale documenting 5 strategy changes from week 9 (q 1→3, raw_samples 512→2048, noise_lb 1e-6→1e-4, Standardize(m=1)→shift transform, MLL restarts 20→40) per FR-001
- [X] T003 Append code cell (CG2 — Imports & Configuration) to functions/f3/f3 - week 10.ipynb — import torch, botorch (SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf), gpytorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood, GreaterThan), matplotlib.pyplot, copy, warnings; define all named constants: KERNEL_NU=2.5, ARD_NUM_DIMS=3, NOISE_LB=1e-4, N_MLL_RESTARTS=40, MC_SAMPLES=512, Q_BATCH=3, NUM_RESTARTS=20, RAW_SAMPLES=2048, GRID_RES=50; each constant with comment explaining value and change from week 9 per FR-002 and FR-007

**Checkpoint**: Configuration cell executes without errors, all constants printed and verified

---

## Phase 3: User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

**Goal**: Apply shift transform to outputs, fit SFGP with Matérn-2.5 ARD (no outcome_transform), run qLogNEI q=3 acquisition, select best candidate via distance filtering, format submission string

**Independent Test**: Run all cells including new optimisation section; verify a formatted submission point `x1-x2-x3` is printed with values in [0.0, 0.999999]

### Implementation for User Story 1

- [X] T004 [US1] Append code cell (CG3 — Data Preparation + Shift Transform) to functions/f3/f3 - week 10.ipynb — convert `inputs` to `X_train` tensor (25,3) float64; compute `y_min = outputs.min()` and store for reverse-transform; compute `Y_train = torch.tensor(outputs - y_min).unsqueeze(-1)` (25,1) float64 with all values ≥ 0; validate no NaN/Inf; print shape summary, raw output range, shifted output range [0, ~0.368], and y_min value per contract CG3 and FR-004
- [X] T005 [US1] Append code cell (CG4 — GP Fitting) to functions/f3/f3 - week 10.ipynb — implement multi-restart MLL loop: for each of N_MLL_RESTARTS=40, construct SingleTaskGP with MaternKernel(nu=KERNEL_NU, ard_num_dims=ARD_NUM_DIMS) wrapped in ScaleKernel, GaussianLikelihood with noise constraint GreaterThan(NOISE_LB), NO outcome_transform (shift already applied); randomise hyperparameters with torch.manual_seed(seed), fit via fit_gpytorch_mll, track best model by lowest MLL loss using copy.deepcopy; set best_model.eval(); print 3 fitted lengthscales, noise, outputscale, best_loss, and count of restarts converging within 0.1 of best_loss per contract CG4, FR-003, FR-005, FR-006
- [X] T006 [US1] Append code cell (CG5 — Acquisition & Selection) to functions/f3/f3 - week 10.ipynb — construct qLogNoisyExpectedImprovement with best_model, X_observed=X_train, sampler with MC_SAMPLES=512; call optimize_acqf with bounds [[0,0,0],[1,1,1]], q=Q_BATCH=3, num_restarts=NUM_RESTARTS=20, raw_samples=RAW_SAMPLES=2048; apply distance-based selection (get posterior means for q=3 candidates in shifted space, keep those with mean ≥ median, from qualified pick max min-distance to X_train via torch.cdist); clamp x_new to [0.0, 0.999999]; format proposed_query as `f"{x1:.6f}-{x2:.6f}-{x3:.6f}"`; check is_duplicate against existing 25 samples; print all 3 candidates with shifted posterior means, selection rationale, and `>>> SUBMISSION: {proposed_query}` per contract CG5, FR-008, FR-009, FR-010, FR-011; retain `acqf` in scope for CG6

**Checkpoint**: Submission string printed — US1 is functionally complete. Verify SC-001, SC-002, SC-003

---

## Phase 4: User Story 2 — Visualise Surrogate and Acquisition Surface (Priority: P2)

**Goal**: 2D contour slices (3 input pairs) showing GP posterior mean and acquisition surface with colour-coded point overlays

**Independent Test**: After running all cells, verify 2×3 figure renders with correct contour surfaces and point colours (blue initial, orange submissions, green star proposed)

### Implementation for User Story 2

- [X] T007 [US2] Append code cell (CG6 — 2D Contour Slice Visualisation) to functions/f3/f3 - week 10.ipynb — create 2×3 figure for pairs (d0,d1), (d0,d2), (d1,d2); for each pair: create 50×50 meshgrid over [0,1]², fix remaining dimension at x_new's corresponding coordinate, compute posterior mean from best_model on grid (shifted space), compute acquisition values from acqf (unsqueeze for q=1 evaluation); row 1: GP posterior mean contourf (viridis + colourbar) for each pair; row 2: acquisition surface contourf (plasma + colourbar) for each pair; overlay on all panels: blue dots for initial (X_train[:N_INITIAL] projected onto pair), orange dots for submissions (X_train[N_INITIAL:] projected onto pair), green star for x_new projected onto pair; add axis labels identifying dimension indices and titles; per contract CG6 and FR-012

**Checkpoint**: 2×3 contour figure renders correctly — US2 is functionally complete. Verify SC-004

---

## Phase 5: User Story 3 — Display Convergence with Proposed Point (Priority: P3)

**Goal**: Updated convergence plot showing running best trajectory with proposed next point reverse-shifted to original scale

**Independent Test**: After running all cells, verify convergence plot shows all 25 data points plus proposed point with green star marker

### Implementation for User Story 3

- [X] T008 [US3] Append code cell (CG7 — Convergence Plot) to functions/f3/f3 - week 10.ipynb — predict at x_new using best_model.posterior() (shifted space), reverse-shift prediction: `pred_original = pred_shifted + y_min`; plot running_best as line in original scale with blue region for initial (1–N_INITIAL) and orange for submissions (N_INITIAL+1–n_total); mark proposed point at position n_total+1 with green star and reverse-shifted predicted mean; linear y-axis (F3 outputs in [-0.399, -0.031]); add vertical dashed line at N_INITIAL boundary; title "F3 Convergence — Running Best + Proposed"; xlabel "Sample", ylabel "Output"; print predicted shifted value and original value; per contract CG7, FR-013, SC-005

**Checkpoint**: Convergence plot renders with proposed point in original scale — US3 is functionally complete. Verify SC-005

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation of the complete notebook

- [X] T009 Run functions/f3/f3 - week 10.ipynb top-to-bottom (Kernel > Restart & Run All) and verify all 19 cells execute without errors per SC-001
- [X] T010 Verify GP hyperparameters are valid: 3 lengthscales finite and positive, noise ≥ 1e-4, outputscale > 0 per SC-002
- [X] T011 Verify submission point is valid: all 3 values in [0.0, 0.999999], duplicate check performed and result reported, format matches `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}` per SC-003
- [X] T012 Verify all hyperparameter constants have comments documenting value and week 9 → week 10 change rationale per SC-006 and FR-007

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify existing notebook
- **Foundational (Phase 2)**: Depends on Phase 1 — append markdown + config cell
- **US1 (Phase 3)**: Depends on Phase 2 — CG3→CG4→CG5 sequential (each cell needs prior cell's outputs)
- **US2 (Phase 4)**: Depends on US1 (needs `best_model`, `acqf`, `X_train`, `x_new`)
- **US3 (Phase 5)**: Depends on US1 (needs `best_model`, `x_new`, `running_best`, `outputs`, `y_min`)
- **Polish (Phase 6)**: Depends on Phases 3, 4, 5 — validate complete notebook

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational (Phase 2). Tasks T004→T005→T006 are strictly sequential (data→model→acquisition).
- **User Story 2 (P2)**: Depends on US1 completion (needs `best_model`, `acqf`, `x_new`). Can start once T006 is done.
- **User Story 3 (P3)**: Depends on US1 completion (needs `best_model`, `x_new`, `y_min`). Can start once T006 is done. Independent of US2.

### Parallel Opportunities

**US2 + US3 can run in parallel** after US1 completes:
- T007 (2D contour slices) and T008 (convergence plot) use different outputs and produce independent cells
- In practice, since both append to the same notebook, they should be implemented sequentially to maintain cell order (T007 before T008)

**Within US1**: No parallelism — T004→T005→T006 is a strict pipeline (each cell consumes the prior cell's outputs)

**Phase 6**: T009–T012 are sequential validation steps after all cells exist

---

## Parallel Example: After US1 Completion

```
After T006 (acquisition) completes:
  → T007 [US2] (2D contour slices) — needs best_model, acqf, X_train, x_new
  → T008 [US3] (convergence plot) — needs best_model, x_new, running_best, outputs, y_min
  Both can conceptually run in parallel but append sequentially to maintain notebook order
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002, T003)
3. Complete Phase 3: User Story 1 (T004→T005→T006)
4. **STOP and VALIDATE**: Verify submission string is printed and valid
5. Proceed to US2 + US3 for visualisation

### Incremental Delivery

1. Complete Setup + Foundational → Configuration ready
2. Add User Story 1 (T004–T006) → Submission point produced (MVP!)
3. Add User Story 2 (T007) → 2D contour slices rendered
4. Add User Story 3 (T008) → Convergence with proposed point
5. Polish (T009–T012) → Full validation
