# Tasks: F8 Week 7 — SFGP + qEI Acquisition

**Input**: Design documents from `/specs/018-f8-sfgp-qei/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cells.md, quickstart.md

**Tests**: No tests (per constitution — no unit tests required; manual execution only).

**Organization**: Tasks grouped by user story. All tasks target a single file: `functions/f8/f8.ipynb` (append cells 50–57 after existing cell 49).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1, US2)
- All paths relative to repository root

---

## Phase 1: Setup

**Purpose**: Verify environment and existing notebook state before adding new cells.

- [X] T001 Verify Python 3.11+ environment is active with torch, botorch, gpytorch, numpy, matplotlib available
- [X] T002 Verify `functions/f8/f8.ipynb` has 49 cells and existing Week 5–6 cells execute without errors; confirm `data/f8/updated_inputs - Week 7.npy` (47×8) and `data/f8/updated_outputs - Week 7.npy` (47,) exist

**Checkpoint**: Notebook runs end-to-end; all Week 6 variables are populated. Data files present with expected shapes.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Section header and data loading that ALL user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Append markdown cell 50 — section header "## Week 7 — SFGP + qEI Acquisition" with approach explanation (returning to GP surrogate after Weeks 5-6 NN) and hyperparameter table (Kernel: Matern 2.5 + ARD, Noise floor: 1e-07, Standardise: Yes, MC samples: 256, xi: 0.01, Restarts: 30, Raw samples: 4096) in `functions/f8/f8.ipynb`
- [X] T004 Append code cell 51 — import numpy, torch, matplotlib, botorch (SingleTaskGP, fit_gpytorch_mll, qExpectedImprovement, SobolQMCNormalSampler, optimize_acqf), gpytorch (ExactMarginalLogLikelihood, ScaleKernel, MaternKernel, GaussianLikelihood, GreaterThan); load `../../data/f8/updated_inputs - Week 7.npy` and `../../data/f8/updated_outputs - Week 7.npy`; create X_train (47,8 float64), Y_train (47,1 float64), BOUNDS (2,8), param_names list; print sample count (47), dims (8), output range, best observed; assert shapes in `functions/f8/f8.ipynb`

**Checkpoint**: `X_raw` (47, 8), `y_raw` (47,), `X_train`, `Y_train`, `BOUNDS`, `param_names` all available in kernel.

---

## Phase 3: User Story 1 — Core SFGP Surrogate + qEI Recommendation (Priority: P1) 🎯 MVP

**Goal**: Fit a BoTorch SingleTaskGP with Matern 2.5 + ARD, compute qEI with 256 MC samples and xi=0.01, optimise over [0,1]^8 to propose the next candidate, and format a submission-ready query string.

**Independent Test**: Run all cells top-to-bottom. Submission query printed in `x1-x2-...-x8` format (6 decimal places, all in [0, 0.999999]). GP reports 8 positive ARD lengthscales. Acquisition value ≥ 0.

### Implementation for User Story 1

- [X] T005 [US1] Append code cell 52 — define constants XI=0.01, MC_SAMPLES=256, NUM_RESTARTS=30, RAW_SAMPLES=4096; create ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8)), GaussianLikelihood(noise_constraint=GreaterThan(1e-7)); fit SingleTaskGP(X_train, Y_train, covar_module, likelihood) via fit_gpytorch_mll(ExactMarginalLogLikelihood); extract and print 8 ARD lengthscales; compute best_f = Y_train.max().item() + XI; print best_f in `functions/f8/f8.ipynb`
- [X] T006 [US1] Append code cell 53 — create SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])), qExpectedImprovement(model=model, best_f=best_f, sampler=sampler); run optimize_acqf(acq_fn, bounds=BOUNDS, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES); clamp candidate to [0,1]; if acq_value <= 0 print warning and fall back to highest GP posterior mean on 4096 Sobol candidates; print candidate coordinates, acquisition value, comparison to current best in `functions/f8/f8.ipynb`
- [X] T007 [US1] Append code cell 57 — define format_query(point) that clamps to [0, 0.999999] and formats 6 decimal places dash-separated; format next_point; validate 8 parts each parseable float in [0, 0.999999]; print submission query with header/footer, coordinate breakdown, and acquisition value summary in `functions/f8/f8.ipynb`

**Checkpoint**: Cells 52, 53, 57 execute without errors. `next_point` shape is (8,), all coordinates in [0, 0.999999], submission query printed. GP reports 8 positive lengthscales.

---

## Phase 4: User Story 2 — Diagnostic Visualisations (Priority: P2)

**Goal**: Produce a feature importance chart, a 3-panel 2D surrogate slice (mean, std, EI), and a convergence plot matching the Week 5-6 visualisation pattern.

**Independent Test**: Execute cells 54–56 after the surrogate is fitted; verify three separate plots render without error. Feature importance bar chart shows 8 dimensions. 3-panel figure has labelled axes. Convergence plot has weekly boundary markers.

### Implementation for User Story 2

- [X] T008 [US2] Append code cell 54 — compute importance = 1/lengthscales, normalise to sum to 1; identify top-2 dimensions (smallest lengthscale); print importance for all 8 dimensions with param_names; plot horizontal bar chart of feature importance in `functions/f8/f8.ipynb`
- [X] T009 [US2] Append code cell 55 — build 50×50 grid over [0,1]² for top-2 dimensions, fix remaining 6 dims at best observed point values; evaluate GP posterior mean and std on grid; use analytic ExpectedImprovement (from botorch.acquisition.analytic) for 2500-point grid evaluation (faster than qEI); plot 3-panel figure [Mean | Std | EI] with contourf, mark best observed point and proposed candidate with distinct symbols in `functions/f8/f8.ipynb`
- [X] T010 [US2] Append code cell 56 — compute running_best = np.maximum.accumulate(y_raw); plot running best vs sample index; add vertical dashed lines at boundaries 40.5 (initial→Week 5), 45.5 (Week 5→Week 6), 46.5 (Week 6→Week 7); print running best at each week boundary in `functions/f8/f8.ipynb`

**Checkpoint**: Cells 54–56 execute without errors. Feature importance bar chart, 3-panel surrogate figure, and convergence plot all render correctly. Top-2 dimensions identified for slice projection.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and commit.

- [X] T011 Run all notebook cells top-to-bottom (cells 0–57) and verify no errors, all assertions pass, all plots render
- [X] T012 Verify cells 0–49 are unmodified (diff check against master — only new cells appended)
- [X] T013 Run quickstart.md verification checklist (11 items) against notebook output

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify environment first
- **Foundational (Phase 2)**: Depends on Setup — data loading blocks everything
- **US1 (Phase 3)**: Depends on Foundational — core GP + acquisition
- **US2 (Phase 4)**: Depends on US1 (needs `model`, `lengthscales`, `best_f`, `acq_fn`, `next_point` from cells 52–53)
- **Polish (Phase 5)**: Depends on all user stories complete

### Task Dependencies

```text
T001 ──► T002 ──► T003 ──► T004 ──► T005 ──► T006 ──► T008 ──► T009 ──► T010 ──► T007
                                                                                    │
                                                                                    ▼
                                                                          T011 ──► T012 ──► T013
```

### Cell Ordering Constraints

All cells are appended sequentially to one notebook — no parallelism within the notebook file. Notebook cell order:

| Order | Cell | Task | Content |
|-------|------|------|---------|
| 1 | 50 | T003 | Markdown header + hyperparameter table |
| 2 | 51 | T004 | Load data & imports |
| 3 | 52 | T005 | Fit SFGP surrogate |
| 4 | 53 | T006 | qEI acquisition + candidate selection |
| 5 | 54 | T008 | Feature importance (lengthscale) |
| 6 | 55 | T009 | 3-panel surrogate visualisation |
| 7 | 56 | T010 | Convergence plot |
| 8 | 57 | T007 | Format submission query |

### FR Coverage

| FR | Description | Task(s) |
|----|-------------|---------|
| FR-001 | Load data + print summary | T004 |
| FR-002 | SingleTaskGP + Matern 2.5 + ARD + noise ≥ 1e-07 | T005 |
| FR-003 | fit_gpytorch_mll | T005 |
| FR-004 | qEI with 256 MC + xi=0.01 | T006 |
| FR-005 | Fantasisation enabled | T006 |
| FR-006 | optimize_acqf 30 restarts, 4096 raw | T006 |
| FR-007 | Clamp + format 8-value submission | T007 |
| FR-008 | Feature importance chart | T008 |
| FR-009 | 3-panel 2D slice | T009 |
| FR-010 | Convergence plot with boundaries | T010 |
| FR-011 | Zero qEI fallback | T006 |
| FR-012 | Append after cell 49, no modifications | T011, T012 |

---

## Parallel Example: User Story 1

```text
# No parallel tasks — all cells append to one file sequentially.
# Execute in notebook cell order: T003 → T004 → T005 → T006 → T008 → T009 → T010 → T007
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup verification (T001–T002)
2. Complete Phase 2: Foundational — markdown header + data loading (T003–T004)
3. Complete Phase 3: US1 core GP + qEI (T005–T006, T007) — cells 52, 53, 57
4. **STOP and VALIDATE**: Submission query produced, GP fitted, acquisition value reported
5. This is a functional submission-ready notebook

### Incremental Delivery

1. Setup → verify environment and notebook state
2. Foundational (cells 50–51) → documentation and data ready
3. US1 (cells 52–53, 57) → core computation, submission-ready
4. US2 (cells 54–56) → visualisation and convergence, examiner-ready
5. Polish → final validation against quickstart checklist (11 items)

---

## Notes

- All tasks modify a single file: `functions/f8/f8.ipynb`
- No new files created, no new dependencies added (BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib already available)
- Cell ordering in notebook: 50 (md), 51–56 (code), 57 (code)
- US3 (Model Documentation & Diagnostics) is satisfied by print statements embedded in US1 cells (51–53, 57) per contracts/cells.md — no separate tasks needed
- Use analytic `ExpectedImprovement` for 2D grid visualisation in cell 55 (2500 evaluations faster than qEI); qEI used only for actual candidate selection in cell 53
- `SobolQMCNormalSampler` uses `sample_shape=torch.Size([256])` — NOT deprecated `num_samples=256`
- `qExpectedImprovement` has no `xi` parameter — encode as `best_f = y_max + 0.01`
- No interior penalty for F8 (unlike F7) — boundary candidates are accepted as-is
- Fallback logic in T006: if acq_value ≤ 0, select argmax(posterior mean) on Sobol candidates
- Total: 13 tasks across 5 phases
- Key hyperparameters: Matern 2.5 ARD (8 dims), noise ≥ 1e-07, Standardize(m=1), 256 MC, xi=0.01, 30 restarts, 4096 raw samples
