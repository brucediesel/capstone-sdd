# Tasks: F5 Week 7 — GP Matérn-5/2 + NEI (Exploration Focus)

**Input**: Design documents from `specs/011-f5-gp-nei/`
**Prerequisites**: plan.md, spec.md (with 6 clarifications), research.md, data-model.md, contracts/week7-cells.md, quickstart.md

**Tests**: No tests — per CONSTITUTION.md, no unit tests are required.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

**Parameter Reference** (all artifacts now consistent post-regeneration):

| Parameter | Value | Sources |
|-----------|-------|---------|
| Lengthscale init | **0.5** | FR-007, RES-002, E-03 |
| Noise init | **0.1 · Var(Y_train)** | FR-008, RES-003, E-03 |
| Batch size (q) | **4** | FR-011, RES-005, E-06 |
| Acquisition | **qLogNoisyExpectedImprovement** | FR-011/FR-012, RES-004/RES-005, E-06 |
| Candidate selection | **farthest from data, mean > median** | FR-014, RES-004, E-07 |
| ξ parameter | **N/A** — BoTorch has no ξ; `eta` only for constraints | FR-012, RES-004 |
| outcome_transform | **None** (manual log1p + z-score avoids double-standardization) | RES-001, E-03 |
| Clamping | **[0.0, 0.999999]** before formatting | FR-015, E-08 |

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- All tasks target `functions/f5/f5.ipynb` (append-only, no existing cell modifications)

---

## Phase 1: Setup

**Purpose**: Verify environment and notebook readiness

- [X] T001 Verify branch is `011-f5-gp-nei`, pyenv `sdd-dev` active, notebook `functions/f5/f5.ipynb` has 50 existing cells (last cell `#VSC-8f8ac8b4`), and data files `data/f5/updated_inputs - Week 7.npy` (27×4) and `data/f5/updated_outputs - Week 7.npy` (27,) exist

**Checkpoint**: Notebook structure and data confirmed, ready to append cells.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Section header cell that all subsequent cells depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Insert Week 7 markdown header cell (Cell 51) after `#VSC-8f8ac8b4` in `functions/f5/f5.ipynb` — heading `## Week 7 — GP Matérn-5/2 + NEI (Exploration Focus)`, rationale paragraph explaining the switch from GBT ensemble (Week 6) to GP-based BO for exploration, and comparison table: Week 6 (GBT+UCB κ=0.5, q=1) vs Week 7 (GP Matérn-5/2+NEI, q=4, ℓ=0.5, noise=0.1·Var(y), distance-based selection)

**Checkpoint**: Week 7 section visible in notebook. All subsequent cells insert after this header.

---

## Phase 3: User Story 1 — Load & Validate Week 7 Data (Priority: P1) 🎯 MVP

**Goal**: Load cumulative Week 7 data (27 samples, 4D) and validate shapes/ranges.

**Independent Test**: Run the data-loading cell; confirm 27 samples, inputs in [0, 1], best value printed.

- [X] T003 [US1] Insert code cell (Cell 52) after the Week 7 header in `functions/f5/f5.ipynb` — imports (numpy, torch, copy, matplotlib, botorch.models.SingleTaskGP, botorch.fit.fit_gpytorch_mll, botorch.acquisition.logei.qLogNoisyExpectedImprovement, botorch.optim.optimize_acqf, botorch.sampling.normal.SobolQMCNormalSampler, gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.kernels.ScaleKernel/MaternKernel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.constraints.GreaterThan) + load `data/f5/updated_inputs - Week 7.npy` and `data/f5/updated_outputs - Week 7.npy` via relative path `../../data/f5/` + validate shapes (27, 4) and (27,), assert all inputs in [0, 1], print sample count, input range, output range, best observed value (~3394.68) and its index (#26). Contract: Cell 52.

**Checkpoint**: US1 complete — data loaded and validated independently.

---

## Phase 4: User Story 2 — Train GP Surrogate with Exploration Hyperparameters (Priority: P1)

**Goal**: Fit SingleTaskGP (Matérn-5/2 ARD, log1p+z-score, ℓ=0.5, noise=0.1·Var(y), outcome_transform=None) via 15-restart MLL.

**Independent Test**: Run training cell; confirm fitted HPs printed, noise ≥ 1e-6, all 15 restarts produce finite neg-MLL.

- [X] T004 [US2] Insert markdown cell (Cell 53) after the data loading cell in `functions/f5/f5.ipynb` — hyperparameter documentation table with 14 entries per contract: (1) kernel: Matérn-5/2, (2) ARD: True/4 lengthscales, (3) ℓ init: **0.5** (broader uncertainty for exploration), (4) outputscale init: ~1.0, (5) noise init: **0.1·Var(Y_train) ≈ 0.1** (prevents over-interpolation), (6) noise floor: 1e-6 (jitter), (7) output transform: manual log1p → z-score, (8) outcome_transform: **None** (avoids double-standardization with Standardize(m=1)), (9) MLL restarts: 15, (10) acquisition: **qLogNoisyExpectedImprovement** (log-space for stability), (11) q: **4** (batch diversity), (12) raw_samples: 3000, (13) num_restarts: 50, (14) selection: **distance-based** (farthest from data, mean > median) — rationale for each entry, note that BoTorch's `eta` only affects constraint smoothing (no ξ exists). Contract: Cell 53.
- [X] T005 [US2] Insert code cell (Cell 54) after the HP markdown cell in `functions/f5/f5.ipynb` — compute `y_log = np.log1p(y_raw)`, z-score standardise (`y_mean, y_std_val = y_log.mean(), y_log.std()`; `y_std = (y_log - y_mean) / y_std_val`), convert to torch tensors (`X_train` double (27,4), `Y_train` double (27,1) via `.unsqueeze(-1)`), 15-restart MLL loop: `torch.manual_seed(seed)`, create `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))` + `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))`, construct `SingleTaskGP(X_train, Y_train, covar_module=covar, likelihood=likelihood, outcome_transform=None)`, init HPs (`model.covar_module.base_kernel.lengthscale = 0.5`, `model.likelihood.noise = 0.1 * Y_train.var().item()`, `model.covar_module.outputscale = 1.0`), fit via `fit_gpytorch_mll(mll)`, eval mode, score `loss = -mll(output, Y_train.squeeze(-1)).item()`, `copy.deepcopy` best, print per-restart neg-MLL, print fitted ℓ₁–ℓ₄ + σ²_f + σ²_n with 6 decimal places. Contract: Cell 54.

**Checkpoint**: US2 complete — GP trained with exploration-promoting hyperparameters, all fitted values reported.

---

## Phase 5: User Story 3 — Propose Next Samples via NEI Acquisition (Priority: P1)

**Goal**: Use qLogNoisyExpectedImprovement (q=4, 3000 Sobol → 50 L-BFGS) to propose 4 candidates and select best via distance-based strategy.

**Independent Test**: Run acquisition cell; confirm 4 candidates in [0, 0.999999]⁴, distance-based selection applied, selected candidate identified.

- [X] T006 [US3] Insert code cell (Cell 55) after the GP training cell in `functions/f5/f5.ipynb` — set `best_model.eval()`, create `SobolQMCNormalSampler(sample_shape=torch.Size([512]))`, create `qLogNoisyExpectedImprovement(model=best_model, X_baseline=X_train, sampler=sampler, prune_baseline=True)` (NO eta/ξ parameter), set `BOUNDS = torch.tensor([[0.0]*4, [1.0]*4], dtype=torch.double)`, call `optimize_acqf(acq_function=nei, bounds=BOUNDS, q=4, num_restarts=50, raw_samples=3000)`, clamp candidates to [0.0, 0.999999], compute posterior means per candidate (inverse z-score: `pred * y_std_val + y_mean` then `np.expm1` to original scale), compute min-distance from each candidate to X_train via `torch.cdist(candidates, X_train).min(dim=1).values`, apply selection: filter to candidates with mean ≥ median(means), among those pick max-distance to nearest observation, print all 4 candidates with coordinates + posterior means (standardised + original) + distances, clearly mark selected candidate with rationale. Contract: Cell 55.

**Checkpoint**: US3 acquisition complete — 4 candidates proposed, best selected by exploration-quality balance.

---

## Phase 6: User Story 4 — Visualise Surrogate & Convergence (Priority: P2)

**Goal**: Render 3-panel surrogate visualisation and convergence plot matching Week 6 layout.

**Independent Test**: Run both visualisation cells; confirm plots render with correct axes, labels, colorbars, and overlaid points.

- [X] T007 [US4] Insert code cell (Cell 56) after the acquisition cell in `functions/f5/f5.ipynb` — extract fitted lengthscales from `best_model.covar_module.base_kernel.lengthscale`, identify top-2 important dims by shortest ARD lengthscales (`np.argsort(lengthscales)[:2]`), fix other 2 dims at best_point values, build 80×80 grid over top-2 dims, get posterior mean+std via `best_model.posterior(grid_tensor)` (de-standardise: `pred * y_std_val + y_mean` then `np.expm1` to original scale), plot 3-panel figure (18×5 inches): Panel 1 mean contour with observed (red dots) + proposed (magenta star) + colourbar, Panel 2 std contour with observed + proposed + colourbar, Panel 3 dimension relevance bar chart (`1/ℓ` normalised, 4 bars labelled x0–x3, taller = more important), `plt.suptitle` includes "Week 7" and "GP". Contract: Cell 56.
- [X] T008 [US4] Insert code cell (Cell 57) after the surrogate plot cell in `functions/f5/f5.ipynb` — compute `running_best = np.maximum.accumulate(y_raw)`, plot vs observation number (1-indexed), add vertical dashed red line at x=26.5 (Week 6→7 boundary), add labels/title ("F5 Convergence — Week 7")/legend/grid, print running best at sample 26 (end Week 6) and sample 27 (Week 7). Contract: Cell 57.

**Checkpoint**: US4 complete — both visualisations rendered matching Week 6 layout.

---

## Phase 7: Polish & Submission

**Purpose**: Format submission query and validate entire notebook end-to-end

- [X] T009 [US3] Insert code cell (Cell 58) after the convergence plot in `functions/f5/f5.ipynb` — format selected best_point as dash-separated string with 6 decimal places (`f"{v:.6f}"` per coordinate), clamp any value ≥ 1.0 to 0.999999 via `torch.clamp(point, 0.0, 0.999999)`, validate: 4 parts, all start with "0.", all in [0.000000, 0.999999], print submission query prominently, print "✓ Submission format validated", print summary table (surrogate: GP Matérn-5/2, acquisition: qLogNEI q=4, selection: distance-based, fitted HPs, query string). Contract: Cell 58.
- [X] T010 Run all 8 new cells (51–58) in `functions/f5/f5.ipynb` sequentially to validate end-to-end execution — verify 58 total cells, verify no existing cells modified (50 original cells unchanged), verify submission query matches `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`, run quickstart.md verification checklist, commit with message `feat(f5): week 7 GP Matern-5/2 + NEI exploration-focused candidates`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — verify environment
- **Phase 2 (Foundational)**: Depends on Phase 1 — header cell must be first; BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 — data loading cell follows header
- **Phase 4 (US2)**: Depends on Phase 3 — training needs loaded data tensors (X_train, Y_train) and transform stats (y_mean, y_std_val)
- **Phase 5 (US3)**: Depends on Phase 4 — acquisition needs fitted `best_model` and X_train
- **Phase 6 (US4)**: Depends on Phase 4 + Phase 5 — visualisation needs `best_model`, fitted lengthscales, and `best_point`
- **Phase 7 (Polish)**: Depends on Phase 5 + Phase 6 — submission formats the selected candidate; validation needs all 8 cells

### User Story Dependencies

- **US1 (Data Loading)**: Independent after foundational — provides X_raw, y_raw
- **US2 (GP Training)**: Depends on US1 — needs X_train, Y_train tensors and transform stats
- **US3 (Acquisition + Submission)**: Depends on US2 — needs `best_model`, X_train; T009 also needs Cells 56–57 to exist before it (cell ordering)
- **US4 (Visualisation)**: Depends on US2 + US3 — needs `best_model`, fitted lengthscales, `best_point`

### Within Each Phase

- Markdown cells before code cells (within same story)
- Code cells execute sequentially — each depends on variables from prior cells
- T007 and T008 share model state but render independent plots

### Parallel Opportunities

```text
Phase 1:  T001
            ↓
Phase 2:  T002
            ↓
Phase 3:  T003
            ↓
Phase 4:  T004 → T005
            ↓
Phase 5:  T006
            ↓
Phase 6:  T007 → T008
            ↓
Phase 7:  T009 → T010
```

> Note: Single-notebook sequential pipeline. No tasks are marked [P] because all cells must be inserted in order within the same file. The linear dependency chain reflects notebook cell execution order: each cell depends on variables defined in preceding cells.

---

## Implementation Strategy

### MVP First (User Stories 1–3, Phases 1–5 + T009)

1. Complete Phase 1: Setup verification (T001)
2. Complete Phase 2: Week 7 header cell (T002)
3. Complete Phase 3: US1 — data loading + validation (T003)
4. Complete Phase 4: US2 — HP docs + GP training (T004, T005)
5. Complete Phase 5: US3 — NEI acquisition with q=4 candidates (T006)
6. **STOP and VALIDATE**: 4 candidates proposed with distance-based selection — submission possible without visualisations

### Full Delivery (add Phases 6–7)

7. Complete Phase 6: US4 — surrogate 3-panel + convergence plots (T007, T008)
8. Complete Phase 7: T009 submission formatting + T010 full validation + commit

### Incremental Delivery Milestones

| After | Milestone | Verifiable |
|-------|-----------|------------|
| T003 | Data loaded: 27 samples, best ~3394.68 at index 26 | Shapes + ranges printed |
| T005 | GP trained: ℓ=0.5 init, noise=0.1·Var, 15 restarts, outcome_transform=None | Fitted HPs printed |
| T006 | 4 candidates via qLogNEI, distance-based selection applied | Candidates + distances printed |
| T009 | Submission query: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` | ✓ Validated |
| T010 | Full feature complete: 58 cells, 3-panel viz, convergence, committed | Git status clean |

---

## Notes

- All 10 tasks target the same file: `functions/f5/f5.ipynb` (append-only)
- No [P] markers because cells must be inserted sequentially in the notebook
- Constitution compliance: no unit tests, no existing cell modifications, BoTorch library, simple code
- Transform pipeline: `y_raw → log1p → z-score → GP(outcome_transform=None) → z-score⁻¹ → expm1 → y_pred`
- **CRITICAL**: Use `outcome_transform=None` in `SingleTaskGP()` to prevent double-standardization (manual z-score + default Standardize(m=1))
- **CRITICAL**: Do NOT pass `eta` as ξ — it has no exploration effect when `constraints=None` (RES-004)
- Key references: spec.md (FR-001–FR-020, 6 clarifications), research.md (RES-001–RES-007), data-model.md (E-01–E-10), contracts/week7-cells.md (Cells 51–58), quickstart.md
- Total: **10 tasks** across **7 phases** covering **4 user stories**
