# Tasks: F6 Week 7 — SFGP Matérn-1.5 + NEI (Exploration Focus)

**Input**: Design documents from `specs/012-f6-sfgp-nei/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/week7-cells.md, quickstart.md

**Tests**: No tests — per CONSTITUTION.md, no unit tests are required.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

**Parameter Reference** (all artifacts consistent):

| Parameter | Value | Sources |
|-----------|-------|---------|
| Kernel | **Matérn ν=1.5** (not 2.5) | FR-003, RES-001, E-03 |
| ARD | **True (5 lengthscales)** | FR-003, E-03 |
| Lengthscale init | **0.5** | FR-006, RES-001, E-03 |
| Output scale init | **1.0** | FR-007, E-03 |
| Noise init | **0.1** (10% of standardised Var≈1.0) | FR-008, RES-003, E-03 |
| Noise floor | **1e-8** (GreaterThan) | FR-003, RES-003 |
| Outcome transform | **Standardize(m=1) default** — no manual transform, no `outcome_transform=None` | FR-004, RES-002 |
| MLL restarts | **15** | FR-005, RES-006, E-04 |
| Acquisition | **qLogNoisyExpectedImprovement** | FR-010, RES-004, E-06 |
| Batch size (q) | **4** | FR-011, RES-005, E-06 |
| raw_samples / num_restarts | **3000 / 50** | FR-011, RES-005, E-06 |
| Candidate selection | **distance-based** (farthest from data, mean ≥ median) | FR-013, RES-004, E-07 |
| Clamping | **[0.0, 0.999999]** before formatting | FR-012, E-09 |
| Posterior space | **Original** — Standardize auto-untransforms | RES-002, E-07, E-08 |
| MLL target | **`model.train_targets`** (standardised) | RES-002, quickstart.md |

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- All tasks target `functions/f6/f6.ipynb` (append-only, no existing cell modifications)

---

## Phase 1: Setup

**Purpose**: Verify environment and notebook readiness

- [X] T001 Verify branch is `012-f6-sfgp-nei`, pyenv `sdd-dev` active, notebook `functions/f6/f6.ipynb` has 47 existing cells (last cell id `a52b2e42`), and data files `data/f6/updated_inputs - Week 7.npy` (27×5) and `data/f6/updated_outputs - Week 7.npy` (27,) exist

**Checkpoint**: Notebook structure and data confirmed, ready to append cells.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Section header cell that all subsequent cells depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Insert Week 7 markdown header cell (Cell 48) after cell 47 (id `a52b2e42`) in `functions/f6/f6.ipynb` — heading `## Week 7 — SFGP Matérn-1.5 + NEI (Exploration Focus)`, rationale paragraph explaining the strategy change from NN + MC Dropout + UCB κ=0.5 (Week 6) to SFGP + NEI q=4 (Week 7), and comparison table: Week 6 (NN, MC Dropout uncertainty, UCB κ=0.5, κ exploitation) vs Week 7 (SFGP Matérn-1.5, GP posterior, qLogNEI q=4, distance-based exploration). Per FR-020.

**Checkpoint**: Week 7 section visible in notebook. All subsequent cells insert after this header.

---

## Phase 3: User Story 1 — Load & Validate Week 7 Data (Priority: P1) 🎯 MVP

**Goal**: Load cumulative Week 7 data (27 samples, 5D) and validate shapes/ranges.

**Independent Test**: Run the data-loading cell; confirm 27 samples, inputs in [0, 1], all outputs negative, best value printed.

- [X] T003 [US1] Insert code cell (Cell 49) after the Week 7 header in `functions/f6/f6.ipynb` — imports (numpy, torch, copy, matplotlib, botorch.models.SingleTaskGP, botorch.fit.fit_gpytorch_mll, botorch.acquisition.logei.qLogNoisyExpectedImprovement, botorch.optim.optimize_acqf, botorch.sampling.normal.SobolQMCNormalSampler, gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.kernels.ScaleKernel/MaternKernel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.constraints.GreaterThan) + load `data/f6/updated_inputs - Week 7.npy` and `data/f6/updated_outputs - Week 7.npy` via relative path `../../data/f6/` + validate shapes (27, 5) and (27,), assert all inputs in [0, 1], confirm all outputs negative, print sample count, input range, output range, best observed value (≈-0.205600) and its index (#26). Per FR-001, FR-002. Contract: Cell 49.

**Checkpoint**: US1 complete — data loaded and validated independently.

---

## Phase 4: User Story 3 — Hyperparameter Documentation (Priority: P3)

**Goal**: Document all SFGP and NEI hyperparameters with plain-English rationale in a markdown cell.

**Independent Test**: Rendered markdown cell shows the complete HP table with 14+ entries.

- [X] T004 [US3] Insert markdown cell (Cell 50) after the data loading cell in `functions/f6/f6.ipynb` — hyperparameter documentation table with 14 entries per contract: (1) kernel: Matérn-1.5 (once-differentiable, rougher than 2.5, wider uncertainty in gaps), (2) ARD: True/5 lengthscales (one per ingredient dimension), (3) ℓ init: **0.5** (broader uncertainty for exploration in 5D), (4) outputscale init: **1.0** (matches standardised variance), (5) noise init: **0.1** (10% of standardised Var(y)≈1.0 — NOT 0.1*Var(y_raw)=0.033; see RES-003), (6) noise floor: **1e-8** (tighter than 1e-6; user-specified), (7) outcome transform: **Standardize(m=1)** (BoTorch default z-score; sufficient for 12.5x range; no manual log transform), (8) MLL restarts: 15, (9) acquisition: **qLogNoisyExpectedImprovement** (log-space for numerical stability with all-negative outputs), (10) q: **4** (batch diversity for exploration), (11) raw_samples: 3000 (Sobol initial points), (12) num_restarts: 50 (L-BFGS starting points), (13) selection: **distance-based** (farthest from data among candidates with mean ≥ median), (14) clamping: [0, 0.999999] before formatting — rationale for each entry. Per FR-009. Contract: Cell 50.

**Checkpoint**: US3 HP documentation complete — constitution requirement satisfied.

---

## Phase 5: User Story 1 — Train GP Surrogate (Priority: P1)

**Goal**: Fit SingleTaskGP (Matérn-1.5 ARD, Standardize(m=1) default, ℓ=0.5, noise=0.1, noise floor 1e-8) via 15-restart MLL.

**Independent Test**: Run training cell; confirm 5 distinct fitted lengthscales, noise ≥ 1e-8, all 15 restarts produce finite neg-MLL.

- [X] T005 [US1] Insert code cell (Cell 51) after the HP markdown cell in `functions/f6/f6.ipynb` — convert raw data to torch double tensors: `X_train = torch.tensor(X_raw, dtype=torch.double)` (27,5), `Y_train = torch.tensor(y_raw, dtype=torch.double).unsqueeze(-1)` (27,1) — NO manual log1p or z-score transform (Standardize(m=1) handles this internally), 15-restart MLL loop: `torch.manual_seed(seed)`, create `GaussianLikelihood(noise_constraint=GreaterThan(1e-8))` + `ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5))`, construct `SingleTaskGP(X_train, Y_train, covar_module=covar, likelihood=likelihood)` — do NOT pass `outcome_transform` argument (default Standardize(m=1) applies), init HPs (`model.covar_module.base_kernel.lengthscale = 0.5`, `model.likelihood.noise = 0.1`, `model.covar_module.outputscale = 1.0`), fit via `fit_gpytorch_mll(mll)`, eval mode, score `loss = -mll(output, model.train_targets).item()` — note: use `model.train_targets` (standardised) NOT `Y_train.squeeze(-1)` (raw), `copy.deepcopy` best, print per-restart neg-MLL, print fitted ℓ₁–ℓ₅ + σ²_f + σ²_n with 6 decimal places. Per FR-003 through FR-008. Contract: Cell 51.

**Checkpoint**: US1 GP training complete — model fitted with exploration-promoting hyperparameters, all 7 fitted values reported.

---

## Phase 6: User Story 1 — Propose Candidates via NEI Acquisition (Priority: P1)

**Goal**: Use qLogNoisyExpectedImprovement (q=4, 3000 Sobol → 50 L-BFGS) to propose 4 candidates and select best via distance-based strategy.

**Independent Test**: Run acquisition cell; confirm 4 candidates in [0, 0.999999]⁵, distance-based selection applied, selected candidate identified with posterior mean (all negative, original space).

- [X] T006 [US1] Insert code cell (Cell 52) after the GP training cell in `functions/f6/f6.ipynb` — set `best_model.eval()`, create `SobolQMCNormalSampler(sample_shape=torch.Size([512]))`, create `qLogNoisyExpectedImprovement(model=best_model, X_baseline=X_train, sampler=sampler, prune_baseline=True)` (NO eta/ξ parameter), set `BOUNDS = torch.tensor([[0.0]*5, [1.0]*5], dtype=torch.double)`, call `optimize_acqf(acq_function=nei, bounds=BOUNDS, q=4, num_restarts=50, raw_samples=3000)`, clamp candidates to [0.0, 0.999999], compute posterior means via `best_model.posterior(candidates).mean.squeeze(-1)` — these are in **original space** automatically (Standardize(m=1) auto-untransforms, NO manual expm1 or inverse z-score needed), compute min-distance from each candidate to X_train via `torch.cdist(candidates, X_train).min(dim=1).values`, apply distance-based selection: filter to candidates with mean ≥ median(means), among those pick max-distance to nearest observation; fallback if all below median: pick candidate with highest mean, print all 4 candidates with coordinates + posterior means (original scale, all negative) + distances, clearly mark selected candidate with rationale. Per FR-010 through FR-013. Contract: Cell 52.

**Checkpoint**: US1 acquisition complete — 4 candidates proposed, best selected by exploration-quality balance.

---

## Phase 7: User Story 2 — Visualise Surrogate & Convergence (Priority: P2)

**Goal**: Render 3-panel surrogate visualisation and convergence plot matching Week 6 layout.

**Independent Test**: Run both visualisation cells; confirm plots render with correct axes, labels, colorbars, 5 importance bars, and overlaid points.

- [X] T007 [US2] Insert code cell (Cell 53) after the acquisition cell in `functions/f6/f6.ipynb` — extract fitted lengthscales from `best_model.covar_module.base_kernel.lengthscale`, identify top-2 important dims by shortest ARD lengthscales (`np.argsort(lengthscales)[:2]`), fix other 3 dims at best_point values (per FR-015), build 80×80 grid over top-2 dims, get posterior mean+std via `best_model.posterior(grid_tensor)` — values are in **original space** automatically (NO manual expm1 or inverse z-score needed, unlike F5), plot 3-panel figure (18×5 inches): Panel 1 mean contour with observed (red dots) + proposed (magenta star) + colourbar, Panel 2 std contour with observed + proposed + colourbar, Panel 3 dimension relevance bar chart (`1/ℓ` normalised, **5 bars** labelled x0–x4, taller = more important), `plt.suptitle` includes "Week 7" and "SFGP". Per FR-014, FR-015. Contract: Cell 53.
- [X] T008 [US2] Insert code cell (Cell 54) after the surrogate plot cell in `functions/f6/f6.ipynb` — compute `running_best = np.maximum.accumulate(y_raw)`, plot vs observation number (1-indexed), add vertical dashed red line at x=26.5 (Week 6→7 boundary), add labels/title ("F6 Convergence — Week 7")/legend/grid, print running best at sample 26 (end Week 6) and sample 27 (Week 7). Per FR-016. Contract: Cell 54.

**Checkpoint**: US2 complete — both visualisations rendered matching Week 6 layout.

---

## Phase 8: Polish & Submission

**Purpose**: Format submission query and validate entire notebook end-to-end

- [X] T009 [US1] Insert code cell (Cell 55) after the convergence plot in `functions/f6/f6.ipynb` — format selected best_point as dash-separated string with 6 decimal places (`f"{v:.6f}"` per coordinate), clamp any value ≥ 1.0 to 0.999999 via `torch.clamp(point, 0.0, 0.999999)`, validate: 5 parts, all parseable as float, all in [0.000000, 0.999999], print submission query prominently, print "✓ Submission format validated", print summary table (surrogate: SFGP Matérn-1.5, outcome transform: Standardize(m=1), acquisition: qLogNEI q=4, selection: distance-based, fitted ℓ₁–ℓ₅ + σ²_f + σ²_n, candidate posterior mean, query string). Per FR-017, FR-018. Contract: Cell 55.
- [X] T010 Run all 8 new cells (48–55) in `functions/f6/f6.ipynb` sequentially to validate end-to-end execution — verify 55 total cells, verify no existing cells modified (47 original cells unchanged), verify submission query matches `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`, run quickstart.md verification checklist, commit with message `feat(f6): week 7 SFGP Matern-1.5 + NEI exploration-focused candidates`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — verify environment
- **Phase 2 (Foundational)**: Depends on Phase 1 — header cell must be first; BLOCKS all user stories
- **Phase 3 (US1 Data)**: Depends on Phase 2 — data loading cell follows header
- **Phase 4 (US3 HP Docs)**: Depends on Phase 3 — markdown cell follows data loading (cell ordering)
- **Phase 5 (US1 Training)**: Depends on Phase 3 + Phase 4 — training needs loaded data tensors; cell ordering after HP docs
- **Phase 6 (US1 Acquisition)**: Depends on Phase 5 — acquisition needs fitted `best_model` and X_train
- **Phase 7 (US2 Visualisation)**: Depends on Phase 5 + Phase 6 — needs `best_model`, fitted lengthscales, and `best_point`
- **Phase 8 (Polish)**: Depends on Phase 6 + Phase 7 — submission formats selected candidate; validation needs all 8 cells

### User Story Dependencies

- **US1 (Data → Training → Acquisition → Submission)**: Sequential chain across Phases 3, 5, 6, 8 — provides X_raw, y_raw, X_train, Y_train, best_model, best_point, query string
- **US2 (Visualisation)**: Depends on US1 training + acquisition — needs `best_model`, fitted lengthscales, `best_point`
- **US3 (HP Documentation)**: Depends on US1 data loading (cell ordering) — no code dependencies, purely documentation

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
Phase 4:  T004
            ↓
Phase 5:  T005
            ↓
Phase 6:  T006
            ↓
Phase 7:  T007 → T008
            ↓
Phase 8:  T009 → T010
```

> **Note**: Due to the sequential nature of notebook cells (each cell depends on variables defined in prior cells), there are limited parallel opportunities within this feature. T007 and T008 must also be sequential within a notebook. The main parallelisable opportunity is that T004 (markdown) could be written independently of T005 (code), but cell ordering requires T004 to be inserted first.

---

## Parallel Example: User Story 2 (Visualisation)

```bash
# Both visualisation cells depend on best_model and best_point,
# but must be sequential in notebook (Cell 53 before Cell 54):
Task T007: "3-panel surrogate plot in functions/f6/f6.ipynb"
Task T008: "Convergence plot in functions/f6/f6.ipynb"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational header (T002)
3. Complete Phase 3: Data loading (T003)
4. Complete Phase 4: HP documentation (T004)
5. Complete Phase 5: GP training (T005)
6. Complete Phase 6: NEI acquisition (T006)
7. **STOP and VALIDATE**: Data loaded, model trained, candidate selected
8. Complete Phase 8: Submission formatting (T009)
9. MVP achieved — submission query ready

### Incremental Delivery

1. Setup + Foundational → Notebook section created
2. US1 Data + US3 HP Docs → Data validated, HPs documented
3. US1 Training → GP fitted, HPs reported
4. US1 Acquisition → Candidate selected, submission-ready (MVP!)
5. US2 Visualisation → Diagnostic plots added
6. Polish → Formatted query, end-to-end validation, commit

### Key Differences from F5 (011-f5-gp-nei)

| Aspect | F5 Tasks | F6 Tasks |
|--------|----------|----------|
| Kernel | `MaternKernel(nu=2.5, ard_num_dims=4)` | `MaternKernel(nu=1.5, ard_num_dims=5)` |
| Transform | Manual log1p + z-score; `outcome_transform=None` | **None** — default `Standardize(m=1)` |
| Noise init | `0.1 * Y_train.var().item()` ≈ 0.1 | **`0.1`** (hardcoded; Standardize guarantees Var≈1.0) |
| Noise constraint | `GreaterThan(1e-6)` | **`GreaterThan(1e-8)`** |
| MLL target | `Y_train.squeeze(-1)` (raw z-scored) | **`model.train_targets`** (auto-standardised) |
| Posterior for viz | Manual `expm1(pred * std + mean)` | **Direct** — `model.posterior()` auto-untransforms |
| Submission format | `x1-x2-x3-x4` (4D) | **`x1-x2-x3-x4-x5`** (5D) |
| Importance bars | 4 (x0–x3) | **5** (x0–x4) |

---

## Notes

- All tasks target a single file: `functions/f6/f6.ipynb`
- No tests required — per constitution
- 8 new cells appended (cells 48–55); 47 existing cells untouched
- Commit after T010 validates end-to-end execution
- The `Standardize(m=1)` auto-untransform eliminates the manual inverse transform that F5 needed
- Noise init `0.1` is correct because Standardize normalises internal variance to ≈1.0 (RES-003)
- `model.train_targets` must be used for MLL scoring (standardised space), NOT `Y_train.squeeze(-1)` (raw space)
