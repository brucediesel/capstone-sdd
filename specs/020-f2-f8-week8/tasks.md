# Tasks: F2–F8 Week 8 — Bayesian Optimisation Iteration

**Input**: Design documents from `/specs/020-f2-f8-week8/`  
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Not requested — no test tasks included.

**Organization**: Tasks are grouped by function (the natural unit of work for self-contained Jupyter notebooks). Each function-phase creates a complete, independently executable notebook covering all 4 user stories (US1: data loading, US2: surrogate fitting, US3: acquisition + submission, US4: visualisation). Functions can be implemented in parallel (different files, no cross-dependencies).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1–US4)
- Include exact file paths in descriptions

## Path Conventions

- Notebooks: `functions/fX/fX - week 8.ipynb`
- Data: `data/fX/updated_inputs - Week 8.npy`, `data/fX/updated_outputs - Week 8.npy`

---

## Phase 1: Setup

**Purpose**: Verify prerequisites before creating any notebooks

- [X] T001 Verify all 7 Week 8 data file pairs exist in data/f2/ through data/f8/ and confirm Python environment has BoTorch, GPyTorch, PyTorch, numpy, matplotlib available in sdd-dev kernel

**Checkpoint**: Environment and data verified — notebook creation can begin

---

## Phase 2: F2 — SFGP + qLogNEI (2D) 🎯 MVP

**Goal**: Create the simplest notebook (2D, single GP fit, no interior penalty) as the template baseline

**Independent Test**: Run all cells in `f2 - week 8.ipynb` — loads 18 samples, fits GP, outputs `0.xxxxxx-0.xxxxxx` submission, shows 3-panel surrogate + convergence plots

- [X] T002 [P] [US1] Create functions/f2/f2 - week 8.ipynb with markdown header (Week 8 intro, strategy rationale), imports cell, hyperparameter constants cell (KERNEL, NOISE_LB, ARD, INPUT_NORM, N_RESTARTS, RAW_SAMPLES, BOUNDS), and data loading cell that loads updated_inputs/outputs Week 8 from data/f2/, converts to float64 tensors, validates 18 samples × 2D, displays tabular data, identifies best observation
- [X] T003 [US2] Add SFGP surrogate training cells to functions/f2/f2 - week 8.ipynb — SingleTaskGP with ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2)), GaussianLikelihood(noise_constraint=GreaterThan(1e-3)), Normalize(d=2) input transform, no output standardisation, .double(), single fit_gpytorch_mll call, print fitted lengthscales and noise
- [X] T004 [US3] Add qLogNEI acquisition and submission formatting cells to functions/f2/f2 - week 8.ipynb — qLogNoisyExpectedImprovement with X_baseline, prune_baseline=True, q=1, optimize_acqf with num_restarts=10, raw_samples=512, bounds [[0,0],[1,1]], clamp to [0, 0.999999], format as 0.xxxxxx-0.xxxxxx, include duplicate check against existing observations
- [X] T005 [US4] Add 3-panel surrogate visualisation (18×5: posterior mean viridis, std YlOrRd, NEI surface plasma, 50×50 grid, red scatter + yellow star) and convergence plot (running max, boundary axvline at 10.5, add Wk7→Wk8 boundary) to functions/f2/f2 - week 8.ipynb

**Checkpoint**: F2 notebook executes end-to-end, produces valid submission query and plots

---

## Phase 3: F8 — SFGP + qEI (8D)

**Goal**: Standard GP but 8D, with qEI acquisition and posterior-mean fallback

**Independent Test**: Run all cells in `f8 - week 8.ipynb` — loads 48 samples, fits GP, outputs 8-component submission, shows feature importance + surrogate + convergence plots

- [X] T006 [P] [US1] Create functions/f8/f8 - week 8.ipynb with markdown header (Week 8, SFGP + qEI strategy), imports cell (including qExpectedImprovement, ExpectedImprovement, SobolQMCNormalSampler, SobolEngine for fallback), constants cell (XI=0.01, MC_SAMPLES=256, NUM_RESTARTS=30, RAW_SAMPLES=4096), and data loading cell that loads Week 8 data from data/f8/, validates 48 samples × 8D, all-positive outputs, displays tabular data with best observation
- [X] T007 [US2] Add SFGP surrogate training cells to functions/f8/f8 - week 8.ipynb — SingleTaskGP with ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8)), GaussianLikelihood(noise_constraint=GreaterThan(1e-7)), default Standardize(m=1), single fit_gpytorch_mll call, print ARD lengthscales and noise
- [X] T008 [US3] Add qEI acquisition cells to functions/f8/f8 - week 8.ipynb — qExpectedImprovement with best_f=Y_train.max()+XI, SobolQMCNormalSampler(256), q=1, num_restarts=30, raw_samples=4096, bounds [0,1]⁸. Include fallback: if acq_val ≤ 0, generate 4096 Sobol candidates and pick highest posterior mean. Format submission as 8-component 0.xxxxxx string using format_query helper function
- [X] T009 [US4] Add feature importance bar chart (8×4: top-2 red, others blue, horizontal, inverted y), 3-panel surrogate visualisation (18×5: mean viridis, std magma, analytic EI plasma via ExpectedImprovement, 50×50 grid, white star for best observed, red X for candidate), and convergence plot (running best blue, grey scatter, weekly boundaries at 40.5/45.5/46.5, add Wk7→Wk8 boundary at 47.5) to functions/f8/f8 - week 8.ipynb

**Checkpoint**: F8 notebook executes end-to-end with valid 8D submission

---

## Phase 4: F3 — SFGP + qLogNEI (3D, 15 MLL restarts)

**Goal**: 3D GP with multi-restart MLL fitting and pairwise slice visualisation

**Independent Test**: Run all cells in `f3 - week 8.ipynb` — loads 23 samples, fits GP with 15 restarts, outputs 3-component submission, shows pairwise 2D slices + convergence

- [X] T010 [P] [US1] Create functions/f3/f3 - week 8.ipynb with markdown header, imports cell (including copy for deepcopy), constants cell (N_RESTARTS=15, LENGTHSCALE_INIT=0.25, SIGNAL_VAR_INIT=1.0, NOISE_VAR_INIT=0.1, BOUNDS), hyperparameter documentation markdown, and data loading cell that loads Week 8 data from data/f3/, validates 23 samples × 3D, applies manual z-score output standardisation (y_mean, y_std), displays per-dimension ranges for Compound A/B/C
- [X] T011 [US2] Add SFGP surrogate training cells to functions/f3/f3 - week 8.ipynb — SingleTaskGP with ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3)), GaussianLikelihood(noise_constraint=GreaterThan(1e-6)), no input transform, 15-restart MLL loop with torch.manual_seed(seed), lengthscale/noise/outputscale init on each restart, best model via copy.deepcopy by lowest negative MLL
- [X] T012 [US3] Add qLogNEI acquisition and submission cells to functions/f3/f3 - week 8.ipynb — qLogNoisyExpectedImprovement with X_baseline, prune_baseline=True, q=1, num_restarts=10, raw_samples=512, bounds [[0,0,0],[0.999999,0.999999,0.999999]], clamp and format as 3-component 0.xxxxxx string
- [X] T013 [US4] Add 3-panel pairwise 2D slice visualisation (18×5: pairs (0,1)/(0,2)/(1,2), third dim fixed at best_point, contourf mean viridis + 2σ white contours, 50×50 grid, de-standardised to original scale, labels Compound A/B/C) and convergence plot (boundary at 15.5, add Wk7→Wk8 boundary) to functions/f3/f3 - week 8.ipynb

**Checkpoint**: F3 notebook executes end-to-end with valid 3D submission

---

## Phase 5: F4 — MFGP + MF-qNEI (4D + fidelity)

**Goal**: Multi-fidelity GP with fidelity column, batch acquisition (q=4), best-of-4 selection

**Independent Test**: Run all cells in `f4 - week 8.ipynb` — loads 38 samples, appends fidelity column, fits MFGP with 15 restarts, proposes q=4 candidates, selects best by posterior mean, outputs 4-component submission

- [X] T014 [P] [US1] Create functions/f4/f4 - week 8.ipynb with markdown header, imports cell (including SingleTaskMultiFidelityGP, SobolQMCNormalSampler), 16-row hyperparameter documentation markdown, constants cell (N_RESTARTS=15, noise_lb=1e-4, q=4, MC_samples=64, acq_restarts=20, raw_samples=512, GRID_RES=80), and data loading cell that loads Week 8 data from data/f4/, validates 38 samples × 4D, applies manual z-score, appends constant fidelity column (all 1.0) to create X_mf shape (38, 5)
- [X] T015 [US2] Add MFGP surrogate training cells to functions/f4/f4 - week 8.ipynb — SingleTaskMultiFidelityGP with nu=2.5, linear_truncated=True, data_fidelities=[4], GaussianLikelihood(noise_constraint=GreaterThan(1e-4)), 15-restart MLL loop, best via copy.deepcopy, extract and print lengthscales ℓ₁–ℓ₄, outputscale, noise, power
- [X] T016 [US3] Add MF-qNEI acquisition cells to functions/f4/f4 - week 8.ipynb — qLogNoisyExpectedImprovement with X_baseline=X_mf, sampler=SobolQMCNormalSampler(64), q=4, num_restarts=20, raw_samples=512, fixed_features={4: 1.0}, 5D bounds. Select best of 4 candidates by highest de-standardised posterior mean. Clamp and format as 4-component submission string with validation
- [X] T017 [US4] Add 2-panel surrogate visualisation (16×6: top-2 dims by shortest ARD lengthscales, mean viridis + std plasma, 80×80 grid, red scatter + yellow star) and convergence plot (boundary at 30.5, add Wk7→Wk8 boundary) to functions/f4/f4 - week 8.ipynb

**Checkpoint**: F4 notebook executes end-to-end with valid 4D submission from best-of-4 selection

---

## Phase 6: F5 — GP + qLogNEI + Interior Penalty (4D)

**Goal**: GP with log1p transform, distance-based candidate selection, and multiplicative interior penalty

**Independent Test**: Run all cells in `f5 - week 8.ipynb` — loads 28 samples, fits GP with 15 restarts, proposes q=4 candidates with distance-based + IP selection, outputs 4-component submission with no boundary values

- [X] T018 [P] [US1] Create functions/f5/f5 - week 8.ipynb with markdown header, imports cell (including SobolQMCNormalSampler), 17-row hyperparameter documentation markdown, constants cell (N_RESTARTS=15, STEEPNESS=1.0, FLOOR=0.01), and data loading cell that loads Week 8 data from data/f5/, validates 28 samples × 4D, applies log1p → z-score transform (y_log = log1p(y_raw), then z-score), displays ranges
- [X] T019 [US2] Add GP surrogate training cells to functions/f5/f5 - week 8.ipynb — SingleTaskGP with ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4)), GaussianLikelihood(noise_constraint=GreaterThan(1e-6)), outcome_transform=None (explicitly disable Standardize), lengthscale init=0.5, noise init=0.1*Y_train.var(), outputscale init=1.0, 15-restart MLL loop, best via copy.deepcopy
- [X] T020 [US3] Add qLogNEI acquisition cells to functions/f5/f5 - week 8.ipynb — qLogNoisyExpectedImprovement with sampler=SobolQMCNormalSampler(512), q=4, num_restarts=50, raw_samples=3000, bounds [0,1]⁴. Distance-based selection: filter candidates to pred_means_orig >= median, pick farthest from training data via torch.cdist. Inverse transform via np.expm1. Clamp and format submission
- [X] T021 [US3] Add interior penalty cells to functions/f5/f5 - week 8.ipynb — markdown rationale, STEEPNESS=1.0, FLOOR=0.01 constants, compute w(x) = FLOOR + (1-FLOOR) × ∏sin(πxᵢ)^(2·STEEPNESS), apply multiplicatively: weighted_means = pred_means_orig × interior_weight, median filter on weighted means, farthest from data selection, min-distance warning < 0.05, format IP-based submission
- [X] T022 [US4] Add 3-panel surrogate visualisation (mean viridis, std magma, dim relevance bar 1/ℓ steelblue, 80×80 grid), convergence plot (boundary at 26.5, add Wk7→Wk8), IP 3-panel (mean, std, penalised mean plasma), and IP convergence plot to functions/f5/f5 - week 8.ipynb

**Checkpoint**: F5 notebook executes end-to-end, IP-selected point avoids boundaries, valid 4D submission

---

## Phase 7: F6 — SFGP + qLogNEI + Interior Penalty, Rank-Based (5D)

**Goal**: GP with feasibility bounds, rank-based interior penalty for all-negative outputs, ingredient labels

**Independent Test**: Run all cells in `f6 - week 8.ipynb` — loads 28 samples (all-negative), fits GP with 15 restarts, feasibility bounds enforce x4 ≥ 0.10, rank-based IP selects non-boundary point, outputs 5-component submission

- [X] T023 [P] [US1] Create functions/f6/f6 - week 8.ipynb with markdown header (strategy change rationale), imports cell (including re for regex validation), 15-row hyperparameter documentation markdown, constants cell (N_RESTARTS=15, STEEPNESS=1.0, FLOOR=0.01, ingredient_names), and data loading cell that loads Week 8 data from data/f6/, validates 28 samples × 5D, asserts all outputs negative, displays ingredient-labelled ranges
- [X] T024 [US2] Add SFGP surrogate training cells to functions/f6/f6 - week 8.ipynb — SingleTaskGP with ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5)), GaussianLikelihood(noise_constraint=GreaterThan(1e-2)), default Standardize(m=1) outcome transform, lengthscale init=0.5, noise init=0.2, outputscale init=1.0, 15-restart MLL loop, best via copy.deepcopy, post-fit assertion noise >= 1e-2
- [X] T025 [US3] Add qLogNEI acquisition cells to functions/f6/f6 - week 8.ipynb — qLogNoisyExpectedImprovement with sampler=SobolQMCNormalSampler(512), q=4, num_restarts=50, raw_samples=3000, feasibility bounds [[0.01,0.01,0.01,0.01,0.10],[1,1,1,1,1]]. Distance-based selection: filter to pred_means >= median, farthest from data, fallback to highest mean. Format submission with regex validation and x4 ≥ 0.10 assertion
- [X] T026 [US3] Add rank-based interior penalty cells to functions/f6/f6 - week 8.ipynb — markdown rationale for rank-based scoring (all-negative outputs), compute w(x) over 5D, rank_mean = argsort(argsort(pred_means))+1, rank_weight = argsort(argsort(interior_weight))+1, combined_score = rank_mean + rank_weight, median filter on combined_score, farthest from data, assert best_point[4] >= 0.10, format IP submission with per-ingredient breakdown
- [X] T027 [US4] Add 3-panel surrogate visualisation (mean viridis, std magma, 5-bar dim relevance, 80×80 grid), convergence plot (boundary at 26.5, add Wk7→Wk8), IP 3-panel (mean, std, penalty RdYlGn vmin=FLOOR, candidates with size ∝ combined_score), and IP convergence (observed line, running best dashed, IP-selected red hline, raw-best orange hline) to functions/f6/f6 - week 8.ipynb

**Checkpoint**: F6 notebook executes end-to-end, rank-based IP point avoids boundaries + milk ≥ 10%, valid 5D submission

---

## Phase 8: F7 — Neural Network + MC Dropout EI + Interior Penalty (6D)

**Goal**: PyTorch neural network surrogate (no BoTorch), MC dropout acquisition, multiplicative IP

**Independent Test**: Run all cells in `f7 - week 8.ipynb` — loads 38 samples, trains 6→5→5→1 NN for 200 epochs, MC dropout EI over 20k candidates with IP, outputs 6-component submission with no boundary values

- [X] T028 [P] [US1] Create functions/f7/f7 - week 8.ipynb with markdown header (NN + MC Dropout EI + IP, 8-row hyperparameter table), imports cell (numpy, matplotlib, torch, torch.nn, torch.optim — NO BoTorch), constants cell (LEARNING_RATE=0.005, EPOCHS=200, DROPOUT=0.1, MC_SAMPLES=50, N_CANDIDATES=20000, STEEPNESS=0.1, FLOOR=0.01), and data loading cell that loads Week 8 data from data/f7/, validates 38 samples × 6D, asserts all-positive outputs, applies z-score normalisation on both X (per-dim, +1e-8 stability) and y, labels: learning_rate/reg_strength/n_layers/dropout/batch_size/optimizer
- [X] T029 [US2] Add SurrogateNN definition and training cells to functions/f7/f7 - week 8.ipynb — define SurrogateNN(nn.Module) with architecture 6→5→5→1, ReLU activations, nn.Dropout(p=0.1) after each hidden layer, train with Adam(lr=0.005) and MSELoss for 200 epochs with torch.manual_seed(42), plot training loss (log scale, 10×4), compute and print R² on original scale via inverse z-score transform
- [X] T030 [US3] Add MC Dropout EI + Interior Penalty acquisition cells to functions/f7/f7 - week 8.ipynb — generate 20000 uniform random candidates in [0,1]⁶ with np.random.seed(42), z-score normalise, 50 stochastic forward passes with model.train(), EI = mean(max(pred_orig - y_best, 0)), compute w(x) with STEEPNESS=0.1/FLOOR=0.01 over 6D, penalised_ei = ei × interior_weight, select np.argmax(penalised_ei) with fallback to np.argmax(interior_weight) if all EI=0, clamp and format 6-component submission with per-hyperparameter breakdown
- [X] T031 [US4] Add feature importance via input gradient magnitude cell, 3-panel visualisation (18×5: NN mean viridis, MC uncertainty YlOrRd, IP heatmap RdYlGn, 50×50 grid, fixed dims at best observed, named axes), convergence plot (running best blue, IP-selected mean green hline, raw-EI best orange dashed hline, weekly boundaries at 30.5/31.5/32.5/33.5/36.5, add Wk7→Wk8 boundary at 37.5) to functions/f7/f7 - week 8.ipynb

**Checkpoint**: F7 notebook executes end-to-end, NN trains without divergence, IP-penalised point avoids boundaries, valid 6D submission

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all notebooks

- [X] T032 Verify all 7 original notebooks (f2.ipynb through f8.ipynb) remain unmodified by running git diff on functions/f2/f2.ipynb through functions/f8/f8.ipynb
- [X] T033 Run quickstart.md validation checklist: confirm each of the 7 notebooks executes without error, produces correct sample counts (F2:18, F3:23, F4:38, F5:28, F6:28, F7:38, F8:48), and outputs valid submission queries with all coordinates in [0.0, 0.999999]
- [X] T034 Verify SC-003: for F5, F6, F7 notebooks, confirm proposed sample points have no coordinate at exactly 0.0 or 0.999999 (interior penalty boundary suppression working)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — must complete first
- **F2–F7 (Phases 2–8)**: All depend on Setup (Phase 1) completion only
  - All 7 function phases can proceed **in parallel** (different notebook files)
  - Or sequentially in recommended order: F2 → F8 → F3 → F4 → F5 → F6 → F7
- **Polish (Phase 9)**: Depends on ALL function phases being complete

### Within Each Function Phase

- Tasks are **sequential** within a function (each cell group depends on previous cells)
- Data loading → Surrogate → Acquisition → Visualisation
- Interior penalty (F5, F6, F7) comes after base acquisition
- No [P] within a function since all tasks target the same notebook file

### Parallel Opportunities

All 7 function phases are fully independent — different files, no shared state:

```text
After T001 (Setup) completes:

  T002─T003─T004─T005                (F2, Phase 2)
  T006─T007─T008─T009                (F8, Phase 3)
  T010─T011─T012─T013                (F3, Phase 4)
  T014─T015─T016─T017                (F4, Phase 5)
  T018─T019─T020─T021─T022           (F5, Phase 6)
  T023─T024─T025─T026─T027           (F6, Phase 7)
  T028─T029─T030─T031                (F7, Phase 8)

All can run in parallel ↑

After all function phases complete:
  T032─T033─T034                     (Polish, Phase 9)
```

---

## Implementation Strategy

### MVP First (F2 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: F2 notebook (T002–T005)
3. **STOP and VALIDATE**: Run F2 notebook end-to-end
4. F2 is the simplest notebook — validates the template pattern before scaling

### Incremental Delivery

1. F2 (2D, no IP) → validates base GP + acquisition pattern
2. F8 (8D, qEI + fallback) → validates high-dimensional GP + fallback logic
3. F3 (3D, multi-restart) → validates MLL restart loop
4. F4 (4D, MFGP) → validates multi-fidelity pattern
5. F5 (4D, IP multiplicative) → validates interior penalty
6. F6 (5D, IP rank-based) → validates rank-based penalty for negative outputs
7. F7 (6D, NN) → validates non-GP surrogate

Each function adds value independently — a valid submission query for one more function.

---

## Notes

- All 7 function phases target **different files** — fully parallelisable
- Each notebook is self-contained per Constitution III
- Week 7 → Week 8 changes are minimal: data path, sample count, convergence boundary marker
- All hyperparameter values come from plan.md Week 7 code research (code is ground truth)
- Interior penalty functions (F5, F6, F7) produce TWO submission candidates: base and IP-adjusted
- Commit after each function phase completion
