# Tasks: F3 Week 7 – SFGP with Matérn-5/2 ARD and NEI Acquisition

**Input**: Design documents from `/specs/009-f3-sfgp-nei/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅  
**Tests**: Not required (constitution: "No unit tests required")  
**Target file**: `functions/f3/f3.ipynb` — all tasks append new cells; zero existing cells modified  

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files or no dependencies on incomplete tasks)
- **[Story]**: Which user story — US1 (Data Loading), US2 (Model Training), US3 (NEI Acquisition), US4 (Visualisations)
- All file paths are relative to repository root

---

## Phase 1: Setup

**Purpose**: Add the Week 7 section header to the notebook

- [ ] T001 Add Week 7 section header markdown cell (Cell 1) to functions/f3/f3.ipynb

  Append a **markdown cell** after the last existing cell (the Research section) with:
  - Heading: `## Week 7 — SFGP with Matérn-5/2 ARD`
  - Brief paragraph explaining: SFGP surrogate on cumulative Week 7 data, NEI acquisition, why Matérn-5/2 with ARD is appropriate for drug dose–response surfaces, why NEI handles observation noise
  - See `specs/009-f3-sfgp-nei/contracts/week7-cells.md` Cell 1 for exact content

---

## Phase 2: User Story 1 — Week 7 Data Loaded and Validated (Priority: P1) 🎯 MVP

**Goal**: Load cumulative Week 7 inputs/outputs, validate ranges, display summary

**Independent Test**: Run only Cells 1-2; confirm sample count (~22), input range [0,1], best observed value printed

### Implementation

- [ ] T002 [US1] Add imports and data loading code cell (Cell 2) to functions/f3/f3.ipynb

  Append a **code cell** with step label comment `# Step 1: Load Week 7 Data`. Must:
  1. Import all libraries needed for the entire Week 7 section (per RES-006):
     - `copy`, `warnings`, `numpy as np`, `torch`
     - `matplotlib.pyplot as plt`
     - `from botorch.models import SingleTaskGP`
     - `from botorch.fit import fit_gpytorch_mll`
     - `from gpytorch.mlls import ExactMarginalLogLikelihood`
     - `from gpytorch.kernels import MaternKernel, ScaleKernel`
     - `from gpytorch.constraints import GreaterThan`
     - `from gpytorch.likelihoods import GaussianLikelihood`
     - `from botorch.acquisition.logei import qLogNoisyExpectedImprovement`
     - `from botorch.optim import optimize_acqf`
  2. Load data:
     - `X_raw = np.load('../../data/f3/updated_inputs - Week 7.npy')` → shape `(n, 3)` float64
     - `Y_raw = np.load('../../data/f3/updated_outputs - Week 7.npy')` → shape `(n,)` float64
  3. Print: sample count `n`, input min/max per column, output min/max, best observed value and its index (`Y_raw.argmax()`)
  4. Validate all inputs in [0.0, 1.0]: if any value outside range, print warning with row index
  - **References**: data-model.md E-01, E-02; contracts Cell 2

**Checkpoint**: Cell 2 runs standalone and prints summary — US1 complete

---

## Phase 3: User Story 2 — SFGP Surrogate Trained with Specified Hyperparameters (Priority: P1)

**Goal**: Document hyperparameters, train model with 15 multi-restart MLL, print fitted values

**Independent Test**: Run Cells 1-4; confirm 5 labelled hyperparameter values printed (ℓ_A, ℓ_B, ℓ_C, σ²_f, σ²_n) with distinct lengthscales

### Implementation

- [ ] T003 [P] [US2] Add hyperparameter explanation markdown cell (Cell 3) to functions/f3/f3.ipynb

  Append a **markdown cell** with step label `### Step 2: SFGP Hyperparameters and Justifications`. Must document all 10 items from the contracts Cell 3 table:

  | Hyperparameter | Value | Document |
  |---|---|---|
  | `N_RESTARTS` | 15 | Why multi-restart MLL is needed (local optima) |
  | `LENGTHSCALE_INIT` | 0.25 | Why ~0.2–0.3 for [0,1]-scaled inputs |
  | `SIGNAL_VAR_INIT` | 1.0 | = Var(y) after z-score by construction |
  | `NOISE_VAR_INIT` | 0.1 | Conservative 10% noise-to-signal for unknown black-box |
  | `JITTER` | 1e-6 | Numerical stability floor |
  | Mean function | Constant (learned) | Why constant for unknown function |
  | Kernel | Matérn-5/2 ARD | Why not RBF; what ARD provides (per-dimension lengthscales) |
  | Likelihood | Gaussian noise | When to consider Student-t instead (heavy-tail outliers) |
  | NEI | `qLogNoisyExpectedImprovement` | Why NEI over analytic EI (noisy observations) |
  | Bounds | [0, 0.999999]³ | Challenge submission format constraint |

  - **References**: contracts Cell 3; research.md RES-001 through RES-004

- [ ] T004 [US2] Add SFGP training with 15 random restarts code cell (Cell 4) to functions/f3/f3.ipynb

  Append a **code cell** with step label comment `# Step 3: Train SFGP with 15 Random Restarts`. Must:
  1. Convert data to tensors:
     - `X_train = torch.tensor(X_raw, dtype=torch.float64)` → shape `(n, 3)`
     - Compute `y_mean = Y_raw.mean()`, `y_std = Y_raw.std()` (as Python floats)
     - `Y_train = torch.tensor((Y_raw - y_mean) / y_std, dtype=torch.float64).unsqueeze(-1)` → shape `(n, 1)`
  2. Define constants at cell top: `N_RESTARTS = 15`, `LENGTHSCALE_INIT = 0.25`, `SIGNAL_VAR_INIT = 1.0`, `NOISE_VAR_INIT = 0.1`
  3. Multi-restart loop (per RES-003):
     ```python
     best_loss = float('inf')
     best_model = None
     for seed in range(N_RESTARTS):
         torch.manual_seed(seed)
         covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
         likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
         model = SingleTaskGP(X_train, Y_train, covar_module=covar_module, likelihood=likelihood)
         # Set initial hyperparameters
         model.covar_module.base_kernel.lengthscale = LENGTHSCALE_INIT
         model.covar_module.outputscale = SIGNAL_VAR_INIT
         model.likelihood.noise = NOISE_VAR_INIT
         mll = ExactMarginalLogLikelihood(model.likelihood, model)
         fit_gpytorch_mll(mll)
         # Score
         model.train()
         with torch.no_grad():
             output = model(X_train)
             loss = -mll(output, Y_train.squeeze(-1)).item()
         if loss < best_loss:
             best_loss = loss
             best_model = copy.deepcopy(model)
     best_model.eval()
     ```
  4. Print fitted hyperparameters (all clearly labelled):
     - `ℓ_A (Compound A) = best_model.covar_module.base_kernel.lengthscale[0, 0].item()`
     - `ℓ_B (Compound B) = best_model.covar_module.base_kernel.lengthscale[0, 1].item()`
     - `ℓ_C (Compound C) = best_model.covar_module.base_kernel.lengthscale[0, 2].item()`
     - `σ²_f = best_model.covar_module.outputscale.item()`
     - `σ²_n = best_model.likelihood.noise.item()`
  5. Print best restart's −MLL loss value
  - **References**: data-model.md E-03 through E-07; research.md RES-001, RES-003; contracts Cell 4

**Checkpoint**: Cells 1-4 run end-to-end; 5 HP values printed; 3 lengthscales are distinct — US2 complete

---

## Phase 4: User Story 3 — NEI Acquisition Proposes Next Sample Point (Priority: P1)

**Goal**: Run NEI optimisation, extract candidate, format for challenge submission

**Independent Test**: Run Cells 1-5 then Cell 8; confirm formatted string `0.xxxxxx-0.yyyyyy-0.zzzzzz` printed

**Note**: US3 spans two non-adjacent notebook cells (Cell 5 and Cell 8). Cell 8 is the final cell in the notebook, appended after US4's Cells 6-7. Task T008 must therefore be implemented after T006 and T007, despite belonging to US3.

### Implementation

- [ ] T005 [US3] Add NEI acquisition code cell (Cell 5) to functions/f3/f3.ipynb

  Append a **code cell** with step label comment `# Step 4: NEI Acquisition — Propose Next Sample`. Must:
  1. Define bounds: `BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [0.999999, 0.999999, 0.999999]], dtype=torch.double)`
  2. Construct acquisition function:
     ```python
     nei = qLogNoisyExpectedImprovement(
         model=best_model, X_baseline=X_train, prune_baseline=True
     )
     ```
  3. Optimise:
     ```python
     candidate, acq_value = optimize_acqf(
         nei, bounds=BOUNDS, q=1, num_restarts=10, raw_samples=512
     )
     ```
  4. Extract result: `next_x_raw = candidate.detach().squeeze(0).cpu().numpy()` → shape `(3,)`
  5. Print: proposed point components (each labelled A, B, C) and NEI acquisition value
  - **References**: data-model.md E-08; research.md RES-004; contracts Cell 5

**Checkpoint (partial)**: Cell 5 produces `next_x_raw` array — acquisition complete; submission formatting in T008

---

## Phase 5: User Story 4 — Surrogate and Convergence Visualisations (Priority: P2)

**Goal**: Produce 3-panel surrogate slice plots and cumulative convergence plot

**Independent Test**: Run Cells 1-7; confirm 3 contour plots with yellow star and 1 convergence plot render

### Implementation

- [ ] T006 [P] [US4] Add surrogate pairwise 2D slice plots code cell (Cell 6) to functions/f3/f3.ipynb

  Append a **code cell** with step label comment `# Step 5: Surrogate Visualisation — Pairwise 2D Slices`. Must:
  1. Find best observed point: `best_idx = Y_raw.argmax()`, `best_point = X_raw[best_idx]`
  2. Create figure: `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`
  3. Define dimension pairs: `pairs = [(0, 1), (0, 2), (1, 2)]` and labels `dim_labels = ['Compound A', 'Compound B', 'Compound C']`
  4. For each pair `(i, j)` with fixed dimension `k`:
     - Build 50×50 meshgrid over [0, 1] for dims i, j
     - Fix dim k at `best_point[k]`
     - Construct grid tensor `(2500, 3)` dtype=float64
     - Get GP posterior: `posterior = best_model.posterior(grid_tensor)`
     - Extract mean and std, de-standardise to original scale: `mean_orig = mean * y_std + y_mean`, `std_orig = std * y_std`
     - Plot `contourf` (cmap='viridis') for mean with colourbar
     - Overlay 2σ contour lines (colour='white', alpha=0.4)
     - Scatter observed points `X_raw[:, i]` vs `X_raw[:, j]` as red dots
     - Mark proposed `next_x_raw[i]`, `next_x_raw[j]` as yellow star (marker='*', s=200)
     - Axis labels: `dim_labels[i]` and `dim_labels[j]`
     - Title: `f"F3 Week 7 Surrogate — {dim_labels[i]} vs {dim_labels[j]} ({dim_labels[k]}={best_point[k]:.3f})"`
  5. `plt.tight_layout(); plt.show()`
  - **References**: data-model.md E-10; research.md RES-005; contracts Cell 6

- [ ] T007 [US4] Add convergence plot code cell (Cell 7) to functions/f3/f3.ipynb

  Append a **code cell** with step label comment `# Step 6: Convergence Plot`. Must:
  1. Compute `running_max = np.maximum.accumulate(Y_raw)`
  2. Create figure: `fig, ax = plt.subplots(figsize=(10, 5))`
  3. Plot `running_max` as blue line with circle markers
  4. Scatter individual observations in gray (alpha=0.4)
  5. Add vertical dashed red line at x=15.5 with label "Initial → Weekly" (15 initial samples + weekly additions)
  6. x-axis label: `"Sample Number"`, y-axis label: `"Best Observed Output"`
  7. Title: `"Function 3 — Convergence Plot (Week 7)"`
  8. Add legend, grid
  9. Print best observed value and the sample number where it was achieved
  - **References**: data-model.md E-11; contracts Cell 7

**Checkpoint**: All visualisations render — US4 complete

---

## Phase 6: User Story 3 (continued) — Submission Output

**Purpose**: Format and display the final submission query (Cell 8 must be last in notebook)

- [ ] T008 [US3] Add submission query formatting code cell (Cell 8) to functions/f3/f3.ipynb

  Append a **code cell** (must be final cell) with step label comment `# Step 7: Format Submission Query`. Must:
  1. Clamp: `clamped = np.clip(next_x_raw, 0.0, 0.999999)`
  2. Format: `submission = "-".join([f"{x:.6f}" for x in clamped])`
  3. Print summary block with:
     - Surrogate type: SFGP (Matérn-5/2 ARD)
     - Acquisition type: NEI (qLogNoisyExpectedImprovement)
     - Fitted hyperparameters: ℓ_A, ℓ_B, ℓ_C, σ²_f, σ²_n (from `best_model`)
     - Raw proposed point: `next_x_raw`
     - Formatted submission string on its own clearly-marked line for copy/paste
  4. Final output must match regex: `r"0\.\d{6}-0\.\d{6}-0\.\d{6}"`
  - **References**: data-model.md E-08, E-09; contracts Cell 8

**Checkpoint**: Full US3 complete — submission string ready for copy/paste

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and commit

- [ ] T009 Run full kernel-restart validation per quickstart.md on functions/f3/f3.ipynb
  - Restart kernel, run all 8 new cells top-to-bottom
  - Verify all items in quickstart.md verification checklist pass:
    - Cell 2 prints sample count and best value
    - Cell 4 prints 5 labelled HP values with distinct lengthscales
    - Cell 5 completes without error; `next_x_raw` defined
    - Cell 6 renders 3 plots with yellow star and labelled axes
    - Cell 7 renders convergence plot with correct labels
    - Cell 8 prints `0.XXXXXX-0.YYYYYY-0.ZZZZZZ` format string

- [ ] T010 Verify zero existing cells modified and commit changes to functions/f3/f3.ipynb
  - Run `git diff functions/f3/f3.ipynb` — only additions at end, no modifications to existing cells
  - `git add functions/f3/f3.ipynb && git commit -m "feat(f3): add Week 7 SFGP Matern-5/2 ARD + NEI section"`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **US1 (Phase 2)**: Depends on Phase 1 — Cell 2 provides data for all subsequent cells
- **US2 (Phase 3)**: Depends on Phase 2 — T003 (markdown) is [P] and can be written before T002 completes, but T004 (training) requires `X_raw`/`Y_raw` from T002
- **US3 (Phase 4)**: T005 depends on T004 (needs `best_model`); T008 depends on T005 (needs `next_x_raw`) AND must be appended after T006/T007 in the notebook
- **US4 (Phase 5)**: T006 depends on T004 (needs `best_model`) and T005 (needs `next_x_raw` for star marker); T007 depends only on T002 (needs `Y_raw`)
- **US3 continued (Phase 6)**: T008 must be the last cell appended; depends on T005
- **Polish (Phase 7)**: Depends on all implementation phases complete

### Notebook Cell Ordering Constraint

All cells must be appended in strict sequential order (Cell 1 → Cell 8). This means:
- T005 (Cell 5) must be appended before T006 (Cell 6)
- T006 (Cell 6) must be appended before T007 (Cell 7)
- T007 (Cell 7) must be appended before T008 (Cell 8)

**Execution order**: T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010

### Parallel Opportunities

Since all tasks modify a single file (`f3.ipynb`) and cells must be appended sequentially, true parallelism is limited. However:
- **T003 [P]**: The markdown cell content can be drafted independently of any code cell
- **T006 [P]**: The plot code can be drafted in parallel with T005, since the code structure is independent (only the variable `next_x_raw` is needed at runtime, not at write-time)
- **T007 [P]**: Convergence plot code only needs `Y_raw`, so its logic is independent of the model training and acquisition cells

### Within Each User Story

- Markdown explanation cells before code cells (T003 before T004)
- Data loading before any computation (T002 is foundational)
- Model training before acquisition (T004 before T005)
- Acquisition before submission formatting (T005 before T008)

---

## Parallel Example: Drafting Phase

```
# These cells can be drafted simultaneously (content is independent):
T003 [P] [US2] — Hyperparameter explanation markdown (no code deps)
T006 [P] [US4] — Surrogate plot code (structure known from contracts)
T007 [P] [US4] — Convergence plot code (only uses Y_raw)

# But must be inserted into the notebook in cell order:
T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 + 3)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: US1 — data loads correctly (T002)
3. Complete Phase 3: US2 — model trains, HPs printed (T003, T004)
4. Complete Phase 4: US3 partial — NEI proposes point (T005)
5. **STOP and VALIDATE**: Model trains, acquisition runs, point is proposed
6. At this point the submission string can be manually formatted — the challenge entry is achievable

### Full Delivery

1. Setup + US1 + US2 + US3 (partial) → MVP achieved
2. Add US4 (T006, T007) → Visualisations for assessment
3. Add US3 final (T008) → Automated submission formatting
4. Polish (T009, T010) → Validated and committed

### Key Constants (from research.md and data-model.md)

| Constant | Value | Used in |
|---|---|---|
| `N_RESTARTS` | 15 | T004 (training loop) |
| `LENGTHSCALE_INIT` | 0.25 | T004 (HP initialisation) |
| `SIGNAL_VAR_INIT` | 1.0 | T004 (HP initialisation) |
| `NOISE_VAR_INIT` | 0.1 | T004 (HP initialisation) |
| `JITTER` | 1e-6 | T004 (noise constraint) |
| `BOUNDS` | [[0,0,0],[0.999999,0.999999,0.999999]] | T005 (optimize_acqf) |
| `num_restarts` | 10 | T005 (optimize_acqf) |
| `raw_samples` | 512 | T005 (optimize_acqf) |
| Grid resolution | 50×50 | T006 (slice plots) |
| Initial samples | 15 | T007 (convergence boundary line at x=15.5) |

---

## Notes

- All code uses `torch.float64` throughout — mixing float32 causes silent MLL fitting failures
- Manual z-score standardisation (not BoTorch `Standardize` transform) per RES-002
- Student-t likelihood is documented in markdown only (T003); Gaussian is implemented (T004)
- The `format_query()` helper already exists in earlier cells but T008 uses inline formatting for self-containment
- [P] marks conceptual parallelism for drafting; actual notebook insertion is strictly sequential
- Commit after each task or logical group for safety
