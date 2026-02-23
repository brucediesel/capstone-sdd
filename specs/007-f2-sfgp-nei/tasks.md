# Tasks: F2 Week 7 — SFGP with NEI Acquisition

**Feature**: 007-f2-sfgp-nei  
**Branch**: `005-week7-pe-surrogates`  
**Input**: `specs/007-f2-sfgp-nei/spec.md`  
**Target notebook**: `functions/f2/f2.ipynb` (append-only — new cells at end)  
**No tests required** (per project constitution)

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can be authored in parallel (independent cells, no content dependency)
- **[Story]**: User story this task belongs to (US1, US2, US3)
- Each task appends one or more cells to `functions/f2/f2.ipynb`

---

## Phase 1: Setup

**Purpose**: Confirm prerequisites are in place before writing any notebook cells.

- [X] T001 Confirm `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy` exist and load cleanly (run `np.load(...)` to verify shapes: 17×2 inputs, 17 outputs)

**Checkpoint**: Data files confirmed present → proceed to Phase 2

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Section scaffold and data loading — both US1 and US2 depend on the model being trained, which depends on data being loaded here.

⚠️ **CRITICAL**: No user story implementation can begin until these two cells are in place

- [X] T002 Add `## Week 7 — SFGP with NEI Acquisition` header markdown cell at end of `functions/f2/f2.ipynb` (context: 17 cumulative data points, SFGP surrogate, NEI acquisition)
- [X] T003 [P] Add Step 1 data loading code cell in `functions/f2/f2.ipynb` (load `data/f2/updated_inputs - Week 7.npy` as `X_w7`, `data/f2/updated_outputs - Week 7.npy` as `y_w7`, print shapes and value ranges)

**Checkpoint**: Section header and data loading cells present → all story phases can begin

---

## Phase 3: User Story 1 — SFGP+NEI Section Delivers Submission Query (Priority: P1) 🎯 MVP

**Goal**: Train the SFGP on Week 7 data, optimise the NEI acquisition function, and produce a valid submission query in `x1-x2` format.

**Independent Test**: Run Phase 2 cells + Phase 3 cells only (T003 → T004 → T005 → T006). A line matching `>>> SUBMISSION: 0.xxxxxx-0.xxxxxx` must be printed with both values in `[0.0, 1.0]`.

### Implementation for User Story 1

- [X] T004 [US1] Add Step 3 SFGP training code cell in `functions/f2/f2.ipynb`:
  - Prepare `X_train_t` (17×2 tensor) and `y_train_t` (17×1 tensor)
  - Instantiate `SingleTaskGP` with `MaternKernel(nu=1.5, ard_num_dims=2)`, `GaussianLikelihood(noise_constraint=GreaterThan(NOISE_LB))`, and `Normalize(d=2)` input transform
  - Fit with `fit_gpytorch_mll(mll)` using `ExactMarginalLogLikelihood`
  - Print fitted lengthscales (one per dimension) and fitted noise level
- [X] T005 [US1] Add Step 4 NEI acquisition code cell in `functions/f2/f2.ipynb`:
  - Instantiate `qNoisyExpectedImprovement(model, X_baseline=X_train_t)`
  - Call `optimize_acqf(acq_function=nei, bounds=BOUNDS, q=1, num_restarts=N_RESTARTS, raw_samples=RAW_SAMPLES)`
  - Store result as `next_x` (1×2 tensor); print raw candidate value
- [X] T006 [US1] Add Step 7 submission formatting code cell in `functions/f2/f2.ipynb`:
  - Extract `x1, x2` from `next_x`, clamp to `[0.0, 1.0]`
  - Print `f">>> SUBMISSION: {x1:.6f}-{x2:.6f}"`

**Checkpoint**: US1 complete — a valid submission query is produced. Week 7 MVP is deliverable.

---

## Phase 4: User Story 2 — Visualise Surrogate and Acquisition Surface (Priority: P2)

**Goal**: Three diagnostic plots showing the GP posterior mean, GP posterior uncertainty, and NEI acquisition surface; plus a convergence running-maximum plot.

**Independent Test**: Run Phase 2 + Phase 3 (model trained, `next_x` available) + Phase 4 cells. Two figures must render: the 3-panel subplot and the convergence plot.

### Implementation for User Story 2

- [X] T007 [P] [US2] Add Step 5 three-panel visualization code cell in `functions/f2/f2.ipynb`:
  - Build 50×50 evaluation grid over `[0,1]²`
  - `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`
  - Panel (a): `contourf` posterior mean, colormap `viridis`, title "GP Posterior Mean"
  - Panel (b): `contourf` posterior std, colormap `YlOrRd`, title "GP Posterior Uncertainty"
  - Panel (c): `contourf` NEI acquisition surface, colormap `plasma`, title "NEI Acquisition Surface"
  - All panels: red scatter for 17 observed points; yellow star marker for `next_x`; colorbars; axis labels x1/x2
- [X] T008 [P] [US2] Add Step 6 convergence plot code cell in `functions/f2/f2.ipynb`:
  - `running_max = np.maximum.accumulate(y_w7)`
  - Plot running max vs sample index
  - `plt.axvline(x=10.5, linestyle='--', color='gray', label='Weekly submissions start')`
  - Title: "Function 2 — Convergence Plot (Week 7)"; x-label: "Sample Index"; y-label: "Best Output"

**Checkpoint**: US2 complete — both visualization outputs confirmed. All three panels and convergence plot visible.

---

## Phase 5: User Story 3 — Explicit Hyperparameter Display (Priority: P3)

**Goal**: Named constants with plain-English justifications printed before training, so an examiner can read the configuration without inspecting source code.

**Independent Test**: Run only the hyperparameter cell (T009 alone) — all parameter names, values, and rationale must print without requiring the model to be trained.

**Note on notebook ordering**: T009's cell must be **inserted between Step 1 (T003) and Step 3 (T004)** in the notebook. Author it last but place it in position 3 of the Week 7 section.

### Implementation for User Story 3

- [X] T009 [US3] Add Step 2 hyperparameter constants code cell in `functions/f2/f2.ipynb` (insert after data loading cell, before training cell):
  - `KERNEL = 'matern15'` — smooth but non-differentiable response, appropriate for unknown regularity
  - `NOISE_LB = 1e-3` — prevents noise collapsing to zero; reflects measurement uncertainty in black-box outputs
  - `ARD = True` — allows each input dimension to have its own lengthscale, so model can detect if one dimension matters more
  - `INPUT_NORMALIZE = True` — centres and scales inputs inside the GP; improves numerical conditioning
  - `N_RESTARTS = 10` — multi-start acquisition optimisation to avoid local optima in 2D
  - `RAW_SAMPLES = 512` — number of Sobol initial candidates for acquisition optimisation
  - `BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]])` — unit hypercube domain
  - Print all values with one-line plain-English justifications

**Checkpoint**: US3 complete — hyperparameter cell runs independently, all parameters and rationale visible.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation and commit.

- [X] T010 Run all Week 7 section cells top-to-bottom in `functions/f2/f2.ipynb` and confirm: no runtime errors, three visualization panels rendered, convergence plot rendered, submission line printed
- [X] T011 [P] Verify SC-002: SFGP reports exactly 2 distinct lengthscale values — confirm ARD is active (both values should differ)
- [X] T012 [P] Verify SC-004: print fitted noise level and confirm it is ≥ 1e-3 (noise lower bound enforced)
- [X] T013 [P] Verify SC-005: confirm proposed `x1-x2` point does not duplicate any of the 17 observed input pairs
- [X] T014 Commit `functions/f2/f2.ipynb` (Week 7 section added) to `005-week7-pe-surrogates` with message `feat(007): add Week 7 SFGP+NEI section to f2.ipynb`

---

## Dependency Graph

```
Phase 1 (T001)
    └── Phase 2 (T002, T003) — BLOCKING
            ├── Phase 3 US1 (T004 → T005 → T006)  ← MVP: produces submission query
            │       └── Phase 4 US2 (T007, T008)   ← needs trained model + next_x
            └── Phase 5 US3 (T009)                 ← notebook position: before T004
                    (depends on Phase 2 for section context, but can be authored after Phase 3)

Phase 6 — runs after all phases complete
```

**User story independence**:
- **US1** (P1): Can be tested with just T003 → T004 → T005 → T006
- **US2** (P2): Can be tested once US1 is complete (needs model and `next_x`)
- **US3** (P3): Cell is standalone (constants only); can be authored any time and inserted in correct notebook position

---

## Parallel Execution Opportunities

Within Phase 2: T002 and T003 can be authored simultaneously (markdown vs code cell, no shared content).

Within Phase 4: T007 and T008 can be authored simultaneously (different plots, both read the same already-computed `next_x` and `y_w7`).

Within Phase 6: T011, T012, and T013 are independent spot-checks that can be run in any order.

---

## Implementation Strategy

| Phase | Delivers | Testable? |
|-------|----------|-----------|
| Phase 1–2 | Data confirmed + section scaffolded | Manually verify |
| Phase 3 (US1) | **MVP** — submission query produced | Run 4 cells, check output |
| Phase 4 (US2) | Diagnostic visualizations | Check 2 figures render |
| Phase 5 (US3) | Hyperparameter transparency | Run 1 cell, read output |
| Phase 6 | Validated + committed | All cells clean |

**Suggested MVP scope**: Complete Phase 1–3 to get a valid submission query. Phases 4–5 can follow without blocking the submission deadline.

---

## Task Summary

| Phase | Tasks | User Story |
|-------|-------|------------|
| Setup | T001 | — |
| Foundational | T002–T003 | — |
| Phase 3 | T004–T006 | US1 (P1) |
| Phase 4 | T007–T008 | US2 (P2) |
| Phase 5 | T009 | US3 (P3) |
| Polish | T010–T014 | — |
| **Total** | **14 tasks** | **3 user stories** |
