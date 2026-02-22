# Tasks: Week 7 — F1 Hurdle Model with Weighted UCB and Local Penalization

**Branch**: `005-week7-pe-surrogates`  
**Feature**: Week 7 F1 Section — Hurdle Model  
**Date**: 2026-02-22  
**Source**: [spec.md](spec.md) · [plan.md](plan.md) · [data-model.md](data-model.md) · [contracts/cell-contracts.md](contracts/cell-contracts.md)

---

## Phase 1 — Setup

**Goal**: Insert the Week 7 section header into `functions/f1/f1.ipynb` before any code cells, establishing the new section boundary and its rationale.

**Independent Test**: Open `functions/f1/f1.ipynb` and confirm a new `## Week 7` markdown heading appears after the last Week 6 cell (cell 54) with a paragraph describing the hurdle model motivation.

- [X] T001 Create Week 7 section header markdown cell (W7-01) in `functions/f1/f1.ipynb` after Week 6 query cell, with heading `## Week 7 — Hurdle Model with Weighted UCB and Local Penalization` and exploration rationale paragraph

---

## Phase 2 — Foundational

**Goal**: Load and validate the Week 7 data. All subsequent user story phases depend on `X_w7`, `y_w7`, `y_binary`, `n_positive`, `X_pos`, `y_pos`, and `y_pos_log` being correctly defined, and the degenerate fallback guard being active.

**Independent Test**: Run only cell W7-02. Verify printed output shows 17 samples × 2 inputs, all inputs within [0.000000, 0.999999], count of positive vs non-positive outputs, and no Python errors. If `n_positive < 3`, confirm `FALLBACK_MODE = True` is printed.

- [X] T002 Create data load and validation code cell (W7-02) in `functions/f1/f1.ipynb` that loads `data/f1/updated_inputs - Week 7.npy` and `data/f1/updated_outputs - Week 7.npy`, validates input range, prints class balance, derives `y_binary`/`X_pos`/`y_pos`/`y_pos_log`, and sets `FALLBACK_MODE` with descriptive warning if `n_positive < MIN_POSITIVE`

---

## Phase 3 — User Story 1: Load and Validate Week 7 Data (P1)

**Story Goal**: As a student, I can confirm the Week 7 dataset is clean and correctly sized before any modelling, so downstream cells are guaranteed to have valid inputs.

**Independent Test**: Execute T002 (cell W7-02) in isolation. Output must show `17 samples, 2 inputs each`, input range confirmation, and positive/negative class counts. No downstream cells need to run.

> T002 (above in Phase 2) is the sole implementation task for this story. It is listed in Phase 2 because it also serves as the foundational prerequisite for all subsequent phases.

---

## Phase 4 — User Story 2: Fit the Hurdle Model with Explicit Hyperparameters (P1)

**Story Goal**: As a student, I can fit a documented two-stage hurdle model on the 17 samples with every hyperparameter named, valued, and justified, so the model is reproducible and explainable.

**Independent Test**: Run cells W7-03 → W7-06 in order (requires T002). Verify: markdown table lists all 8 hyperparameters with rationale; constants cell prints all 8 values; Stage 1 prints training accuracy; Stage 2 prints training R² on the log scale (or a `FALLBACK_MODE` warning if fewer than 3 positive samples).

- [X] T003 [P] [US2] Create hyperparameter rationale markdown cell (W7-03) in `functions/f1/f1.ipynb` listing `C_STAGE1`, `N_ESTIMATORS`, `MAX_DEPTH`, `KAPPA`, `PENALTY_RADIUS`, `N_CANDIDATES`, `GRID_RES`, `MIN_POSITIVE` — each as `**NAME = value**: one-sentence rationale` following the Week 5/6 pattern
- [X] T004 [P] [US2] Create hyperparameter constants code cell (W7-04) in `functions/f1/f1.ipynb` defining all 8 named constants with inline comments and a `print` statement per constant
- [X] T005 [US2] Create Stage 1 classifier code cell (W7-05) in `functions/f1/f1.ipynb` that fits `CalibratedClassifierCV(LogisticRegression(C=C_STAGE1, max_iter=1000, class_weight='balanced'), cv=3, method='sigmoid')` on `(X_w7, y_binary)`, prints training accuracy, and stores `stage1_clf` and `p_train`
- [X] T006 [US2] Create Stage 2 regressor code cell (W7-06) in `functions/f1/f1.ipynb` that fits `RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)` on `(X_pos, y_pos_log)`, prints training R² on the log scale, and stores `stage2_rf` (skipped with warning if `FALLBACK_MODE` is `True`)

---

## Phase 5 — User Story 3: Apply Weighted UCB with Local Penalization and Propose Query (P1)

**Story Goal**: As a student, I can run the weighted UCB acquisition function with multiplicative Gaussian local penalization over all 17 existing data points to produce a single exploration-focused query, formatted for submission.

**Independent Test**: Run cells W7-07 and W7-10 (requires T002 → T006). Verify: printed output shows best penalized UCB score, proposed `next_x`, and minimum distance to existing data ≥ 0.05. W7-10 prints `X.XXXXXX-X.XXXXXX` with exactly six decimal places, both values in [0.000000, 0.999999].

- [X] T007 [US3] Create weighted UCB and local penalization code cell (W7-07) in `functions/f1/f1.ipynb` that generates `N_CANDIDATES` uniform random candidates, evaluates `a(x) = p_cand*mu_cand + KAPPA*p_cand*sigma_rf_cand`, applies multiplicative Gaussian penalty mask against all 17 `X_w7` points with radius `PENALTY_RADIUS`, selects `argmax(acq_penalized)` as `next_x` (clipped to [0.000000, 0.999999]), and prints best score and min distance to existing data
- [X] T008 [US3] Create submission query formatter code cell (W7-10) in `functions/f1/f1.ipynb` that prints `Week 7 Submission Query: X.XXXXXX-X.XXXXXX` using the already-clipped `next_x` from T007

---

## Phase 6 — User Story 4: Reproduce Week 6 Visualizations (P2)

**Story Goal**: As a student, I can review the Week 7 results using the same three visualization types used in Week 6 — surrogate surface, uncertainty surface, acquisition surface, and convergence plot — so progress is consistently comparable across submissions.

**Independent Test**: Run cells W7-08 and W7-09 (requires T002 → T007). Verify: a 3-panel `(18, 5)` figure renders with correct titles, colorbars, training point scatter, and yellow star at `next_x`; a separate `(10, 5)` convergence figure renders showing running maximum over 17 observations.

- [X] T009 [P] [US4] Create 3-panel surrogate and acquisition surface plot code cell (W7-08) in `functions/f1/f1.ipynb` using `plt.subplots(1, 3, figsize=(18, 5))` with panels: (1) `grid_hurdle` / viridis / `'Hurdle Mean Prediction (ŷ = p·expm1(μ))'`, (2) `grid_uncertainty` / YlOrRd / `'Hurdle Uncertainty (p·σ_RF)'`, (3) `grid_ucb` / plasma / `f'Penalized UCB Acquisition (κ={KAPPA})'`; training points as red/blue scatter (positive/non-positive, matching Week 5/6 style), proposed point as yellow star `s=200`, one colorbar per panel
- [X] T010 [P] [US4] Create convergence plot code cell (W7-09) in `functions/f1/f1.ipynb` using `plt.figure(figsize=(10, 5))` with running maximum `'b-o'` line, individual observations as gray scatter, vertical dashed red line at x=10.5, title `'Function 1 — Convergence Plot (Week 7)'`, x-label `'Observation Number'`, y-label `'Objective Value'`, legend and grid

---

## Final Phase — Polish and Verification

**Goal**: Confirm the complete Week 7 section executes end-to-end without errors and meets all five success criteria from the spec.

- [X] T011 Run all Week 7 cells (W7-01 through W7-10) top-to-bottom in `functions/f1/f1.ipynb` and verify SC-001 (no errors), SC-002 (zero undocumented magic numbers), SC-003 (query ≥ 0.05 distance from all existing points, correct format), SC-004 (all 3 plots render with 17-point history and proposed query), SC-005 (proposed point has fewer than 3 existing data points within 0.15-radius)

---

## Dependencies

```
T001 (Setup: section header)
  └──► T002 (Foundational: data load) ←── blocks all phases below
          ├──► T003 [P] (US2: hyperparameter markdown)
          ├──► T004 [P] (US2: hyperparameter constants)
          │       └──► T005 (US2: Stage 1 fit)
          │               └──► T006 (US2: Stage 2 fit)
          │                       └──► T007 (US3: UCB + penalization)
          │                               ├──► T008 (US3: query formatter)
          │                               ├──► T009 [P] (US4: 3-panel plot)
          │                               └──► T010 [P] (US4: convergence plot)
          │
          └──────────────────────────────────► T011 (Polish: end-to-end verify)
```

**Story completion order**: US1 (T002) → US2 (T003–T006) → US3 (T007–T008) → US4 (T009–T010) → Polish (T011)

---

## Parallel Execution Examples

**Within Phase 4 (US2):** T003 and T004 can be implemented simultaneously — T003 is a markdown cell with no code dependencies; T004 is a pure constants cell with no imports.

**Within Final Phase (US4):** T009 (3-panel plot) and T010 (convergence plot) operate on different variables (`X_grid`/`next_x` vs `y_w7`) and produce independent figures — implement simultaneously.

---

## Implementation Strategy

**MVP scope (Phase 1 through Phase 5, tasks T001–T008)**: These 8 tasks implement the core notebook section that produces the submission query. T009 and T010 (plots) are required for the submission but do not affect the query value itself.

**Incremental delivery:**
1. T001 → T002: Confirm data loads correctly before writing any model code
2. T003 → T004: Establish all constants in one cell before fitting any model (avoids scattered magic numbers)
3. T005 → T006: Fit Stage 1, confirm probabilities are non-degenerate, then fit Stage 2
4. T007: Confirm penalized UCB produces a candidate ≥ 0.05 from all existing points before adding formatting/plots
5. T008 → T009 → T010: Finalize output format and add visualizations
6. T011: End-to-end run for final verification

---

## Task Count Summary

| Scope | Count |
|-------|-------|
| Total tasks | 11 |
| US1 (Load & Validate) | 1 (T002, shared with Foundational) |
| US2 (Hurdle Model) | 4 (T003–T006) |
| US3 (UCB + Query) | 2 (T007–T008) |
| US4 (Visualizations) | 2 (T009–T010) |
| Setup + Polish | 2 (T001, T011) |
| Parallelizable [P] tasks | 4 (T003, T004, T009, T010) |

**Independent test criteria per story:**

| Story | Independent Test |
|-------|----------------|
| US1 | Cell W7-02 alone: 17 samples printed, input range ✓, class balance shown |
| US2 | Cells W7-03 → W7-06: hyperparameter table rendered, Stage 1 accuracy printed, Stage 2 R² printed |
| US3 | Cells W7-07 + W7-10: min distance ≥ 0.05 printed, submission string `X.XXXXXX-X.XXXXXX` printed |
| US4 | Cells W7-08 + W7-09: both figures render with correct titles, colorbars, and proposed point |
