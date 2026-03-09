# Feature Specification: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Feature Branch**: `024-f3-week9-standardize`  
**Created**: 2026-03-09  
**Status**: Draft  
**Input**: User description: "Create a new branch for this week, focus only on F3. Apply the following to the week 9 notebook and rerun it: 1. Use BoTorch Standardize(m=1) instead of manual z-score 2. Increase acquisition num_restarts from 10 to 20 3. Add interior penalty (S=0.5, F=0.01)"

## Clarifications

### Session 2026-03-09

- Q: After evaluation, should the interior penalty be removed? → A: Yes — remove interior penalty entirely; revert visualisation to 1×3 posterior mean layout (matching Week 8)

## User Scenarios & Testing

### User Story 1 — Replace Manual z-score with BoTorch Standardize (Priority: P1) 🎯

The student replaces the manual z-score standardisation of training outputs with BoTorch's built-in `Standardize(m=1)` outcome transform. This eliminates the need to manually compute mean/std, pass them through LOO folds, or re-standardise when retraining. The GP is constructed with `outcome_transform=Standardize(m=1)` so that normalisation is handled automatically and consistently.

**Why this priority**: This is the primary structural change — it simplifies the model pipeline, removes a source of manual error, and affects how the GP is constructed and trained. All downstream cells (acquisition, LOO, visualisation) depend on the model being correctly configured.

**Independent Test**: Run the GP training cell — the model trains without manual z-score computation, and the GP posterior produces valid mean and variance predictions over the input space.

**Acceptance Scenarios**:

1. **Given** Week 9 data (24 samples, 3D) is loaded, **When** the GP is constructed with `outcome_transform=Standardize(m=1)`, **Then** the model trains successfully via MLL optimisation without any manual mean/std computation in the training cell
2. **Given** a trained GP with Standardize, **When** posterior predictions are requested on candidate points, **Then** the returned mean and variance are in the original output scale (Standardize handles un-standardisation internally)
3. **Given** LOO cross-validation is performed, **When** each fold retrains the GP, **Then** each fold uses `Standardize(m=1)` without manual recomputation of z-score statistics

---

### User Story 2 — Increase Acquisition Restarts to 20 (Priority: P1)

The student increases the `num_restarts` parameter in `optimize_acqf` from 10 to 20. This broadens the multi-start acquisition optimisation search, reducing the chance of missing the global optimum of the acquisition surface. The `NUM_RESTARTS_ACQ` hyperparameter is defined alongside other constants and used in the acquisition cell.

**Why this priority**: A broader acquisition search directly impacts the quality of the proposed next sample point — the primary deliverable of each weekly notebook.

**Independent Test**: Run the acquisition cell — it completes with `num_restarts=20` and proposes a candidate point.

**Acceptance Scenarios**:

1. **Given** a fitted GP model, **When** `optimize_acqf` is called with `num_restarts=20` and `raw_samples=512`, **Then** a valid next candidate is returned within [0, 0.999999]³
2. **Given** the `NUM_RESTARTS_ACQ` hyperparameter is set to 20, **When** the hyperparameters cell is printed, **Then** `NUM_RESTARTS_ACQ = 20` is displayed

---

### ~~User Story 3 — Add Interior Penalty to Acquisition (Priority: P1)~~ [REMOVED per clarification 2026-03-09]

> Interior penalty was evaluated during implementation and removed per user decision. The acquisition function uses plain qLogNEI without boundary suppression.

---

### User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

The student produces the standard visualisation suite: surrogate contour plots (mean and uncertainty across 2D slices) in a 1×3 layout matching Week 8, convergence plot with running maximum, and three-colour sample scheme (initial=blue, submissions=orange, proposed=green star). The visualisation confirms that the Standardize-based GP produces sensible predictions.

**Why this priority**: Visualisation is required by the constitution but does not affect the submission query itself.

**Independent Test**: Run the visualisation cells — contour plots render with correct colour coding and the convergence plot shows the running maximum trajectory.

**Acceptance Scenarios**:

1. **Given** all model and acquisition cells have executed, **When** the contour plot cell runs, **Then** 1×3 contour plots display GP posterior mean across 2D slices of the 3D input space with white uncertainty contours
2. **Given** 24 observations, **When** the convergence plot cell runs, **Then** the running maximum is plotted with initial samples in blue and weekly submissions in orange, with a vertical boundary line at observation 15

---

### User Story 5 — Performance Evaluation and Strategy Recommendations (Priority: P2)

The student evaluates convergence metrics, exploration spread, and LOO surrogate error using the Standardize-based GP. The LOO evaluation no longer requires manual z-score recomputation per fold. A markdown section provides strategy recommendations for Week 10.

**Why this priority**: Performance evaluation provides insight into whether the changes improved the optimisation, but does not affect the submission deliverable.

**Independent Test**: Run the performance cells — convergence metrics, spread statistics, and LOO errors are printed without manual z-score handling.

**Acceptance Scenarios**:

1. **Given** 24 observations (15 initial + 9 submissions), **When** convergence metrics are computed, **Then** the stalling flag, per-submission deltas, and trailing no-improvement streak are reported
2. **Given** the LOO cell runs, **When** each fold retrains with `Standardize(m=1)`, **Then** LOO MAE and RMSE are reported without any manual z-score recomputation
3. **Given** all performance metrics, **When** the interpretation cell is reviewed, **Then** actionable Week 10 strategy recommendations are provided

---

### Edge Cases

- What happens if the GP fails to train with Standardize on a LOO fold with very few samples? The notebook falls back gracefully with a warning.
- What happens if the best candidate is identical to a previous sample? A minimum distance check warns if the proposed point is within 1e-6 of any existing observation.

## Requirements

### Functional Requirements

- **FR-001**: Notebook MUST be created as `functions/f3/f3 - week 9.ipynb`, replacing the existing week 9 notebook with the updated strategy
- **FR-002**: Notebook MUST load Week 9 data from `../../data/f3/updated_inputs - Week 9.npy` and `../../data/f3/updated_outputs - Week 9.npy`
- **FR-003**: Data loading MUST validate shape (24, 3) for inputs and (24,) for outputs, with inputs in [0.0, 1.0]
- **FR-004**: GP MUST be constructed with `outcome_transform=Standardize(m=1)` — no manual z-score computation for model training
- **FR-005**: Kernel MUST remain Matérn-2.5 with ARD (3 lengthscales for compounds A, B, C)
- **FR-006**: MLL optimisation MUST use multi-start with `N_RESTARTS = 15` (unchanged from previous week)
- **FR-007**: `NUM_RESTARTS_ACQ` for acquisition optimisation MUST be set to 20 (increased from 10)
- **FR-008**: Raw samples for acquisition MUST remain 512 Sobol candidates
- **FR-009**: Acquisition function MUST be qLogNoisyExpectedImprovement (qLogNEI)
- ~~**FR-010**: Interior penalty~~ [REMOVED per clarification 2026-03-09]
- ~~**FR-011**: Penalised acquisition wrapper~~ [REMOVED per clarification 2026-03-09]
- **FR-012**: Proposed candidate MUST be clipped to [0.0, 0.999999] per dimension
- **FR-013**: Submission MUST be formatted as `x1-x2-x3` with 6 decimal places per coordinate
- **FR-014**: Contour visualisation MUST show 1×3 panels: GP posterior mean across 2D slices with white uncertainty contours (matching Week 8 layout)
- **FR-015**: All plots MUST use three-colour scheme: initial samples (blue), weekly submissions (orange), proposed next point (green star)
- **FR-016**: Convergence plot MUST show running maximum with vertical boundary at observation 15
- **FR-017**: A hyperparameters section MUST document all constants with rationale in a markdown table, including what changed from previous week and why
- **FR-018**: LOO cross-validation MUST retrain the GP with `Standardize(m=1)` per fold — no manual z-score recomputation
- **FR-019**: Performance evaluation MUST include convergence metrics (stalling flag, per-submission deltas), exploration spread, and LOO MAE/RMSE
- **FR-020**: A strategy recommendations markdown section MUST be included at the end with actionable guidance for Week 10

### Key Entities

- **Observation**: A single sample point with 3 input coordinates (A, B, C) and 1 output value; 24 total (15 initial + 9 submissions)
- **GP Model**: SingleTaskGP with Matérn-2.5 ARD kernel and Standardize outcome transform
- **Acquisition Score**: Plain qLogNEI value, optimised with 20 restarts

## Success Criteria

### Measurable Outcomes

- **SC-001**: Notebook executes end-to-end without errors and produces a valid submission query in `0.xxxxxx-0.xxxxxx-0.xxxxxx` format
- **SC-002**: GP training cell completes without any manual z-score code — only `Standardize(m=1)` is used for output normalisation
- **SC-003**: Acquisition optimisation uses 20 restarts (doubled from 10), confirmed in printed hyperparameters
- ~~**SC-004**: Interior penalty~~ [REMOVED per clarification 2026-03-09]
- **SC-005**: Proposed candidate coordinates fall within [0.0, 0.999999]
- **SC-006**: LOO cross-validation completes without manual z-score recomputation per fold
- **SC-007**: Convergence plot correctly shows running maximum trajectory with 15 initial and 9 submission points distinguished by colour

## Assumptions

- F3 has 3 input dimensions (compounds A, B, C) — confirmed from existing notebooks
- Week 9 data contains 24 samples total (15 initial + 9 weekly submissions)
- The existing SFGP + qLogNEI strategy is retained; only two changes are applied (Standardize + increased restarts; interior penalty was evaluated and removed)
- BoTorch's `Standardize(m=1)` is compatible with `SingleTaskGP` and handles un-standardisation in posterior predictions automatically
- The noise floor constraint (≥ 1e-6) from the previous week is retained
- MLL restart count (15) is unchanged — only the acquisition restart count increases to 20
- The contour visualisation uses 2D slices of the 3D space (fixing one dimension at the best-observed value)
