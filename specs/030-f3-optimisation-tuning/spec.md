# Feature Specification: F3 Week 10 — Optimisation Tuning

**Feature Branch**: `030-f3-optimisation-tuning`  
**Created**: 2026-03-12  
**Status**: Draft  
**Input**: User description: "F3 optimisation tuning: increase batch size from q=1 to q=3, increase raw samples from 512 to 2048, review noise floor (noise_lb=1e-6 may be too tight), change output transform from Standardize(m=1) to shift transform (y - y_min), increase MLL restarts from 20 to 40."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

As a student working on the F3 black box optimisation challenge, I need to add a new optimisation section to the existing F3 week 10 notebook that applies the approved tuning changes (q=3 batch, 2048 raw samples, relaxed noise floor, shift transform, 40 MLL restarts) and proposes the next sample point for submission.

**Why this priority**: The challenge requires a submission each week. Without a proposed sample point, no submission can be made.

**Independent Test**: Open the F3 week 10 notebook, run all cells including the new optimisation section, and verify a properly formatted next sample point is produced.

**Acceptance Scenarios**:

1. **Given** the existing F3 week 10 notebook with 12 cells (review + evaluation), **When** the new optimisation cells are appended, **Then** the notebook runs top-to-bottom without errors and the existing review cells remain unchanged.
2. **Given** the SFGP is fitted with Matérn-2.5 ARD and the shift transform (y - y_min), **When** I run the optimisation cell, **Then** the GP fits successfully and the printed hyperparameters are valid values (lengthscales finite and positive, noise ≥ 1e-4, outputscale > 0).
3. **Given** q=3 candidates are proposed by qLogNEI, **When** distance-based selection is applied, **Then** the single best candidate is selected and formatted as `x1-x2-x3` with 6 decimal places, each value in [0.0, 0.999999].

---

### User Story 2 — Visualise Surrogate and Acquisition Surface (Priority: P2)

As a student, I need to see the GP posterior and acquisition landscape to understand where the model thinks the optimum might be. Since F3 is 3D, I need pair-wise 2D contour slices to interpret the surfaces.

**Why this priority**: Visualisation supports learning objectives and the constitution requires it. Without it, the surrogate behaviour cannot be verified.

**Independent Test**: After running all cells, verify contour panels render showing posterior mean and acquisition surface with colour-coded data point overlays.

**Acceptance Scenarios**:

1. **Given** the GP is fitted, **When** the visualisation cell is executed, **Then** 2D contour slice plots are displayed showing GP posterior mean and qLogNEI acquisition surface for each input dimension pair.
2. **Given** the contour plots are rendered, **When** I examine the point overlays, **Then** initial samples are shown in blue, submission samples in orange, and the proposed next point as a green star.

---

### User Story 3 — Display Convergence with Proposed Point (Priority: P3)

As a student, I need an updated convergence plot that includes the proposed next point to see how it fits into the overall trajectory.

**Why this priority**: Convergence context helps verify the proposal makes sense relative to the optimisation history.

**Independent Test**: Verify the convergence plot shows all existing data points plus the proposed point marked distinctly.

**Acceptance Scenarios**:

1. **Given** the optimisation produces a proposed next point, **When** the convergence plot cell is executed, **Then** existing data and the proposed point (marked with a green star) are displayed with the running best trajectory.

---

### Edge Cases

- What happens when all q=3 candidates cluster in the same region? → The distance-based selection step picks the one farthest from training data, but if all three are close, the result is still the best available candidate.
- What happens when the shift transform produces all-zero shifted outputs? → This would only happen if all outputs are identical, which is impossible with 25 diverse samples.
- What happens when the GP fit fails to converge during MLL restarts? → Each restart is independent; the best (lowest loss) model is kept. With 40 restarts, degenerate fits are unlikely.
- What happens when the proposed point is a duplicate of an existing sample? → A duplicate check prints a warning; the point is still valid for submission but noted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: New cells MUST be appended to the existing `functions/f3/f3 - week 10.ipynb` notebook after the current Step 5 (Proposed Strategy Improvements). Existing cells (1–12) MUST NOT be modified.
- **FR-002**: The notebook MUST include all necessary library imports and hyperparameter constants in a dedicated configuration cell before any optimisation logic.
- **FR-003**: The surrogate MUST be a single-fidelity Gaussian Process (SFGP) with a Matérn-2.5 ARD kernel (one lengthscale per input dimension, d=3).
- **FR-004**: The output transform MUST be a shift transform that maps all outputs to non-negative values by subtracting the minimum observed output: `Y_shifted = Y - Y.min()`. This replaces the Standardize(m=1) transform used in week 9.
- **FR-005**: The noise lower bound MUST be relaxed from 1e-6 to 1e-4 to prevent overfitting on individual observations in the negative-valued landscape.
- **FR-006**: The GP MUST be fitted using at least 40 MLL restarts (increased from week 9's 15–20 restarts) to better explore the multi-modal likelihood surface of the 3D problem. The model with the lowest negative log-likelihood is retained.
- **FR-007**: All hyperparameters MUST be declared as named constants at the top of the configuration cell with clear comments explaining their purpose and chosen values, including justification for changes from week 9.
- **FR-008**: The acquisition function MUST be q-Log Noisy Expected Improvement (qLogNEI) with q=3 (increased from week 9's q=1) to propose 3 candidates per batch, improving coverage of the 3D space.
- **FR-009**: The acquisition optimisation MUST use at least 2,048 Sobol-generated raw samples (increased from week 9's 512) and at least 20 multi-start restarts to ensure broad search coverage across the [0,1]³ domain.
- **FR-010**: From the q=3 candidates, a single submission point MUST be selected using distance-based filtering: keep candidates with posterior mean ≥ median of the batch, then select the one with maximum minimum-distance to all existing training points.
- **FR-011**: The selected point MUST be formatted as `x1-x2-x3` with 6 decimal places, each value clamped to [0.0, 0.999999]. A duplicate check against all existing samples MUST be performed.
- **FR-012**: Visualisation MUST show 2D contour slices (fixing one dimension at the proposed point's coordinate) of the GP posterior mean and acquisition surface for each of the 3 input dimension pairs, with initial samples (blue), submissions (orange), and proposed point (green star) overlaid.
- **FR-013**: An updated convergence plot MUST be displayed showing the running best with the proposed next point marked distinctly (green star).

### Key Entities

- **Optimisation config**: Named hyperparameter constants (KERNEL_NU=2.5, ARD_NUM_DIMS=3, NOISE_LB=1e-4, N_MLL_RESTARTS=40, MC_SAMPLES=512, Q_BATCH=3, NUM_RESTARTS=20, RAW_SAMPLES=2048, GRID_RES=50) declared with explanatory comments per FR-007, including justification for week 9 → week 10 changes.
- **SFGP Model**: Single-fidelity Gaussian Process with Matérn-2.5 ARD kernel (3 lengthscales) and shift transform on outputs, trained via multi-restart MLL optimisation.
- **Acquisition Candidates**: q=3 candidates from `optimize_acqf`, reduced to 1 via distance-based selection.
- **Submission Point**: Single 3D point formatted as `x1-x2-x3`, 6 decimal places, in [0.0, 0.999999].

## Assumptions

- The existing F3 week 10 notebook (cells 1–12) is already complete with data loading, convergence plot, pair plots, evaluation, and improvement suggestions. Data variables (`inputs`, `outputs`, `n_total`, `N_INITIAL`, `N_DIMS`, etc.) are available from earlier cells.
- Per the project constitution, BoTorch/GPyTorch is the mandated GP library, installed in the `sdd-dev` environment.
- F3 outputs are all negative (approximately [-0.399, -0.031]). The shift transform maps them to approximately [0, 0.368], giving the GP a non-negative target surface. The GP operates in shifted space; predictions are reverse-shifted for display and convergence.
- All inputs are in [0, 1] and the GP can model the 3D input space directly.
- The notebook follows the constitution convention: new sections are appended, existing cells are never modified.
- F3 has N_DIMS=3, N_INITIAL=15, and 10 weekly submissions (weeks 3–10 inclusive, except week 4 missing), totalling 25 samples.
- The shift transform requires storing `y_min` to reverse-transform predictions back to original scale for visualisation and convergence plots.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes without errors from top to bottom, including all new optimisation cells.
- **SC-002**: The GP fits successfully with valid hyperparameters (lengthscales finite and positive, noise ≥ 1e-4, outputscale > 0) after multi-restart MLL optimisation.
- **SC-003**: The proposed next sample point is in the valid range [0.0, 0.999999] for all three dimensions. A duplicate check against all existing samples is performed and the result reported.
- **SC-004**: Visualisation contour panels render correctly with GP posterior mean and acquisition surface slices, showing correct point overlays (blue initial, orange submissions, green star proposed).
- **SC-005**: The convergence plot includes the proposed point marked distinctly, with the predicted value reverse-shifted to original scale.
- **SC-006**: All hyperparameters are explicitly documented with values and justifications for changes from week 9 visible in the notebook.
