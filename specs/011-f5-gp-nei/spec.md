# Feature Specification: F5 Week 7 — GP Matérn-5/2 + NEI

**Feature Branch**: `011-f5-gp-nei`
**Created**: 2026-02-23
**Status**: Draft
**Input**: User description: "update f5 for week 7 - process outputs from week 6 - perform bayesian optimisation using following surrogate: GP with Matern52, log, noise=0.03·Var, ls=0.25 - acquisition function: NEI, q=2, ξ=0.01, Optimiser=3000 Sobol starts → best 50 → L‑BFGS - use same visualization as week 6"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load & Validate Week 7 Cumulative Data (Priority: P1)

As a data scientist, I need to load the Week 7 cumulative data (27 samples × 4 dimensions) for F5 and validate its integrity so that the surrogate model trains on the correct dataset.

**Why this priority**: Without valid data, no modelling can proceed. This is the foundation for all downstream tasks.

**Independent Test**: Run the data-loading cell and confirm 27 samples in [0,1]⁴ with outputs matching expected ranges.

**Acceptance Scenarios**:

1. **Given** Week 7 data files exist in `data/f5/`, **When** the data-loading cell executes, **Then** inputs have shape (27, 4), outputs have shape (27,), all inputs in [0, 1], and the best observed value and its index are printed.
2. **Given** the loaded data, **When** summary statistics are displayed, **Then** the output range, sample count, and best-observed point are clearly reported.

---

### User Story 2 — Train GP Surrogate with Specified Hyperparameters (Priority: P1)

As a data scientist, I need to fit a Gaussian Process with Matérn-5/2 kernel, log-transformed outputs, noise initialised at 0.03·Var(y), and lengthscales initialised at 0.25, so that I have a well-calibrated surrogate for acquisition.

**Why this priority**: The surrogate model is the core of the BO loop. Its configuration directly determines proposal quality.

**Independent Test**: Run the training cell and confirm the model fits without error, fitted hyperparameters are printed, and the negative marginal log-likelihood converges.

**Acceptance Scenarios**:

1. **Given** 27 validated samples, **When** the GP is trained with Matérn-5/2 kernel (ARD), log-transformed outputs, noise=0.03·Var(y), ls=0.25, and MLL optimisation with multiple restarts, **Then** the fitted lengthscales, output scale, and noise variance are printed.
2. **Given** the trained GP, **When** hyperparameters are reported, **Then** all lengthscales, signal variance, and noise variance are displayed with 6 decimal places.

---

### User Story 3 — Propose Next Samples via NEI Acquisition (Priority: P1)

As a data scientist, I need to run the Noisy Expected Improvement acquisition function with q=2 candidates, ξ=0.01, optimised via 3000 Sobol starts → best 50 → L-BFGS, to propose two candidate points for the next experiment.

**Why this priority**: Producing the submission query is the deliverable of each weekly iteration.

**Independent Test**: Run the acquisition cell and confirm 2 candidate points are returned in [0,1]⁴ with their posterior means reported.

**Acceptance Scenarios**:

1. **Given** a trained GP surrogate, **When** NEI is optimised with q=2 and 3000 Sobol initialisations narrowed to 50 L-BFGS restarts, **Then** 2 candidate points in [0,1]⁴ are returned.
2. **Given** 2 candidate points, **When** posterior means are computed, **Then** the best candidate (highest posterior mean) is selected for submission and its coordinates are displayed.
3. **Given** the best candidate, **When** formatted as a submission query, **Then** the output follows the `x1-x2-x3-x4` format with 6 decimal places, all values in [0, 0.999999].

---

### User Story 4 — Visualise Surrogate & Convergence (Priority: P2)

As a data scientist, I need the same visualisation as Week 6 (2D surrogate slice + convergence plot) so that I can assess model quality and optimisation progress.

**Why this priority**: Visualisation supports model interpretation and progress assessment but does not block submission.

**Independent Test**: Run the visualisation cells and confirm plots render with correct axes and labels.

**Acceptance Scenarios**:

1. **Given** a trained GP, **When** the surrogate visualisation cell executes, **Then** a multi-panel figure is rendered showing predicted mean and uncertainty over the two most important dimensions, with observed points and the proposed point overlaid.
2. **Given** 27 cumulative observations, **When** the convergence plot cell executes, **Then** a running-best plot is displayed with a vertical line marking the Week 6→7 boundary.

---

### Edge Cases

- What happens if the log transform encounters zero or negative outputs? The log1p transform handles values near zero safely.
- What happens if MLL optimisation fails to converge? Multiple random restarts (10–20) mitigate this; the best restart is selected.
- What happens if all q=2 candidates collapse to the same point? The NEI formulation with ξ=0.01 provides exploration pressure to diversify candidates.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy` from `data/f5/` and validate shapes (27, 4) and (27,).
- **FR-002**: System MUST validate all input values are in [0, 1].
- **FR-003**: System MUST print sample count, input range, output range, and best observed value with its index.
- **FR-004**: System MUST apply a log transformation to outputs before fitting the GP (log1p for numerical safety).
- **FR-005**: System MUST standardise inputs to [0, 1] (already in range per data validation).
- **FR-006**: System MUST construct a GP with Matérn-5/2 kernel with ARD (4 lengthscale parameters).
- **FR-007**: System MUST initialise lengthscales at 0.25.
- **FR-008**: System MUST initialise noise variance at 0.03 × Var(y_transformed).
- **FR-009**: System MUST optimise the marginal log-likelihood with multiple random restarts (10–20) using L-BFGS-B.
- **FR-010**: System MUST print all fitted hyperparameters: lengthscales (ℓ₁–ℓ₄), signal variance (σ²_f), noise variance (σ²_n).
- **FR-011**: System MUST use qNoisyExpectedImprovement (NEI) as the acquisition function with q=2.
- **FR-012**: System MUST set the NEI exploration parameter ξ = 0.01 (via eta or prune_baseline adjustment).
- **FR-013**: System MUST optimise the acquisition function using 3000 Sobol initial points, selecting the best 50 for L-BFGS refinement.
- **FR-014**: System MUST report all q=2 candidate points with their posterior means and select the best for submission.
- **FR-015**: System MUST format the submission query as `x1-x2-x3-x4` with 6 decimal places, all values in [0, 0.999999] (values at 1.0 MUST be clamped to 0.999999).
- **FR-016**: System MUST validate the submission query format (4 dimensions, 6 decimal places, values in [0, 0.999999]).
- **FR-017**: System MUST produce a surrogate visualisation with the same structure as Week 6: 2D slice over the two most important dimensions showing predicted mean and uncertainty, with observed and proposed points overlaid.
- **FR-018**: System MUST produce a convergence plot showing running best across all observations with a vertical line at the Week 6→7 boundary (sample 26.5).
- **FR-019**: System MUST document all hyperparameters in a markdown cell before the training code, including the rationale for the switch from GBT to GP.
- **FR-020**: System MUST append all new cells after the existing Week 6 content without modifying any prior cells.

### Key Entities

- **Week 7 Data**: 27 cumulative observations (26 from weeks 1–6 + 1 new from week 6 submission). Inputs: (27, 4) in [0,1]. Outputs: (27,) representing chemical yield.
- **GP Surrogate**: Matérn-5/2 kernel with ARD, log-transformed outputs, Gaussian likelihood.
- **Hyperparameter Set**: lengthscales ℓ₁–ℓ₄ (init 0.25), signal variance σ²_f (init Var(y)), noise variance σ²_n (init 0.03·Var(y)), jitter 1e-6.
- **Acquisition Result**: q=2 candidate points from NEI, ranked by posterior mean.
- **Submission Query**: Single best candidate formatted as `x1-x2-x3-x4`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The data-loading cell successfully loads 27 samples and all validation checks pass.
- **SC-002**: The GP trains without errors and the negative MLL converges across restarts.
- **SC-003**: All 4 fitted lengthscales, signal variance, and noise variance are reported with at least 4 significant figures.
- **SC-004**: The NEI acquisition returns exactly 2 candidate points, each with 4 coordinates in [0, 1].
- **SC-005**: The submission query matches the format `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` (each value in [0.000000, 0.999999]) and passes validation.
- **SC-006**: The surrogate visualisation renders a multi-panel figure consistent with Week 6's layout (mean surface, uncertainty surface, and dimension relevance).
- **SC-007**: The convergence plot shows a non-decreasing running best with the Week 6→7 boundary correctly marked at sample 26.5.
- **SC-008**: All new content is appended as new cells after cell 50 (`#VSC-8f8ac8b4`), preserving all existing notebook content.

## Assumptions

1. **Data availability**: Week 7 cumulative data files exist at the expected paths and contain valid numpy arrays.
2. **Output transformation**: `log1p` is used rather than raw `log` to handle any outputs near zero safely. The GP fits the transformed outputs; predictions are inverse-transformed (`expm1`) for reporting and visualisation.
3. **Lengthscale initialisation**: The value 0.25 applies to all 4 ARD lengthscales uniformly as a starting point; the optimiser adjusts them during MLL fitting.
4. **Noise initialisation**: 0.03·Var(y) refers to the variance of the log-transformed outputs, ensuring the noise prior is calibrated to the modelling space.
5. **NEI ξ parameter**: In BoTorch, NEI does not have a direct ξ parameter. The exploration-exploitation balance is achieved via the `prune_baseline` mechanism and the inherent noise-aware baseline. The ξ=0.01 intent is captured by using standard NEI defaults which provide slight exploration bias.
6. **Week 6 visualisation style**: The Week 6 visualisation used 3 panels (mean, std, feature importance). For the GP surrogate, feature importance is replaced with lengthscale-based dimension relevance (shorter lengthscale = more important dimension). The 2D slice approach and convergence plot layout are preserved.
7. **Existing cells**: The notebook has 50 cells. All new Week 7 content is appended starting at cell 51.

## Clarifications

### Session 2026-02-23

- Q: What is the valid range for submission query values? → A: [0.000000, 0.999999] — values at 1.0 must be clamped to 0.999999. This applies to all function notebooks (f1–f8).
