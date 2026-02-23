# Feature Specification: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Feature Branch**: `010-f4-mfgp-nei`  
**Created**: 2026-02-23  
**Status**: Draft  
**Input**: User description: "update f4 for week 7 — process outputs from week 6 — perform bayesian optimisation using MFGP with nu=2.5, LinTrunc, standardise, noise>=1e-4 — acquisition function: Cost-aware MF-qNEI (q=4), fantasies=64 — visualise the surrogate functions and convergence performance"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load and Validate Week 7 Data (Priority: P1)

As a student, I want to load the cumulative Week 7 data for Function 4 so that I have all 37 observations (30 initial + 7 weekly) available for model training.

**Why this priority**: Without correct data loading, no subsequent modelling or optimisation is possible.

**Independent Test**: Run the data-loading cell and confirm the shape, value ranges, and sample count match expectations.

**Acceptance Scenarios**:

1. **Given** the file `data/f4/updated_inputs - Week 7.npy` exists, **When** the cell runs, **Then** a tensor of shape (37, 4) is loaded with all values in [0, 1].
2. **Given** the file `data/f4/updated_outputs - Week 7.npy` exists, **When** the cell runs, **Then** a 1-D array of 37 output values is loaded and the best observed value and its index are printed.
3. **Given** any value falls outside [0, 1] for inputs, **When** loading completes, **Then** a warning is printed identifying the out-of-range samples.

---

### User Story 2 — Train MFGP Surrogate (Priority: P1)

As a student, I want to fit a Multi-Fidelity Gaussian Process (Matérn-5/2 ARD, LinearTruncatedFidelityKernel, z-score standardised outputs, noise floor ≥ 1e-4) so that I have a calibrated surrogate model of the F4 objective landscape.

**Why this priority**: The surrogate is the foundation for acquisition; if it doesn't train correctly, proposals will be meaningless.

**Independent Test**: Run the training cell. Confirm that the fitted hyperparameters (4 ARD lengthscales, signal variance, noise variance) are printed, the noise variance is ≥ 1e-4, and no numerical errors occur.

**Acceptance Scenarios**:

1. **Given** 37 samples are loaded and z-score standardised, **When** the MFGP is fitted via marginal log-likelihood with multiple random restarts, **Then** the best model's negative MLL, lengthscales (ℓ₁–ℓ₄), signal variance, and noise variance are printed.
2. **Given** the noise constraint is set to ≥ 1e-4, **When** the model is fitted, **Then** the reported noise variance is ≥ 1e-4.
3. **Given** multiple random restarts are performed, **When** training completes, **Then** the restart with the lowest negative MLL is selected and its parameters are used.

---

### User Story 3 — Propose Next Sample via MF-qNEI (Priority: P1)

As a student, I want to use Cost-aware Multi-Fidelity qNEI (q=4, fantasies=64) to propose the next sampling point(s) so that I can submit an optimised query for the challenge.

**Why this priority**: The acquisition proposal is the deliverable output — the submission query for the weekly challenge.

**Independent Test**: Run the acquisition cell. Confirm it outputs 4 candidate points in [0, 0.999999]⁴ and identifies the best one for submission.

**Acceptance Scenarios**:

1. **Given** a fitted MFGP model, **When** MF-qNEI is optimised with q=4 and 64 fantasies, **Then** 4 candidate points are returned, each with 4 coordinates in [0, 0.999999].
2. **Given** the 4 candidates are returned, **When** the acquisition values are computed, **Then** the candidate with the highest posterior mean is identified as the primary submission point.
3. **Given** the primary submission point, **When** formatted, **Then** a dash-separated string of 4 values each with 6 decimal places (e.g., `0.123456-0.654321-0.111111-0.222222`) is produced.

---

### User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

As a student, I want to visualise the MFGP surrogate surface (2D slices of the 4D space) and the convergence of the objective across all weeks so that I can assess model quality and optimisation progress.

**Why this priority**: Visualisations are required by the constitution but are not blocking the submission query.

**Independent Test**: Run the visualisation cells and confirm that the surrogate plot and convergence plot render without errors.

**Acceptance Scenarios**:

1. **Given** a fitted MFGP, **When** the surrogate plot cell runs, **Then** a multi-panel figure shows predicted mean and posterior uncertainty as 2D contour slices through the two most important dimensions, with observed points and the proposed point marked.
2. **Given** 37 cumulative observations, **When** the convergence plot cell runs, **Then** a line chart of running-best output values is displayed with a vertical boundary line separating initial data from weekly submissions.
3. **Given** both plots, **When** examined, **Then** axis labels, titles, colorbars, and legends are present and readable.

---

### Edge Cases

- What happens when the MLL optimisation produces NaN gradients? The training loop catches and skips that restart, continuing with the remaining restarts.
- What happens when all 4 candidates from q=4 are nearly identical? The submission cell still selects and formats the best one; the others are printed for reference.
- How does the system handle Week 7 data with fewer or more than 37 samples? A validation check prints the actual count and warns if it differs from the expected 37.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST add a new "Week 7" section after the existing Week 6 cells without modifying any existing cells.
- **FR-002**: The data loading cell MUST read `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy` from `data/f4/`.
- **FR-003**: The data loading cell MUST validate that inputs are shape (N, 4), outputs are shape (N,), and all input values are in [0, 1].
- **FR-004**: The data loading cell MUST print the number of samples, best observed value, and the sample index at which it occurs.
- **FR-005**: Outputs MUST be z-score standardised (subtract mean, divide by standard deviation) before model fitting.
- **FR-006**: The surrogate model MUST be a Multi-Fidelity GP with Matérn-5/2 kernel (nu=2.5), ARD lengthscales for all 4 input dimensions, and LinearTruncatedFidelityKernel.
- **FR-007**: A constant fidelity column (all 1.0) MUST be appended to the input tensor before passing to the MFGP.
- **FR-008**: The noise model MUST enforce a noise floor of 1e-4.
- **FR-009**: The MLL optimisation MUST use multiple random restarts (10–20) and select the model with the lowest negative MLL.
- **FR-010**: All fitted hyperparameters (ℓ₁–ℓ₄, signal variance, noise variance, fidelity kernel power) MUST be printed after training.
- **FR-011**: The acquisition function MUST be a multi-fidelity noisy expected improvement variant configured with q=4 and 64 MC fantasies.
- **FR-012**: Acquisition optimisation MUST use Sobol-initialised candidates followed by multi-start optimisation, with bounds [0, 0.999999]⁴ for the input dimensions and fidelity fixed at 1.0.
- **FR-013**: The primary submission point MUST be formatted as a dash-separated string with 6 decimal places per coordinate.
- **FR-014**: The surrogate visualisation MUST show at least two panels: predicted mean and posterior standard deviation, as 2D contour slices through the most important pair of dimensions.
- **FR-015**: The convergence plot MUST show running-best observed values with a vertical line separating the initial 30 samples from the weekly submissions.
- **FR-016**: The submission summary MUST print the surrogate type, acquisition function, fitted hyperparameters, and the formatted submission query.

### Key Entities

- **Week 7 Cumulative Data**: 37 observations (30 initial + 7 weekly) with 4D inputs and scalar outputs representing the warehouse placement objective.
- **MFGP Surrogate**: Multi-Fidelity GP with Matérn-5/2 ARD spatial kernel + LinearTruncatedFidelityKernel, trained via MLL with random restarts.
- **Acquisition Batch**: 4 candidate points (q=4) from MF-qNEI with 64 fantasies; the highest-valued candidate is the primary submission.
- **Submission Query**: Dash-separated string of 4 coordinates, each 6 decimal places, in [0, 0.999999].

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The data loading cell reports exactly 37 samples with all inputs in [0, 1] and no errors.
- **SC-002**: The MFGP trains successfully with noise variance ≥ 1e-4 and all 4 ARD lengthscales, signal variance, and fidelity power reported.
- **SC-003**: The MF-qNEI acquisition returns 4 candidate points, each with 4 coordinates in [0, 0.999999].
- **SC-004**: A valid submission query string is produced in the format `0.XXXXXX-0.YYYYYY-0.ZZZZZZ-0.WWWWWW`.
- **SC-005**: The surrogate visualisation renders a multi-panel figure with predicted mean and uncertainty contour plots without rendering errors.
- **SC-006**: The convergence plot renders with a clear running-best line and a week boundary marker.
- **SC-007**: No existing cells in the F4 notebook are modified — all changes are appended as new cells.

## Assumptions

- **A-001**: The `.npy` files are cumulative — `updated_inputs - Week 7.npy` contains all 37 samples (30 initial + 7 weekly), not just the Week 7 increment.
- **A-002**: Although the MFGP expects multi-fidelity data, all F4 observations are at a single (high) fidelity. A constant fidelity column of 1.0 is appended to leverage the LinearTruncatedFidelityKernel's inductive bias as a regularisation mechanism (consistent with the approach validated in the F4 prequential evaluation).
- **A-003**: The "Cost-aware" aspect of MF-qNEI is implemented by fixing the fidelity target to 1.0 during acquisition optimisation, effectively requesting only high-fidelity evaluations.
- **A-004**: Standard notebook performance expectations apply — cells complete within minutes, not hours.
- **A-005**: The BoTorch/GPyTorch versions installed in the environment support the multi-fidelity GP with `linear_truncated=True`.
- **A-006**: The z-score standardisation is applied manually (subtract mean, divide by std) for explicitness and consistency with the project's approach.
- **A-007**: With q=4 candidates returned, the best single candidate (by posterior mean) serves as the primary submission; the remaining 3 are informational.
