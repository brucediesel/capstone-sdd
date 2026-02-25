# Feature Specification: F8 Week 7 — SFGP + qEI Acquisition

**Feature Branch**: `018-f8-sfgp-qei`  
**Created**: 2025-07-17  
**Status**: Draft  
**Input**: User description: "process week 7 outputs for F8 - Use a SFGP surrogate function with hyperparameters: matern_2.5, standardise, noise>=1e-07 - Use qEI acquisition with following parameters: MC samples: 256 (512 if stable). Fantasization: enabled for pending points. xi: 0.01. - evaluate the problem and data and select appropriate other appropriate hyperparameters - Provide the same visualisations as previous weeks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Core SFGP Surrogate + qEI Recommendation (Priority: P1)

A researcher loads the 47 cumulative Week 7 observations for Function 8 (8-dimensional, all-positive outputs in [5.59, 9.95]) and fits a Single-task GP with a Matérn 2.5 kernel, ARD lengthscales, output standardisation, and minimum noise of 1e-07. The system then computes quasi-Expected Improvement (qEI) using 256 MC samples (or 512 if numerically stable), with fantasisation enabled for pending points and an improvement threshold (xi) of 0.01, to propose the next candidate point. The candidate is formatted as an 8-coordinate submission string clamped to [0, 1].

**Why this priority**: Without a fitted surrogate and acquisition output, there is no Week 7 submission — this is the core deliverable.

**Independent Test**: Run the data-load, GP-fitting, and qEI cells end-to-end; verify a valid 8D submission query is printed.

**Acceptance Scenarios**:

1. **Given** 47 samples (8D inputs, 1D outputs) loaded from `data/f8/updated_inputs - Week 7.npy` and `data/f8/updated_outputs - Week 7.npy`, **When** the SFGP is fitted with Matérn 2.5 + ARD + standardise + noise >= 1e-07, **Then** training completes without error and the GP reports positive lengthscales for all 8 dimensions.
2. **Given** a trained SFGP and current best y = 9.953025, **When** qEI is maximised with 256+ MC samples, fantasisation enabled, xi = 0.01, and bounds [0, 1]^8, **Then** the system outputs a candidate point with all coordinates in [0, 1] and prints a formatted submission string.
3. **Given** the recommended point, **When** validated, **Then** the submission string has exactly 8 dash-separated values, each with 6 decimal places, each starting with 0.

---

### User Story 2 — Diagnostic Visualisations (Priority: P2)

The researcher generates the same set of visualisations used in Weeks 5-6: a feature-importance chart (GP lengthscale-based for Week 7), a 3-panel 2D surrogate slice (mean, uncertainty, qEI) through the top-2 most important dimensions, and a convergence plot showing the running best with weekly sample boundaries.

**Why this priority**: Visualisations provide interpretability and confidence in the recommendation, but the submission can proceed without them.

**Independent Test**: Execute the visualisation cells after the surrogate is fitted; verify three separate plots render without error.

**Acceptance Scenarios**:

1. **Given** a fitted SFGP, **When** ARD lengthscales are extracted, **Then** a bar chart showing relative feature importance for all 8 dimensions is displayed, with shorter lengthscale = higher importance.
2. **Given** the top-2 most important dimensions, **When** a 2D slice grid is evaluated through the best observed point, **Then** a 3-panel figure (GP mean, GP posterior std, qEI surface) renders with labelled axes.
3. **Given** 47 cumulative observations, **When** the convergence plot is drawn, **Then** it shows a running-best curve with vertical lines marking weekly sample boundaries (initial 40, Week 5 at 45, Week 6 at 46, Week 7 at 47).

---

### User Story 3 — Model Documentation & Diagnostics (Priority: P2)

The researcher inspects printed diagnostics confirming data shape, output range, GP training completion, lengthscale values, best observed value, and qEI acquisition value to verify the pipeline operated correctly.

**Why this priority**: Print diagnostics support debugging and reproducibility without affecting the core recommendation.

**Independent Test**: Check that every code cell prints summary statistics matching the loaded data.

**Acceptance Scenarios**:

1. **Given** Week 7 data is loaded, **When** the data-load cell executes, **Then** it prints sample count (47), dimensionality (8), output range, and current best value.
2. **Given** the GP is trained, **When** diagnostics are printed, **Then** they include the 8 ARD lengthscales and confirmation of successful fitting.
3. **Given** qEI optimisation completes, **When** the result cell runs, **Then** it prints the acquisition value, candidate coordinates, and formatted submission string.

---

### Edge Cases

- **All qEI values are zero**: If the GP is overconfident and predicts no improvement over best_f = 9.953, the system should detect this condition, log a warning, and fall back to selecting the candidate with the highest posterior mean as a pure exploitation choice.
- **GP fitting failure / non-convergence** *(informational)*: GP on 47×8 with Matern 2.5 is expected to converge reliably. If issues arise, increase noise floor from 1e-07 to 1e-04 in cell 52. No try/except is implemented to keep code simple per constitution.
- **Candidate violates bounds**: If `optimize_acqf` returns any coordinate outside [0, 1], all coordinates must be clamped to [0, 1] before formatting the submission.
- **Numerical instability with 512 MC samples** *(informational)*: Implementation uses 256 MC samples. The 512 option is deferred as 256 is sufficient for q=1 single-candidate optimisation.
- **Duplicate candidate** *(informational)*: If the proposed point is within Euclidean distance < 1e-6 of any existing observation, this indicates convergence. No detection logic is implemented — the GP's posterior naturally discourages exact duplicates via noise modelling.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load `data/f8/updated_inputs - Week 7.npy` (shape 47 x 8) and `data/f8/updated_outputs - Week 7.npy` (shape 47,) and print data summary including sample count, dimensionality, output range, and best observed value.
- **FR-002**: System MUST fit a BoTorch `SingleTaskGP` with Matern 2.5 kernel, ARD lengthscales (one per input dimension), output standardisation enabled, and Gaussian likelihood noise lower-bounded at 1e-07.
- **FR-003**: System MUST optimise the GP hyperparameters by maximising the exact marginal log-likelihood using `fit_gpytorch_mll`.
- **FR-004**: System MUST compute qEI (`qExpectedImprovement`) with `best_f = y_max + xi` where `xi = 0.01`, using 256 MC samples via `SobolQMCNormalSampler`. The 512 option is deferred — 256 is the implementation default.
- **FR-005**: System MUST enable fantasisation for any pending evaluation points when computing qEI.
- **FR-006**: System MUST optimise the qEI acquisition function over bounds [0, 1]^8 using `optimize_acqf` with `num_restarts=30` and `raw_samples=4096`.
- **FR-007**: System MUST clamp the proposed candidate to [0, 1] in every dimension and format it as an 8-value dash-separated string with 6 decimal places.
- **FR-008**: System MUST display a feature importance chart based on the GP ARD lengthscales for all 8 input dimensions.
- **FR-009**: System MUST produce a 3-panel 2D slice visualisation (GP posterior mean, GP posterior standard deviation, analytic EI acquisition surface) through the top-2 most important dimensions, fixing remaining dimensions at the best observed point. Analytic EI is used for the 2500-point grid for performance; qEI is used only for actual candidate selection.
- **FR-010**: System MUST generate a convergence plot showing running best across all 47 observations with vertical markers at weekly sample boundaries (40, 45, 46, 47).
- **FR-011**: System MUST detect the case where all qEI values are zero and fall back to selecting the candidate with the highest GP posterior mean.
- **FR-012**: System MUST append all new cells after existing Week 6 content (cell 49 onwards) without modifying any prior cells.

### Key Entities

- **Observation Set**: 47 evaluated 8D input-output pairs accumulated across Weeks 1-6, loaded from `.npy` files in `data/f8/`.
- **SFGP Surrogate**: A BoTorch `SingleTaskGP` with Matern 2.5 + ARD kernel, trained on standardised outputs and bounded noise — the probabilistic model of the objective surface.
- **qEI Acquisition**: Quasi-Expected Improvement function computed via MC sampling over the GP posterior, used to select the next evaluation point that maximises expected gain above `best_f + xi`.
- **Submission Query**: The formatted 8-coordinate string representing the next proposed point for the Week 7 evaluation oracle.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The SFGP trains to completion on 47 x 8 data without errors and reports 8 positive ARD lengthscales.
- **SC-002**: The qEI acquisition function returns a candidate point with all 8 coordinates in [0, 1] and a non-negative acquisition value.
- **SC-003**: The formatted submission string contains exactly 8 dash-separated decimal values, each with 6 decimal places.
- **SC-004**: The feature importance chart renders and clearly distinguishes the relative influence of each of the 8 input dimensions.
- **SC-005**: The 3-panel surrogate visualisation renders correctly with labelled axes corresponding to the top-2 most important dimensions.
- **SC-006**: The convergence plot displays a monotonically non-decreasing running-best curve spanning all 47 observations with weekly boundary markers.

## Assumptions

- Week 7 data files (`updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy`) exist and follow the same format as prior weeks.
- All 47 outputs are positive (observed range [5.59, 9.95]), so qEI with `best_f = y_max` is well-defined.
- The input domain is [0, 1]^8 as established in prior weeks.
- BoTorch and GPyTorch are available in the runtime environment (already used in the initial submission cells).
- The notebook follows the existing weekly-section convention: a markdown header announces Week 7, followed by numbered code cells for data load, GP fit, acquisition, visualisation, convergence, and submission.
- 30 restarts and 4096 raw samples (matching the initial submission's 8D settings) are sufficient for `optimize_acqf` in 8D; these may be adjusted if qEI optimisation is slow.
- Fantasisation refers to BoTorch's built-in mechanism for handling pending points; since there are no confirmed pending points in this workflow, fantasisation is enabled but may not materially affect the result.
