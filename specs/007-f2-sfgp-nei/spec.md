# Feature Specification: F2 Week 7 — SFGP with NEI Acquisition

**Feature Branch**: `005-week7-pe-surrogates`
**Created**: 2026-02-22
**Status**: Draft
**Input**: User description: "remain on 005-week7-pe-surrogates branch, do not create a new branch - change F2 surrogate function to SFGP with parameters matern15,noise=1e-03,ard=True,norm=T - use NEI acquisition function balancing exploration with exploitation - create new section in f2 notebook - use same visualization as week 6 - process the outputs and propose a new sample point"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Add Week 7 SFGP+NEI Section to f2 Notebook (Priority: P1)

A student reviewing the capstone notebook opens `functions/f2/f2.ipynb` and navigates to a new **Week 7** section. The section trains a Single-Fidelity Gaussian Process (SFGP) on the 17 cumulative data points (10 initial + 7 weekly), applies Noisy Expected Improvement (NEI) to balance exploration with exploitation, and proposes the next sample point.

**Why this priority**: Delivering the Week 7 submission query is the primary deliverable. Without this, no new sample can be submitted.

**Independent Test**: Can be tested by running only the Week 7 section cells; they load Week 7 data independently, produce visualizations, and print a submission-format query without depending on earlier section state.

**Acceptance Scenarios**:

1. **Given** the Week 7 f2 data files are present, **When** the Week 7 section cells are executed top-to-bottom, **Then** a surrogate mean and uncertainty surface are plotted and a next sample point is printed in `x1-x2` format with 6 decimal places clamped to `[0.0, 1.0]`.
2. **Given** the SFGP is trained with the specified parameters (Matern-1.5 kernel, noise lower bound 1e-3, ARD, input normalisation), **When** the model is fitted, **Then** the model reports one lengthscale value per input dimension, confirming that each dimension is weighted independently.
3. **Given** the NEI acquisition function is evaluated, **When** any candidate falls outside `[0.0, 0.999999]`, **Then** that candidate is rejected and the proposed point is clamped within valid bounds.

---

### User Story 2 — Visualise Surrogate and Acquisition Surface (Priority: P2)

The student examines three diagnostic plots in the Week 7 section: the GP posterior mean over the 2D search space, the GP posterior uncertainty, and the NEI acquisition surface showing the region targeted for the next sample.

**Why this priority**: Visualization is required for submission assessment and demonstrates model behaviour, but does not block the query calculation.

**Independent Test**: Visualization cells can be run once the model is fitted; plots are produced independently of the submission formatting step.

**Acceptance Scenarios**:

1. **Given** the SFGP is trained, **When** the visualization cell runs, **Then** three side-by-side panels are displayed: (a) posterior mean contour with all observed points overlaid in red and proposed next point as a yellow star, (b) posterior uncertainty contour, (c) NEI acquisition surface contour showing the candidate region.
2. **Given** all 17 observations, **When** the convergence cell runs, **Then** a running-maximum plot is produced with an indicator line between the initial 10 samples and the 7 weekly submissions.

---

### User Story 3 — Explicit Hyperparameter Display (Priority: P3)

The notebook section prints all SFGP and NEI hyperparameters with plain-English justifications before training begins, so an examiner can verify the configuration without reading source code.

**Why this priority**: Transparency of hyperparameters is required by the constitution; however it does not affect the numerical result.

**Independent Test**: The hyperparameter cell can be executed alone and will print all parameter names, values, and rationale.

**Acceptance Scenarios**:

1. **Given** the hyperparameter cell runs, **When** output is displayed, **Then** it shows kernel type (Matern-1.5), noise lower bound (1e-3), ARD enabled (True), input normalisation enabled (True), acquisition type (NEI), and optimisation settings.

---

### Edge Cases

- What happens if Week 7 data files are missing? → File-not-found error raised at load time with the expected path shown clearly.
- What if all observed outputs are equal (zero variance)? → The noise lower bound and input normalisation prevent a singular covariance; the model still fits with noise dominating.
- What if the NEI optimiser returns a point on the boundary? → The point is clamped to `[0.0, 0.999999]` before submission formatting.
- What if the model fitting procedure does not converge on first attempt? → The fitting routine retries internally; a warning is displayed but training completes and the best available model is used.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The section MUST load Week 7 cumulative data from `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy`.
- **FR-002**: The surrogate MUST be a Single-Fidelity Gaussian Process configured with a **Matern-1.5 kernel**, **ARD lengthscales** (one per input dimension), **noise lower bound of 1e-3**, and **input normalisation enabled**.
- **FR-003**: All SFGP parameters MUST be defined as named constants with a plain-English justification printed before model training begins.
- **FR-004**: The acquisition function MUST be **Noisy Expected Improvement (NEI)**, balancing exploration with exploitation by accounting for observation noise when computing the expected improvement over the current best observed value.
- **FR-005**: The acquisition function MUST be optimised over the domain `[0.0, 1.0]²` using multi-start optimisation to avoid local traps in the acquisition landscape.
- **FR-006**: The section MUST produce a **three-panel surrogate visualization** matching the Week 6 style: (a) posterior mean contour with observed points and proposed next point overlaid, (b) posterior standard deviation contour, (c) NEI acquisition surface contour.
- **FR-007**: The section MUST produce a **convergence plot** (running maximum of observed output values across all 17 observations) with a vertical indicator separating initial 10 samples from the 7 weekly submissions, matching the Week 6 style.
- **FR-008**: The proposed next sample point MUST be printed in submission format `x1-x2` with 6 decimal places, each value clamped to `[0.000000, 1.000000]`.
- **FR-009**: The Week 7 section MUST be appended as new cells at the end of `functions/f2/f2.ipynb`; no existing cells MUST be modified or removed.
- **FR-010**: The section MUST NOT introduce a new Git branch; all work stays on `005-week7-pe-surrogates`.

### Key Entities

- **SFGP Configuration**: Kernel = Matern-1.5, ARD = True, noise lower bound = 1e-3, input normalisation = True — the four parameters that fully specify the surrogate for this feature.
- **Training Data (Week 7)**: 17 × 2 input array and 17-element output array loaded from `.npy` files; used with the surrogate's built-in normalisation handling output scaling.
- **NEI Acquisition**: Noisy Expected Improvement configured with the trained SFGP; optimised over the unit hypercube; outputs a single 2D candidate point.
- **Proposed Sample Point**: A 2-element array clamped to `[0.0, 1.0]`, formatted as `x1-x2` with 6 decimal places for submission.
- **Visualisation Grid**: 50×50 evaluation grid over `[0,1]²` used for contour plots of posterior mean, posterior std, and NEI acquisition surface.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Running the Week 7 section cells top-to-bottom completes without runtime errors, producing three visualisation panels and one convergence plot.
- **SC-002**: The SFGP fits all 17 data points and reports two distinct lengthscale values — one per input dimension — confirming that each dimension is weighted independently (ARD active).
- **SC-003**: A valid next sample point is produced in the correct submission format (`x1-x2`, 6 decimal places, both values in `[0.0, 1.0]`).
- **SC-004**: The fitted noise level is at or above the specified lower bound of 1e-3, confirming the noise constraint is enforced.
- **SC-005**: The proposed next point differs from all 17 previously observed points (no duplicate submission).
- **SC-006**: No existing cells in `f2.ipynb` are altered — only new cells are appended at the end of the notebook.

---

## Assumptions

- Week 7 data files (`updated_inputs - Week 7.npy`, `updated_outputs - Week 7.npy`) already exist in `data/f2/`, generated by the results processing notebook from the submission feedback.
- "Same visualization as week 6" means: three-panel subplot (viridis contour for mean, YlOrRd for uncertainty, plasma for acquisition surface) plus a separate convergence plot with a boundary marker. The third panel changes from Week 6's feature importance to the NEI acquisition surface, which is the natural GP equivalent.
- The domain is `[0.0, 1.0]²` in all dimensions per submission format requirements.
- "norm=T" in the user specification corresponds to input normalisation applied inside the GP model.
- NEI is implemented using BoTorch's `qNoisyExpectedImprovement` (or its log variant for numerical stability); `X_baseline` is set to the training inputs.
- Multi-start acquisition optimisation uses at least 10 restarts and 512 raw samples as reasonable defaults for a 2D domain with 17 training points.
