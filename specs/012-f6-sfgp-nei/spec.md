# Feature Specification: F6 Week 7 — SFGP Matérn-1.5 + NEI (Exploration Focus)

**Feature Branch**: `012-f6-sfgp-nei`
**Created**: 2026-02-24
**Status**: Draft
**Input**: User description: "Update F6 for week 7 - Process outputs from week 6 - perform bayesian optimisation using SFGP with parameters: SF: matern_1.5, standardise, noise>=1e-08 - acquisition function: NEI, q=4, Optimiser=3000 Sobol starts → best 50 → L-BFGS, encourage exploration - use same visualization as week 6"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Add Week 7 SFGP+NEI Section to f6 Notebook (Priority: P1) 🎯 MVP

A student reviewing the capstone notebook opens `functions/f6/f6.ipynb` and navigates to a new **Week 7** section. The section loads the cumulative Week 7 data (27 samples, 5D), trains a Single-Fidelity Gaussian Process (SFGP) with a Matérn-1.5 kernel on the standardised outputs, applies batch Noisy Expected Improvement (qLogNEI, q=4) with an exploration-focused selection strategy, and proposes the next sample point formatted for submission.

**Why this priority**: Delivering the Week 7 submission query is the primary deliverable. Without this, no new sample can be submitted.

**Independent Test**: Run only the Week 7 section cells top-to-bottom; they load Week 7 data independently and produce a submission-format query without depending on earlier section state (Weeks 5–6).

**Acceptance Scenarios**:

1. **Given** the Week 7 f6 data files are present, **When** the Week 7 cells are executed top-to-bottom, **Then** the SFGP is trained, 4 candidates are proposed via NEI, and a single next sample point is printed in `x1-x2-x3-x4-x5` format with 6 decimal places clamped to `[0.0, 0.999999]`.
1. **Given** the SFGP is trained with Matérn-1.5, ARD, and noise lower bound 1e-2, **When** the model is fitted, **Then** the model reports five distinct lengthscale values (one per input dimension), confirming ARD is active.
3. **Given** 4 candidates are produced by NEI, **When** distance-based selection is applied, **Then** the candidate with the highest minimum distance to existing data (among those with predicted mean above the batch median) is selected as the submission point.

---

### User Story 2 — Visualise Surrogate and Convergence (Priority: P2)

The student examines diagnostic plots in the Week 7 section matching the Week 6 layout: a 2D slice of the GP posterior mean over the two most important dimensions, a 2D slice of the posterior uncertainty, a feature importance bar chart, and a convergence plot.

**Why this priority**: Visualization is required for submission assessment and demonstrates model behaviour, but does not block the query calculation.

**Independent Test**: Visualization cells can be run once the model is fitted; plots render independently of the submission formatting step.

**Acceptance Scenarios**:

1. **Given** the SFGP is trained, **When** the 3-panel visualization cell runs, **Then** three side-by-side panels are displayed: (a) posterior mean contour over the top-2 important dimensions with all observed points overlaid and proposed next point marked, (b) posterior uncertainty contour, (c) dimension relevance bar chart (1/ℓ, normalised, 5 bars for x0–x4).
2. **Given** all 27 observations, **When** the convergence cell runs, **Then** a running-maximum plot is produced with an indicator line at the Week 6→7 boundary (sample 26.5), matching Week 6 style.

---

### User Story 3 — Explicit Hyperparameter Documentation (Priority: P3)

The notebook section displays all SFGP and NEI hyperparameters with plain-English justifications before training begins, so an examiner can verify the configuration without reading source code.

**Why this priority**: Transparency of hyperparameters is required by the constitution; however it does not affect the numerical result.

**Independent Test**: The hyperparameter markdown cell can be read independently and will show all parameter names, values, and rationale.

**Acceptance Scenarios**:

1. **Given** the hyperparameter markdown cell is present, **When** it is rendered, **Then** it shows a table with: kernel type (Matérn-1.5), ARD (True, 5 lengthscales), noise lower bound (1e-2), standardisation (Standardize(m=1)), acquisition type (qLogNEI), q (4), optimiser settings (3000 Sobol → 50 L-BFGS), and selection strategy (distance-based, exploration focus).

---

### Edge Cases

- What happens if Week 7 data files are missing? → File-not-found error raised at load time with the expected path shown.
- What if all 4 NEI candidates have predicted means below the median? → Fallback: select the candidate with the highest predicted mean among all 4.
- What if multiple candidates tie on distance? → Select the one with the higher predicted mean as tiebreaker.
- What if GP fitting produces numerical instability? → The noise lower bound of 1e-2 and BoTorch's default `Standardize(m=1)` prevent singular covariance matrices.
- What if a candidate exceeds [0, 1]? → All candidates are clamped to [0.0, 0.999999] before evaluation and submission.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The section MUST load Week 7 cumulative data from `data/f6/updated_inputs - Week 7.npy` (shape 27×5) and `data/f6/updated_outputs - Week 7.npy` (shape 27,) using relative path `../../data/f6/`.
- **FR-002**: The section MUST validate that the loaded data has shape (27, 5) for inputs and (27,) for outputs, that all input values lie in [0, 1], and MUST print the sample count, input range, output range, and best observed value with its index.
- **FR-003**: The surrogate MUST be a **SingleTaskGP** with a **Matérn ν=1.5 kernel** (via `MaternKernel(nu=1.5, ard_num_dims=5)`), **ARD lengthscales** (one per input dimension), wrapped in a **ScaleKernel**, with a **GaussianLikelihood** whose noise is constrained to ≥ **1e-2** (via `GreaterThan(1e-2)`). The aggressive noise floor prevents exact interpolation and ensures sufficient posterior uncertainty across the input space, promoting exploration over exploitation.
- **FR-004**: The GP MUST use BoTorch's default **`Standardize(m=1)` outcome transform** to z-score standardise outputs, since the output dynamic range (12.5x) is moderate and all values are negative — no manual log transform is needed.
- **FR-005**: The GP MUST be trained via **maximum marginal log-likelihood** using `fit_gpytorch_mll`, with **15 random restarts** (each with a different `torch.manual_seed`). The restart with the lowest negative log-likelihood MUST be selected as the best model via `copy.deepcopy`.
- **FR-006**: Lengthscale initialisation MUST be set to **0.5** to promote broader initial uncertainty for exploration of the 5D space.
- **FR-007**: Output scale MUST be initialised to **1.0**.
- **FR-008**: Noise MUST be initialised to **0.2** (fixed value) to give the optimiser a starting point well above the 1e-2 floor, preventing exact interpolation of the 27 observations.
- **FR-009**: All SFGP hyperparameters MUST be documented in a markdown cell with plain-English rationale before training, per the constitution.
- **FR-010**: The acquisition function MUST be **`qLogNoisyExpectedImprovement`** (log-space formulation for numerical stability), configured with `X_baseline=X_train`, `sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]))`, and `prune_baseline=True`.
- **FR-011**: The acquisition function MUST be optimised with **q=4** candidates, **num_restarts=50**, **raw_samples=3000**, over feasibility-constrained bounds: **x4 (milk) ∈ [0.10, 1.0]**, **all other dims ∈ [0.01, 1.0]**. This prevents the GP from exploiting the x4=0 boundary trap and ensures all ingredient proportions remain physically feasible.
- **FR-012**: All 4 candidates MUST be clamped to **[0.0, 0.999999]** after optimisation, before any further processing.
- **FR-013**: Candidate selection MUST use a **distance-based exploration strategy**: compute the posterior mean for each candidate, compute the minimum Euclidean distance from each candidate to X_train (via `torch.cdist`), filter to candidates with predicted mean ≥ median of the 4 means, then select the candidate with the maximum minimum distance to training data among those filtered.
- **FR-014**: The section MUST produce a **3-panel surrogate visualization** matching Week 6 layout: (a) 2D slice of posterior mean over the top-2 important dimensions (shortest ARD lengthscales), observed points in red, proposed point marked distinctly, with colourbar; (b) 2D slice of posterior uncertainty (GP posterior std replaces MC Dropout uncertainty); (c) feature importance bar chart using **1/ℓ normalised** (5 bars, one per dimension).
- **FR-015**: Fixed dimensions in the 2D slice MUST be set to the **selected best point's values**.
- **FR-016**: The section MUST produce a **convergence plot** (running maximum across all 27 observations) with a vertical dashed line at x=26.5 marking the Week 6→7 boundary, matching Week 6 style.
- **FR-017**: The proposed next sample point MUST be printed in submission format `x1-x2-x3-x4-x5` with 6 decimal places, each value in `[0.000000, 0.999999]`, followed by validation assertions.
- **FR-018**: A summary MUST print: surrogate type, kernel, acquisition function, q, selection strategy, fitted lengthscales, fitted output scale, fitted noise, and the selected candidate's posterior mean.
- **FR-019**: The Week 7 section MUST be appended as new cells at the end of `functions/f6/f6.ipynb`; **no existing cells MUST be modified or removed**.
- **FR-020**: The section header MUST explain the **strategy change** from Week 6 (NN + MC Dropout + UCB κ=0.5) to Week 7 (SFGP + NEI q=4, exploration focus), with a comparison table.

### Key Entities

- **SFGP Configuration**: Kernel = Matérn-1.5, ARD = True (5 lengthscales), noise lower bound = 1e-2, outcome transform = Standardize(m=1), lengthscale init = 0.5, noise init = 0.2, output scale init = 1.0.
- **Training Data (Week 7)**: 27×5 input array and 27-element output array; all outputs are negative (range [-2.571, -0.206]); all inputs in [0, 1].
- **NEI Acquisition**: qLogNoisyExpectedImprovement with q=4, 512 Sobol samples, prune_baseline=True; optimised with 3000 raw samples → 50 L-BFGS restarts.
- **Candidate Selection**: Distance-based — from the 4 candidates, filter to those with mean ≥ median, pick the farthest from existing observations. Promotes exploration of under-sampled regions while maintaining quality.
- **Proposed Sample Point**: A 5-element array clamped to [0.0, 0.999999], formatted as `x1-x2-x3-x4-x5` with 6 decimal places.
- **Visualisation Grid**: 80×80 evaluation grid over the top-2 important dimensions; other 3 dimensions fixed at best-point values.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Running the Week 7 section cells top-to-bottom completes without runtime errors, producing one 3-panel visualisation and one convergence plot.
- **SC-002**: The SFGP fits all 27 data points and reports five distinct lengthscale values — one per input dimension — confirming ARD is active and each dimension is weighted independently.
- **SC-003**: A valid next sample point is produced in the correct submission format (`x1-x2-x3-x4-x5`, 6 decimal places, all values in `[0.000000, 0.999999]`).
- **SC-004**: The fitted noise level is at or above the specified lower bound of 1e-2, confirming the noise constraint is enforced and the GP is not exactly interpolating observations.
- **SC-005**: The proposed next point differs from all 27 previously observed points (no duplicate submission).
- **SC-006**: No existing cells in `f6.ipynb` are altered — only new cells are appended at the end of the notebook.
- **SC-007**: The 3-panel visualization renders with correct axis labels, colorbars, and annotations matching the Week 6 layout style.
- **SC-008**: The convergence plot shows a running maximum that is monotonically non-decreasing, with the Week 6→7 boundary at sample 26.5.

---

## Assumptions

- Week 7 data files (`updated_inputs - Week 7.npy`, `updated_outputs - Week 7.npy`) already exist in `data/f6/`, generated by the results processing notebook.
- The optimisation domain uses **feasibility-constrained bounds**: x4 (milk) ∈ [0.10, 1.0], all other dims ∈ [0.01, 1.0]. Submission values are clamped to [0.0, 0.999999]. The tighter bounds prevent the GP from driving milk to zero (a physically infeasible recipe) and avoid exact-boundary proposals on other dimensions.
- "Same visualization as week 6" means: 3-panel subplot (contour for mean, contour for uncertainty, bar chart for dimension importance) plus a separate convergence plot. The NN-specific panels (MC Dropout mean/uncertainty) are replaced by GP posterior equivalents; the feature importance bar chart uses inverse lengthscales (1/ℓ) instead of gradient magnitude, which serves the same interpretive purpose.
- Matérn ν=1.5 (once-differentiable) is chosen over ν=2.5 (twice-differentiable) per the user's explicit specification, suggesting the function may have rougher behaviour than assumed by smoother kernels.
- "Encourage exploration" is implemented through: (a) the distance-based candidate selection strategy that favours points far from existing data, (b) initialising lengthscales at 0.5 to promote broad uncertainty, (c) batch q=4 to generate diverse candidates, (d) an aggressive noise floor of 1e-2 that prevents exact interpolation and ensures posterior uncertainty everywhere, and (e) feasibility-constrained bounds (x4 ≥ 0.10, others ≥ 0.01) that prevent boundary-trap exploitation.
- The BoTorch `Standardize(m=1)` outcome transform handles z-score standardisation automatically — no manual log or z-score transform is needed since the output dynamic range is only 12.5x.
- The NN surrogate and related variables from Weeks 5–6 are not available in the Week 7 section's execution context; all imports and data loading are re-done in Week 7 cells.
- Per the constitution, 15 random restarts of MLL fitting is a reasonable default for 5D with 27 data points, matching the approach used in other function specifications.

---

## Clarifications

### Session 2026-02-24

- Q: What minimum feasible bounds should be applied to prevent x4 (milk) = 0 proposals? → A: x4 ≥ 0.10, all other dims ≥ 0.01
- Q: What noise floor and noise initialisation should be used to prevent exact interpolation? → A: Noise floor = 1e-2, noise init = 0.2 (most aggressive option to maximise posterior uncertainty and exploration)
