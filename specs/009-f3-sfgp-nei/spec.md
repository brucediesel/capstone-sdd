# Feature Specification: F3 Week 7 – SFGP with Matérn-5/2 ARD and NEI Acquisition

**Feature Branch**: `009-f3-sfgp-nei`  
**Created**: 2026-02-23  
**Status**: Draft  
**Spec Directory**: `specs/009-f3-sfgp-nei`

## Overview

Function 3 (`f3.ipynb`) requires a new Week 7 section that loads the updated Week 7 dataset, fits a Single-Fidelity Gaussian Process (SFGP) surrogate with a fully specified configuration, runs one Bayesian Optimisation iteration using the Noisy Expected Improvement (NEI) acquisition function, and reports the proposed next sampling point together with surrogate and convergence visualisations.

The intent is to continue the black-box optimisation challenge submission cycle: receive updated results, retrain the surrogate, and propose the next query point in the required submission format.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 – Week 7 Data Loaded and Validated (Priority: P1)

A student opens `f3.ipynb`, runs the Week 7 section, and sees the updated inputs and outputs loaded from `data/f3/updated_inputs - Week 7.npy` and `data/f3/updated_outputs - Week 7.npy`. The loaded data is displayed in a table and all values are confirmed to lie in the valid input range [0.0, 1.0].

**Why this priority**: Without correct data loading the entire section is invalid. This is the foundation for all subsequent steps.

**Independent Test**: Can be tested by executing only the data-loading cell. The cell displays a table of inputs and outputs with no errors and confirms all inputs are in [0.0, 1.0].

**Acceptance Scenarios**:

1. **Given** the Week 7 `.npy` files exist in `data/f3/`, **When** the data-loading cell is run, **Then** a table displays all inputs (3 columns, one per dimension) and the scalar output for each sample, with no NaN or out-of-range values.
2. **Given** an input value outside [0.0, 1.0] exists in the file, **When** the validation step runs, **Then** a clear warning message is printed identifying the offending row.
3. **Given** the data files are missing, **When** the cell is run, **Then** a clear `FileNotFoundError` message is shown identifying the missing file path.

---

### User Story 2 – SFGP Surrogate Trained with Specified Hyperparameters (Priority: P1)

A student runs the surrogate-fitting cell and sees the SFGP trained with the exact configuration listed below. The cell prints a summary of the fitted hyperparameters (lengthscales for each dimension, signal variance, noise variance) so the student can verify the model learned meaningful values.

**Why this priority**: The surrogate model is the core of this submission; its configuration must be explicit, documented, and reproducible.

**Independent Test**: Execute the surrogate training cell alone. The cell completes without error and prints the fitted hyperparameter values clearly labelled.

**Acceptance Scenarios**:

1. **Given** the Week 7 training data is loaded, **When** the SFGP training cell is run with 10–20 random restarts, **Then** the fitted model converges and the cell prints lengthscales ℓ_A, ℓ_B, ℓ_C, signal variance, and noise variance.
2. **Given** the inputs are scaled to [0, 1] and outputs are z-score standardised before training, **When** the model is evaluated, **Then** predictions are returned in the original output scale.
3. **Given** the Matérn-5/2 kernel with ARD is used, **When** the cell runs, **Then** three separate lengthscales (one per input dimension) are reported, each labelled with its corresponding input dimension.

---

### User Story 3 – NEI Acquisition Proposes Next Sample Point (Priority: P1)

A student runs the optimisation cell and receives a single proposed next query point formatted according to the challenge submission format (`x1-x2-x3`, each beginning with 0 and specified to six decimal places).

**Why this priority**: Producing the next submission point is the deliverable of every weekly cycle. Without this the challenge entry cannot be submitted.

**Independent Test**: Execute the acquisition cell independently (after the surrogate is trained). A single formatted query string is printed.

**Acceptance Scenarios**:

1. **Given** the trained SFGP surrogate, **When** the NEI optimisation cell runs, **Then** one candidate point is returned as a formatted string `0.xxxxxx-0.yyyyyy-0.zzzzzz`.
2. **Given** the proposed point is generated, **When** it is displayed, **Then** each component starts with `0` and is expressed to exactly 6 decimal places, with all components in [0.000000, 0.999999].
3. **Given** inputs have been normalised for model training, **When** the acquisition function proposes a point, **Then** the point is de-normalised back to the original [0, 1] scale before display.

---

### User Story 4 – Surrogate and Convergence Visualisations (Priority: P2)

A student runs the visualisation cells and sees:
1. Pairwise 2D surrogate slice plots (one per input-dimension pair) showing the predicted mean and uncertainty across the input space.
2. A convergence plot showing the best observed output value across all accumulated weekly submissions up to and including Week 7.

**Why this priority**: Visualisations are required by the challenge and are the primary means of communicating model behaviour and progress to assessors.

**Independent Test**: Execute the visualisation cells after training. Both plots render without error and are clearly labelled.

**Acceptance Scenarios**:

1. **Given** the trained surrogate, **When** the surrogate visualisation cell runs, **Then** at least one plot is shown with labelled axes, a title referencing "F3 – Week 7 Surrogate", and a colour bar or legend distinguishing predicted mean from uncertainty.
2. **Given** all accumulated data from initial samples through Week 7, **When** the convergence plot cell runs, **Then** a line or step chart is displayed showing the best-observed output value at each sample, with the x-axis labelled "Sample Number" and y-axis labelled "Best Observed Output".
3. **Given** the proposed next point, **When** the surrogate plot is shown, **Then** the proposed point is marked distinctly (e.g., a star or cross marker) on the plot.

---

### Edge Cases

- What if the Week 7 `.npy` files contain a different number of samples than expected? The cell must print the actual sample count and proceed, clearly noting the count.
- What if all random restarts converge to the same local optimum? The fitted hyperparameter summary displays the single result; no error is raised.
- What if NEI optimisation returns a point on the boundary of the input space? It is accepted as valid provided all components are in [0.000000, 0.999999].
- What if the noise variance collapses to near zero during training? The jitter term ensures numerical stability; the fitted noise value is reported as-is.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A new section titled **"Week 7"** MUST be added to `functions/f3/f3.ipynb` as a new markdown heading followed by new code cells. Existing cells MUST NOT be modified.
- **FR-002**: The Week 7 section MUST load inputs from `data/f3/updated_inputs - Week 7.npy` and outputs from `data/f3/updated_outputs - Week 7.npy`.
- **FR-003**: Loaded inputs MUST be validated to lie in [0.0, 1.0]; any out-of-range value MUST trigger a printed warning identifying the offending row.
- **FR-004**: Inputs MUST be scaled to [0, 1] before training; outputs MUST be z-score standardised (subtract mean, divide by std). Predictions MUST be converted back to the original output scale before display and submission.
- **FR-005**: The surrogate MUST be a Single-Fidelity Gaussian Process with the following explicit configuration:
  - **Mean function**: Constant (value learned during training)
  - **Kernel**: Matérn-5/2 with Automatic Relevance Determination (ARD) — one lengthscale per input dimension (ℓ_A, ℓ_B, ℓ_C)
  - **Likelihood**: Gaussian noise by default; a markdown note MUST document the option to switch to Student-t likelihood if heavy-tail outliers are observed
  - **Training**: Maximise marginal log-likelihood with 10–20 random restarts
- **FR-006**: A markdown cell MUST precede the model definition cell explaining each hyperparameter, its role, and the rationale for its starting value.
- **FR-007**: After training, the cell MUST print the fitted values of ℓ_A, ℓ_B, ℓ_C, signal variance σ²_f, and noise variance σ²_n, each clearly labelled.
- **FR-008**: The acquisition function MUST be Noisy Expected Improvement (NEI).
- **FR-009**: One Bayesian Optimisation iteration MUST be executed, optimising the NEI acquisition over the input space [0, 0.999999]³, and the resulting candidate MUST be printed in the challenge submission format: `0.xxxxxx-0.yyyyyy-0.zzzzzz`.
- **FR-010**: The proposed candidate MUST satisfy: each component starts with `0`, is in [0.000000, 0.999999], and is expressed to exactly 6 decimal places.
- **FR-011**: A surrogate visualisation MUST be produced using pairwise 2D contour or heatmap slices (A vs B, A vs C, B vs C) with the remaining dimension held at its current best-observed value. Each plot MUST show predicted mean and overlaid uncertainty (e.g., ±2σ contours or shaded bands).
- **FR-012**: The proposed next sampling point MUST be marked on the surrogate visualisation (e.g., a star or cross marker).
- **FR-013**: A convergence plot MUST show the running best observed output across all cumulative samples (initial dataset + updated weekly data through Week 7), with x-axis labelled "Sample Number" and y-axis labelled "Best Observed Output".
- **FR-014**: All code cells MUST include concise inline comments explaining each step in plain language accessible to a non-expert reader.
- **FR-015**: No existing cells in `f3.ipynb` MUST be modified by this change; all additions are appended as new cells only.

### Key Entities

- **SFGP Model**: A Gaussian Process with constant mean, Matérn-5/2 ARD kernel, and Gaussian noise likelihood. Hyperparameters documented in a dedicated markdown cell.
- **Training Dataset**: Union of all F3 samples (initial + Weeks 3–7). Inputs scaled to [0, 1]; outputs z-score standardised.
- **NEI Acquisition Function**: Noisy Expected Improvement evaluated over the [0, 0.999999]³ input space to select the most promising next query point.
- **Proposal**: The single candidate output by NEI, formatted as `0.xxxxxx-0.yyyyyy-0.zzzzzz` for direct challenge submission.
- **Surrogate Visualisation**: A set of pairwise 2D slice contour plots showing surrogate mean and uncertainty, with the proposed point marked.
- **Convergence Plot**: A chart of best-observed output vs cumulative sample index spanning initial data through Week 7.

---

## Assumptions

- F3 has a 3-dimensional input space; lengthscales ℓ_A, ℓ_B, ℓ_C refer to dimensions 1, 2, and 3 respectively.
- The challenge evaluates the maximum of the black-box function; convergence tracking uses the running maximum of observed outputs.
- All existing notebook cells from previous weeks remain intact and unmodified.
- Week 7 data files exist at the expected paths when the notebook is executed.
- The Student-t likelihood is documented as an option but Gaussian noise is implemented as the default.
- Lengthscale initialisations around 0.2–0.3 (inputs scaled to [0, 1]) are appropriate starting values for a 3-dimensional function.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The Week 7 section runs end-to-end without error when all new cells are executed top-to-bottom in a fresh kernel.
- **SC-002**: Three distinct lengthscale values (one per input dimension) are printed after training, confirming ARD is active.
- **SC-003**: The proposed next query point is printed in the exact format `0.xxxxxx-0.yyyyyy-0.zzzzzz` with all components in [0.000000, 0.999999].
- **SC-004**: At least three surrogate visualisation plots are produced (one per input-dimension pair), each with labelled axes, a title referencing "F3 Week 7", and the proposed point marked.
- **SC-005**: A convergence plot is produced covering all samples from the initial dataset through Week 7, with no gaps in the series.
- **SC-006**: Zero existing cells in `f3.ipynb` are modified; the notebook before and after the change is identical except for the new Week 7 cells appended at the end.
- **SC-007**: Each hyperparameter is described in a markdown cell before the model code, with its purpose and initial value rationale documented.
