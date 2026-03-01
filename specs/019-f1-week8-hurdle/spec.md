# Feature Specification: F1 Week 8 — Hurdle Model Bayesian Optimisation Iteration

**Feature Branch**: `019-f1-week8-hurdle`  
**Created**: 2026-03-01  
**Status**: Draft  
**Input**: User description: "Process week 8 outputs for F1, keeping the same strategy as week 7, propose the next sample point. Create a new notebook for this iteration as per constitution."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load and Validate Week 8 Data (Priority: P1)

As a student running the weekly Bayesian Optimisation loop for Function 1 (2D radiation source detection), I want to load the Week 8 updated inputs and outputs so that all 18 evaluated sample points are available for the surrogate model.

**Why this priority**: Without current data, no optimisation can proceed. This is the foundation for the entire iteration.

**Independent Test**: Run the data-loading cells — the notebook should display 18 input rows (2 columns each) and 18 output values in tabular format, with no NaN or out-of-range values.

**Acceptance Scenarios**:

1. **Given** the files `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` exist in `./data/f1/`, **When** the notebook loads and displays the data, **Then** 18 input rows with 2 dimensions and 18 output values are shown.
2. **Given** the loaded data, **When** validation checks run, **Then** all input values are within [0.0, 1.0] and no outputs contain NaN.
3. **Given** the loaded data, **When** the current best observation is identified, **Then** its value and location are printed clearly.

---

### User Story 2 — Fit Hurdle Model Surrogate (Priority: P1)

As a student, I want to train the same two-stage hurdle model used in Week 7 on the Week 8 data so that the surrogate captures both the probability of a positive response and the magnitude of positive outputs.

**Why this priority**: The hurdle model is the chosen surrogate strategy; fitting it is prerequisite to acquisition and proposal.

**Independent Test**: After the surrogate cells execute, the classifier and regressor are fitted without errors and per-point predictions can be queried.

**Acceptance Scenarios**:

1. **Given** 18 data points (note: Week 8 data has all outputs ≤ 0, so fallback mode is the expected path — see scenario 3), **When** Stage 1 (classifier) is trained, **Then** calibrated probability predictions are available for all candidates.
2. **Given** at least 3 positive-output samples exist, **When** Stage 2 (regressor) is trained on log1p-transformed positive outputs, **Then** point predictions and uncertainty estimates are available.
3. **Given** fewer than 3 positive samples, **When** Stage 2 cannot be trained, **Then** the notebook falls back to pure exploration mode and documents this clearly.

---

### User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

As a student, I want the notebook to maximise the penalised weighted UCB acquisition function (with local penalization and interior penalty) to propose the next sample point for Week 9 submission.

**Why this priority**: Proposing the next sample point is the primary deliverable of each weekly iteration.

**Independent Test**: The notebook outputs a formatted submission string in `x1-x2` format with 6 decimal places, all values within [0.0, 0.999999].

**Acceptance Scenarios**:

1. **Given** the hurdle model is fitted, **When** the weighted UCB acquisition is evaluated over 20 000 random candidates, **Then** a best candidate is selected after applying local penalization and interior penalty.
2. **Given** the proposed point, **When** validation checks run, **Then** both coordinates are in [0.0, 0.999999] and the point is at least 0.05 Euclidean distance from all 18 existing observations.
3. **Given** the selected candidate, **When** the submission query is formatted, **Then** it is printed as `0.xxxxxx-0.xxxxxx`.

---

### User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

As a student submitting a capstone notebook, I want the same 3-panel contour plots and convergence plot used in Week 7 so that the examiner can assess the surrogate quality and optimisation progress.

**Why this priority**: Visualisation is a capstone marking requirement and essential for interpretability, but the proposal itself is the primary deliverable.

**Independent Test**: Four plots are generated: hurdle mean, hurdle uncertainty, penalised acquisition surface, and convergence. All are legible and correctly annotated.

**Acceptance Scenarios**:

1. **Given** the surrogate is fitted, **When** the 3-panel plot is generated, **Then** Panel 1 shows the hurdle mean $\hat{y} = p \cdot \text{expm1}(\mu)$, Panel 2 shows the hurdle uncertainty $p \cdot \sigma_{\text{RF}}$, Panel 3 shows the penalised acquisition surface with interior penalty applied.
2. **Given** the plots, **When** training points are overlaid, **Then** positive observations are shown in one colour and non-positive in another, with the proposed next point highlighted.
3. **Given** 18 historical observations, **When** the convergence plot is generated, **Then** the running maximum of observed values is shown across all 18 points, with a vertical marker at the initial/weekly boundary.

---

### Edge Cases

- What happens when Week 8 data files are missing? The notebook should fail early with a clear file-not-found message identifying the expected path.
- What happens when no positive outputs exist in the updated data? The fallback mode (pure exploration) is activated and documented.
- What happens when the proposed point is very close to an existing observation? Minimum distance validation (≥ 0.05) rejects it and selects the next-best candidate.
- What happens when all 20 000 candidates are within 0.05 of existing points? Increase the candidate pool size or relax the distance constraint, documenting the decision.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A new self-contained Jupyter notebook MUST be created at `./functions/f1/f1 - week 8.ipynb` following the constitution convention. The original `f1.ipynb` MUST NOT be modified.
- **FR-002**: The notebook MUST load `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` from `./data/f1/`.
- **FR-003**: The notebook MUST display all 18 input/output data points in tabular format, highlighting the current best observation.
- **FR-004**: The notebook MUST implement the same two-stage hurdle model as Week 7: Stage 1 — calibrated logistic classifier for P(y > 0); Stage 2 — random forest regressor on log1p(y) for positive outputs.
- **FR-004a**: When FALLBACK_MODE is active (n_positive < MIN_POSITIVE), Stage 2 MUST be skipped. The acquisition function MUST degrade to pure exploration with μ_cand = 0, σ_RF_cand = 1, producing scores driven only by local penalization and interior penalty.
- **FR-005**: The notebook MUST use the same hyperparameters as Week 7 unless explicitly changed:

  | Parameter        | Value   |
  |------------------|---------|
  | C_STAGE1         | 1.0     |
  | N_ESTIMATORS     | 100     |
  | MAX_DEPTH        | 3       |
  | KAPPA            | 3.0     |
  | PENALTY_RADIUS   | 0.15    |
  | N_CANDIDATES     | 20 000  |
  | GRID_RES         | 50      |
  | MIN_POSITIVE     | 3       |
  | STEEPNESS        | 0.1     |
  | FLOOR            | 0.01    |

  > **Note**: STEEPNESS was reduced from the spec-014 default of 2.0 to 0.1 in Week 7 for gentler boundary suppression, allowing candidates closer to edges. This value is carried forward unchanged.

- **FR-006**: The acquisition function MUST be weighted UCB: $a(x) = p(x) \cdot \mu(x) + \kappa \cdot p(x) \cdot \sigma_{\text{RF}}(x)$, multiplied by local penalization and interior penalty factors.
- **FR-007**: Local penalization MUST be computed over all 18 evaluated points: $\text{penalty}(x) = \prod_{i=1}^{18}\left(1 - e^{-\|x - x_i\|^2 / (2r^2)}\right)$ where $r$ = PENALTY_RADIUS.
- **FR-008**: Interior penalty MUST be: $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{d} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$.
- **FR-009**: The notebook MUST produce a 3-panel contour visualisation: (1) hurdle mean prediction, (2) hurdle uncertainty, (3) penalised acquisition surface — each overlaying training points (colour-coded by output sign) and the proposed sample point.
- **FR-010**: The notebook MUST produce a convergence plot showing the running maximum of observed output values across all 18 data points.
- **FR-011**: The notebook MUST output the proposed next sample as a formatted submission query string: `0.xxxxxx-0.xxxxxx` with all values clipped to [0.0, 0.999999].
- **FR-012**: The notebook MUST validate that the proposed point has (a) all coordinates in [0.0, 0.999999], (b) minimum Euclidean distance ≥ 0.05 from any existing data point.
- **FR-013**: The notebook MUST be fully self-contained — all imports, data loading, model fitting, acquisition, visualisation, and submission output in one notebook.
- **FR-014**: All hyperparameters MUST be defined as named constants in a dedicated cell at the top of the notebook with markdown documentation explaining each parameter's role.

### Key Entities

- **Hurdle Model**: A two-stage surrogate combining a calibrated classifier (Stage 1: probability of positive output) and a random forest regressor (Stage 2: magnitude of positive output via log1p transform). Together they predict $\hat{y} = p(x) \cdot \text{expm1}(\mu(x))$.
- **Weighted UCB Acquisition**: Exploration-focused acquisition function that weights both exploitation ($p \cdot \mu$) and exploration ($\kappa \cdot p \cdot \sigma$) by the classifier probability, naturally suppressing unproductive regions.
- **Local Penalization**: Multiplicative penalty that discourages candidates near already-evaluated points, promoting diversity in the search.
- **Interior Penalty**: Multiplicative boundary suppression via sinusoidal product, preventing candidates from clustering on search-space edges.
- **Week 8 Data**: 18 evaluated observations (10 initial + 8 weekly submissions) for F1 (2D input, 1D output).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook `f1 - week 8.ipynb` executes end-to-end without errors, producing all expected outputs.
- **SC-002**: The proposed sample point has both coordinates within [0.05, 0.95], confirming the interior penalty is effective.
- **SC-003**: The submission query string is in valid format `0.xxxxxx-0.xxxxxx` with values in [0.0, 0.999999].
- **SC-004**: The convergence plot shows clearly whether the Week 8 observation improved upon the previous best.
- **SC-005**: All 18 data points are correctly loaded and displayed, matching the contents of the Week 8 data files.
- **SC-006**: The 3-panel surrogate visualisation is legible and correctly annotated with training points and the proposed candidate.

## Assumptions

- The data files `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` already exist in `./data/f1/` and contain 18 observations (10 initial + 8 weekly submissions).
- The F1 problem is a 2-dimensional search over [0, 1]² (radiation source detection).
- The same hyperparameters from Week 7 (including the interior penalty added in spec 014) remain appropriate for Week 8. No hyperparameter tuning is specified for this iteration.
- The notebook follows the constitution convention: a new file `f1 - week 8.ipynb` in `./functions/f1/`, fully self-contained.
- The original `f1.ipynb` is not modified.
- The strategy remains exploration-focused (KAPPA = 3.0) as F1 has shown limited improvement in prior weeks.
