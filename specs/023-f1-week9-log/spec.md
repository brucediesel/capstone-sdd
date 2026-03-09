# Feature Specification: F1 Week 9 — log Transform, No Penalties

**Feature Branch**: `023-f1-week9-log`  
**Created**: 2026-03-09  
**Status**: Draft  
**Input**: User description: "Create a new branch for this week. Limit to only F1 for this specification. Focussing on F1, create a new notebook for week 9 but change the outputs of the RF regressor with 'log' instead of log1p. Provide visualisation of surrogate using log instead of log1p. Propose next sample point using correct format."

## Motivation

Function 1's positive outputs span extreme magnitudes (approximately 1e-245 to 1e-16). The Week 9 performance evaluation identified that using `log1p(y)` is ineffective for these ultra-small values because by Taylor expansion, `log1p(ε) ≈ ε` when ε is near zero, collapsing the range and eliminating any benefit of the transformation. Switching to `log(y)` maps these values to an interpretable range (approximately -565 to -35), giving the Random Forest regressor meaningful signal to model.

## Clarifications

### Session 2026-03-09

- Q: Should surrogate contour panels display values in log-space or back-transformed original space? → A: Log-space (values -565 to -35) for readable, informative contours.
- Q: With interior penalty and local penalization removed, should KAPPA be adjusted? → A: Keep KAPPA=3.0 unchanged (isolate the effect of removing penalties).
- Q: With penalties removed, what should contour Panel 3 display? → A: Raw weighted UCB acquisition surface, titled "Acquisition (Weighted UCB)".
- Q: Should acquisition prioritise exploitation or exploration given improved surrogate scaling? → A: Exploitation. Reduce KAPPA from 3.0 to 0.5 — only 4 submissions remain in budget and the log transform gives the surrogate meaningful signal to exploit.

## Per-Function Strategy Summary

This specification covers **F1 only**. All other functions are unchanged.

| Function | Dims | Week 9 Samples | Initial Samples | Surrogate | Acquisition | Interior Penalty |
|----------|------|-----------------|-----------------|-----------|-------------|------------------|
| F1 | 2 | 19 | 10 | Hurdle Model (Classifier + RF Regressor with **log** transform) | Weighted UCB (KAPPA=0.5, exploitation-focused, no local penalization) | No |

**Key changes from Week 8**: (1) Stage 2 RF regressor trains on `log(y)` instead of `log1p(y)` for positive outputs. Back-transformation uses `exp()` instead of `expm1()`. (2) Interior penalty and local penalization removed from acquisition function to simplify the optimisation. (3) KAPPA reduced from 3.0 to 0.5 to prioritise exploitation — the log transform gives the surrogate meaningful signal and only 4 budget submissions remain.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load and Validate Week 9 Data (Priority: P1)

As a student running the weekly Bayesian Optimisation loop for F1, I want the notebook to load the Week 9 updated inputs and outputs so that all 19 evaluated sample points are available for surrogate modelling.

**Why this priority**: Without current data, no optimisation can proceed. This is the foundation for the iteration.

**Independent Test**: Run the data-loading cells — they should display 19 samples in tabular format with no NaN or out-of-range values, and the current best observation identified.

**Acceptance Scenarios**:

1. **Given** updated_inputs - Week 9.npy and updated_outputs - Week 9.npy exist in ./data/f1/, **When** the notebook loads and displays the data, **Then** 19 samples with 2 input dimensions are shown in tabular form.
2. **Given** the loaded data, **When** validation checks run, **Then** all input values are within [0.0, 1.0] and no outputs contain NaN or Inf.
3. **Given** the loaded data, **When** the current best observation is identified, **Then** its value and location are printed clearly.

---

### User Story 2 — Fit Hurdle Model with log Transform (Priority: P1)

As a student, I want the notebook to train the hurdle model using `log(y)` instead of `log1p(y)` for the RF regressor stage, so that the surrogate can model the extreme-magnitude positive outputs meaningfully.

**Why this priority**: This is the core change requested — switching the output transformation is the primary objective of this iteration.

**Independent Test**: After the surrogate cells execute, the RF regressor is trained on log-transformed positive outputs and predictions can be queried with back-transformation via `exp()`.

**Acceptance Scenarios**:

1. **Given** F1 data (19 samples, 2D) with some positive outputs, **When** Stage 1 (calibrated logistic classifier) is trained on binary labels (y > 0), **Then** the classifier fits successfully and produces calibrated probability estimates.
2. **Given** the positive-output subset, **When** the RF regressor is trained on `log(y)` (not `log1p(y)`), **Then** the model fits on values in the range approximately [-565, -35] instead of near-zero values.
3. **Given** predicting at new candidate points, **When** the RF produces mean and uncertainty in log-space, **Then** back-transformation uses `exp(mu)` instead of `expm1(mu)`.
4. **Given** fewer than 3 positive outputs exist, **When** the notebook enters fallback mode, **Then** Stage 2 is skipped and acquisition defaults to pure exploration.

---

### User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

As a student, I want the notebook to optimise the acquisition function and output a correctly formatted submission query so that I can submit the next sample point.

**Why this priority**: The submission query is the primary deliverable of each weekly iteration.

**Independent Test**: The notebook outputs a formatted string in 0.xxxxxx-0.xxxxxx format with values clipped to [0.0, 0.999999].

**Acceptance Scenarios**:

1. **Given** the hurdle model is fitted with log transform, **When** weighted UCB acquisition is evaluated over 20,000 random candidates (no local penalization or interior penalty), **Then** a proposed sample point is selected.
2. **Given** the proposed point, **When** it is formatted for submission, **Then** the output string follows the pattern 0.xxxxxx-0.xxxxxx with exactly 6 decimal places per coordinate.
3. **Given** the proposed point, **When** clipping is applied, **Then** all values are within [0.0, 0.999999].
4. **Given** the proposed point, **When** minimum distance to existing data is computed, **Then** a warning is raised if the distance is less than 0.05.

---

### User Story 4 — Visualise Surrogate with log Transform (Priority: P1)

As a student, I want the notebook to produce surrogate visualisations using the log-transformed model so that I can visually assess the surrogate's behaviour with the new transformation.

**Why this priority**: The user explicitly requested visualisation of the surrogate using log instead of log1p. This provides insight into whether the transformation change improves surrogate quality.

**Independent Test**: Three-panel contour plots are produced with correct colour coding, and values are displayed in log-space (not back-transformed) for readable contours.

**Acceptance Scenarios**:

1. **Given** the hurdle model with log transform is fitted, **When** the 3-panel contour plot is generated, **Then** Panel 1 shows hurdle mean prediction in log-space (p(x) · mu(x), values approximately -565 to -35), Panel 2 shows hurdle uncertainty in log-space (p(x) · sigma_RF(x)), and Panel 3 shows the raw weighted UCB acquisition surface titled "Acquisition (Weighted UCB)".
2. **Given** the contour plots, **When** training points are overlaid, **Then** initial samples (10) appear in blue and weekly submissions (9) appear in orange with the proposed next point as a green star.
3. **Given** the convergence plot, **When** the running maximum is displayed, **Then** initial samples are in blue and weekly submissions are in orange with the boundary clearly marked.
4. **Given** all plots, **When** a legend is displayed, **Then** it includes three entries: "Initial samples", "Weekly submissions", and "Proposed next point".

---

### User Story 5 — Performance Evaluation (Priority: P2)

As a student, I want the notebook to include a performance evaluation section assessing whether the log transformation improves model quality compared to log1p.

**Why this priority**: Understanding the impact of the transformation change is important for the capstone writeup, but is secondary to producing the submission query.

**Independent Test**: Code cells compute convergence metrics, stalling flag, and exploration spread; a final markdown cell interprets results.

**Acceptance Scenarios**:

1. **Given** the complete dataset (10 initial + 9 submissions), **When** the performance evaluation runs, **Then** the best-value trajectory, per-submission improvement, and stalling flag are computed and displayed.
2. **Given** the submission points, **When** exploration spread analysis runs, **Then** mean pairwise distance and max nearest-neighbour distance among submissions are displayed.
3. **Given** a stalling flag is True, **When** the concluding markdown cell is written, **Then** it proposes at least one actionable strategy change.
4. **Given** a stalling flag is False, **When** the concluding markdown cell is written, **Then** it confirms the current strategy is performing well.

---

### Edge Cases

- **Missing data files**: If updated_inputs - Week 9.npy or updated_outputs - Week 9.npy is absent, the notebook should fail early with a clear file-not-found message directing the user to run the results processing notebook first.
- **Unexpected sample count**: If loaded data has fewer or more than 19 rows, the notebook should print a warning and adjust the initial/submission split accordingly.
- **No positive outputs**: If no positive outputs exist (y > 0), the hurdle model enters fallback pure exploration mode — Stage 2 is skipped entirely.
- **log(0) guard**: The log transform is only applied to positive outputs (y > 0). Zero or negative values are handled by Stage 1 classification, never passed to log().
- **Very small positive values**: Values like 1e-245 produce log values near -564. The RF should handle these large-magnitude negative values without issues.
- **Boundary candidates**: With no interior penalty, edge candidates are permitted. The acquisition function may select boundary points.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A new self-contained Jupyter notebook MUST be created at `./functions/f1/f1 - week 9.ipynb`. Existing notebooks MUST NOT be modified.
- **FR-002**: The notebook MUST load `updated_inputs - Week 9.npy` and `updated_outputs - Week 9.npy` from `./data/f1/`.
- **FR-003**: The notebook MUST display all 19 data points in tabular format, identifying the current best observation and its location.
- **FR-004**: The notebook MUST implement the same two-stage hurdle model as Week 8 with one change: Stage 2 RF regressor MUST train on `log(y)` for positive outputs instead of `log1p(y)`.
- **FR-005**: Stage 1 MUST use CalibratedClassifierCV(LogisticRegression) with C=1.0, class_weight='balanced', max_iter=1000, cv=3, method='sigmoid'.
- **FR-006**: Stage 2 MUST use RandomForestRegressor with n_estimators=100, max_depth=3. Training targets MUST be `np.log(y_pos)` (not `np.log1p(y_pos)`).
- **FR-007**: If Stage 2 predictions are displayed in original space (e.g., data table comparison), back-transformation MUST use `np.exp(mu)` (not `np.expm1(mu)`). Contour panels display in log-space and do not require back-transformation.
- **FR-008**: If fewer than MIN_POSITIVE=3 positive outputs exist, FALLBACK_MODE MUST be set; Stage 2 is skipped and acquisition defaults to pure exploration (mu=0, sigma=1).
- **FR-009**: Acquisition function MUST be weighted UCB: a(x) = p(x)·mu(x) + kappa·p(x)·sigma_RF(x). No local penalization or interior penalty applied.
- **FR-010**: Hyperparameters MUST be: N_INITIAL=10, N_TOTAL=19, MIN_POSITIVE=3, C_STAGE1=1.0, N_ESTIMATORS=100, MAX_DEPTH=3, KAPPA=0.5, N_CANDIDATES=20000, GRID_RES=50.
- **FR-011**: The proposed sample point MUST be formatted as `0.xxxxxx-0.xxxxxx` with values clipped to [0.0, 0.999999].
- **FR-012**: The notebook MUST produce a 3-panel contour visualisation: (1) hurdle mean in log-space (p(x) · mu, where mu is log-scale RF prediction, yielding values approximately -565 to -35), (2) hurdle uncertainty in log-space (p(x) · sigma_RF), (3) raw weighted UCB acquisition surface titled "Acquisition (Weighted UCB)" — each overlaying training points (initial samples in blue, weekly submissions in orange, proposed next point as green star).
- **FR-013**: The convergence plot MUST show the running maximum of observed outputs, with initial samples in blue and weekly submissions in orange.
- **FR-014**: All plots MUST include a legend with three entries: "Initial samples", "Weekly submissions", and "Proposed next point".
- **FR-015**: All hyperparameters MUST be defined as named constants in a dedicated cell at the top of the notebook with markdown documentation.
- **FR-016**: The notebook MUST be fully self-contained — all imports, data loading, model fitting, acquisition, visualisation, and submission output in one notebook.
- **FR-017**: The notebook MUST end with a "Performance Evaluation" section containing code cells computing best-value trajectory, per-submission improvement deltas, stalling flag (no improvement in last 3 submissions), mean pairwise distance among submissions, max nearest-neighbour distance, and a concluding markdown cell interpreting results.

### Key Entities

- **Hurdle Model**: Two-stage surrogate for zero-inflated outputs — Stage 1 classifier for P(y > 0), Stage 2 RF regressor for log(y) conditioned on y > 0.
- **Weighted UCB**: Acquisition function combining classifier probability weighting and exploration-exploitation trade-off (kappa). No local penalization or interior penalty applied.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes end-to-end without errors, producing a valid submission query for F1 in the correct format.
- **SC-002**: The RF regressor trains on log-transformed values (range approximately -565 to -35) instead of near-zero log1p values, demonstrating improved surrogate modelling of extreme-magnitude outputs.
- **SC-003**: The 3-panel surrogate visualisation renders correctly in log-space (values approximately -565 to -35), showing meaningful variation across the input space that would be invisible in back-transformed original space.
- **SC-004**: All 19 data points are correctly loaded, validated, and displayed with the initial/submission colour distinction maintained across all plots.
- **SC-005**: The proposed next sample point is within [0.0, 0.999999]^2. A proximity warning is issued if the point is less than 0.05 Euclidean distance from the nearest existing observation.
- **SC-006**: The performance evaluation section quantitatively assesses convergence and provides an actionable strategy recommendation if stalling is detected.

## Assumptions

- Week 9 data files (`updated_inputs - Week 9.npy`, `updated_outputs - Week 9.npy`) will be available in `./data/f1/` before notebook execution. If not yet created, the user should run the results processing notebook (`./functions/results/`) first.
- F1 has 10 initial samples plus 9 weekly submissions = 19 total observations for week 9.
- The F1 problem remains a 2D input, scalar output maximisation task.
- All positive outputs for F1 are strictly greater than zero (no exact-zero positive values), making `log(y)` safe to compute without encountering log(0).
- The same hyperparameters as Week 8 are retained except KAPPA (reduced from 3.0 to 0.5 for exploitation focus) and the removal of interior penalty and local penalization.
- The hurdle model architecture (classifier + regressor) is unchanged — the log transform within Stage 2 is modified, and the acquisition function is simplified by removing penalization terms.
