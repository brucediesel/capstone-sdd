# Feature Specification: Week 7 — F1 Hurdle Model with Weighted UCB and Local Penalization

**Feature Branch**: `005-week7-pe-surrogates`  
**Parent Branch**: `master`  
**Created**: 2026-02-22  
**Status**: Draft  
**Scope**: Function 1 (f1) notebook only

## Summary

Add a "Week 7" section to the f1 notebook. The cumulative 17-sample dataset (10 initial + 7 weekly updates) is loaded from `data/f1/updated_inputs - Week 7.npy` and `data/f1/updated_outputs - Week 7.npy`. The polynomial surrogate used in Weeks 5–6 is replaced with a **hurdle model**: a two-stage approach that models the probability of a positive output separately from the conditional magnitude of positive outputs. This is combined with a **weighted UCB acquisition function** that accounts for the probability of a positive result, and **local penalization** to propose sampling points that collectively cover the search space rather than clustering around a single mode.

The strategy for Week 7 is **exploration-focused**: f1 has returned no improvements across three submissions; the model must search new regions rather than refine around the current best.

**Surrogate Architecture (Hurdle Model):**

| Stage | Model | Purpose |
|-------|-------|---------|
| Stage 1 — Classifier | Logistic Regression wrapped in `CalibratedClassifierCV`; hyperparameter: regularisation strength C | Estimates P(y > 0) at each candidate point with well-calibrated probabilities |
| Stage 2 — Regressor | Random Forest Regressor trained on log1p(y) for all y > 0; uncertainty = std across trees | Estimates E[log1p(y) \| y > 0] and per-point uncertainty via tree ensemble |

**Acquisition Function:**

$$a(x) = p(x) \cdot \mu(x) + \kappa \cdot p(x) \cdot \sigma_{\text{RF}}(x)$$

where $p(x)$ is the Stage 1 probability estimate, $\mu(x)$ is the Stage 2 conditional mean prediction (back-transformed from log scale), and $\sigma_{\text{RF}}(x)$ is the Random Forest tree standard deviation. Multiplying the exploration bonus by $p(x)$ naturally suppresses it in regions unlikely to produce positive outputs.

**Local Penalization:** After the primary candidate is selected, subsequent candidates are penalized proportionally to their proximity to already-selected points, encouraging the Bayesian optimisation loop to propose diverse sampling locations.

## Clarifications

### Session 2026-02-22

- Q: What algorithm should be used for Stage 2 (the regressor trained on log1p(y) for positive outputs)? → A: **Random Forest Regressor** — per-point uncertainty derived from std across trees; simple to explain and handles small positive-sample counts without a separate bootstrap loop.
- Q: How should √Var(p(x)μ(x)) be computed in the weighted UCB formula? → A: **Simplified approximation: p(x)·σ_RF(x)** — scale the Random Forest tree-std by the Stage 1 probability; exploration bonus is naturally suppressed in low-probability regions and fully expressed where positive output is likely.
- Q: What algorithm should be used for Stage 1 (the binary classifier estimating P(y > 0))? → A: **Logistic Regression + CalibratedClassifierCV** — single hyperparameter C, transparent, and produces well-calibrated probabilities required by the weighted UCB formula.
- Q: Should local penalization apply to all existing data points or only batch candidates? → A: **All 17 existing evaluated data points** — prevents the acquisition from re-visiting any previously sampled region, maximising exploration across the full input space.
- Q: What is the minimum number of positive-output samples required before Stage 2 is fitted? → A: **3** — fewer than 3 positive samples is insufficient for a Random Forest to form reliable splits on 2 features; below this threshold the notebook warns and falls back.

## Assumptions

- f1 is a 2-dimensional function; all inputs lie in [0.0, 0.999999].
- The Week 7 data files for f1 already exist at `data/f1/updated_inputs - Week 7.npy` and `data/f1/updated_outputs - Week 7.npy` and contain all 17 cumulative samples.
- Existing notebook cells (Weeks 5 & 6) are not modified; all new code is added as a new "Week 7" section appended after the Week 6 section.
- The submission format (`x1-x2` with six decimal places, values in [0.000000, 0.999999]) is unchanged.
- A single new sampling point is proposed as the Week 7 submission query.
- Positive outputs are defined as y > 0; the threshold can be adjusted if the output range differs.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load and Validate Week 7 Data for F1 (Priority: P1)

As a student working on the Week 7 submission for f1, I need to load the cumulative dataset and confirm it contains the expected number of samples and no out-of-range values, so that subsequent modelling uses clean data.

**Why this priority**: All downstream steps depend on correct data loading. This is the fastest-failing independent check.

**Independent Test**: Execute only the data-loading cell in the Week 7 section. Verify 17 rows of 2D inputs and 17 scalar outputs are displayed, all inputs in [0.0, 1.0], with a tabular summary shown.

**Acceptance Scenarios**:

1. **Given** the Week 7 `.npy` files for f1, **When** the data-loading cell runs, **Then** a table displays 17 input rows (2 columns each) and 17 corresponding output values
2. **Given** the loaded data, **When** the validation check runs, **Then** all input values are confirmed within [0.000000, 0.999999] with no NaN or infinite outputs
3. **Given** the loaded data, **When** the class balance check runs, **Then** the count of positive outputs (y > 0) and non-positive outputs (y ≤ 0) are both displayed to confirm viability of the hurdle model

---

### User Story 2 — Fit the Hurdle Model with Explicit Hyperparameters (Priority: P1)

As a student modelling f1 for Week 7, I need to fit a two-stage hurdle model on the 17-sample dataset, with all hyperparameters explicitly listed and their selection rationale documented, so that the model choice can be understood and reproduced.

**Why this priority**: This is the core modelling change for Week 7 and must succeed before acquisition or visualisation can proceed.

**Independent Test**: Execute the model-fitting cells. Verify Stage 1 outputs a probability for each training point, Stage 2 outputs a mean and uncertainty estimate on the log-transformed training outputs, and both models report training metrics.

**Acceptance Scenarios**:

1. **Given** the 17-sample training set, **When** Stage 1 (classifier) is fitted, **Then** the model reports accuracy or log-loss on training data, and each training point has a probability estimate between 0 and 1
2. **Given** the subset of training samples where y > 0, **When** Stage 2 (regressor on log1p(y)) is fitted, **Then** the model reports R² or MAE on the log-scale training data, and each positive-output training point has a mean and uncertainty prediction
3. **Given** both stages fitted, **When** a grid of candidate points is evaluated, **Then** each candidate receives a combined hurdle prediction: $\hat{y}(x) = p(x) \cdot \exp(\mu(x)) - 1$ (inverse of log1p), displayed as a heatmap
4. **Given** the fitted model, **When** the hyperparameter summary cell runs, **Then** a markdown or printed table lists every tunable hyperparameter name, value used, and one-sentence rationale for that value

---

### User Story 3 — Apply Weighted UCB with Local Penalization and Propose Query (Priority: P1)

As a student running the Bayesian optimisation loop for Week 7, I need the weighted UCB acquisition function with local penalization to identify a diverse, exploration-focused sampling point, so that the Week 7 submission explores a new region of the f1 input space.

**Why this priority**: This directly produces the submission query; without it the week's task is incomplete.

**Independent Test**: Execute the acquisition and penalization cells. Verify the acquisition surface is plotted, the selected point lies in [0.0, 0.999999]², and the output is formatted correctly.

**Acceptance Scenarios**:

1. **Given** the hurdle model predictions over a 2D candidate grid, **When** the weighted UCB acquisition function is evaluated, **Then** the acquisition value at each point is computed as $a(x) = p(x)\cdot\mu(x) + \kappa\cdot p(x)\cdot\sigma_{\text{RF}}(x)$, and κ is set to an exploration-focused value (κ ≥ 2.0) with its selection documented
2. **Given** the acquisition surface, **When** local penalization is applied, **Then** a penalty function centred on already-selected candidates reduces acquisition values within a specified radius $r$ of those points, with $r$ explicitly stated and its selection explained
3. **Given** the penalized acquisition surface, **When** the optimiser selects the next query point, **Then** the point is printed in the format `x1-x2` with six decimal places, both values within [0.000000, 0.999999]
4. **Given** an acquisition surface dominated by exploration (high κ), **When** the selected point is compared to all previous 17 data points, **Then** the minimum Euclidean distance from the selected point to any existing data point is at least 0.05

---

### User Story 4 — Reproduce Week 6 Visualizations (Priority: P2)

As a student reviewing Week 7 results for f1, I need the same set of visualizations used in Week 6 so that progress can be assessed consistently across weeks.

**Why this priority**: Visualizations are required for the submission but do not block the core modelling work.

**Independent Test**: Execute only the visualization cells. Verify each plot renders without errors and includes the Week 7 data points.

**Acceptance Scenarios**:

1. **Given** the Week 7 data and hurdle model, **When** the convergence plot cell runs, **Then** a line or scatter chart shows the best observed output value across all 17 cumulative evaluations, matching the style used in Week 6
2. **Given** the hurdle model predictions on a 2D grid, **When** the surrogate surface plot runs, **Then** a heatmap or contour plot of the predicted output $\hat{y}(x)$ is displayed with axis labels, the 17 training points overlaid, and the proposed query point highlighted
3. **Given** the acquisition function values on the 2D grid, **When** the acquisition surface plot runs, **Then** a heatmap or contour plot shows the weighted UCB values, the penalization effect around any existing candidates, and the location of the selected next point

---

### Edge Cases

- **Fewer than 3 positive outputs (including the all-zero case)**: Stage 2 (Random Forest Regressor) requires at least 3 positive-output samples to form meaningful splits. If fewer than 3 positive outputs are present (which covers the degenerate all-zero case), the notebook must print a descriptive warning and fall back to a pure exploration acquisition (random sampling over the penalized candidate space).
- **Proposed point too close to an existing sample** (distance < 0.01): The local penalization radius should prevent this; if not, a proximity check should warn the student before the query is formatted.
- **Input values at boundary** (0.000000 or 0.999999): The optimiser must clip or constrain candidates to remain within [0.000000, 0.999999] on both dimensions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST load `data/f1/updated_inputs - Week 7.npy` and `data/f1/updated_outputs - Week 7.npy` as the cumulative dataset for Week 7
- **FR-002**: The notebook MUST validate that all loaded input values are within [0.000000, 0.999999] and display the count of positive vs. non-positive outputs before fitting any model
- **FR-003**: The notebook MUST fit a two-stage hurdle model: Stage 1 is a **Logistic Regression wrapped in `CalibratedClassifierCV`** (hyperparameter: C) to classify P(y > 0); Stage 2 uses a **Random Forest Regressor** on log1p(y) for the positive-output subset only, with per-point uncertainty estimated as the standard deviation across trees
- **FR-004**: The notebook MUST present every hyperparameter of both stages in a dedicated cell, with the name, value, and one-sentence rationale for each
- **FR-005**: The acquisition function MUST implement $a(x) = p(x) \cdot \mu(x) + \kappa \cdot p(x) \cdot \sigma_{\text{RF}}(x)$, where $\sigma_{\text{RF}}(x)$ is the Random Forest tree standard deviation, $\kappa$ is explicitly set to an exploration-focused value (κ ≥ 2.0), and its selection is documented
- **FR-006**: The notebook MUST implement local penalization that applies a Gaussian penalty centred on **all 17 existing evaluated data points**, discouraging the optimiser from proposing points within a defined radius $r$ of any previously evaluated location; $r$ must be explicitly stated and its selection documented
- **FR-007**: The proposed next query MUST be printed in the format `x1-x2` with exactly six decimal places, within [0.000000, 0.999999]
- **FR-008**: The notebook MUST display a convergence plot showing the best observed output across all 17 cumulative evaluations
- **FR-009**: The notebook MUST display a 2D surrogate surface plot (heatmap or contour) of the combined hurdle prediction with training points and the proposed query overlaid
- **FR-010**: The notebook MUST display a 2D acquisition surface plot showing the penalized weighted UCB values and the location of the selected next point
- **FR-011**: All new code MUST be added as a new "Week 7" section appended to the f1 notebook; no existing cells from prior weeks may be modified or deleted
- **FR-012**: The notebook MUST check that at least **3 positive-output samples** (y > 0) are present before fitting Stage 2; if fewer than 3 are present (including the all-zero case), it MUST print a descriptive warning and fall back to a pure exploration acquisition strategy

### Key Entities

- **Hurdle Model**: Two-stage predictive model; Stage 1 is a Logistic Regression (`CalibratedClassifierCV`) outputting a calibrated probability scalar in (0,1) for each input; Stage 2 is a Random Forest Regressor that outputs a mean and per-point std (across trees) on the log1p scale for inputs where y > 0
- **Weighted UCB Score**: A scalar acquisition value per candidate point computed as $a(x) = p(x)\cdot\mu(x) + \kappa\cdot p(x)\cdot\sigma_{\text{RF}}(x)$; the exploration bonus is naturally suppressed in regions unlikely to produce positive outputs
- **Local Penalization Mask**: A scalar-valued penalty field over the input space; reduces acquisition values within radius $r$ of each already-selected candidate; radius $r$ is a named, documented hyperparameter
- **Candidate Grid**: A regular grid of points in [0.0, 0.999999]² used to evaluate the acquisition function; grid resolution is a named hyperparameter

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The Week 7 section in f1.ipynb executes from top to bottom without errors when run on the Week 7 data files
- **SC-002**: Every tunable hyperparameter across both hurdle model stages, the acquisition function, and the local penalization is identified in a dedicated summary cell — zero undocumented "magic numbers" in the modelling code
- **SC-003**: The proposed next query point is formatted correctly (`x1-x2`, six decimal places, both values in [0.000000, 0.999999]) and is at least 0.05 Euclidean distance from every previously evaluated point
- **SC-004**: All three required visualizations (convergence plot, surrogate surface, acquisition surface) render without error and include the full 17-sample history plus the proposed query location
- **SC-005**: The exploration intent is demonstrable: the selected query point lies in a region of the input space with fewer than 3 existing data points within a 0.15-radius neighbourhood
