# Feature Specification: Week 7 — SFGP and MFGP Prequential Evaluation on Function 2

**Feature Branch**: `006-sfgp-mfgp-pe`  
**Parent Branch**: `master`  
**Created**: 2026-02-22  
**Status**: Draft  
**Scope**: Function 2 (f2) notebook only

## Summary

Replace the three existing surrogate model families (GP, BART, RF) in `preq-eval-f2.ipynb` with two Gaussian Process variants — a **Single-Fidelity Gaussian Process (SFGP)** and a **Multi-Fidelity Gaussian Process (MFGP)** — and extend the dataset to include the Week 7 data, giving 17 total samples (10 initial + 7 prequential steps).

Each surrogate family is evaluated across **50 hyperparameter configurations** (100 total), replacing the previous 10-per-family scheme. The best configuration from each family is compared in a head-to-head summary. The best overall surrogate is identified and its predictions are visualised in detail. All existing visualisation patterns (per-step prediction plot, bar chart comparison, horizontal bar sensitivity chart, ranked results table) are preserved.

**Surrogate Architectures:**

| Family | Description | Key Hyperparameter Axes |
|--------|-------------|------------------------|
| SFGP | Standard exact GP trained on all available data at a single fidelity level; analytic posterior | Kernel type (Matérn ν, RBF), ARD, noise lower bound, input normalisation, log-transform |
| MFGP | Multi-fidelity GP that models observations from two fidelity levels jointly using BoTorch `MultiTaskGP` (ICM); posterior accounts for cross-fidelity covariance | ICM rank, inner GP kernel, noise lower bound, output standardization |

**Evaluation Metrics** (unchanged from existing notebook):

$$\text{NLP}_i = \frac{1}{2}\log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}$$

| Metric | Direction | Description |
|--------|-----------|-------------|
| MAE | Lower is better | Mean absolute error between predicted and actual values |
| NLP | Lower is better | Negative log predictive density; penalises inaccurate means and miscalibrated uncertainty |
| 95% Coverage | Closer to 0.95 is better | Proportion of actuals within the 95% prediction interval |

## Assumptions

- The Week 7 data files for f2 already exist at `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy` and contain all 17 cumulative samples.
- f2 is a 2-dimensional maximisation problem with output range approximately [-0.07, 0.67].
- All 50 SFGP configurations and all 50 MFGP configurations are independent hyperparameter trials evaluated by sequentially running the prequential loop on the fixed dataset — not active-learning runs.
- For the MFGP, fidelity levels are assigned based on observation order: the initial 10 samples form the low-fidelity set and the 7 weekly updates are treated as the high-fidelity set. This assignment is fixed and does not vary across configurations.
- The prequential loop always starts from 10 training points and evaluates 7 one-step-ahead predictions, matching the existing `N_INIT = 10` convention.
- The submission format and any actual BO query point are out of scope for this notebook; this spec covers evaluation only.
- NLP is the primary selection metric, consistent with the existing notebook.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load Week 7 Data and Confirm Prequential Setup (Priority: P1)

As a researcher extending the f2 prequential evaluation to Week 7, I need to load the 17-sample cumulative dataset and confirm the correct prequential configuration (10 initial + 7 steps), so that both SFGP and MFGP evaluations operate on the right data window.

**Why this priority**: All modelling depends on correct data loading; a misconfigured step count silently invalidates every metric.

**Independent Test**: Execute only the data-loading cell. Verify it prints 17 rows of 2D inputs, 17 scalar outputs, `Initial training points: 10`, and `Evaluation steps: 7`.

**Acceptance Scenarios**:

1. **Given** the Week 7 `.npy` files for f2, **When** the data-loading cell runs, **Then** the printed summary shows `X shape: (17, 2)`, `y shape: (17,)`, `Initial training points: 10`, and `Evaluation steps: 7`
2. **Given** the loaded dataset, **When** an output range check runs, **Then** the output range is within the known f2 bounds (approximately -0.07 to 0.67) with no NaN or infinite values
3. **Given** the loaded dataset, **When** the fidelity-split summary runs (for MFGP context), **Then** the notebook confirms 10 observations designated as low-fidelity and 7 as high-fidelity

---

### User Story 2 — Evaluate SFGP Across 50 Hyperparameter Configurations (Priority: P1)

As a researcher comparing Gaussian Process variants, I need to run the prequential loop for 50 SFGP hyperparameter configurations and identify the best one by NLP, so that the SFGP benchmark is properly representative of the family.

**Why this priority**: SFGP forms one half of the head-to-head comparison and directly results in the recommendation for the BO pipeline.

**Independent Test**: Execute only the SFGP sections. Verify a 50-row results table is populated, the best configuration by NLP is printed, and a default-config prediction plot renders.

**Acceptance Scenarios**:

1. **Given** the 17-sample dataset and 50 SFGP configuration dictionaries, **When** the hyperparameter sweep cell runs, **Then** a table with 50 rows is produced, each row containing a label, MAE, NLP, and Coverage_95 value; no more than 5 rows may contain NaN (failed fits are tolerated but must be clearly flagged)
2. **Given** the 50-row SFGP results table, **When** the best-configuration selection cell runs, **Then** the configuration with the lowest NLP is identified, its label is printed, and MAE / NLP / Coverage are displayed
3. **Given** the best SFGP configuration, **When** the prediction plot cell runs, **Then** a per-step chart is displayed showing the 7 one-step-ahead predictions, their 95% uncertainty bands, and the actual observed values, matching the existing `plot_prequential_results()` style

---

### User Story 3 — Evaluate MFGP Across 50 Hyperparameter Configurations (Priority: P1)

As a researcher comparing Gaussian Process variants, I need to run the prequential loop for 50 MFGP hyperparameter configurations and identify the best one by NLP, so that the MFGP family is fully characterised.

**Why this priority**: MFGP forms the other half of the comparison and is the novel element of this feature; without it the feature is incomplete.

**Independent Test**: Execute only the MFGP sections. Verify a 50-row results table populates, best MFGP by NLP is selected, and a prediction plot renders.

**Acceptance Scenarios**:

1. **Given** the 17-sample dataset with fidelity assignments and 50 MFGP configuration dictionaries, **When** the hyperparameter sweep cell runs, **Then** a 50-row results table is produced with the same columns (label, MAE, NLP, Coverage_95); no more than 10 rows may contain NaN
2. **Given** the 50-row MFGP results table, **When** the best-configuration selection cell runs, **Then** the lowest-NLP configuration is identified, its label and hyperparameter values are printed, and MAE / NLP / Coverage are displayed
3. **Given** the best MFGP configuration, **When** the prediction plot cell runs, **Then** a per-step chart renders with the same visual style as the SFGP and prior-notebook plots

---

### User Story 4 — Head-to-Head Comparison and Best Surrogate Visualisation (Priority: P2)

As a researcher deciding which surrogate to use in the BO pipeline for f2, I need a side-by-side comparison of the best SFGP and best MFGP configurations, with the overall winner clearly visualised, so I can make an evidence-based recommendation.

**Why this priority**: This is the decision-making output of the notebook; the comparison is only meaningful once both surrogates are evaluated.

**Independent Test**: Execute only the comparison and visualisation cells. Verify a 2-row comparison table renders, a bar chart appears with SFGP and MFGP side by side, and the winner is announced.

**Acceptance Scenarios**:

1. **Given** the best SFGP and best MFGP results, **When** the comparison table cell runs, **Then** a 2-row table is displayed with Model, Configuration, MAE, NLP, and Coverage_95 columns, and the per-metric winner is printed (e.g., "Best NLP: SFGP")
2. **Given** the comparison table, **When** the bar chart cell runs, **Then** a 3-panel figure (MAE, NLP, Coverage) is rendered with one bar per model per panel, value labels on each bar, and the 0.95 reference line on the coverage panel — matching the existing notebook's bar-chart style
3. **Given** the overall winner by NLP, **When** the sensitivity chart cell runs, **Then** a horizontal bar chart displays all 100 configurations (50 SFGP + 50 MFGP) ranked by NLP, colour-coded by family (SFGP blue, MFGP orange), matching the existing 30-configuration sensitivity chart convention
4. **Given** the overall winner, **When** the winner-detail visualisation cell runs, **Then** the best model's per-step predictions are shown alongside the actual f2 values, with the uncertainty band visualised, and a title clearly naming the winning surrogate and its configuration

---

### Edge Cases

- What happens when a MFGP fit fails due to insufficient low-fidelity observations? → Model catches the exception, records NaN for that configuration, prints a warning, and continues to the next configuration
- What happens if all 50 MFGP configurations fail? → The best-configuration selection cell falls back to reporting "No valid MFGP configurations" and the comparison proceeds with SFGP only
- What happens when the prequential step size is only 1 point (coverage computed from just 7 predictions)? → Coverage is still reported numerically; a note in the conclusions warns that 7 steps is a small sample for reliable coverage estimation
- What happens if SFGP and MFGP tie on NLP? → The tie is broken first by MAE (lower wins), then by Coverage proximity to 0.95; the tiebreaker rule is documented in the conclusions cell

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST load data from `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy` and set `WEEK = 7`, `N_INIT = 10`
- **FR-002**: The notebook MUST remove all existing GP, BART, and RF surrogate sections and replace them with SFGP and MFGP sections
- **FR-003**: The SFGP section MUST evaluate exactly 50 hyperparameter configurations covering at minimum: kernel type, noise lower bound, and ARD setting; all configurations MUST be listed as a Python list named `sfgp_configs`
- **FR-004**: The MFGP section MUST evaluate exactly 50 hyperparameter configurations covering at minimum: ICM rank (`rank`), inner GP kernel, and noise lower bound; all configurations MUST be listed as a Python list named `mfgp_configs`
- **FR-005**: The prequential loop for each configuration MUST start from `N_INIT = 10` training points and evaluate 7 one-step-ahead predictions, producing MAE, NLP, and Coverage_95 for each configuration
- **FR-006**: NLP MUST be computed using the same formula as the existing notebook: $\frac{1}{2}\log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}$
- **FR-007**: The notebook MUST reuse the existing `compute_metrics()` function without modification; if the function is moved, it MUST remain functionally identical
- **FR-008**: The notebook MUST reuse the existing `plot_prequential_results()` visualisation function (or an equivalent function with the same visual output) for per-step prediction charts
- **FR-009**: The best SFGP configuration MUST be selected as the row with the minimum NLP in `sfgp_hp_df`, stored in a variable named `best_sfgp`
- **FR-010**: The best MFGP configuration MUST be selected as the row with the minimum NLP in `mfgp_hp_df`, stored in a variable named `best_mfgp`
- **FR-011**: The comparison section MUST produce a 2-row table (Best SFGP vs Best MFGP) reporting MAE, NLP, and Coverage_95
- **FR-012**: The comparison section MUST include a 3-panel bar chart (MAE / NLP / Coverage) using the same colour convention and value-label style as the existing notebook
- **FR-013**: The sensitivity section MUST include a horizontal bar chart showing all 100 configurations ranked by NLP, colour-coded by family
- **FR-014**: The winner-detail section MUST show the best overall surrogate's per-step predictions with uncertainty bands
- **FR-015**: Failed hyperparameter configurations MUST record NaN in the results dataframe and MUST NOT raise an unhandled exception

### Key Entities

- **SFGP configuration**: A dict with at minimum `kernel_type`, `noise_lb`, and `label`; may include additional keys such as `ard`, `log_transform`, `input_normalize`, `lengthscale_prior`
- **MFGP configuration**: A dict with at minimum `rank`, `kernel_type`, `noise_lb`, and `label`; `rank` controls the ICM dimensionality; BoTorch `MultiTaskGP` shares a noise lower bound constraint across both tasks; may include additional keys such as `output_standardize`, `step0_fallback`
- **Prequential result record**: A dict with keys `label`, `MAE`, `NLP`, `Coverage_95`; stored as rows in `sfgp_hp_df` and `mfgp_hp_df` DataFrames
- **Fidelity split**: Fixed partition of the 17-sample dataset where indices 0–9 are low-fidelity and indices 10–16 are high-fidelity, used by the MFGP loader

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 50 SFGP configurations produce a result record (valid metric or explicit NaN) without the notebook halting on an error
- **SC-002**: All 50 MFGP configurations produce a result record (valid metric or explicit NaN) without the notebook halting on an error
- **SC-003**: At least 40 of the 50 SFGP configurations produce valid (non-NaN) metrics
- **SC-004**: At least 30 of the 50 MFGP configurations produce valid (non-NaN) metrics
- **SC-005**: The best SFGP and best MFGP are identified within 5 seconds of completing their respective sweeps (table display and best-config print rendered)
- **SC-006**: The comparison bar chart, sensitivity chart, and winner-detail prediction plot all render without error when run sequentially from top to bottom on a clean kernel
- **SC-007**: The notebook runs end-to-end (all cells, top to bottom) in under 30 minutes on a standard laptop with 8 GB RAM
- **SC-008**: The winner is unambiguously identified: the conclusions cell prints a single recommendation sentence naming the winning surrogate family and its key hyperparameter values
