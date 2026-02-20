# Feature Specification: Prequential Evaluation of Surrogate Models

**Feature Branch**: `004-prequential-evaluation`  
**Created**: 2026-02-20  
**Status**: Draft  
**Input**: User description: "Create notebooks for Prequential Evaluation of surrogate performance — F1 (GP vs BART) and F2 (GP vs BART vs Random Forest)"

## Overview

Create a new Jupyter notebook (`functions/f1/preq-eval-f1.ipynb`) that performs **prequential (one-step-ahead) evaluation** of surrogate model predictive performance for the radiation source detection problem (Function 1).

The evaluation trains each surrogate on the initial 10 data points, then sequentially predicts the next observation, records the error, retrains with the new point included, and repeats until all 16 available data points have been processed. This produces 6 one-step-ahead prediction evaluations per surrogate configuration.

Two surrogate families are compared:
1. **Gaussian Process** (via BoTorch/GPyTorch)
2. **BART — Bayesian Additive Regression Trees** (via PyMC-BART)

For each surrogate family, hyperparameters are optimised over 10 iterations (configurations), and the best configuration of each family is compared head-to-head.

### Function 1 Context

| Property | Value |
|----------|-------|
| Problem | Radiation source detection in 2D area |
| Input dimensions | 2 |
| Output dimensions | 1 |
| Objective | Maximise |
| Input range | [0, 1] |
| Output characteristics | Very small values (near 0 unless close to source); spans many orders of magnitude |
| Initial samples | 10 |
| Total samples (Week 6) | 16 |
| Evaluation steps | 6 one-step-ahead predictions |

## User Scenarios & Testing

### User Story 1 — Run GP Prequential Evaluation (Priority: P1)

As a student doing capstone analysis, I want to train a Gaussian Process on the initial 10 F1 data points, then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage at each step, so that I can assess GP predictive quality.

**Why this priority**: GP is the primary surrogate used in the project and is the baseline against which alternatives are measured.

**Independent Test**: Run cells 1–13 of the notebook. Verify that 6 prediction steps are executed, metrics are printed, and plots render correctly.

**Acceptance Scenarios**:

1. **Given** the Week 6 data (16 points) is loaded, **When** the GP prequential function runs with default Matérn 5/2 kernel, **Then** 6 one-step-ahead predictions are produced with MAE, NLP, and 95% coverage reported.
2. **Given** the GP prequential results, **When** the visualisation cell is executed, **Then** three plots are displayed: predictions vs actuals with uncertainty bands, absolute error per step, and NLP per step.

---

### User Story 2 — Optimise GP Hyperparameters (Priority: P1)

As a student, I want to evaluate 10 different GP configurations (varying kernel type, output transform, noise bounds), so that I can identify the GP configuration with the best predictive calibration.

**Why this priority**: Hyperparameter optimisation is a core requirement of the specification.

**Independent Test**: Run cells 14–17. Verify that 10 configurations are evaluated and a results table with MAE, NLP, and Coverage is displayed, with best configuration identified.

**Acceptance Scenarios**:

1. **Given** 10 GP configurations are defined, **When** the HP optimisation loop runs, **Then** a DataFrame of results is produced with one row per configuration showing MAE, NLP, and Coverage_95.
2. **Given** the GP HP results, **When** the best configuration cell runs, **Then** the configuration with the lowest NLP is identified and printed.

---

### User Story 3 — Run BART Prequential Evaluation (Priority: P1)

As a student, I want to train a BART model on the initial 10 F1 data points and perform the same one-step-ahead evaluation, so that I have a non-GP surrogate baseline.

**Why this priority**: BART is explicitly required as the second surrogate model to compare against GP.

**Independent Test**: Run cells 18–23. Verify that BART produces 6 predictions with the same metric format as GP, and visualisations render.

**Acceptance Scenarios**:

1. **Given** the Week 6 data is loaded, **When** BART prequential evaluation runs with default hyperparameters (m=50, draws=500, tune=200), **Then** 6 one-step-ahead predictions are produced with MAE, NLP, and 95% coverage reported.
2. **Given** BART results, **When** the visualisation cell runs, **Then** the same 3-panel plot format as GP is displayed.

---

### User Story 4 — Optimise BART Hyperparameters (Priority: P1)

As a student, I want to evaluate 10 BART configurations (varying number of trees, MCMC draws, and burn-in), so that I can find the best-calibrated BART configuration.

**Why this priority**: Equal treatment of both surrogate families requires HP optimisation for both.

**Independent Test**: Run cells 24–27. Verify that 10 BART configurations are evaluated with a results DataFrame, and the best configuration is identified.

**Acceptance Scenarios**:

1. **Given** 10 BART configurations are defined, **When** the HP optimisation loop runs, **Then** a results DataFrame is produced with MAE, NLP, and Coverage_95 per configuration.
2. **Given** the BART HP results, **When** the best configuration cell runs, **Then** the configuration with the lowest NLP is identified.

---

### User Story 5 — Compare GP vs BART (Priority: P1)

As a student, I want a side-by-side comparison of the best GP and best BART configurations, with bar charts and a ranked summary table, so that I can determine which surrogate is better for F1.

**Why this priority**: The comparison is the final deliverable that answers the research question.

**Independent Test**: Run cells 28–36. Verify comparison table, bar charts, and full ranked results table across all 20 configurations are displayed.

**Acceptance Scenarios**:

1. **Given** both GP and BART HP optimisation results exist, **When** the comparison cell runs, **Then** a table showing MAE, NLP, and Coverage_95 for the best of each model is displayed, with a metric-by-metric winner identified.
2. **Given** comparison results, **When** the visual comparison cell runs, **Then** three bar charts (MAE, NLP, Coverage) are displayed with values annotated and an ideal 95% coverage line shown.
3. **Given** all 20 configurations (10 GP + 10 BART), **When** the full results table is generated, **Then** all configurations are ranked by NLP (ascending) in a single table.

---

### Edge Cases

- What happens when GP fitting fails for a configuration? → The code catches the exception and records NaN metrics; the configuration is included in results but marked as failed.
- What happens when BART MCMC sampling diverges? → Same exception handling; NaN metrics are recorded.
- What happens when output values are extremely small (near machine epsilon)? → The GP log-transform configurations address this by operating in log-space. NLP computation clips standard deviations to avoid log(0).
- What if all 6 test-point outputs are zero? → Metrics are still computed; coverage will depend on whether the prediction interval includes 0.

## Clarifications

### Session 2026-02-20

- Q: Should the notebook parameterize the data week number so it can be re-run with future data? → A: Yes — define a `WEEK = 6` variable at the top of the notebook and construct file paths from it.
- Q: How should the "best" configuration be selected when ranking all 20 surrogates? → A: Rank by NLP only (lower is better). NLP captures both calibration and accuracy in a single score.
- Q: Should the notebook generalise to other functions (f2–f8) via a FUNCTION parameter? → A: No — keep it F1-only. Duplicate and manually adapt for other functions when needed.

## Requirements

### Functional Requirements

- **FR-001**: Notebook MUST define a `WEEK` variable (default `6`) at the top and load data from `../../data/f1/updated_inputs - Week {WEEK}.npy` and `../../data/f1/updated_outputs - Week {WEEK}.npy`.
- **FR-002**: Notebook MUST use the first 10 data points as the initial training set.
- **FR-003**: Notebook MUST perform one-step-ahead prequential evaluation: for step t, train on points 1...(10+t), predict point (11+t), record error, repeat for t=0...5.
- **FR-004**: Notebook MUST compute three metrics per evaluation: MAE (Mean Absolute Error), NLP (Negative Log Predictive Density), and Coverage of 95% prediction interval.
- **FR-005**: Notebook MUST evaluate Gaussian Process surrogates using BoTorch `SingleTaskGP` with marginal likelihood fitting.
- **FR-006**: Notebook MUST evaluate BART surrogates using PyMC-BART with MCMC posterior sampling.
- **FR-007**: Notebook MUST optimise hyperparameters for each surrogate family across 10 configurations.
- **FR-008**: GP hyperparameters to vary: kernel type (Matérn 5/2, RBF), output log-transform (yes/no), noise lower bound (1e-4, 1e-5, 1e-6).
- **FR-009**: BART hyperparameters to vary: number of trees (10, 20, 50, 100, 200), MCMC draws (200, 500), burn-in/tune (100, 200).
- **FR-010**: Notebook MUST produce a final side-by-side comparison of best GP vs best BART by NLP.
- **FR-011**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty bands, absolute error per step, NLP per step, and bar-chart comparisons.
- **FR-012**: Notebook MUST produce a full ranked results table of all 20 configurations sorted by NLP.
- **FR-013**: Each code step MUST be clearly explained in markdown cells preceding it.
- **FR-014**: Notebook MUST be stored at `functions/f1/preq-eval-f1.ipynb`.

### Key Entities

- **Surrogate Model**: A predictive model (GP or BART) trained on observed data to predict unseen outputs and provide uncertainty estimates.
- **Prequential Evaluation**: Sequential train-predict-retrain protocol where each new data point is first predicted, then added to the training set.
- **Configuration**: A specific set of hyperparameters for a surrogate model (one of 10 per family).
- **Metrics**: MAE, NLP, and 95% Coverage computed from the 6 one-step-ahead predictions.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Notebook executes end-to-end without errors (all cells run successfully).
- **SC-002**: 6 one-step-ahead predictions are produced for each of the 20 configurations (10 GP + 10 BART).
- **SC-003**: Final comparison table clearly identifies the best surrogate model for F1.
- **SC-004**: All three metrics (MAE, NLP, Coverage) are reported for every configuration.
- **SC-005**: Visualisations are clear, labelled, and suitable for inclusion in a capstone report.
- **SC-006**: Code is simple with each step clearly explained (per project constitution).

## Technical Notes

### Libraries

| Library | Purpose |
|---------|---------|
| BoTorch / GPyTorch | Gaussian Process surrogate and fitting |
| PyMC + PyMC-BART | BART surrogate with MCMC sampling |
| NumPy | Data loading and array operations |
| Pandas | Results tables |
| Matplotlib | Visualisations |

### Data Flow

```
data/f1/updated_inputs - Week 6.npy  ──┐
data/f1/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 16 points
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                              ▼
                   GP Evaluation                  BART Evaluation
                   (10 configs)                   (10 configs)
                         │                              │
                         ▼                              ▼
                  GP Results DF                  BART Results DF
                         │                              │
                         └──────────────┬──────────────┘
                                        ▼
                              Comparison & Ranking
                                        │
                                        ▼
                              Tables + Visualisations
```


---

# Function 2: Prequential Evaluation — GP vs BART vs Random Forest

## Overview (F2)

Create a new Jupyter notebook (`functions/f2/preq-eval-f2.ipynb`) that performs **prequential (one-step-ahead) evaluation** of surrogate model predictive performance for the noisy log-likelihood maximisation problem (Function 2).

The evaluation follows the same prequential protocol as F1: train on the initial 10 data points, predict the next observation one step ahead, record the error, retrain, and repeat until all 16 available points have been processed (6 evaluation steps).

**Three** surrogate families are compared:
1. **Gaussian Process** (via BoTorch/GPyTorch)
2. **BART — Bayesian Additive Regression Trees** (via PyMC-BART)
3. **Random Forest** (via scikit-learn)

For each surrogate family, hyperparameters are optimised over 10 configurations, and the best configuration of each family is compared in a three-way comparison.

### Function 2 Context

| Property | Value |
|----------|-------|
| Problem | Noisy log-likelihood estimation (mystery ML model) |
| Input dimensions | 2 |
| Output dimensions | 1 |
| Objective | Maximise |
| Input range | [0, 1] |
| Output characteristics | Moderate range [-0.07, 0.67]; noisy; possible local optima |
| Initial samples | 10 |
| Total samples (Week 6) | 16 |
| Evaluation steps | 6 one-step-ahead predictions |

### Key Differences from F1

| Aspect | F1 | F2 |
|--------|----|----|
| Surrogate families | GP, BART | GP, BART, **Random Forest** |
| Output scale | Extreme (near machine epsilon) | Moderate (-0.07 to 0.67) |
| Log-transform needed | Yes (GP configs) | Likely not (outputs in normal range) |
| Total configurations | 20 (10 GP + 10 BART) | **30** (10 GP + 10 BART + 10 RF) |

## User Scenarios & Testing (F2)

### User Story F2-1 — Run GP Prequential Evaluation on F2 (Priority: P1)

As a student, I want to train a GP on the initial 10 F2 data points, then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage, so that I can assess GP predictive quality on F2.

**Independent Test**: Run GP section cells. Verify 6 predictions with metrics and 3-panel plots.

---

### User Story F2-2 — Optimise GP Hyperparameters on F2 (Priority: P1)

As a student, I want to evaluate 10 GP configurations for F2, so that I can identify the best-calibrated GP.

**Independent Test**: Run GP HP cells. Verify 10-row results DataFrame.

---

### User Story F2-3 — Run BART Prequential Evaluation on F2 (Priority: P1)

As a student, I want to train BART on the initial 10 F2 data points and perform the same evaluation, so that BART provides a second surrogate baseline.

**Independent Test**: Run BART section cells. Verify 6 predictions with metrics and plots.

---

### User Story F2-4 — Optimise BART Hyperparameters on F2 (Priority: P1)

As a student, I want to evaluate 10 BART configurations for F2.

**Independent Test**: Run BART HP cells. Verify 10-row results DataFrame.

---

### User Story F2-5 — Run Random Forest Prequential Evaluation on F2 (Priority: P1)

As a student, I want to train a Random Forest on the initial 10 F2 data points and perform one-step-ahead evaluation with uncertainty estimates, so that I have a third surrogate to compare.

**Why RF**: Random Forest is well-suited to F2's moderate output range and does not require Gaussian assumptions. Uncertainty is estimated via the variance across individual tree predictions.

**Independent Test**: Run RF section cells. Verify 6 predictions with MAE, NLP, Coverage, and plots.

---

### User Story F2-6 — Optimise Random Forest Hyperparameters on F2 (Priority: P1)

As a student, I want to evaluate 10 RF configurations (varying number of trees, max depth, min samples, bootstrap), so that I can find the best-calibrated RF.

**Independent Test**: Run RF HP cells. Verify 10-row results DataFrame.

---

### User Story F2-7 — Compare GP vs BART vs RF (Priority: P1)

As a student, I want a three-way comparison of the best GP, best BART, and best RF configurations, with bar charts and a ranked summary table of all 30 configurations, so that I can determine the best surrogate for F2.

**Independent Test**: Run comparison cells. Verify 3-way comparison table, bar charts, and full 30-row ranked table.

---

### Edge Cases (F2)

- RF uncertainty estimation: Tree-level variance can underestimate true uncertainty. NLP may be poor if std estimates are too narrow.
- RF with very few trees (e.g., 50): May produce high-variance uncertainty estimates.
- GP on F2 should not need log-transform (outputs are moderate), but include as a configuration option for robustness.

## Requirements (F2)

### Functional Requirements (F2)

- **FR-F2-001**: Notebook MUST define a `WEEK` variable (default `6`) and load data from `../../data/f2/updated_inputs - Week {WEEK}.npy` and `../../data/f2/updated_outputs - Week {WEEK}.npy`.
- **FR-F2-002**: Notebook MUST use the first 10 data points as the initial training set.
- **FR-F2-003**: Notebook MUST perform one-step-ahead prequential evaluation (same protocol as F1).
- **FR-F2-004**: Notebook MUST compute MAE, NLP, and Coverage of 95% prediction interval for each configuration.
- **FR-F2-005**: Notebook MUST evaluate GP surrogates using BoTorch `SingleTaskGP`.
- **FR-F2-006**: Notebook MUST evaluate BART surrogates using PyMC-BART.
- **FR-F2-007**: Notebook MUST evaluate RF surrogates using scikit-learn `RandomForestRegressor`, deriving uncertainty from individual tree predictions (mean +/- std of tree outputs).
- **FR-F2-008**: Notebook MUST optimise hyperparameters for each of the 3 surrogate families across 10 configurations (30 total).
- **FR-F2-009**: GP hyperparameters to vary: kernel type (Matern 5/2, RBF), output log-transform (yes/no), noise lower bound (1e-4, 1e-5, 1e-6).
- **FR-F2-010**: BART hyperparameters to vary: number of trees (10, 20, 50, 100, 200), MCMC draws (200, 500), burn-in/tune (100, 200).
- **FR-F2-011**: RF hyperparameters to vary: n_estimators (50, 100, 200, 500), max_depth (None, 5, 10), min_samples_leaf (1, 2, 5), bootstrap (True/False).
- **FR-F2-012**: Notebook MUST produce a 3-way comparison of best GP vs best BART vs best RF by NLP.
- **FR-F2-013**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty, absolute error, NLP per step, and bar-chart comparisons.
- **FR-F2-014**: Notebook MUST produce a full ranked results table of all 30 configurations sorted by NLP.
- **FR-F2-015**: Each code step MUST be clearly explained in markdown cells.
- **FR-F2-016**: Notebook MUST be stored at `functions/f2/preq-eval-f2.ipynb`.

### Key Entities (F2)

- **Random Forest Uncertainty**: Predicted mean = mean of individual tree predictions; predicted std = std of individual tree predictions. This enables NLP and coverage computation.

## Success Criteria (F2)

- **SC-F2-001**: Notebook executes end-to-end without errors.
- **SC-F2-002**: 6 one-step-ahead predictions for each of the 30 configurations.
- **SC-F2-003**: Final comparison table identifies the best surrogate for F2.
- **SC-F2-004**: All three metrics reported for every configuration.
- **SC-F2-005**: Visualisations clear, labelled, capstone-report ready.
- **SC-F2-006**: Code is simple with each step clearly explained.

## Technical Notes (F2)

### Random Forest Uncertainty Estimation

```python
# Each tree predicts independently
tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
mean = tree_preds.mean(axis=0)
std = tree_preds.std(axis=0)
# std used for NLP and 95% CI: mean +/- 1.96 * std
```

### Additional Library

| Library | Purpose | Install |
|---------|---------|---------|
| scikit-learn | Random Forest surrogate | Already available in sdd-dev environment |

### Data Flow (F2)

````
data/f2/updated_inputs - Week 6.npy  ──┐
data/f2/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 16 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              GP Evaluation       BART Evaluation      RF Evaluation
              (10 configs)        (10 configs)         (10 configs)
                    │                   │                   │
                    ▼                   ▼                   ▼
             GP Results DF        BART Results DF     RF Results DF
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                           3-way Comparison & Ranking
                                        │
                                        ▼
                              Tables + Visualisations
````
