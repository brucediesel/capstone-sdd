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

---

# Function 3: Prequential Evaluation — GP vs BART vs Random Forest

## Overview (F3)

Create a new Jupyter notebook (`functions/f3/preq-eval-f3.ipynb`) that performs **prequential (one-step-ahead) evaluation** of surrogate model predictive performance for the drug discovery problem (Function 3).

The evaluation follows the same prequential protocol as F1/F2: train on the initial 10 data points, predict the next observation one step ahead, record the error, retrain, and repeat until all 16 available points have been processed (6 evaluation steps).

**Three** surrogate families are compared:
1. **Gaussian Process** (via BoTorch/GPyTorch)
2. **BART — Bayesian Additive Regression Trees** (via PyMC-BART)
3. **Random Forest** (via scikit-learn)

For each surrogate family, hyperparameters are optimised over **15 configurations** (45 total), and the best configuration of each family is compared in a three-way comparison.

### Function 3 Context

| Property | Value |
|----------|-------|
| Problem | Drug discovery — minimise adverse reactions from 3 compounds |
| Input dimensions | 3 |
| Output dimensions | 1 |
| Objective | Maximise (transformed: negative of side effects) |
| Input range | [0, 1] |
| Output characteristics | Number of adverse reactions (transformed); moderate range |
| Initial samples | 10 |
| Total samples (Week 6) | 16 |
| Evaluation steps | 6 one-step-ahead predictions |

### Key Differences from F2

| Aspect | F2 | F3 |
|--------|----|----|
| Input dimensions | 2 | **3** |
| Problem domain | Log-likelihood estimation | **Drug discovery** |
| HP configs per family | 10 | **15** |
| Total configurations | 30 | **45** |
| GP kernels tested | Matérn 5/2, RBF | Matérn 5/2, **Matérn 3/2**, RBF |

## User Scenarios & Testing (F3)

### User Story F3-1 — Run GP Prequential Evaluation on F3 (Priority: P1)

As a student, I want to train a GP on the initial 10 F3 data points (3D compound concentrations), then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage, so that I can assess GP predictive quality on the 3D drug discovery problem.

**Independent Test**: Run GP section cells. Verify 6 predictions with metrics and 3-panel plots.

---

### User Story F3-2 — Optimise GP Hyperparameters on F3 (Priority: P1)

As a student, I want to evaluate 15 GP configurations for F3 (including Matérn 3/2 for potentially less-smooth drug response surfaces), so that I can identify the best-calibrated GP.

**Independent Test**: Run GP HP cells. Verify 15-row results DataFrame.

---

### User Story F3-3 — Run BART Prequential Evaluation on F3 (Priority: P1)

As a student, I want to train BART on the initial 10 F3 data points and perform the same evaluation, so that BART provides a second surrogate baseline that can naturally capture compound interactions.

**Independent Test**: Run BART section cells. Verify 6 predictions with metrics and plots.

---

### User Story F3-4 — Optimise BART Hyperparameters on F3 (Priority: P1)

As a student, I want to evaluate 15 BART configurations for F3 (including higher MCMC draws of 1000 for convergence assessment).

**Independent Test**: Run BART HP cells. Verify 15-row results DataFrame.

---

### User Story F3-5 — Run Random Forest Prequential Evaluation on F3 (Priority: P1)

As a student, I want to train a Random Forest on the initial 10 F3 data points and perform one-step-ahead evaluation with uncertainty estimates, so that I have a third surrogate to compare.

**Independent Test**: Run RF section cells. Verify 6 predictions with MAE, NLP, Coverage, and plots.

---

### User Story F3-6 — Optimise Random Forest Hyperparameters on F3 (Priority: P1)

As a student, I want to evaluate 15 RF configurations (including shallow trees and higher min_samples_leaf for the small-data 3D regime).

**Independent Test**: Run RF HP cells. Verify 15-row results DataFrame.

---

### User Story F3-7 — Compare GP vs BART vs RF on F3 (Priority: P1)

As a student, I want a three-way comparison of the best GP, best BART, and best RF configurations, with bar charts and a ranked summary table of all 45 configurations, so that I can determine the best surrogate for F3.

**Independent Test**: Run comparison cells. Verify 3-way comparison table, bar charts, and full 45-row ranked table.

---

### Edge Cases (F3)

- 3D input space with only 10 initial points: Curse of dimensionality may affect GP lengthscale estimation. ARD helps by learning per-dimension lengthscales.
- Drug response surfaces may have discontinuities or flat regions: Matérn 3/2 and tree-based models (BART, RF) may handle these better than smoother kernels.
- RF uncertainty with shallow trees (`max_depth=3`): May produce more conservative (wider) prediction intervals.

## Requirements (F3)

### Functional Requirements (F3)

- **FR-F3-001**: Notebook MUST define a `WEEK` variable (default `6`) and load data from `../../data/f3/updated_inputs - Week {WEEK}.npy` and `../../data/f3/updated_outputs - Week {WEEK}.npy`.
- **FR-F3-002**: Notebook MUST use the first 10 data points as the initial training set.
- **FR-F3-003**: Notebook MUST perform one-step-ahead prequential evaluation (same protocol as F1/F2).
- **FR-F3-004**: Notebook MUST compute MAE, NLP, and Coverage of 95% prediction interval for each configuration.
- **FR-F3-005**: Notebook MUST evaluate GP surrogates using BoTorch `SingleTaskGP`.
- **FR-F3-006**: Notebook MUST evaluate BART surrogates using PyMC-BART.
- **FR-F3-007**: Notebook MUST evaluate RF surrogates using scikit-learn `RandomForestRegressor`, deriving uncertainty from individual tree predictions.
- **FR-F3-008**: Notebook MUST optimise hyperparameters for each of the 3 surrogate families across **15 configurations** (45 total).
- **FR-F3-009**: GP hyperparameters to vary: kernel type (Matérn 5/2, Matérn 3/2, RBF), output log-transform (yes/no), noise lower bound (1e-4, 1e-5, 1e-6).
- **FR-F3-010**: BART hyperparameters to vary: number of trees (10, 20, 50, 100, 200), MCMC draws (200, 500, 1000), burn-in/tune (100, 200).
- **FR-F3-011**: RF hyperparameters to vary: n_estimators (50, 100, 200, 500), max_depth (None, 3, 5, 10), min_samples_leaf (1, 2, 3, 5), bootstrap (True/False).
- **FR-F3-012**: Notebook MUST produce a 3-way comparison of best GP vs best BART vs best RF by NLP.
- **FR-F3-013**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty, absolute error, NLP per step, and bar-chart comparisons.
- **FR-F3-014**: Notebook MUST produce a full ranked results table of all 45 configurations sorted by NLP.
- **FR-F3-015**: Each code step MUST be clearly explained in markdown cells.
- **FR-F3-016**: Notebook MUST be stored at `functions/f3/preq-eval-f3.ipynb`.

### Key Entities (F3)

- **ARD (Automatic Relevance Determination)**: Per-dimension lengthscales in GP kernels, particularly important for 3D input to learn which compounds matter most.
- **Matérn 3/2 kernel**: Less smooth than Matérn 5/2; included for F3 because drug response surfaces may not be twice-differentiable.

## Success Criteria (F3)

- **SC-F3-001**: Notebook executes end-to-end without errors.
- **SC-F3-002**: 6 one-step-ahead predictions for each of the 45 configurations (15 GP + 15 BART + 15 RF).
- **SC-F3-003**: Final comparison table identifies the best surrogate for F3.
- **SC-F3-004**: All three metrics reported for every configuration.
- **SC-F3-005**: Visualisations clear, labelled, capstone-report ready.
- **SC-F3-006**: Code is simple with each step clearly explained.

## Technical Notes (F3)

### Data Flow (F3)

````
data/f3/updated_inputs - Week 6.npy  ──┐
data/f3/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 16 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              GP Evaluation       BART Evaluation      RF Evaluation
              (15 configs)        (15 configs)         (15 configs)
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

---

# Function 4: Prequential Evaluation — Single Fidelity GP vs Multi Fidelity GP

## Overview (F4)

Create a new Jupyter notebook (`functions/f4/preq-eval-f4.ipynb`) that performs **prequential (one-step-ahead) evaluation** of surrogate model predictive performance for the warehouse product placement problem (Function 4).

The evaluation follows the same prequential protocol as F1–F3: train on the initial 30 data points, predict the next observation one step ahead, record the error, retrain, and repeat until all 36 available points have been processed (6 evaluation steps).

**Three** surrogate families are compared:
1. **Single Fidelity GP** (via BoTorch/GPyTorch `SingleTaskGP`)
2. **Multi Fidelity GP** (via BoTorch `SingleTaskMultiFidelityGP` with an autoregressive co-kriging approach)
3. **Gradient Boosted Trees (GBT)** (via scikit-learn `GradientBoostingRegressor`)

For each surrogate family, hyperparameters are optimised over **15 configurations** (45 total), and the best configuration of each family is compared in a three-way comparison.

### Function 4 Context

| Property | Value |
|----------|-------|
| Problem | Warehouse product placement — optimally placing products across warehouses |
| Input dimensions | 4 |
| Output dimensions | 1 |
| Objective | Maximise |
| Input range | [0, 1] |
| Output characteristics | Wide negative range (-32.63 to 0.53); mean ≈ -14.72; std ≈ 8.66; many local optima |
| Initial samples | 30 |
| Total samples (Week 6) | 36 |
| Evaluation steps | 6 one-step-ahead predictions |

### Key Differences from F3

| Aspect | F3 | F4 |
|--------|----|----|
| Input dimensions | 3 | **4** |
| Problem domain | Drug discovery | **Warehouse product placement** |
| Initial samples | 10 | **30** |
| Total samples | 16 | **36** |
| Surrogate families | GP, BART, RF | **Single Fidelity GP, Multi Fidelity GP, GBT** |
| Total configurations | 45 | **45** (15 SF-GP + 15 MF-GP + 15 GBT) |
| Output range | Moderate | **Wide negative (-32.6 to 0.5)** |

### Why Multi Fidelity GP for F4

F4 is a warehouse product placement problem with wide output range and many local optima. The Multi Fidelity GP (co-kriging / autoregressive model) treats the dataset as containing multiple fidelity levels, modelling the relationship:

$$f_{\text{high}}(x) = \rho \cdot f_{\text{low}}(x) + \delta(x)$$

For F4, we construct a synthetic fidelity dimension by splitting the training data: the initial 30 points form the "low fidelity" base, and the sequentially acquired points are treated as "high fidelity" observations. This allows the multi fidelity GP to share information across the fidelity levels while modelling systematic differences.

Even when only single-fidelity data is available, the MF-GP can be configured with a single fidelity level, acting as a regularised GP with the autoregressive kernel structure providing an inductive bias.

## User Scenarios & Testing (F4)

### User Story F4-1 — Run Single Fidelity GP Prequential Evaluation on F4 (Priority: P1)

As a student, I want to train a Single Fidelity GP on the initial 30 F4 data points (4D warehouse parameters), then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage, so that I can assess standard GP predictive quality on the 4D warehouse problem.

**Independent Test**: Run SF-GP section cells. Verify 6 predictions with metrics and 3-panel plots.

---

### User Story F4-2 — Optimise Single Fidelity GP Hyperparameters on F4 (Priority: P1)

As a student, I want to evaluate 15 SF-GP configurations for F4 (varying kernel type, output transform, and noise constraints for the wide-range outputs), so that I can identify the best-calibrated standard GP.

**Independent Test**: Run SF-GP HP cells. Verify 15-row results DataFrame.

---

### User Story F4-3 — Run Multi Fidelity GP Prequential Evaluation on F4 (Priority: P1)

As a student, I want to train a Multi Fidelity GP on the initial 30 F4 data points using the autoregressive co-kriging approach, then perform one-step-ahead evaluation, so that I can assess whether the MF-GP structure improves predictions.

**Why MF-GP**: The autoregressive kernel structure provides an inductive bias that may improve prediction quality when data points have varying information content (e.g., early vs later observations in the optimisation trajectory).

**Independent Test**: Run MF-GP section cells. Verify 6 predictions with metrics and plots.

---

### User Story F4-4 — Optimise Multi Fidelity GP Hyperparameters on F4 (Priority: P1)

As a student, I want to evaluate 15 MF-GP configurations (varying kernel type, fidelity kernel structure, noise bounds, and output normalisation), so that I can find the best-calibrated MF-GP.

**Independent Test**: Run MF-GP HP cells. Verify 15-row results DataFrame.

---

### User Story F4-5 — Run GBT Prequential Evaluation on F4 (Priority: P1)

As a student, I want to train a Gradient Boosted Trees model on the initial 30 F4 data points and perform one-step-ahead evaluation with uncertainty estimates, so that I have a non-GP surrogate to compare against both GP variants.

**Why GBT**: Gradient Boosted Trees are well-suited to F4's wide output range and 4D input space. GBT can model complex non-linear relationships and is robust to outliers. Uncertainty is estimated via quantile regression (fitting separate models for the 2.5th and 97.5th percentiles).

**Independent Test**: Run GBT section cells. Verify 6 predictions with MAE, NLP, Coverage, and plots.

---

### User Story F4-6 — Optimise GBT Hyperparameters on F4 (Priority: P1)

As a student, I want to evaluate 15 GBT configurations (varying number of estimators, learning rate, max depth, min samples leaf, and subsample fraction), so that I can find the best-calibrated GBT configuration.

**Independent Test**: Run GBT HP cells. Verify 15-row results DataFrame.

---

### User Story F4-7 — Compare SF-GP vs MF-GP vs GBT (Priority: P1)

As a student, I want a three-way comparison of the best SF-GP, best MF-GP, and best GBT configurations, with bar charts and a ranked summary table of all 45 configurations, so that I can determine which surrogate is best for F4.

**Independent Test**: Run comparison cells. Verify 3-way comparison table, bar charts, and full 45-row ranked table.

---

### Edge Cases (F4)

- Wide output range (-32.6 to 0.5): May benefit from output standardisation or log-transform. Include as a configuration option.
- 30 initial training points in 4D: Relatively well-sampled compared to F1–F3, but lengthscale estimation in 4D still requires ARD.
- MF-GP fidelity dimension: When all data is at the same fidelity level, the fidelity kernel may degenerate. Configurations should include fallback to a single-fidelity structure.
- GP fitting with wide output range: The marginal likelihood optimisation may converge to poor local optima. Include output normalisation as a configuration option.
- GBT uncertainty: Quantile regression provides asymmetric prediction intervals. NLP computation uses the average of upper and lower half-widths as the standard deviation estimate.
- GBT with few estimators (<50): May underfit the 4D problem. Include larger numbers as well.

## Requirements (F4)

### Functional Requirements (F4)

- **FR-F4-001**: Notebook MUST define a `WEEK` variable (default `6`) and load data from `../../data/f4/updated_inputs - Week {WEEK}.npy` and `../../data/f4/updated_outputs - Week {WEEK}.npy`.
- **FR-F4-002**: Notebook MUST use the first 30 data points as the initial training set.
- **FR-F4-003**: Notebook MUST perform one-step-ahead prequential evaluation (same protocol as F1–F3).
- **FR-F4-004**: Notebook MUST compute MAE, NLP, and Coverage of 95% prediction interval for each configuration.
- **FR-F4-005**: Notebook MUST evaluate Single Fidelity GP surrogates using BoTorch `SingleTaskGP`.
- **FR-F4-006**: Notebook MUST evaluate Multi Fidelity GP surrogates using BoTorch with an autoregressive/co-kriging kernel structure (e.g., `SingleTaskMultiFidelityGP` or custom multi-fidelity model).
- **FR-F4-007**: Notebook MUST optimise hyperparameters for each of the 3 surrogate families across **15 configurations** (45 total).
- **FR-F4-007b**: Notebook MUST evaluate GBT surrogates using scikit-learn `GradientBoostingRegressor`, deriving uncertainty from quantile regression (fitting separate models for upper/lower prediction intervals).
- **FR-F4-008**: SF-GP hyperparameters to vary: kernel type (Matérn 5/2, Matérn 3/2, RBF), output transform (raw, standardise, log-transform), noise lower bound (1e-4, 1e-5, 1e-6).
- **FR-F4-009**: MF-GP hyperparameters to vary: kernel type (Matérn 5/2, Matérn 3/2, RBF), fidelity kernel (linear, exponential decay), output transform (raw, standardise), noise lower bound (1e-4, 1e-5, 1e-6).
- **FR-F4-009b**: GBT hyperparameters to vary: n_estimators (50, 100, 200, 500), learning_rate (0.01, 0.05, 0.1, 0.2), max_depth (3, 4, 5, 6), min_samples_leaf (1, 2, 5), subsample (0.8, 1.0).
- **FR-F4-010**: Notebook MUST produce a 3-way comparison of best SF-GP vs best MF-GP vs best GBT by NLP.
- **FR-F4-011**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty, absolute error per step, NLP per step, and bar-chart comparisons.
- **FR-F4-012**: Notebook MUST produce a full ranked results table of all 45 configurations sorted by NLP.
- **FR-F4-013**: Each code step MUST be clearly explained in markdown cells.
- **FR-F4-014**: Notebook MUST be stored at `functions/f4/preq-eval-f4.ipynb`.

### Key Entities (F4)

- **Single Fidelity GP (SF-GP)**: Standard Gaussian Process via BoTorch `SingleTaskGP`. Treats all observations as having the same information quality.
- **Multi Fidelity GP (MF-GP)**: Gaussian Process with autoregressive/co-kriging kernel structure. Models correlations between different fidelity levels of observations. Implemented via BoTorch's multi-fidelity GP facilities or a custom autoregressive model.
- **Fidelity Dimension**: An additional input dimension indicating the fidelity level of each observation. For F4, we construct this synthetically from the data ordering.
- **ARD (Automatic Relevance Determination)**: Per-dimension lengthscales in GP kernels, important for 4D input to learn which warehouse parameters matter most.
- **Gradient Boosted Trees (GBT)**: An ensemble of weak decision tree learners trained sequentially, each correcting the errors of the previous. Uncertainty is estimated via quantile regression.

## Success Criteria (F4)

- **SC-F4-001**: Notebook executes end-to-end without errors.
- **SC-F4-002**: 6 one-step-ahead predictions for each of the 45 configurations (15 SF-GP + 15 MF-GP + 15 GBT).
- **SC-F4-003**: Final comparison table identifies the best surrogate for F4 across all three families.
- **SC-F4-004**: All three metrics reported for every configuration.
- **SC-F4-005**: Visualisations clear, labelled, capstone-report ready.
- **SC-F4-006**: Code is simple with each step clearly explained.

## Technical Notes (F4)

### Multi Fidelity GP Implementation

The MF-GP uses an autoregressive approach where a synthetic fidelity column is appended to the input features. In the simplest form, this assigns fidelity = 1.0 to all training data (single fidelity), and the `SingleTaskMultiFidelityGP` applies a specialised kernel that decomposes the covariance into a task-kernel (fidelity) component and a spatial kernel component.

For the prequential evaluation, we use a **simplified MF-GP** approach:
- Append a constant fidelity column (all 1.0) to the training inputs
- Use `SingleTaskMultiFidelityGP` which applies a `DownsamplingKernel` or linear fidelity kernel
- The MF-GP's specialised kernel structure provides a different inductive bias than standard `SingleTaskGP`

```python
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize, Normalize

# Append fidelity column
X_train_mf = torch.cat([X_train, torch.ones(n, 1, dtype=torch.float64)], dim=-1)
X_test_mf = torch.cat([X_test, torch.ones(1, 1, dtype=torch.float64)], dim=-1)

# Build MF-GP (fidelity column is the last dimension)
model = SingleTaskMultiFidelityGP(
    X_train_mf, y_train,
    data_fidelities=[X_train_mf.shape[-1] - 1]  # last column is fidelity
)
```

### Additional Library

| Library | Purpose | Install |
|---------|---------|---------|
| BoTorch (multi-fidelity) | `SingleTaskMultiFidelityGP`, fidelity kernels | Already available in sdd-dev environment |

### Data Flow (F4)

````
data/f4/updated_inputs - Week 6.npy  ──┐
data/f4/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 36 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              SF-GP Evaluation    MF-GP Evaluation     GBT Evaluation
              (15 configs)        (15 configs)         (15 configs)
                    │                   │                   │
                    ▼                   ▼                   ▼
             SF-GP Results DF    MF-GP Results DF     GBT Results DF
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                           3-way Comparison & Ranking
                                        │
                                        ▼
                              Tables + Visualisations
````

---

# Function 5: Prequential Evaluation — GP, GBT & MFGP Comparison

## Overview (F5)

Create / extend the Jupyter notebook (`functions/f5/preq-eval-f5.ipynb`) that performs **prequential (one-step-ahead) evaluation** of surrogate model predictive performance for the chemical process yield optimisation problem (Function 5).

Three surrogate families are compared:
1. **Gaussian Process (GP)** — 15 configurations varying kernel, output transform, noise, and lengthscale
2. **Gradient Boosted Trees (GBT)** — 15 configurations varying n_estimators, learning rate, max depth, min_samples_leaf, and subsample
3. **Multi-Fidelity GP (MFGP)** — 15 configurations varying Matérn smoothness, fidelity kernel type, output transform, and noise lower bound

The evaluation follows the same prequential protocol: train on the initial 20 data points, predict the next observation one step ahead, record the error, retrain, and repeat until all 26 available points have been processed (6 evaluation steps). A total of **45 configurations** (15 per family) are evaluated and ranked.

### Function 5 Context

| Property | Value |
|----------|-------|
| Problem | Chemical process yield optimisation — maximise yield from 4 chemical inputs |
| Input dimensions | 4 |
| Output dimensions | 1 |
| Objective | Maximise |
| Input range | [0, 1] |
| Output characteristics | Typically unimodal; wide range (0.11 to 3331.80); mean ≈ 558.78; std ≈ 902.73 |
| Initial samples | 20 |
| Total samples (Week 6) | 26 |
| Evaluation steps | 6 one-step-ahead predictions |

### Starting GP Configuration

The GP starting configuration is specifically tailored for F5's characteristics:

- **Kernel**: Matérn 5/2 with ARD (smooth chemical response surface)
- **Mean function**: Constant prior
- **Likelihood**: Gaussian
- **Output transformation**: Standardise yields (z-score) — critical given the wide output range (0.11 to 3331.80)
- **Hyperparameter priors / initialisation**:
  - Lengthscales (ℓ₁..ℓ₄): Initialise around 0.2–0.3 (inputs scaled to [0,1])
  - Signal variance σ²_f: Initialised to Var(y)
  - Noise variance σ²_n: Initialised to (0.02–0.05)·Var(y)
  - Jitter: 1e-6
  - Optimisation: 10–20 random restarts of MLL (L-BFGS-B)

### Key Differences from F4

| Aspect | F4 | F5 |
|--------|----|----|
| Input dimensions | 4 | 4 |
| Problem domain | Warehouse placement | **Chemical process yield** |
| Initial samples | 30 | **20** |
| Total samples | 36 | **26** |
| Surrogate families | SF-GP, MF-GP, GBT | **GP, GBT, MFGP** |
| HP configs | 45 (15 per family) | **45** (15 per family) |
| Output range | -32.6 to 0.5 | **0.11 to 3331.80** (very wide positive) |
| Function type | Many local optima | **Unimodal** |

## User Scenarios & Testing (F5)

### User Story F5-1 — Run GP Prequential with Starting Configuration (Priority: P1) 🎯 MVP

As a student, I want to train a GP on the initial 20 F5 data points using the specified starting configuration (Matérn 5/2, ARD, z-score standardisation, specific hyperparameter initialisation), then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage, so that I can establish a baseline GP performance for F5.

**Independent Test**: Run cells 1–13 of the notebook. Verify 6 predictions with metrics and 3-panel plots.

---

### User Story F5-2 — Optimise GP Hyperparameters (Priority: P1)

As a student, I want to evaluate 15 GP configurations varying kernel type, output transform, noise initialisation, and lengthscale initialisation, so that I can identify the GP configuration with the best predictive calibration for F5.

**Why HP optimisation**: The extremely wide output range (0.11 to 3331.80) and unimodal structure of F5 mean that the GP's output normalisation and noise handling are critical. The 15 configurations systematically explore these dimensions.

**Independent Test**: Run cells 14–17. Verify 15-row results DataFrame with MAE, NLP, and Coverage.

---

### User Story F5-4 — Run GBT Prequential Evaluation (Priority: P1)

As a student, I want to evaluate 15 Gradient Boosted Tree configurations varying n_estimators, learning rate, max depth, min_samples_leaf, and subsample, so that I can assess whether GBT provides competitive predictive performance compared to GP on F5's wide-range chemical yield data.

**Why GBT**: GBT is a non-parametric ensemble method that doesn't assume a smooth function. For wide-range outputs, GBT's quantile regression can provide well-calibrated uncertainty estimates.

**Independent Test**: Run GBT cells. Verify 15-row results DataFrame with MAE, NLP, and Coverage.

---

### User Story F5-5 — Run MFGP Prequential Evaluation (Priority: P1)

As a student, I want to evaluate 15 Multi-Fidelity GP configurations varying Matérn smoothness (nu), fidelity kernel type (linear truncated vs exponential decay), output transform, and noise lower bound, so that I can assess MFGP as an alternative GP-based surrogate for F5.

**Why MFGP**: MFGP can capture additional structure through its fidelity kernel. For a unimodal 4D function with wide output range, the fidelity kernel and output standardisation choices may significantly impact predictive quality.

**Independent Test**: Run MFGP cells. Verify 15-row results DataFrame with MAE, NLP, and Coverage.

---

### User Story F5-3 — Compare and Select Best Configuration (Priority: P1)

As a student, I want a side-by-side comparison of the best GP, best GBT, and best MFGP configurations with bar charts and a ranked summary table of all 45 configurations, so that I can determine which surrogate family is best for F5 and understand which hyperparameters matter most.

**Independent Test**: Run cells 18–24. Verify sensitivity bar charts, ranked table, and conclusions.

---

### Edge Cases (F5)

- Very wide output range (0.11 to 3331.80): z-score standardisation is critical for GP/MFGP. Log-transform may also be important given the range spans orders of magnitude.
- Unimodal function: Matérn 5/2 (smooth) should be well-suited for GP, but RBF may also work well.
- 20 initial points in 4D: Relatively well-sampled for a unimodal function. Lengthscale estimation should be feasible.
- GP fitting with very large output values: Without standardisation, the MLL optimisation may struggle.
- GBT with small data: Only 20–25 training points — GBT may overfit with deep trees or large ensembles. min_samples_leaf regularisation critical.
- MFGP with constant fidelity: All data is at the same fidelity level (1.0). The fidelity kernel may not add useful information, potentially making MFGP equivalent to or worse than standard GP.
- GBT quantile uncertainty: GBT uncertainty comes from quantile regression, which may be poorly calibrated with small data.

## Requirements (F5)

### Functional Requirements (F5)

- **FR-F5-001**: Notebook MUST define a `WEEK` variable (default `6`) and load data from `../../data/f5/updated_inputs - Week {WEEK}.npy` and `../../data/f5/updated_outputs - Week {WEEK}.npy`.
- **FR-F5-002**: Notebook MUST use the first 20 data points as the initial training set.
- **FR-F5-003**: Notebook MUST perform one-step-ahead prequential evaluation (same protocol as F1–F4).
- **FR-F5-004**: Notebook MUST compute MAE, NLP, and Coverage of 95% prediction interval for each configuration.
- **FR-F5-005**: Notebook MUST evaluate GP surrogates using BoTorch `SingleTaskGP` with the specified starting configuration (Matérn 5/2, ARD, z-score standardisation).
- **FR-F5-006**: Notebook MUST initialise GP hyperparameters as specified: lengthscales 0.2–0.3, signal variance = Var(y), noise variance = 0.02–0.05 · Var(y), jitter = 1e-6.
- **FR-F5-007**: Notebook MUST use 10–20 random restarts of MLL optimisation (L-BFGS-B) for hyperparameter fitting.
- **FR-F5-008**: Notebook MUST optimise GP hyperparameters across 15 configurations.
- **FR-F5-009**: GP hyperparameters to vary: kernel type (Matérn 5/2, Matérn 3/2, RBF), output transform (z-score, log, raw), noise initialisation (0.02·Var(y), 0.05·Var(y)), lengthscale initialisation (0.2, 0.3).
- **FR-F5-010**: Notebook MUST produce a ranked results table of all 45 configurations sorted by NLP.
- **FR-F5-011**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty, absolute error per step, NLP per step, and sensitivity bar charts.
- **FR-F5-012**: Each code step MUST be clearly explained in markdown cells.
- **FR-F5-013**: Notebook MUST be stored at `functions/f5/preq-eval-f5.ipynb`.
- **FR-F5-014**: Notebook MUST evaluate GBT (Gradient Boosted Trees) surrogates using `sklearn.ensemble.GradientBoostingRegressor` with quantile regression for uncertainty. 15 GBT configurations varying n_estimators, learning_rate, max_depth, min_samples_leaf, subsample.
- **FR-F5-015**: Notebook MUST evaluate MFGP (Multi-Fidelity GP) surrogates using BoTorch `SingleTaskMultiFidelityGP`. 15 MFGP configurations varying nu (2.5, 1.5), linear_truncated (True/False), output_transform (raw, standardise), noise_lb.
- **FR-F5-016**: Notebook MUST produce a 3-way comparison table of the best GP vs best GBT vs best MFGP configurations, identifying the metric winner for MAE, NLP, and Coverage.
- **FR-F5-017**: Notebook MUST produce sensitivity bar charts for all 45 configurations colour-coded by model family (GP blue, MFGP pink, GBT green).

### Key Entities (F5)

- **Z-score standardisation**: Output transform $(y - \mu) / \sigma$ where $\mu$ and $\sigma$ are computed from the training set. Critical for F5's wide output range.
- **Lengthscale initialisation**: Starting values for the GP lengthscale parameters, set between 0.2–0.3 for inputs in [0,1]. Affects convergence of MLL optimisation.
- **Signal variance initialisation**: Starting value for the GP output scale, set to Var(y) to match the data scale.
- **Noise variance initialisation**: Starting value for observation noise, set to 2–5% of Var(y) to reflect that chemical yield measurements have some noise but the signal is strong.
- **Random restarts**: Multiple L-BFGS-B starts from different initial points to avoid poor local optima in MLL.

## Success Criteria (F5)

- **SC-F5-001**: Notebook executes end-to-end without errors.
- **SC-F5-002**: 6 one-step-ahead predictions for each of the 45 configurations (15 GP + 15 GBT + 15 MFGP = 270 total predictions).
- **SC-F5-003**: Ranked results table identifies the best configuration overall and per family for F5.
- **SC-F5-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration.
- **SC-F5-005**: Visualisations clear, labelled, capstone-report ready.
- **SC-F5-006**: Code is simple with each step clearly explained.
- **SC-F5-007**: GP hyperparameter initialisation matches the specified starting configuration.
- **SC-F5-008**: 3-way comparison table shows best GP vs best GBT vs best MFGP with metric winners.
- **SC-F5-009**: Sensitivity bar charts colour-coded by model family (45 bars).

## Technical Notes (F5)

### GP Hyperparameter Initialisation

```python
# Initialise lengthscales
model.covar_module.base_kernel.lengthscale = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

# Initialise outputscale (signal variance)
model.covar_module.outputscale = torch.tensor(y_train_var)

# Initialise noise
model.likelihood.noise = torch.tensor(0.03 * y_train_var)

# Add jitter for numerical stability
gpytorch.settings.cholesky_jitter(float=1e-6, double=1e-6)
```

### Data Flow (F5)

````
data/f5/updated_inputs - Week 6.npy  ──┐
data/f5/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 26 points
                                        │
                        ┌───────────────┼────────────────┐
                        ▼               ▼                ▼
                  GP Evaluation   GBT Evaluation   MFGP Evaluation
                  (15 configs)    (15 configs)     (15 configs)
                        │               │                │
                        ▼               ▼                ▼
                   GP Results      GBT Results     MFGP Results
                        │               │                │
                        └───────┬───────┘────────────────┘
                                ▼
                     3-Way Best-of-Family Comparison
                                │
                                ▼
                   Ranking & Sensitivity Analysis (45 configs)
                                │
                                ▼
                      Tables + Visualisations
````

---

# Function 6: Prequential Evaluation — NN, SFGP & MFGP Comparison

## Overview (F6)

Evaluate three surrogate families — **Neural Network (NN)**, **Single Fidelity GP (SFGP)**, and **Multi-Fidelity GP (MFGP)** — across **135 total hyperparameter configurations** for Function 6, a 5D cake-recipe optimisation problem. The goal is to identify the best surrogate model and hyperparameters that produce the most accurate and well-calibrated predictions (lowest NLP) for use in the Bayesian Optimisation pipeline.

1. **Neural Network (NN)** — 45 configurations varying hidden layers, nodes per layer, and learning rate
2. **Single Fidelity GP (SFGP)** — 40 configurations varying kernel type, output transform, and noise floor
3. **Multi-Fidelity GP (MFGP)** — 50 configurations varying Matérn smoothness, fidelity kernel type, output transform, and noise lower bound

### Function 6 Context

| Property | Value |
|----------|-------|
| **Function** | F6 — Cake recipe scoring (5 ingredient inputs) |
| **Input dimensions** | 5 (flour, sugar, eggs, butter, milk) |
| **Output dimensions** | 1 (composite score, negative) |
| **Objective** | Maximise (bring score closest to 0) |
| **Input range** | [0, 1] |
| **Output range** | [−2.571, −0.219] |
| **Output mean** | −1.316 |
| **Output std** | 0.546 |
| **Data source** | `data/f6/updated_inputs - Week 6.npy`, `data/f6/updated_outputs - Week 6.npy` |
| **Total samples** | 26 (Week 6) |
| **Initial training** | N_INIT = 20 |
| **Evaluation steps** | 6 one-step-ahead predictions |

### Starting NN Configuration

Based on the existing F6 notebook (Week 5/6):

| Hyperparameter | Starting Value | Notes |
|----------------|---------------|-------|
| **Hidden layers** | 2 | Mid-range architecture |
| **Nodes per layer** | 5 | Small architecture appropriate for 26 samples |
| **Activation** | ReLU | Standard for regression |
| **Dropout** | 0.2 | Fixed — required for MC Dropout uncertainty |
| **Learning rate** | 0.01 | Adam optimiser |
| **Epochs** | 500 | Loss convergence monitored |
| **MC samples** | 50 | Forward passes for uncertainty estimation |
| **Input normalisation** | z-score | $(x - \bar{x}) / \sigma_x$ |
| **Output normalisation** | z-score | $(y - \bar{y}) / \sigma_y$ |

### NN Hyperparameter Search Space (45 Configs)

The 45 configurations are generated from a structured grid over three axes:

| Axis | Values | Count |
|------|--------|-------|
| **Hidden layers** | 1, 2, 3 | 3 |
| **Nodes per layer** | 4, 5, 6 | 3 |
| **Learning rate** | 0.001, 0.005, 0.01, 0.05, 0.1 | 5 |

Full grid: layers ∈ {1, 2, 3} × nodes ∈ {4, 5, 6} × lr ∈ {0.001, 0.005, 0.01, 0.05, 0.1} = **45 configs**.

**Rationale**: Layers 1–3 test shallow to moderately deep architectures; nodes 4–6 are appropriate for the small dataset (26 samples, 5D input). Wider networks (16, 32, 64 nodes) are excluded to avoid overfitting risk.

**Fixed hyperparameters** (not varied):
- Activation: ReLU
- Dropout: 0.2 (required for MC Dropout)
- Epochs: 500
- MC samples: 50
- Input/output normalisation: z-score

### Key Differences from F5

| Aspect | F5 | F6 |
|--------|----|----|
| Surrogate families | GP, GBT, MFGP (3 families × 15) | **NN (45) + SFGP (40) + MFGP (50) → 3-way comparison** |
| Total configs | 45 | **135** (45 NN + 40 SFGP + 50 MFGP) |
| HP axes | Kernel, transform, noise, lengthscale | NN: layers, nodes, lr · SFGP: kernel, transform, noise · MFGP: nu, fidelity kernel, transform, noise |
| Uncertainty method | Analytic (GP/MFGP), Quantile (GBT) | MC Dropout (NN), Analytic (SFGP/MFGP) |
| Input dims | 4 | 5 (+ 1 fidelity column for MFGP = 6D) |
| Output nature | Wide range (0.11–3332) | Narrow negative range (−2.57–−0.22) |

## User Scenarios & Testing (F6)

### User Story F6-1: NN Default Evaluation
**As a** data scientist evaluating surrogate performance for F6,
**I want** to run prequential evaluation with the starting NN config (2 layers, 5 nodes, lr=0.01),
**so that** I have a baseline to compare against.

### User Story F6-2: NN Hyperparameter Optimisation
**As a** data scientist tuning NN architectures,
**I want** to evaluate 45 NN configurations varying layers (1–3), nodes (4–6), and learning rate,
**so that** I can identify the most accurate and well-calibrated NN architecture for F6.

### User Story F6-3: Sensitivity & Ranking
**As a** researcher analysing the hyperparameter landscape,
**I want** to see a ranked table and sensitivity visualisation for all configurations across all three families,
**so that** I understand which hyperparameters and surrogate families matter most for F6 performance.

### User Story F6-4: MFGP Hyperparameter Optimisation (Priority: P1)

**As a** data scientist evaluating alternative surrogate families for F6,
**I want** to evaluate 50 Multi-Fidelity GP configurations varying kernel type, smoothness, output transform, and noise floor,
**so that** I can identify the best MFGP configuration and compare it fairly against the other surrogates.

**Why this priority**: MFGP is already implemented and provides a strong GP-based baseline for the 3-way comparison.

**Independent Test**: Can be fully tested by running the 50-config MFGP evaluation loop and verifying a 50-row results table with MAE, NLP, and Coverage columns.

**Acceptance Scenarios**:

1. **Given** F6 data (26 × 5 inputs, 26 outputs) and N_INIT = 20, **When** the MFGP evaluation loop runs 50 configurations, **Then** a 50-row DataFrame `mfgp_hp_df` is produced with columns label, MAE, NLP, Coverage_95
2. **Given** 50 evaluated MFGP configs, **When** selecting the best by NLP, **Then** the configuration with the lowest NLP is identified and displayed

### User Story F6-7: SFGP Hyperparameter Optimisation (Priority: P1)

**As a** data scientist comparing standard GP against multi-fidelity GP and neural network surrogates,
**I want** to evaluate 40 Single Fidelity GP configurations varying kernel type, output transform, and noise floor,
**so that** I can assess whether standard GP is competitive with NN and MFGP on the 5D cake-recipe problem.

**Why this priority**: SFGP provides the classical GP baseline — essential for understanding whether the added complexity of MFGP or NN is justified.

**Independent Test**: Can be fully tested by running the 40-config SFGP evaluation loop and verifying a 40-row results table with MAE, NLP, and Coverage columns.

**Acceptance Scenarios**:

1. **Given** F6 data (26 × 5 inputs, 26 outputs) and N_INIT = 20, **When** the SFGP evaluation loop runs 40 configurations, **Then** a 40-row DataFrame `sfgp_hp_df` is produced with columns label, MAE, NLP, Coverage_95
2. **Given** 40 evaluated SFGP configs, **When** selecting the best by NLP, **Then** the configuration with the lowest NLP is identified and displayed

### User Story F6-5: 3-Way Comparison — NN vs SFGP vs MFGP (Priority: P1)

**As a** researcher deciding which surrogate family to use for F6 Bayesian Optimisation,
**I want** to see a direct 3-way comparison of the best NN, best SFGP, and best MFGP configurations,
**so that** I can make an informed decision about which model to deploy.

**Why this priority**: The head-to-head comparison is the core deliverable that informs the surrogate selection decision.

**Independent Test**: Can be fully tested by verifying a side-by-side table and bar chart comparing the best of each family on all three metrics.

**Acceptance Scenarios**:

1. **Given** best NN, best SFGP, and best MFGP results, **When** displayed in a comparison table, **Then** all three models appear side-by-side with MAE, NLP, and Coverage_95 values and metric-by-metric winners identified
2. **Given** all 135 configs (45 NN + 40 SFGP + 50 MFGP), **When** ranked together by NLP, **Then** a combined ranked table and sensitivity charts show all three families with distinct colours (NN = orange, SFGP = blue, MFGP = pink)

### User Story F6-6: Best Model Visualisation (Priority: P2)

**As a** researcher validating the winning surrogate,
**I want** to see the prequential evaluation plot (predictions vs actuals, error bands, NLP per step) for the overall best model,
**so that** I can visually confirm its prediction quality across all evaluation steps.

**Why this priority**: Visualisation validates the quantitative metrics and builds confidence in the recommendation.

**Independent Test**: Can be fully tested by verifying a 3-panel plot (predictions, error, NLP) is produced for the overall winner.

**Acceptance Scenarios**:

1. **Given** the overall best model (lowest NLP across all 135 configs), **When** its prequential results are plotted, **Then** a 3-panel visualisation (predictions vs actuals with 95% CI, absolute error, NLP contribution per step) is displayed

## Requirements (F6)

### Functional Requirements

| ID | Requirement | Stories |
|----|-------------|---------|
| FR-F6-001 | Load F6 data from `data/f6/updated_inputs - Week 6.npy` and outputs | F6-1 |
| FR-F6-002 | Set `N_INIT = 20`, compute 6 evaluation steps | F6-1 |
| FR-F6-003 | Implement `compute_metrics()` returning MAE, NLP, 95% Coverage | F6-1 |
| FR-F6-004 | Implement `nn_prequential_with_config()` building a configurable NN (layers, nodes, lr) | F6-2 |
| FR-F6-005 | NN uses MC Dropout (50 forward passes) for uncertainty estimation | F6-1, F6-2 |
| FR-F6-006 | NN normalises inputs and outputs via z-score before training | F6-1, F6-2 |
| FR-F6-007 | NN trains with Adam optimiser for 500 epochs with MSELoss | F6-1, F6-2 |
| FR-F6-008 | Define 45 NN configurations: layers {1,2,3} × nodes {4,5,6} × lr {0.001,0.005,0.01,0.05,0.1} | F6-2 |
| FR-F6-009 | Evaluate all 45 NN configs via prequential loop, store results in `nn_hp_df` | F6-2 |
| FR-F6-010 | Select best NN configuration by NLP (primary) and MAE (secondary) | F6-2 |
| FR-F6-011 | Display horizontal sensitivity bar chart for all configs (NLP, MAE, Coverage) | F6-3 |
| FR-F6-012 | Display full ranked table of all configs sorted by NLP | F6-3 |
| FR-F6-013 | Visualise default config prequential results (predictions vs actuals, error, NLP per step) | F6-1 |
| FR-F6-014 | Import BoTorch/GPyTorch dependencies (SingleTaskGP, SingleTaskMultiFidelityGP, fit_gpytorch_mll, GaussianLikelihood, GreaterThan, kernels) | F6-4, F6-7 |
| FR-F6-015 | Implement `mfgp_prequential_with_config()` — appends fidelity=1.0 column to 5D inputs, builds SingleTaskMultiFidelityGP, returns metrics | F6-4 |
| FR-F6-016 | Define 50 MFGP configurations varying nu (2.5, 1.5, 0.5), linear_truncated (True/False), output_transform (raw/standardise), and noise_lb (1e-4, 1e-5, 1e-6, 1e-7) | F6-4 |
| FR-F6-017 | Evaluate all 50 MFGP configs via prequential loop, store results in `mfgp_hp_df` | F6-4 |
| FR-F6-018 | Select best MFGP configuration by NLP (primary) and MAE (secondary) | F6-4 |
| FR-F6-019 | Display 3-way comparison table: best NN vs best SFGP vs best MFGP with MAE, NLP, Coverage_95 | F6-5 |
| FR-F6-020 | Display comparison bar chart: best NN vs best SFGP vs best MFGP across all three metrics | F6-5 |
| FR-F6-021 | Update sensitivity charts to show all 135 configs (NN = orange #FF9800, SFGP = blue #2196F3, MFGP = pink #E91E63) | F6-5 |
| FR-F6-022 | Update ranked table to include all 135 configs sorted by NLP with Family column | F6-5 |
| FR-F6-023 | Visualise prequential results for the overall best model — 3-panel plot (predictions vs actuals with 95% CI, absolute error, NLP per step) | F6-6 |
| FR-F6-024 | Implement `sfgp_prequential_with_config()` — builds SingleTaskGP with configurable kernel (Matérn 2.5, 1.5, 0.5, RBF), output transform, noise constraint | F6-7 |
| FR-F6-025 | Define 40 SFGP configurations: kernel {Matérn 2.5, 1.5, 0.5, RBF} × transform {raw, standardise} × noise_lb {1e-4, 1e-5, 1e-6, 1e-7, 1e-8} | F6-7 |
| FR-F6-026 | Evaluate all 40 SFGP configs via prequential loop, store results in `sfgp_hp_df` | F6-7 |
| FR-F6-027 | Select best SFGP configuration by NLP (primary) and MAE (secondary) | F6-7 |

### Key Entities (F6)

- **NN Config**: dict with `n_layers`, `n_nodes`, `lr`, `label`
- **SFGP Config**: dict with `kernel_type`, `output_transform`, `noise_lb`, `label`
- **MFGP Config**: dict with `nu`, `linear_truncated`, `output_transform`, `noise_lb`, `label`
- **Prequential Result**: dict with `predictions`, `actuals`, `stds`, `metrics`
- **nn_hp_df**: DataFrame with columns `label`, `MAE`, `NLP`, `Coverage_95` (45 rows)
- **sfgp_hp_df**: DataFrame with columns `label`, `MAE`, `NLP`, `Coverage_95` (40 rows)
- **mfgp_hp_df**: DataFrame with columns `label`, `MAE`, `NLP`, `Coverage_95` (50 rows)
- **combined_df**: Merged DataFrame of all 135 configs with a `Family` column (NN / SFGP / MFGP)

## Success Criteria (F6)

| ID | Criterion |
|----|-----------|
| SC-F6-001 | All cells execute without errors |
| SC-F6-002 | 135 configs × 6 predictions each = 810 total predictions (270 NN + 240 SFGP + 300 MFGP) |
| SC-F6-003 | Ranked results table identifies the best configuration across all three families |
| SC-F6-004 | All three metrics (MAE, NLP, Coverage) reported for every configuration |
| SC-F6-005 | Visualisations clear and labelled: NN in orange (#FF9800), SFGP in blue (#2196F3), MFGP in pink (#E91E63) |
| SC-F6-006 | Code simple, each step explained in markdown |
| SC-F6-007 | NN configurations use layers {1,2,3} × nodes {4,5,6} as specified |
| SC-F6-008 | 50 MFGP configs evaluated with SingleTaskMultiFidelityGP (fidelity column appended) |
| SC-F6-009 | 3-way comparison table and chart show best NN vs best SFGP vs best MFGP side-by-side |
| SC-F6-010 | Best overall model visualised with 3-panel prequential plot |
| SC-F6-011 | Combined sensitivity charts show all 135 configs with distinct family colours |
| SC-F6-012 | 40 SFGP configs evaluated with SingleTaskGP (ARD for 5D, configurable kernel/noise) |

## Technical Notes (F6)

### NN Architecture (Configurable)

```python
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_nodes, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev_dim, n_nodes), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = n_nodes
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### MC Dropout Uncertainty

With model in `train()` mode (dropout active), run 50 forward passes per test point:
- `mean_pred` = mean of 50 predictions (denormalised)
- `std_pred` = std of 50 predictions (denormalised)
- Floor std at `max(std, 1e-10)` for numerical stability

### Edge Cases

- **Very small networks** (1 layer, 4 nodes): May underfit, producing high MAE but possibly reasonable NLP
- **3 hidden layers with 4–6 nodes**: Moderately deep for 20 training points; dropout mitigates overfitting
- **Learning rate extremes**: lr=0.1 may cause training instability; lr=0.001 may not converge in 500 epochs
- **Narrow output range**: F6 outputs span ~2.35 units — z-score normalisation is appropriate
- **SFGP with Matérn 0.5**: Very rough kernel; may struggle if the underlying function is smooth
- **SFGP noise floor too low** (1e-8): May cause numerical instability during Cholesky decomposition
- **SFGP with negative outputs**: Z-score standardisation handles negative outputs; log-transform not applicable
- **MFGP with nu=0.5**: Very rough kernel; may struggle with smooth underlying function
- **MFGP noise floor too low** (1e-7): May cause numerical instability during Cholesky decomposition
- **MFGP standardised output + negative values**: Z-score normalisation handles negative outputs correctly since it centres on the mean

### SFGP Architecture

Single Fidelity Gaussian Process using BoTorch's `SingleTaskGP`:
- Standard GP with 5D inputs (no fidelity column)
- ARD (Automatic Relevance Determination) lengthscales for all 5 dimensions
- Configurable base kernel: Matérn 2.5, Matérn 1.5, Matérn 0.5, or RBF
- Output transform options: raw or standardise (z-score)
- Noise constraint via `GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))`
- Hyperparameters fitted via `fit_gpytorch_mll` (L-BFGS-B optimisation of marginal log-likelihood)

### SFGP Hyperparameter Search Space (40 configs)

| Axis | Values | Count |
|------|--------|-------|
| Kernel type | Matérn 2.5, Matérn 1.5, Matérn 0.5, RBF | 4 |
| Output transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 | 5 |

Full grid: 4 × 2 × 5 = **40 configs**

### MFGP Architecture

Multi-Fidelity Gaussian Process using BoTorch's `SingleTaskMultiFidelityGP`:
- Appends a constant fidelity column (1.0) to the 5D input → 6D training tensor
- `data_fidelities=[5]` (last column index) tells the model which column is fidelity
- Supports `LinearTruncatedFidelityKernel` or `ExponentialDecayFidelityKernel`
- Matérn base kernel with configurable smoothness `nu` (0.5, 1.5, or 2.5)
- Noise constraint via `GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))`
- Hyperparameters fitted via `fit_gpytorch_mll` (L-BFGS-B optimisation of marginal log-likelihood)

### MFGP Hyperparameter Search Space (50 configs)

| Axis | Values | Count |
|------|--------|-------|
| nu (Matérn smoothness) | 0.5, 1.5, 2.5 | 3 |
| linear_truncated (fidelity kernel) | True, False | 2 |
| output_transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7 | 4 |

Full grid: 3 × 2 × 2 × 4 = **48 configs** + 2 additional configs with nu=2.5, noise_lb=5e-5 (LinTrunc raw, ExpDecay standardise) = **50 total**

### Data Flow (F6)

````
data/f6/updated_inputs - Week 6.npy  ──┐
data/f6/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 26 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            NN Evaluation       SFGP Evaluation     MFGP Evaluation
            (45 configs)        (40 configs)        (50 configs)
                    │                   │                   │
                    ▼                   ▼                   ▼
            nn_hp_df            sfgp_hp_df          mfgp_hp_df
                    │                   │                   │
                    └─────────┬─────────┴───────────────────┘
                              ▼
                    3-Way Comparison
                    (Best NN vs Best SFGP vs Best MFGP)
                              │
                              ▼
                Combined Ranking & Sensitivity
                        (135 configs)
                              │
                              ▼
                Best Model Visualisation
                        (Overall winner)
                              │
                              ▼
                    Tables + Visualisations
````

---

# Function 7: Prequential Evaluation — NN, SFGP & MFGP Comparison

## Overview (F7)

Evaluate three surrogate families — **Neural Network (NN)**, **Single Fidelity GP (SFGP)**, and **Multi-Fidelity GP (MFGP)** — across **135 total hyperparameter configurations** for Function 7, a 6D ML hyperparameter tuning problem. The goal is to identify the best surrogate model and hyperparameters that produce the most accurate and well-calibrated predictions (lowest NLP) for use in the Bayesian Optimisation pipeline.

This notebook follows the identical evaluation strategy established in F6:

1. **Neural Network (NN)** — 45 configurations varying hidden layers, nodes per layer, and learning rate
2. **Single Fidelity GP (SFGP)** — 40 configurations varying kernel type, output transform, and noise floor
3. **Multi-Fidelity GP (MFGP)** — 50 configurations varying Matérn smoothness, fidelity kernel type, output transform, and noise lower bound

### Function 7 Context

| Property | Value |
|----------|-------|
| **Function** | F7 — ML model hyperparameter tuning (6 hyperparameter inputs) |
| **Input dimensions** | 6 (learning rate, regularisation strength, hidden layers, etc.) |
| **Output dimensions** | 1 (model performance score, e.g. accuracy or F1) |
| **Objective** | Maximise |
| **Input range** | [0, 1] |
| **Output range** | [0.003, 2.305] |
| **Output mean** | 0.387 |
| **Output std** | 0.530 |
| **Output var** | 0.281 |
| **Data source** | `data/f7/updated_inputs - Week 6.npy`, `data/f7/updated_outputs - Week 6.npy` |
| **Total samples** | 36 (Week 6) |
| **Initial training** | N_INIT = 30 |
| **Evaluation steps** | 6 one-step-ahead predictions |

### Key Differences from F6

| Aspect | F6 | F7 |
|--------|----|----|
| Input dimensions | 5 (cake ingredients) | **6** (ML hyperparameters) |
| Output range | [−2.571, −0.219] (negative) | **[0.003, 2.305]** (positive) |
| Total samples | 26 | **36** |
| N_INIT | 20 | **30** |
| MFGP input dims | 5 + 1 fidelity = 6D | **6 + 1 fidelity = 7D** |
| SFGP ARD dims | 5 | **6** |

### NN Hyperparameter Search Space (45 Configs)

Same structured grid as F6:

| Axis | Values | Count |
|------|--------|-------|
| **Hidden layers** | 1, 2, 3 | 3 |
| **Nodes per layer** | 4, 5, 6 | 3 |
| **Learning rate** | 0.001, 0.005, 0.01, 0.05, 0.1 | 5 |

Full grid: 3 × 3 × 5 = **45 configs**

**Rationale**: Layers 1–3 test shallow to moderately deep architectures. Nodes 4–6 are appropriate for the dataset size (36 samples). The additional input dimension (6D vs 5D) does not require wider layers given the 30-point training set.

**Fixed hyperparameters** (not varied):
- Activation: ReLU
- Dropout: 0.2 (required for MC Dropout)
- Epochs: 500
- MC samples: 50
- Input/output normalisation: z-score

### SFGP Hyperparameter Search Space (40 Configs)

Same structured grid as F6:

| Axis | Values | Count |
|------|--------|-------|
| Kernel type | Matérn 2.5, Matérn 1.5, Matérn 0.5, RBF | 4 |
| Output transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 | 5 |

Full grid: 4 × 2 × 5 = **40 configs**

SFGP uses ARD with 6 lengthscale parameters (one per input dimension).

### MFGP Hyperparameter Search Space (50 Configs)

Same structured grid as F6:

| Axis | Values | Count |
|------|--------|-------|
| nu (Matérn smoothness) | 0.5, 1.5, 2.5 | 3 |
| linear_truncated (fidelity kernel) | True, False | 2 |
| output_transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7 | 4 |

Full grid: 3 × 2 × 2 × 4 = 48 + 2 additional configs = **50 total**

MFGP appends a fidelity=1.0 column to the 6D input → 7D training tensor. `data_fidelities=[6]` (last column index).

## User Scenarios & Testing (F7)

### User Story F7-1: NN Default Evaluation
**As a** data scientist evaluating surrogate performance for F7,
**I want** to run prequential evaluation with the starting NN config (2 layers, 5 nodes, lr=0.01),
**so that** I have a baseline to compare against.

### User Story F7-2: NN Hyperparameter Optimisation
**As a** data scientist tuning NN architectures,
**I want** to evaluate 45 NN configurations varying layers (1–3), nodes (4–6), and learning rate,
**so that** I can identify the most accurate and well-calibrated NN architecture for F7.

### User Story F7-3: SFGP Hyperparameter Optimisation
**As a** data scientist comparing standard GP against other surrogates,
**I want** to evaluate 40 SFGP configurations varying kernel type, output transform, and noise floor,
**so that** I can assess whether standard GP is competitive with NN and MFGP on the 6D problem.

### User Story F7-4: MFGP Hyperparameter Optimisation
**As a** data scientist evaluating alternative surrogate families,
**I want** to evaluate 50 MFGP configurations varying kernel type, smoothness, output transform, and noise floor,
**so that** I can identify the best MFGP configuration.

### User Story F7-5: 3-Way Comparison — NN vs SFGP vs MFGP
**As a** researcher deciding which surrogate family to use for F7 Bayesian Optimisation,
**I want** to see a direct 3-way comparison of the best NN, best SFGP, and best MFGP configurations,
**so that** I can make an informed decision about which model to deploy.

**Acceptance Scenarios**:

1. **Given** best NN, best SFGP, and best MFGP results, **When** displayed in a comparison table, **Then** all three models appear side-by-side with MAE, NLP, and Coverage_95 values and metric-by-metric winners identified
2. **Given** all 135 configs, **When** ranked together by NLP, **Then** a combined ranked table and sensitivity charts show all three families with distinct colours (NN = orange, SFGP = blue, MFGP = pink)

### User Story F7-6: Best Model Visualisation
**As a** researcher validating the winning surrogate,
**I want** to see the prequential evaluation plot for the overall best model,
**so that** I can visually confirm its prediction quality across all 6 evaluation steps.

### User Story F7-7: Sensitivity & Ranking
**As a** researcher analysing the hyperparameter landscape,
**I want** to see a ranked table and sensitivity visualisation for all 135 configurations,
**so that** I understand which hyperparameters and surrogate families matter most for F7.

### Edge Cases (F7)

- **Positive output range**: F7 outputs are in [0.003, 2.305] — log transform could be considered but z-score is used for consistency with F6
- **Near-zero outputs**: Some outputs are very close to zero (0.003), which may cause NLP instability if predicted std is very small
- **6D input space**: One additional dimension over F6 increases the curse of dimensionality; ARD lengthscales help identify relevant dimensions
- **30 initial training points**: More training data than F6 (20) which may favour GP models that benefit from larger training sets

## Requirements (F7)

### Functional Requirements

| ID | Requirement | Stories |
|----|-------------|---------|
| FR-F7-001 | Load F7 data from `data/f7/updated_inputs - Week 6.npy` and outputs | F7-1 |
| FR-F7-002 | Set `N_INIT = 30`, compute 6 evaluation steps | F7-1 |
| FR-F7-003 | Implement `compute_metrics()` returning MAE, NLP, 95% Coverage | F7-1 |
| FR-F7-004 | Implement `nn_prequential_with_config()` for configurable NN (layers, nodes, lr) | F7-2 |
| FR-F7-005 | NN uses MC Dropout (50 forward passes) for uncertainty estimation | F7-1, F7-2 |
| FR-F7-006 | NN normalises inputs and outputs via z-score | F7-1, F7-2 |
| FR-F7-007 | NN trains with Adam optimiser for 500 epochs with MSELoss | F7-1, F7-2 |
| FR-F7-008 | Define 45 NN configurations: layers {1,2,3} × nodes {4,5,6} × lr {0.001,0.005,0.01,0.05,0.1} | F7-2 |
| FR-F7-009 | Evaluate all 45 NN configs via prequential loop, store results in `nn_hp_df` | F7-2 |
| FR-F7-010 | Select best NN configuration by NLP (primary) and MAE (secondary) | F7-2 |
| FR-F7-011 | Implement `sfgp_prequential_with_config()` — builds SingleTaskGP with configurable kernel, ARD for 6D | F7-3 |
| FR-F7-012 | Define 40 SFGP configurations: kernel {Matérn 2.5, 1.5, 0.5, RBF} × transform {raw, standardise} × noise_lb {1e-4..1e-8} | F7-3 |
| FR-F7-013 | Evaluate all 40 SFGP configs via prequential loop, store results in `sfgp_hp_df` | F7-3 |
| FR-F7-014 | Select best SFGP configuration by NLP | F7-3 |
| FR-F7-015 | Implement `mfgp_prequential_with_config()` — appends fidelity=1.0 to 6D inputs → 7D, data_fidelities=[6] | F7-4 |
| FR-F7-016 | Define 50 MFGP configurations varying nu, linear_truncated, output_transform, noise_lb | F7-4 |
| FR-F7-017 | Evaluate all 50 MFGP configs via prequential loop, store results in `mfgp_hp_df` | F7-4 |
| FR-F7-018 | Select best MFGP configuration by NLP | F7-4 |
| FR-F7-019 | Display 3-way comparison table: best NN vs best SFGP vs best MFGP | F7-5 |
| FR-F7-020 | Display comparison bar chart across all three metrics | F7-5 |
| FR-F7-021 | Sensitivity charts for all 135 configs (NN=orange, SFGP=blue, MFGP=pink) | F7-5, F7-7 |
| FR-F7-022 | Full ranked table of all 135 configs sorted by NLP | F7-7 |
| FR-F7-023 | Visualise prequential results for overall best model — 3-panel plot | F7-6 |
| FR-F7-024 | Import BoTorch/GPyTorch dependencies (SingleTaskGP, SingleTaskMultiFidelityGP, kernels) | F7-3, F7-4 |
| FR-F7-025 | Implement `plot_prequential_results()` — 3-panel viz (predictions vs actuals, error, NLP per step) | F7-1, F7-6 |
| FR-F7-026 | Display default NN prequential results (2L×5N, lr=0.01) with 3-panel plot | F7-1 |
| FR-F7-027 | Conclusions section summarising 3-way comparison findings | F7-5 |

### Key Entities (F7)

- **NN Config**: dict with `n_layers`, `n_nodes`, `lr`, `label`
- **SFGP Config**: dict with `kernel_type`, `output_transform`, `noise_lb`, `label`
- **MFGP Config**: dict with `nu`, `linear_truncated`, `output_transform`, `noise_lb`, `label`
- **Prequential Result**: dict with `predictions`, `actuals`, `stds`, `metrics`
- **nn_hp_df**: DataFrame — 45 rows, columns: label, MAE, NLP, Coverage_95
- **sfgp_hp_df**: DataFrame — 40 rows, columns: label, MAE, NLP, Coverage_95
- **mfgp_hp_df**: DataFrame — 50 rows, columns: label, MAE, NLP, Coverage_95

## Success Criteria (F7)

| ID | Criterion |
|----|-----------|
| SC-F7-001 | All cells execute without errors |
| SC-F7-002 | 135 configs × 6 predictions each = 810 total predictions (270 NN + 240 SFGP + 300 MFGP) |
| SC-F7-003 | Ranked results table identifies the best configuration across all three families |
| SC-F7-004 | All three metrics (MAE, NLP, Coverage) reported for every configuration |
| SC-F7-005 | Visualisations clear and labelled: NN=orange (#FF9800), SFGP=blue (#2196F3), MFGP=pink (#E91E63) |
| SC-F7-006 | Code simple, each step explained in markdown |
| SC-F7-007 | NN configurations use layers {1,2,3} × nodes {4,5,6} |
| SC-F7-008 | 50 MFGP configs evaluated with fidelity column (data_fidelities=[6]) |
| SC-F7-009 | 3-way comparison table and chart present |
| SC-F7-010 | Best overall model visualised with 3-panel prequential plot |
| SC-F7-011 | Combined sensitivity charts show all 135 configs with distinct colours |
| SC-F7-012 | 40 SFGP configs evaluated with ARD for 6D |

## Technical Notes (F7)

### Data Flow (F7)

````
data/f7/updated_inputs - Week 6.npy  ──┐
data/f7/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 36 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            NN Evaluation       SFGP Evaluation     MFGP Evaluation
            (45 configs)        (40 configs)        (50 configs)
                    │                   │                   │
                    ▼                   ▼                   ▼
            nn_hp_df            sfgp_hp_df          mfgp_hp_df
                    │                   │                   │
                    └─────────┬─────────┴───────────────────┘
                              ▼
                    3-Way Comparison
                    (Best NN vs Best SFGP vs Best MFGP)
                              │
                              ▼
                Combined Ranking & Sensitivity
                        (135 configs)
                              │
                              ▼
                Best Model Visualisation
                        (Overall winner)
                              │
                              ▼
                    Tables + Visualisations
````

### F7-Specific Considerations

- **Positive output range** [0.003, 2.305]: Z-score normalisation is appropriate. Log-transform is theoretically possible but not used for consistency with F6.
- **6D SFGP**: `ard_num_dims=6` — one ARD lengthscale per ML hyperparameter dimension
- **7D MFGP**: 6 input dims + 1 fidelity column; `data_fidelities=[6]` (0-indexed last column)
- **30 training points**: More data than F6 (20 points), which benefits GP models via better covariance estimation
- **Near-zero outputs**: The minimum output (0.003) is near zero but positive; no special handling needed beyond z-score

---

# Function 8: Prequential Evaluation — NN, SFGP & MFGP Comparison

## Overview (F8)

Evaluate three surrogate families — **Neural Network (NN)**, **Single Fidelity GP (SFGP)**, and **Multi-Fidelity GP (MFGP)** — across **135 total hyperparameter configurations** for Function 8, an 8D high-dimensional black-box optimisation problem. The goal is to identify the best surrogate model and hyperparameters that produce the most accurate and well-calibrated predictions (lowest NLP) for use in the Bayesian Optimisation pipeline.

This notebook follows the identical evaluation strategy established in F6 and F7:

1. **Neural Network (NN)** — 45 configurations varying hidden layers, nodes per layer, and learning rate
2. **Single Fidelity GP (SFGP)** — 40 configurations varying kernel type, output transform, and noise floor
3. **Multi-Fidelity GP (MFGP)** — 50 configurations varying Matérn smoothness, fidelity kernel type, output transform, and noise lower bound

### Function 8 Context

| Property | Value |
|----------|-------|
| **Function** | F8 — High-dimensional black-box optimisation (8 input parameters) |
| **Input dimensions** | 8 |
| **Output dimensions** | 1 |
| **Objective** | Maximise |
| **Input range** | [0, 1] |
| **Output range** | [5.592, 9.953] |
| **Output mean** | 8.066 |
| **Output std** | 1.098 |
| **Output var** | 1.206 |
| **Data source** | `data/f8/updated_inputs - Week 6.npy`, `data/f8/updated_outputs - Week 6.npy` |
| **Total samples** | 46 (Week 6) |
| **Initial training** | N_INIT = 40 |
| **Evaluation steps** | 6 one-step-ahead predictions |

### Key Differences from F6 and F7

| Aspect | F6 | F7 | F8 |
|--------|----|----|-----|
| Input dims | 5 | 6 | **8** |
| Output range | [−2.57, −0.22] | [0.003, 2.305] | **[5.59, 9.95]** |
| Total samples | 26 | 36 | **46** |
| N_INIT | 20 | 30 | **40** |
| MFGP dims | 5+1=6D | 6+1=7D | **8+1=9D** |
| SFGP ARD | 5 | 6 | **8** |

### NN Hyperparameter Search Space (45 Configs)

Same structured grid as F6/F7:

| Axis | Values | Count |
|------|--------|-------|
| **Hidden layers** | 1, 2, 3 | 3 |
| **Nodes per layer** | 4, 5, 6 | 3 |
| **Learning rate** | 0.001, 0.005, 0.01, 0.05, 0.1 | 5 |

Full grid: 3 × 3 × 5 = **45 configs**

**Rationale**: With 40 training points and 8D input, the small node counts (4–6) help prevent overfitting. The network's first layer maps 8→4/5/6 nodes, providing implicit dimensionality reduction.

**Fixed hyperparameters** (not varied):
- Activation: ReLU
- Dropout: 0.2 (required for MC Dropout)
- Epochs: 500
- MC samples: 50
- Input/output normalisation: z-score

### SFGP Hyperparameter Search Space (40 Configs)

Same structured grid as F6/F7:

| Axis | Values | Count |
|------|--------|-------|
| Kernel type | Matérn 2.5, Matérn 1.5, Matérn 0.5, RBF | 4 |
| Output transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 | 5 |

Full grid: 4 × 2 × 5 = **40 configs**

SFGP uses ARD with 8 lengthscale parameters (one per input dimension).

### MFGP Hyperparameter Search Space (50 Configs)

Same structured grid as F6/F7:

| Axis | Values | Count |
|------|--------|-------|
| nu (Matérn smoothness) | 0.5, 1.5, 2.5 | 3 |
| linear_truncated (fidelity kernel) | True, False | 2 |
| output_transform | raw, standardise | 2 |
| noise_lb (noise floor) | 1e-4, 1e-5, 1e-6, 1e-7 | 4 |

Full grid: 3 × 2 × 2 × 4 = 48 + 2 additional configs = **50 total**

MFGP appends a fidelity=1.0 column to the 8D input → 9D training tensor. `data_fidelities=[8]` (last column index).

## User Scenarios & Testing (F8)

### User Story F8-1: NN Default Evaluation
**As a** data scientist evaluating surrogate performance for F8,
**I want** to run prequential evaluation with the starting NN config (2 layers, 5 nodes, lr=0.01),
**so that** I have a baseline to compare against.

### User Story F8-2: NN Hyperparameter Optimisation
**As a** data scientist tuning NN architectures,
**I want** to evaluate 45 NN configurations varying layers (1–3), nodes (4–6), and learning rate,
**so that** I can identify the most accurate and well-calibrated NN architecture for F8.

### User Story F8-3: SFGP Hyperparameter Optimisation
**As a** data scientist comparing standard GP against other surrogates,
**I want** to evaluate 40 SFGP configurations varying kernel type, output transform, and noise floor,
**so that** I can assess whether standard GP is competitive with NN and MFGP on the 8D problem.

### User Story F8-4: MFGP Hyperparameter Optimisation
**As a** data scientist evaluating alternative surrogate families,
**I want** to evaluate 50 MFGP configurations varying kernel type, smoothness, output transform, and noise floor,
**so that** I can identify the best MFGP configuration for the high-dimensional F8 problem.

### User Story F8-5: 3-Way Comparison — NN vs SFGP vs MFGP
**As a** researcher deciding which surrogate family to use for F8 Bayesian Optimisation,
**I want** to see a direct 3-way comparison of the best NN, best SFGP, and best MFGP configurations,
**so that** I can make an informed decision about which model to deploy.

**Acceptance Scenarios**:

1. **Given** best NN, best SFGP, and best MFGP results, **When** displayed in a comparison table, **Then** all three models appear side-by-side with MAE, NLP, and Coverage_95 values and metric-by-metric winners identified
2. **Given** all 135 configs, **When** ranked together by NLP, **Then** a combined ranked table and sensitivity charts show all three families with distinct colours (NN = orange, SFGP = blue, MFGP = pink)

### User Story F8-6: Best Model Visualisation
**As a** researcher validating the winning surrogate,
**I want** to see the prequential evaluation plot for the overall best model,
**so that** I can visually confirm its prediction quality across all 6 evaluation steps.

### User Story F8-7: Sensitivity & Ranking
**As a** researcher analysing the hyperparameter landscape,
**I want** to see a ranked table and sensitivity visualisation for all 135 configurations,
**so that** I understand which hyperparameters and surrogate families matter most for F8.

### Edge Cases (F8)

- **8D input space**: Highest dimensionality in the project; GPs may struggle with the curse of dimensionality, potentially favouring NN
- **Large training set (40 points)**: Benefits GP models via better covariance estimation; NN also benefits from more data
- **High positive output range** [5.59, 9.95]: Z-score normalisation centres around mean ≈ 8.07 — no issues expected
- **Moderate output variance** (std=1.098): Well-behaved range that should not cause numerical issues for any surrogate
- **9D MFGP**: The fidelity column adds a 9th dimension; LinearTruncatedFidelityKernel may struggle at this dimensionality

## Requirements (F8)

### Functional Requirements

| ID | Requirement | Stories |
|----|-------------|---------|
| FR-F8-001 | Load F8 data from `data/f8/updated_inputs - Week 6.npy` and outputs | F8-1 |
| FR-F8-002 | Set `N_INIT = 40`, compute 6 evaluation steps | F8-1 |
| FR-F8-003 | Implement `compute_metrics()` returning MAE, NLP, 95% Coverage | F8-1 |
| FR-F8-004 | Implement `nn_prequential_with_config()` for configurable NN (layers, nodes, lr) | F8-2 |
| FR-F8-005 | NN uses MC Dropout (50 forward passes) for uncertainty estimation | F8-1, F8-2 |
| FR-F8-006 | NN normalises inputs and outputs via z-score | F8-1, F8-2 |
| FR-F8-007 | NN trains with Adam optimiser for 500 epochs with MSELoss | F8-1, F8-2 |
| FR-F8-008 | Define 45 NN configurations: layers {1,2,3} × nodes {4,5,6} × lr {0.001,0.005,0.01,0.05,0.1} | F8-2 |
| FR-F8-009 | Evaluate all 45 NN configs via prequential loop, store results in `nn_hp_df` | F8-2 |
| FR-F8-010 | Select best NN configuration by NLP (primary) and MAE (secondary) | F8-2 |
| FR-F8-011 | Implement `sfgp_prequential_with_config()` — builds SingleTaskGP with configurable kernel, ARD for 8D | F8-3 |
| FR-F8-012 | Define 40 SFGP configurations: kernel {Matérn 2.5, 1.5, 0.5, RBF} × transform {raw, standardise} × noise_lb {1e-4..1e-8} | F8-3 |
| FR-F8-013 | Evaluate all 40 SFGP configs via prequential loop, store results in `sfgp_hp_df` | F8-3 |
| FR-F8-014 | Select best SFGP configuration by NLP | F8-3 |
| FR-F8-015 | Implement `mfgp_prequential_with_config()` — appends fidelity=1.0 to 8D inputs → 9D, data_fidelities=[8] | F8-4 |
| FR-F8-016 | Define 50 MFGP configurations varying nu, linear_truncated, output_transform, noise_lb | F8-4 |
| FR-F8-017 | Evaluate all 50 MFGP configs via prequential loop, store results in `mfgp_hp_df` | F8-4 |
| FR-F8-018 | Select best MFGP configuration by NLP | F8-4 |
| FR-F8-019 | Display 3-way comparison table: best NN vs best SFGP vs best MFGP | F8-5 |
| FR-F8-020 | Display comparison bar chart across all three metrics | F8-5 |
| FR-F8-021 | Sensitivity charts for all 135 configs (NN=orange, SFGP=blue, MFGP=pink) | F8-5, F8-7 |
| FR-F8-022 | Full ranked table of all 135 configs sorted by NLP | F8-7 |
| FR-F8-023 | Visualise prequential results for overall best model — 3-panel plot | F8-6 |
| FR-F8-024 | Import BoTorch/GPyTorch dependencies (SingleTaskGP, SingleTaskMultiFidelityGP, kernels) | F8-3, F8-4 |
| FR-F8-025 | Implement `plot_prequential_results()` — 3-panel viz | F8-1, F8-6 |
| FR-F8-026 | Display default NN prequential results (2L×5N, lr=0.01) with 3-panel plot | F8-1 |
| FR-F8-027 | Conclusions section summarising 3-way comparison findings | F8-5 |

### Key Entities (F8)

- **NN Config**: dict with `n_layers`, `n_nodes`, `lr`, `label`
- **SFGP Config**: dict with `kernel_type`, `output_transform`, `noise_lb`, `label`
- **MFGP Config**: dict with `nu`, `linear_truncated`, `output_transform`, `noise_lb`, `label`
- **Prequential Result**: dict with `predictions`, `actuals`, `stds`, `metrics`
- **nn_hp_df**: DataFrame — 45 rows, columns: label, MAE, NLP, Coverage_95
- **sfgp_hp_df**: DataFrame — 40 rows, columns: label, MAE, NLP, Coverage_95
- **mfgp_hp_df**: DataFrame — 50 rows, columns: label, MAE, NLP, Coverage_95

## Success Criteria (F8)

| ID | Criterion |
|----|-----------|
| SC-F8-001 | All cells execute without errors |
| SC-F8-002 | 135 configs × 6 predictions each = 810 total predictions (270 NN + 240 SFGP + 300 MFGP) |
| SC-F8-003 | Ranked results table identifies the best configuration across all three families |
| SC-F8-004 | All three metrics (MAE, NLP, Coverage) reported for every configuration |
| SC-F8-005 | Visualisations clear and labelled: NN=orange (#FF9800), SFGP=blue (#2196F3), MFGP=pink (#E91E63) |
| SC-F8-006 | Code simple, each step explained in markdown |
| SC-F8-007 | NN configurations use layers {1,2,3} × nodes {4,5,6} |
| SC-F8-008 | 50 MFGP configs evaluated with fidelity column (data_fidelities=[8]) |
| SC-F8-009 | 3-way comparison table and chart present |
| SC-F8-010 | Best overall model visualised with 3-panel prequential plot |
| SC-F8-011 | Combined sensitivity charts show all 135 configs with distinct colours |
| SC-F8-012 | 40 SFGP configs evaluated with ARD for 8D |

## Technical Notes (F8)

### Data Flow (F8)

````
data/f8/updated_inputs - Week 6.npy  ──┐
data/f8/updated_outputs - Week 6.npy ──┤
                                        ▼
                              Load all 46 points
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            NN Evaluation       SFGP Evaluation     MFGP Evaluation
            (45 configs)        (40 configs)        (50 configs)
                    │                   │                   │
                    ▼                   ▼                   ▼
            nn_hp_df            sfgp_hp_df          mfgp_hp_df
                    │                   │                   │
                    └─────────┬─────────┴───────────────────┘
                              ▼
                    3-Way Comparison
                    (Best NN vs Best SFGP vs Best MFGP)
                              │
                              ▼
                Combined Ranking & Sensitivity
                        (135 configs)
                              │
                              ▼
                Best Model Visualisation
                        (Overall winner)
                              │
                              ▼
                    Tables + Visualisations
````

### F8-Specific Considerations

- **8D input space**: Highest dimensionality across all functions; GPs face the curse of dimensionality with 8 ARD parameters to estimate from 40 training points
- **9D MFGP**: 8 input + 1 fidelity → 9D is at the practical limit for GP-based models; may favour NN which handles higher dims naturally
- **Large output values** [5.59, 9.95]: Z-score normalisation centres around mean ≈ 8.07 with std ≈ 1.1 — numerically well-behaved
- **40 training points in 8D**: The samples-to-dimension ratio (40/8 = 5) is relatively low for GPs; NN's implicit feature extraction may provide advantage
- **NN dimensionality reduction**: The first layer (8→4/5/6 nodes) naturally compresses the 8D input, which can help with curse of dimensionality
