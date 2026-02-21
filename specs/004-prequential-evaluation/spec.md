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

# Function 5: Prequential Evaluation — GP Hyperparameter Optimisation

## Overview (F5)

Create a new Jupyter notebook (`functions/f5/preq-eval-f5.ipynb`) that performs **prequential (one-step-ahead) evaluation** of Gaussian Process surrogate model hyperparameters for the chemical process yield optimisation problem (Function 5).

Unlike F1–F4 which compared multiple surrogate families, F5 focuses on **optimising GP hyperparameters only**, using a specific starting configuration and varying key hyperparameters across 10 configurations to identify the best GP setup for this 4D unimodal function.

The evaluation follows the same prequential protocol: train on the initial 20 data points, predict the next observation one step ahead, record the error, retrain, and repeat until all 26 available points have been processed (6 evaluation steps).

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
| Surrogate families | SF-GP, MF-GP, GBT | **GP only** (hyperparameter optimisation focus) |
| HP configs | 45 (15 per family) | **10** (GP configurations) |
| Output range | -32.6 to 0.5 | **0.11 to 3331.80** (very wide positive) |
| Function type | Many local optima | **Unimodal** |

## User Scenarios & Testing (F5)

### User Story F5-1 — Run GP Prequential with Starting Configuration (Priority: P1) 🎯 MVP

As a student, I want to train a GP on the initial 20 F5 data points using the specified starting configuration (Matérn 5/2, ARD, z-score standardisation, specific hyperparameter initialisation), then sequentially predict each of the 6 remaining observations one step ahead, recording MAE, NLP, and 95% coverage, so that I can establish a baseline GP performance for F5.

**Independent Test**: Run cells 1–13 of the notebook. Verify 6 predictions with metrics and 3-panel plots.

---

### User Story F5-2 — Optimise GP Hyperparameters (Priority: P1)

As a student, I want to evaluate 10 GP configurations varying kernel type, output transform, noise initialisation, and lengthscale initialisation, so that I can identify the GP configuration with the best predictive calibration for F5.

**Why HP optimisation**: The extremely wide output range (0.11 to 3331.80) and unimodal structure of F5 mean that the GP's output normalisation and noise handling are critical. The 10 configurations systematically explore these dimensions.

**Independent Test**: Run cells 14–17. Verify 10-row results DataFrame with MAE, NLP, and Coverage.

---

### User Story F5-3 — Compare and Select Best Configuration (Priority: P1)

As a student, I want a ranked summary of all 10 GP configurations with sensitivity visualisations, so that I can determine the best GP setup for F5 and understand which hyperparameters matter most.

**Independent Test**: Run cells 18–24. Verify sensitivity bar charts, ranked table, and conclusions.

---

### Edge Cases (F5)

- Very wide output range (0.11 to 3331.80): z-score standardisation is critical. Log-transform may also be important given the range spans orders of magnitude.
- Unimodal function: Matérn 5/2 (smooth) should be well-suited, but RBF may also work well.
- 20 initial points in 4D: Relatively well-sampled for a unimodal function. Lengthscale estimation should be feasible.
- GP fitting with very large output values: Without standardisation, the MLL optimisation may struggle. Initialising noise variance proportional to Var(y) helps.

## Requirements (F5)

### Functional Requirements (F5)

- **FR-F5-001**: Notebook MUST define a `WEEK` variable (default `6`) and load data from `../../data/f5/updated_inputs - Week {WEEK}.npy` and `../../data/f5/updated_outputs - Week {WEEK}.npy`.
- **FR-F5-002**: Notebook MUST use the first 20 data points as the initial training set.
- **FR-F5-003**: Notebook MUST perform one-step-ahead prequential evaluation (same protocol as F1–F4).
- **FR-F5-004**: Notebook MUST compute MAE, NLP, and Coverage of 95% prediction interval for each configuration.
- **FR-F5-005**: Notebook MUST evaluate GP surrogates using BoTorch `SingleTaskGP` with the specified starting configuration (Matérn 5/2, ARD, z-score standardisation).
- **FR-F5-006**: Notebook MUST initialise GP hyperparameters as specified: lengthscales 0.2–0.3, signal variance = Var(y), noise variance = 0.02–0.05 · Var(y), jitter = 1e-6.
- **FR-F5-007**: Notebook MUST use 10–20 random restarts of MLL optimisation (L-BFGS-B) for hyperparameter fitting.
- **FR-F5-008**: Notebook MUST optimise GP hyperparameters across 10 configurations.
- **FR-F5-009**: GP hyperparameters to vary: kernel type (Matérn 5/2, Matérn 3/2, RBF), output transform (z-score, log, raw), noise initialisation (0.02·Var(y), 0.05·Var(y)), lengthscale initialisation (0.2, 0.3).
- **FR-F5-010**: Notebook MUST produce a ranked results table of all 10 configurations sorted by NLP.
- **FR-F5-011**: Notebook MUST produce visualisations: predictions vs actuals with uncertainty, absolute error per step, NLP per step, and sensitivity bar charts.
- **FR-F5-012**: Each code step MUST be clearly explained in markdown cells.
- **FR-F5-013**: Notebook MUST be stored at `functions/f5/preq-eval-f5.ipynb`.

### Key Entities (F5)

- **Z-score standardisation**: Output transform $(y - \mu) / \sigma$ where $\mu$ and $\sigma$ are computed from the training set. Critical for F5's wide output range.
- **Lengthscale initialisation**: Starting values for the GP lengthscale parameters, set between 0.2–0.3 for inputs in [0,1]. Affects convergence of MLL optimisation.
- **Signal variance initialisation**: Starting value for the GP output scale, set to Var(y) to match the data scale.
- **Noise variance initialisation**: Starting value for observation noise, set to 2–5% of Var(y) to reflect that chemical yield measurements have some noise but the signal is strong.
- **Random restarts**: Multiple L-BFGS-B starts from different initial points to avoid poor local optima in MLL.

## Success Criteria (F5)

- **SC-F5-001**: Notebook executes end-to-end without errors.
- **SC-F5-002**: 6 one-step-ahead predictions for each of the 10 configurations.
- **SC-F5-003**: Ranked results table identifies the best GP configuration for F5.
- **SC-F5-004**: All three metrics (MAE, NLP, Coverage) reported for every configuration.
- **SC-F5-005**: Visualisations clear, labelled, capstone-report ready.
- **SC-F5-006**: Code is simple with each step clearly explained.
- **SC-F5-007**: GP hyperparameter initialisation matches the specified starting configuration.

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
                                        ▼
                              GP Evaluation
                              (10 configs)
                                        │
                                        ▼
                              GP Results DF
                                        │
                                        ▼
                           Ranking & Sensitivity Analysis
                                        │
                                        ▼
                              Tables + Visualisations
````
