# Feature Specification: F1-F8 Week 9 -- Bayesian Optimisation Iteration with Enhanced Visualisation

**Feature Branch**: `021-f1-f8-week9`  
**Created**: 2026-03-02  
**Status**: Draft  
**Input**: User description: "Process week 9 outputs for F1-F8, keep same strategy as week 8, update the visualisation of all functions to show last 8 sample points in a different colour so that I can visualise how the proposed sample points are clustered through the iterations"

## Per-Function Strategy Summary

Each function retains its Week 8 surrogate and acquisition strategy unchanged. The only change across all functions is an enhanced visualisation showing all weekly submission points in a distinct colour to reveal clustering patterns.

| Function | Dims | Week 9 Samples | Initial Samples | Surrogate | Acquisition | Interior Penalty |
|----------|------|-----------------|-----------------|-----------|-------------|------------------|
| F1 | 2 | 19 | 10 | Hurdle Model (Classifier + RF Regressor) | Weighted UCB + Local Penalization | Yes (S=0.1, F=0.01) |
| F2 | 2 | 19 | 10 | SFGP (Matern 1.5, ARD, noise >= 1e-3) | qLogNEI (10 restarts, 512 raw) | No |
| F3 | 3 | 24 | 15 | SFGP (Matern 2.5, ARD, z-score) | qLogNEI (10 restarts, 512 raw) | No |
| F4 | 4 | 39 | 30 | MFGP (Matern 2.5, ARD, LinTrunc, noise >= 1e-4) | Cost-aware MF-qNEI (q=4, 64 fantasies) | No |
| F5 | 4 | 29 | 20 | GP (Matern 2.5, ARD, log1p transform) | qLogNEI (q=4, 3000 Sobol -> 50 L-BFGS) | Yes (S=1.0, F=0.01) |
| F6 | 5 | 29 | 20 | SFGP (Matern 1.5, ARD, noise >= 1e-2) | qLogNEI (q=4, 512 QMC, 3000 raw -> 50 restarts) | Yes (S=1.0, F=0.01) |
| F7 | 6 | 39 | 30 | Neural Network (6->5->5->1, dropout=0.1, lr=0.005) | MC Dropout EI (50 samples, 20000 candidates) | Yes (S=0.1, F=0.01) |
| F8 | 8 | 49 | 40 | SFGP (Matern 2.5, ARD, noise >= 1e-7) | qEI (q=1, 256 MC, 30 restarts, 4096 raw) | No |

> **Visualisation Enhancement**: All functions now show the weekly submission points (all points beyond the initial set) in a distinct colour to reveal whether proposals are clustering in a sub-region or exploring broadly.

## Clarifications

### Session 2026-03-02

- Q: Each function has 9 weekly submissions. Should only the last 8 be highlighted, or all 9? → A: All 9 submissions (indices N_initial to N_total-1).
- Q: What quantitative signal defines "stalling"? → A: No new best observed value in the last 3+ consecutive submissions OR total improvement over all submissions < 5% relative to initial best.
- Q: Should performance evaluation be code cells or markdown-only? → A: Code cells compute metrics & plots, then a markdown cell interprets results and proposes strategy changes.
- Q: What metrics should the performance evaluation code cells compute? → A: Convergence metrics (best-value trajectory, per-submission improvement, stalling flag) + exploration spread analysis (mean pairwise distance, clustering metric) + model quality (leave-one-out prediction error on submissions).

## User Scenarios & Testing *(mandatory)*

### User Story 1 -- Load and Validate Week 9 Data for All Functions (Priority: P1)

As a student running the weekly Bayesian Optimisation loop for Functions 1-8, I want each notebook to load the corresponding Week 9 updated inputs and outputs so that all evaluated sample points are available for surrogate modelling.

**Why this priority**: Without current data, no optimisation can proceed. This is the foundation for every function's iteration.

**Independent Test**: Run the data-loading cells in each notebook -- each should display the correct number of samples in tabular format, with no NaN or out-of-range values, and the current best observation identified.

**Acceptance Scenarios**:

1. **Given** updated_inputs - Week 9.npy and updated_outputs - Week 9.npy exist in ./data/fX/, **When** a notebook loads and displays the data, **Then** the expected number of samples (F1: 19, F2: 19, F3: 24, F4: 39, F5: 29, F6: 29, F7: 39, F8: 49) with the correct input dimensions are shown.
2. **Given** the loaded data, **When** validation checks run, **Then** all input values are within [0.0, 1.0] and no outputs contain NaN.
3. **Given** the loaded data for each function, **When** the current best observation is identified, **Then** its value and location are printed clearly.

---

### User Story 2 -- Fit the Same Surrogate Model as Week 8 (Priority: P1)

As a student, I want each notebook to train the identical surrogate model (type and hyperparameters) used in Week 8 on the Week 9 data so that model continuity is maintained across iterations.

**Why this priority**: The surrogate model is the core of each optimisation loop; using the same strategy ensures consistency and comparability with prior weeks.

**Independent Test**: After the surrogate cells execute in each notebook, the model is fitted without errors, and predictions can be queried across the input space.

**Acceptance Scenarios**:

1. **Given** F1 data (19 samples, 2D), **When** the hurdle model (calibrated classifier + random forest regressor for positive outputs) is trained, **Then** the model fits successfully (or enters fallback exploration mode if fewer than 3 positive outputs exist).
2. **Given** F2 data (19 samples, 2D), **When** SingleTaskGP with Matern 1.5, ARD, noise >= 1e-3 is trained, **Then** the model fits successfully and posterior predictions are available.
3. **Given** F3 data (24 samples, 3D), **When** SingleTaskGP with Matern 2.5, ARD, z-score standardisation is trained, **Then** the model fits successfully.
4. **Given** F4 data (39 samples, 4D), **When** Multi-Fidelity GP with LinearTruncatedFidelityKernel and constant fidelity column is trained, **Then** the model fits successfully.
5. **Given** F5 data (29 samples, 4D), **When** GP with Matern 2.5, ARD, log1p-transformed outputs is trained, **Then** the model fits successfully.
6. **Given** F6 data (29 samples, 5D), **When** SingleTaskGP with Matern 1.5, ARD, noise >= 1e-2, Standardize transform is trained, **Then** the model fits successfully.
7. **Given** F7 data (39 samples, 6D), **When** the neural network (6->5->5->1, ReLU, dropout=0.1) is trained with Adam (lr=0.005, 200 epochs), **Then** training completes without errors and predictions are available via MC dropout.
8. **Given** F8 data (49 samples, 8D), **When** SingleTaskGP with Matern 2.5, ARD, noise >= 1e-7, Standardize transform is trained, **Then** the model fits successfully.

---

### User Story 3 -- Propose Next Sample Point via Acquisition (Priority: P1)

As a student, I want each notebook to optimise its acquisition function (same as Week 8) and propose the next sample point for the next submission, formatted as a valid submission query.

**Why this priority**: Proposing the next sample point is the primary deliverable of each weekly iteration.

**Independent Test**: Each notebook outputs a formatted submission string in x1-x2-...-xn format with 6 decimal places, all values within [0.0, 0.999999].

**Acceptance Scenarios**:

1. **Given** F1 hurdle model is fitted, **When** weighted UCB acquisition is evaluated over 20000 random candidates with local penalization and interior penalty, **Then** a proposed sample point is selected and formatted as 0.xxxxxx-0.xxxxxx.
2. **Given** F2 SFGP is fitted, **When** qLogNEI is optimised over [0, 0.999999]^2 with 10 restarts and 512 raw samples, **Then** a proposed sample point is formatted as 0.xxxxxx-0.xxxxxx.
3. **Given** F3 SFGP is fitted, **When** qLogNEI is optimised over [0, 0.999999]^3 with 10 restarts and 512 raw samples, **Then** a proposed sample point is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx.
4. **Given** F4 MFGP is fitted, **When** cost-aware MF-qNEI generates q=4 candidates with fidelity fixed at 1.0, **Then** the best candidate is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx.
5. **Given** F5 GP is fitted, **When** qLogNEI generates q=4 candidates with interior penalty re-scoring, **Then** the point is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx.
6. **Given** F6 SFGP is fitted, **When** qLogNEI generates q=4 candidates with feasibility bounds and interior penalty re-scoring, **Then** the point is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx.
7. **Given** F7 NN is trained, **When** MC Dropout EI is evaluated over 20000 candidates with interior penalty (STEEPNESS=0.1, FLOOR=0.01), **Then** the best candidate is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx.
8. **Given** F8 SFGP is fitted, **When** qEI is optimised over [0, 0.999999]^8 with 256 MC samples and 30 restarts, **Then** a proposed point is formatted as 0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx.

---

### User Story 4 -- Visualise Surrogate, Convergence, and Submission Clustering (Priority: P1)

As a student submitting capstone notebooks, I want each notebook to produce surrogate function visualisations and convergence plots with all weekly submission points shown in a distinct colour, so that I can assess whether proposals are clustering in a particular region or exploring the search space.

**Why this priority**: This is a new requirement that directly addresses the need to visually assess whether the optimisation loop is exploring or converging. Elevated to P1 because it is explicitly requested and critical for understanding iteration behaviour.

**Independent Test**: Each notebook generates legible, correctly annotated plots where initial samples appear in one colour and all weekly submissions appear in a visually distinct second colour.

**Acceptance Scenarios**:

1. **Given** any function's data with N total samples (of which the first K are initial and the remaining 9 are weekly submissions), **When** surrogate plots are generated, **Then** the initial samples are shown in one colour (e.g., blue) and all weekly submissions are shown in a contrasting colour (e.g., red/orange).
2. **Given** the proposed new sample point, **When** it is overlaid on the surrogate plot, **Then** it is shown in a third distinct marker style (e.g., star marker in green) clearly labelled.
3. **Given** the convergence plot, **When** the running maximum is displayed, **Then** vertical markers or shading distinguish the initial observation period from the weekly submission period, and all submitted points are visually distinguishable.
4. **Given** F1 (2D), **When** the 3-panel contour plot is generated, **Then** initial points and recent submissions are shown with different colours on all three panels.
5. **Given** F2 or F3 (2D/3D), **When** contour/surface plots are generated, **Then** initial points and recent submissions are shown with different colours enabling visual assessment of clustering.
6. **Given** F4-F8 (4D-8D), **When** pairwise slice plots or other appropriate visualisations are generated, **Then** the colour distinction between initial and recent points is maintained across all subplot panels.
7. **Given** any notebook, **When** a plot legend is displayed, **Then** it clearly labels the three categories: "Initial samples", "Weekly submissions", and "Proposed next point".

---

### User Story 5 -- Performance Evaluation and Strategy Recommendation (Priority: P1)

As a student submitting capstone notebooks, I want each notebook to contain a performance evaluation section at the end that quantitatively assesses whether the optimisation is stalling and, if so, proposes a concrete strategy change for the next iteration.

**Why this priority**: The user explicitly requested this analysis to identify underperforming strategies before they waste further submissions. This is critical for demonstrating analytical rigour in a capstone project.

**Independent Test**: Each notebook ends with code cells computing performance metrics and a markdown cell providing interpretation and strategy recommendations. Stalling functions are flagged with specific, actionable changes proposed.

**Acceptance Scenarios**:

1. **Given** any function's complete data set (initial + 9 submissions), **When** the performance evaluation code cells execute, **Then** the following metrics are computed and displayed: (a) best-value trajectory across all submissions, (b) per-submission improvement (delta from previous best), (c) stalling flag (True if no new best in 3+ consecutive submissions OR total improvement < 5% relative to initial best).
2. **Given** the submission points, **When** exploration spread analysis runs, **Then** the mean pairwise Euclidean distance between submission points and a clustering metric (e.g., max nearest-neighbour distance) are computed and displayed.
3. **Given** the fitted surrogate model, **When** leave-one-out prediction error is computed on submission points, **Then** MAE and RMSE of the surrogate's predictions at each submission point (trained without that point) are displayed.
4. **Given** a function where the stalling flag is True, **When** the concluding markdown cell is written, **Then** it proposes at least one specific, actionable strategy change (e.g., change surrogate type, adjust acquisition parameters, increase exploration).
5. **Given** a function where the stalling flag is False, **When** the concluding markdown cell is written, **Then** it confirms the current strategy is performing well and notes any observations about exploration coverage.

---

### Edge Cases

- **Missing data files**: If updated_inputs - Week 9.npy or updated_outputs - Week 9.npy is absent for any function, the notebook should fail early with a clear file-not-found message.
- **Unexpected sample count**: If loaded data has fewer or more rows than expected, the notebook should fail with a clear assertion error.
- **Model fitting failure**: If the GP fails to converge (e.g., numerical instability in F4 MFGP or F8 high-dimensional GP), the notebook should catch the error, increase jitter, and retry.
- **All acquisition values zero**: If all candidates have zero acquisition value (possible in F8), a fallback to highest posterior mean selection should activate.
- **Candidates near boundaries**: For functions with interior penalty (F1, F5, F6, F7), the penalty should suppress boundary candidates.
- **F7 NN training instability**: If the neural network diverges during training, reduce learning rate and retry.
- **F1 no positive outputs**: If F1 still has no positive outputs, fallback to pure exploration mode as in previous weeks.
- **Colour distinction on dense plots**: For high-dimensional functions with many data points (F4: 39, F7: 39, F8: 49), ensure the colour distinction remains visible by using larger markers for the weekly submission points.
- **All submissions improve**: If every submission improves upon the previous best (no stalling), the performance evaluation should still run but report "no stalling detected" and affirm the current strategy.
- **LOO error with few submissions**: Leave-one-out prediction error is computed over 9 submission points; with small counts the error estimate may be noisy — report the metric but note the limited sample size.
- **Tied best values**: If multiple submissions return the exact same best value, this counts as stalling (no improvement).
- **Negative or zero outputs**: For functions where all outputs are negative (e.g., F1 historically), the 5% relative improvement threshold should be computed on the absolute magnitude of improvement.

## Requirements *(mandatory)*

### Functional Requirements

#### General (apply to all 8 notebooks)

- **FR-001**: A new self-contained Jupyter notebook MUST be created at ./functions/fX/fX - week 9.ipynb for each function (X = 1, 2, 3, 4, 5, 6, 7, 8). Existing notebooks MUST NOT be modified.
- **FR-002**: Each notebook MUST load updated_inputs - Week 9.npy and updated_outputs - Week 9.npy from ./data/fX/.
- **FR-003**: Each notebook MUST display all data points in tabular format, identifying the current best observation and its location.
- **FR-004**: Each notebook MUST use the identical surrogate model type and hyperparameters as its Week 8 implementation (see Per-Function Strategy Summary table).
- **FR-005**: Each notebook MUST use the identical acquisition function type and parameters as its Week 8 implementation.
- **FR-006**: Each notebook MUST output the proposed next sample as a formatted submission query: 0.xxxxxx-0.xxxxxx-...-0.xxxxxx with all values clipped to [0.0, 0.999999].
- **FR-007**: Each notebook MUST produce a convergence plot showing the running maximum of observed outputs, with all weekly submissions visually distinguishable from initial samples.
- **FR-008**: Each notebook MUST produce surrogate function visualisation appropriate to the function's dimensionality.
- **FR-009**: Each notebook MUST be fully self-contained -- all imports, data loading, model fitting, acquisition, visualisation, and submission output in one notebook.
- **FR-010**: All hyperparameters MUST be defined as named constants in a dedicated cell at the top of each notebook with markdown documentation.

#### Performance Evaluation (apply to all 8 notebooks)

- **FR-PERF-001**: Each notebook MUST end with a "Performance Evaluation" section containing code cells followed by a concluding markdown cell.
- **FR-PERF-002**: A code cell MUST compute and display the best-value trajectory: for each submission week, the running maximum observed output, the per-submission improvement (delta), and whether that submission found a new best.
- **FR-PERF-003**: A code cell MUST compute and display a stalling flag set to True when either: (a) no new best has been found in the last 3 or more consecutive submissions, OR (b) total improvement across all 9 submissions is less than 5% relative to the initial best value.
- **FR-PERF-004**: A code cell MUST compute and display exploration spread metrics: mean pairwise Euclidean distance between all submission points, and the maximum nearest-neighbour distance among submissions.
- **FR-PERF-005**: A code cell MUST compute and display leave-one-out (LOO) surrogate prediction error on the 9 submission points: for each submission point, retrain the surrogate on all data excluding that point, predict its output, and report per-point error, overall MAE, and RMSE.
- **FR-PERF-006**: A concluding markdown cell MUST interpret the metrics and, if the stalling flag is True, propose at least one specific, actionable strategy change for the next iteration. Strategy changes MUST reference concrete modifications (e.g., "switch from Matérn 1.5 to Matérn 2.5", "increase exploration by raising kappa from 3.0 to 5.0", "add Latin Hypercube diversification").
- **FR-PERF-007**: If the stalling flag is False, the concluding markdown cell MUST confirm the current strategy is performing well and note observations about exploration coverage.

#### Visualisation Enhancement (apply to all 8 notebooks)

- **FR-VIS-001**: All surrogate and data plots MUST distinguish between two groups of training points: (a) initial samples (indices 0 to N_initial-1) and (b) all weekly submissions (indices N_initial to N_total-1). The initial sample counts per function are: F1=10, F2=10, F3=15, F4=30, F5=20, F6=20, F7=30, F8=40. This yields 9 highlighted submission points per function.
- **FR-VIS-002**: Initial samples MUST be displayed in blue (`tab:blue`).
- **FR-VIS-003**: All weekly submission points (9 per function) MUST be displayed in orange (`tab:orange`) with slightly larger markers to ensure visibility.
- **FR-VIS-004**: The proposed next sample point MUST be displayed with a distinct marker style (e.g., green star) and labelled "Proposed next point".
- **FR-VIS-005**: Every plot with data point overlays MUST include a legend with three entries: "Initial samples", "Weekly submissions", and "Proposed next point".
- **FR-VIS-006**: The convergence plot MUST use the same colour scheme (blue for initial, orange/red for weekly submissions) to colour the observation markers along the running-maximum curve.
- **FR-VIS-007**: For high-dimensional functions (F4-F8), the colour distinction MUST be maintained consistently across all pairwise or slice subplot panels.

#### F1-Specific

- **FR-F1-001**: Surrogate MUST be the same two-stage hurdle model as Week 8: Stage 1 -- calibrated logistic classifier for P(y > 0); Stage 2 -- random forest regressor on log1p(y) for positive outputs.
- **FR-F1-002**: When fewer than 3 positive outputs exist (FALLBACK_MODE), Stage 2 MUST be skipped and the acquisition function MUST degrade to pure exploration.
- **FR-F1-003**: Acquisition MUST be weighted UCB: a(x) = p(x)*mu(x) + kappa*p(x)*sigma_RF(x), multiplied by local penalization and interior penalty.
- **FR-F1-004**: Hyperparameters MUST be: C_STAGE1=1.0, N_ESTIMATORS=100, MAX_DEPTH=3, KAPPA=3.0, PENALTY_RADIUS=0.15, N_CANDIDATES=20000, STEEPNESS=0.1, FLOOR=0.01.
- **FR-F1-005**: The notebook MUST produce a 3-panel contour visualisation: (1) hurdle mean, (2) hurdle uncertainty, (3) penalised acquisition surface -- each overlaying training points with the enhanced colour scheme.

#### F2-Specific

- **FR-F2-001**: Surrogate MUST be SingleTaskGP with Matern 1.5 kernel, ARD=True, noise lower bound=1e-3, input normalisation=True.
- **FR-F2-002**: Acquisition MUST be qLogNoisyExpectedImprovement (qLogNEI) optimised over [0, 0.999999]^2 with num_restarts=10, raw_samples=512.

#### F3-Specific

- **FR-F3-001**: Surrogate MUST be SingleTaskGP with Matern 2.5 kernel, ARD=True (3 lengthscales), Gaussian noise likelihood with noise >= 1e-6, outputs z-score standardised, inputs normalised to [0, 1].
- **FR-F3-002**: MLL training MUST use 15 random restarts.
- **FR-F3-003**: Acquisition MUST be qLogNoisyExpectedImprovement (qLogNEI) optimised over [0, 0.999999]^3 with num_restarts=10, raw_samples=512.

#### F4-Specific

- **FR-F4-001**: Surrogate MUST be Multi-Fidelity GP with Matern 2.5 kernel, ARD (4 lengthscales), LinearTruncatedFidelityKernel, constant fidelity column (all 1.0) appended to inputs.
- **FR-F4-002**: Noise floor MUST be >= 1e-4. Outputs MUST be z-score standardised.
- **FR-F4-003**: Acquisition MUST be cost-aware MF-qNEI with q=4, 64 MC fantasies, Sobol-initialised + multi-start.
- **FR-F4-004**: Bounds MUST be [0, 0.999999]^4 with fidelity fixed at 1.0. Best of 4 candidates MUST be selected by highest posterior mean.

#### F5-Specific

- **FR-F5-001**: Surrogate MUST be GP with Matern 2.5 kernel, ARD (4 lengthscales), outputs log1p-transformed, lengthscale init=0.5, noise init=0.1*Var(y_transformed), jitter=1e-6.
- **FR-F5-002**: MLL training MUST use 15 random restarts.
- **FR-F5-003**: Acquisition MUST be qLogNoisyExpectedImprovement with q=4. Optimiser: 3000 Sobol starts -> best 50 -> L-BFGS.
- **FR-F5-004**: Candidate selection MUST filter to posterior mean > median, then pick the candidate farthest from existing data (Euclidean distance).
- **FR-F5-005**: Interior penalty MUST be applied with STEEPNESS=1.0, FLOOR=0.01.

#### F6-Specific

- **FR-F6-001**: Surrogate MUST be SingleTaskGP with Matern 1.5 kernel, ScaleKernel wrapper, ARD (5 lengthscales), GaussianLikelihood with noise >= 1e-2, noise init=0.2, Standardize(m=1) outcome transform, lengthscale init=0.5, outputscale init=1.0.
- **FR-F6-002**: MLL training MUST use fit_gpytorch_mll with 15 random restarts with manual seed.
- **FR-F6-003**: Acquisition MUST be qLogNoisyExpectedImprovement with q=4, 512 Sobol QMC samples, prune_baseline=True. Optimiser: raw_samples=3000, num_restarts=50.
- **FR-F6-004**: Feasibility bounds MUST constrain x4 (milk) in [0.10, 1.0], all other dimensions in [0.01, 1.0].
- **FR-F6-005**: Distance-based exploration MUST filter candidates to mean > median, then select the farthest from existing data.
- **FR-F6-006**: Interior penalty MUST be applied with STEEPNESS=1.0, FLOOR=0.01. Rank-based scoring MUST be used for negative output values.

#### F7-Specific

- **FR-F7-001**: Surrogate MUST be a neural network with architecture 6->5->5->1, ReLU activations, dropout p=0.1. Inputs and outputs MUST be z-score normalised.
- **FR-F7-002**: Training MUST use Adam optimiser with lr=0.005 for 200 epochs.
- **FR-F7-003**: Acquisition MUST be MC Dropout EI with MC_SAMPLES=50 stochastic forward passes. EI is computed as mean(max(pred_i - y_best, 0)) over MC samples.
- **FR-F7-004**: N_CANDIDATES=20000 uniformly sampled in [0, 1]^6.
- **FR-F7-005**: Interior penalty MUST be applied multiplicatively with STEEPNESS=0.1, FLOOR=0.01.

#### F8-Specific

- **FR-F8-001**: Surrogate MUST be SingleTaskGP with Matern 2.5 kernel, ARD (8 lengthscales), Standardize(m=1) output transform, noise >= 1e-7.
- **FR-F8-002**: MLL training MUST use fit_gpytorch_mll (exact marginal log-likelihood).
- **FR-F8-003**: Acquisition MUST be qExpectedImprovement with q=1, 256 MC samples (SobolQMCNormalSampler), fantasisation enabled, xi=0.01 (best_f = y_max + xi).
- **FR-F8-004**: Optimisation MUST use num_restarts=30, raw_samples=4096, bounds [0, 0.999999]^8.
- **FR-F8-005**: Fallback: if all qEI values are 0, the candidate with the highest posterior mean MUST be selected.

### Key Entities

- **Initial Samples**: The first K data points provided with the original problem (K varies per function: F1=10, F2=10, F3=15, F4=30, F5=20, F6=20, F7=30, F8=40). Displayed in blue on all visualisations.
- **Weekly Submissions**: All 9 data points submitted across the weekly iterations (indices N_initial to N_total-1). Displayed in orange/red on all visualisations to reveal clustering or exploration patterns.
- **Proposed Next Point**: The candidate selected by the acquisition function for the next submission. Displayed as a green star marker on all visualisations.
- **Hurdle Model (F1)**: Two-stage surrogate combining a calibrated classifier (Stage 1) and random forest regressor (Stage 2) for positive outputs.
- **SingleTaskGP (SFGP)**: BoTorch standard GP surrogate for single-output problems. Used by F2, F3, F6, F8.
- **Multi-Fidelity GP (MFGP)**: GP with LinearTruncatedFidelityKernel for multi-fidelity modelling. Used by F4.
- **Standard GP**: BoTorch GP with log-transformed outputs. Used by F5.
- **Neural Network Surrogate**: Compact feedforward network with MC dropout for uncertainty estimation. Used by F7.
- **Interior Penalty**: Boundary suppression via sinusoidal product: w(x) = FLOOR + (1-FLOOR) * prod(sin(pi*xi)^(2*STEEPNESS)). Applied to F1, F5, F6, F7.
- **Week 9 Data**: Function-specific observation counts: F1=19, F2=19, F3=24, F4=39, F5=29, F6=29, F7=39, F8=49.
- **Stalling**: A quantitative flag indicating the optimisation is not making progress. Defined as True when (a) no new best observed value in the last 3+ consecutive submissions, OR (b) total improvement over all submissions is less than 5% relative to the initial best value.
- **Performance Evaluation Section**: A mandatory end-of-notebook section containing code cells (convergence metrics, exploration spread, LOO error) and a concluding markdown cell with interpretation and strategy recommendations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 8 notebooks (f1 - week 9.ipynb through f8 - week 9.ipynb) execute end-to-end without errors, producing all expected outputs.
- **SC-002**: Each notebook's proposed sample point has all coordinates within [0.0, 0.999999] and the submission query is in valid format 0.xxxxxx-0.xxxxxx-...-0.xxxxxx.
- **SC-003**: For functions with interior penalty (F1, F5, F6, F7), the proposed sample point is not on the boundary of the search space (no coordinate is 0.0 or 0.999999).
- **SC-004**: Each notebook's convergence plot shows clearly whether the latest observation improved upon the previous best.
- **SC-005**: All data points are correctly loaded for each function, matching the expected sample counts (F1: 19, F2: 19, F3: 24, F4: 39, F5: 29, F6: 29, F7: 39, F8: 49).
- **SC-006**: Surrogate function visualisations are legible and correctly annotated with training points and the proposed candidate.
- **SC-007**: All 8 notebooks clearly display initial samples in one colour and all weekly submissions in a visually distinct second colour across all plots.
- **SC-008**: A viewer can determine, at a glance, whether the submitted sample points are clustered in a specific region or spread across the search space.
- **SC-009**: Every plot with data points includes a legend distinguishing "Initial samples", "Weekly submissions", and "Proposed next point".
- **SC-010**: All 8 notebooks contain a "Performance Evaluation" section at the end with code cells computing convergence metrics, exploration spread, and LOO surrogate error, followed by a markdown interpretation cell.
- **SC-011**: Any function where the stalling flag is True has a markdown cell proposing at least one specific, actionable strategy change for the next iteration.
- **SC-012**: The stalling detection correctly identifies functions with no improvement in 3+ consecutive submissions or < 5% total relative improvement.

## Assumptions

- The data files updated_inputs - Week 9.npy and updated_outputs - Week 9.npy already exist in ./data/fX/ for all 8 functions and contain the correct number of observations.
- The same hyperparameters from Week 8 remain appropriate for Week 9. No hyperparameter tuning is specified for this iteration.
- Each notebook follows the constitution convention: a new file fX - week 9.ipynb in ./functions/fX/, fully self-contained.
- Existing notebooks are not modified.
- All weekly submissions (9 per function) are highlighted in orange/red; the split is simply: initial samples in blue, all points beyond the initial set in orange/red.
- Blue, orange/red, and green star are reasonable default colours; the exact hex values are not constrained, only the visual distinctiveness.
- The Week 8 notebooks serve as the code template for each Week 9 notebook, with data file references updated and visualisation colours added.
- The LOO prediction error computation retrains the surrogate 9 times (once per submission point omitted). This is acceptable given the small data sizes.
- Strategy recommendations in the performance evaluation section are advisory -- they will be reviewed by the student before being adopted in Week 10.
- The 5% relative improvement threshold and 3-consecutive-submission window are reasonable defaults; they may be adjusted in future iterations based on experience.
