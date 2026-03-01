# Feature Specification: F2–F8 Week 8 — Bayesian Optimisation Iteration

**Feature Branch**: `020-f2-f8-week8`  
**Created**: 2026-03-01  
**Status**: Draft  
**Input**: User description: "Process week 8 outputs for F2 through F8, keeping the same strategy as week 7, propose the next sample point. Create a new notebook for this iteration as per constitution."

## Per-Function Strategy Summary

Each function retains its Week 7 surrogate and acquisition strategy unchanged. The table below summarises key parameters per function:

| Function | Dims | Samples | Surrogate | Acquisition | Interior Penalty |
|----------|------|---------|-----------|-------------|------------------|
| F2 | 2 | 18 | SFGP (Matérn 1.5, ARD, noise ≥ 1e-3) | qLogNEI (10 restarts, 512 raw) | No |
| F3 | 3 | 23 | SFGP (Matérn 2.5, ARD, z-score) | qLogNEI (10 restarts, 512 raw) | No |
| F4 | 4 | 38 | MFGP (Matérn 2.5, ARD, LinTrunc, noise ≥ 1e-4) | Cost-aware MF-qNEI (q=4, 64 fantasies) | No |
| F5 | 4 | 28 | GP (Matérn 2.5, ARD, log1p transform) | qLogNEI (q=4, 3000 Sobol → 50 L-BFGS) | Yes (S=1.0, F=0.01) |
| F6 | 5 | 28 | SFGP (Matérn 1.5, ARD, noise ≥ 1e-2) | qLogNEI (q=4, 512 QMC, 3000 raw → 50 restarts) | Yes (S=1.0, F=0.01) |
| F7 | 6 | 38 | Neural Network (6→5→5→1, dropout=0.1, lr=0.005) | MC Dropout EI (50 samples, 20000 candidates) | Yes (S=0.1, F=0.01) |
| F8 | 8 | 48 | SFGP (Matérn 2.5, ARD, noise ≥ 1e-7) | qEI (q=1, 256 MC, 30 restarts, 4096 raw) | No |

> **Note on F7 STEEPNESS**: The Week 7 notebook code uses STEEPNESS=0.1 (gentler boundary suppression), which differs from the spec-017 stated value of 1.0. The notebook implementation is ground truth — Week 8 carries forward STEEPNESS=0.1.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load and Validate Week 8 Data for Each Function (Priority: P1)

As a student running the weekly Bayesian Optimisation loop for Functions 2–8, I want each notebook to load the corresponding Week 8 updated inputs and outputs so that all evaluated sample points are available for surrogate modelling.

**Why this priority**: Without current data, no optimisation can proceed. This is the foundation for every function's iteration.

**Independent Test**: Run the data-loading cells in each notebook — each should display the correct number of samples in tabular format, with no NaN or out-of-range values, and the current best observation identified.

**Acceptance Scenarios**:

1. **Given** `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` exist in `./data/fX/`, **When** a notebook loads and displays the data, **Then** the expected number of samples (see table above) with the correct input dimensions are shown.
2. **Given** the loaded data, **When** validation checks run, **Then** all input values are within [0.0, 1.0] and no outputs contain NaN.
3. **Given** the loaded data for each function, **When** the current best observation is identified, **Then** its value and location are printed clearly.

---

### User Story 2 — Fit the Same Surrogate Model as Week 7 (Priority: P1)

As a student, I want each notebook to train the identical surrogate model (type and hyperparameters) used in Week 7 on the Week 8 data so that model continuity is maintained across iterations.

**Why this priority**: The surrogate model is the core of each optimisation loop; using the same strategy ensures consistency and comparability with prior weeks.

**Independent Test**: After the surrogate cells execute in each notebook, the model is fitted without errors, and predictions can be queried across the input space.

**Acceptance Scenarios**:

1. **Given** F2 data (18 samples, 2D), **When** SingleTaskGP with Matérn 1.5, ARD, noise ≥ 1e-3 is trained, **Then** the model fits successfully and posterior predictions are available.
2. **Given** F3 data (23 samples, 3D), **When** SingleTaskGP with Matérn 2.5, ARD, z-score standardisation is trained, **Then** the model fits successfully.
3. **Given** F4 data (38 samples, 4D), **When** Multi-Fidelity GP with LinearTruncatedFidelityKernel and constant fidelity column is trained, **Then** the model fits successfully.
4. **Given** F5 data (28 samples, 4D), **When** GP with Matérn 2.5, ARD, log1p-transformed outputs is trained, **Then** the model fits successfully.
5. **Given** F6 data (28 samples, 5D), **When** SingleTaskGP with Matérn 1.5, ARD, noise ≥ 1e-2, Standardize transform is trained, **Then** the model fits successfully.
6. **Given** F7 data (38 samples, 6D), **When** the neural network (6→5→5→1, ReLU, dropout=0.1) is trained with Adam (lr=0.005, 200 epochs), **Then** training completes without errors and predictions are available via MC dropout.
7. **Given** F8 data (48 samples, 8D), **When** SingleTaskGP with Matérn 2.5, ARD, noise ≥ 1e-7, Standardize transform is trained, **Then** the model fits successfully.

---

### User Story 3 — Propose Next Sample Point via Acquisition (Priority: P1)

As a student, I want each notebook to optimise its acquisition function (same as Week 7) and propose the next sample point for Week 9 submission, formatted as a valid submission query.

**Why this priority**: Proposing the next sample point is the primary deliverable of each weekly iteration.

**Independent Test**: Each notebook outputs a formatted submission string in `x1-x2-...-xn` format with 6 decimal places, all values within [0.0, 0.999999].

**Acceptance Scenarios**:

1. **Given** F2's SFGP is fitted, **When** qLogNEI is optimised over [0, 0.999999]² with 10 restarts and 512 raw samples, **Then** a proposed sample point is selected and formatted as `0.xxxxxx-0.xxxxxx`.
2. **Given** F3's SFGP is fitted, **When** qLogNEI is optimised over [0, 0.999999]³ with 10 restarts and 512 raw samples, **Then** a proposed sample point is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx`.
3. **Given** F4's MFGP is fitted, **When** cost-aware MF-qNEI generates q=4 candidates with fidelity fixed at 1.0, **Then** the best candidate (highest posterior mean) is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`.
4. **Given** F5's GP is fitted, **When** qLogNEI generates q=4 candidates and the best is selected by distance-based exploration with interior penalty re-scoring, **Then** the point is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`.
5. **Given** F6's SFGP is fitted, **When** qLogNEI generates q=4 candidates with feasibility bounds (x4 ∈ [0.10, 1.0], others ∈ [0.01, 1.0]) and interior penalty re-scoring, **Then** the point is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`.
6. **Given** F7's NN is trained, **When** MC Dropout EI is evaluated over 20 000 candidates with interior penalty (STEEPNESS=0.1, FLOOR=0.01), **Then** the best candidate is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`.
7. **Given** F8's SFGP is fitted, **When** qEI is optimised over [0, 0.999999]⁸ with 256 MC samples and 30 restarts, **Then** a proposed point is formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`.

---

### User Story 4 — Visualise Surrogate and Convergence (Priority: P2)

As a student submitting capstone notebooks, I want each notebook to produce surrogate function visualisations and convergence plots consistent with Week 7 so that the examiner can assess model quality and optimisation progress.

**Why this priority**: Visualisation is a capstone marking requirement and essential for interpretability, but the proposal itself is the primary deliverable.

**Independent Test**: Each notebook generates legible, correctly annotated plots showing the surrogate surface and convergence history.

**Acceptance Scenarios**:

1. **Given** F2 or F3 (2D/3D), **When** surrogate plots are generated, **Then** contour plots of the posterior mean and variance are shown with training points overlaid and the proposed point highlighted.
2. **Given** F4 or F5 (4D), **When** surrogate plots are generated, **Then** pairwise 2D slice plots or the equivalent visualisation from Week 7 are shown.
3. **Given** F6 (5D), **When** surrogate plots are generated, **Then** appropriate dimensionality-reduced or slice visualisations are produced.
4. **Given** F7 (6D), **When** surrogate plots are generated, **Then** the neural network's uncertainty landscape is visualised via pairwise slices or similar.
5. **Given** F8 (8D), **When** surrogate plots are generated, **Then** appropriate high-dimensional visualisations (slice plots, parallel coordinates, or similar) are produced.
6. **Given** any function, **When** the convergence plot is generated, **Then** the running maximum of observed output values is shown across all data points, with clear markers at initial/weekly boundaries.

---

### Edge Cases

- **Missing data files**: If `updated_inputs - Week 8.npy` or `updated_outputs - Week 8.npy` is absent for any function, the notebook should fail early with a clear file-not-found message.
- **Unexpected sample count**: If loaded data has fewer or more rows than expected, the notebook should warn but continue (data may have been corrected between weeks).
- **Model fitting failure**: If the GP fails to converge (e.g., numerical instability in F4's MFGP or F8's high-dimensional GP), the notebook should catch the error, increase jitter, and retry.
- **All acquisition values zero**: If all candidates have zero acquisition value (possible in F8), a fallback to highest posterior mean selection should activate.
- **Candidates near boundaries**: For functions with interior penalty (F5, F6, F7), the penalty should suppress boundary candidates. For functions without interior penalty, candidates at boundaries are acceptable.
- **F7 NN training instability**: If the small neural network diverges during training, reduce learning rate and retry.

## Requirements *(mandatory)*

### Functional Requirements

#### General (apply to all 7 notebooks)

- **FR-001**: A new self-contained Jupyter notebook MUST be created at `./functions/fX/fX - week 8.ipynb` for each function (X = 2, 3, 4, 5, 6, 7, 8). The original `fX.ipynb` MUST NOT be modified.
- **FR-002**: Each notebook MUST load `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` from `./data/fX/`.
- **FR-003**: Each notebook MUST display all data points in tabular format, identifying the current best observation and its location.
- **FR-004**: Each notebook MUST use the identical surrogate model type and hyperparameters as its Week 7 implementation (see Per-Function Strategy Summary table).
- **FR-005**: Each notebook MUST use the identical acquisition function type and parameters as its Week 7 implementation.
- **FR-006**: Each notebook MUST output the proposed next sample as a formatted submission query: `0.xxxxxx-0.xxxxxx-...-0.xxxxxx` with all values clipped to [0.0, 0.999999].
- **FR-007**: Each notebook MUST produce a convergence plot showing the running maximum of observed outputs.
- **FR-008**: Each notebook MUST produce surrogate function visualisation appropriate to the function's dimensionality.
- **FR-009**: Each notebook MUST be fully self-contained — all imports, data loading, model fitting, acquisition, visualisation, and submission output in one notebook.
- **FR-010**: All hyperparameters MUST be defined as named constants in a dedicated cell at the top of each notebook with markdown documentation.

#### F2-Specific

- **FR-F2-001**: Surrogate MUST be SingleTaskGP with Matérn 1.5 kernel, ARD=True, noise lower bound=1e-3, input normalisation=True.
- **FR-F2-002**: Acquisition MUST be qLogNoisyExpectedImprovement (qLogNEI) optimised over [0, 0.999999]² with num_restarts=10, raw_samples=512.

#### F3-Specific

- **FR-F3-001**: Surrogate MUST be SingleTaskGP with Matérn 2.5 kernel, ARD=True (3 lengthscales), Gaussian noise likelihood with noise ≥ 1e-6, outputs z-score standardised, inputs normalised to [0, 1].
- **FR-F3-002**: MLL training MUST use 15 random restarts.
- **FR-F3-003**: Acquisition MUST be qLogNoisyExpectedImprovement (qLogNEI) optimised over [0, 0.999999]³ with num_restarts=10, raw_samples=512.

#### F4-Specific

- **FR-F4-001**: Surrogate MUST be Multi-Fidelity GP with Matérn 2.5 kernel, ARD (4 lengthscales), LinearTruncatedFidelityKernel, constant fidelity column (all 1.0) appended to inputs.
- **FR-F4-002**: Noise floor MUST be ≥ 1e-4. Outputs MUST be z-score standardised.
- **FR-F4-003**: Acquisition MUST be cost-aware MF-qNEI with q=4, 64 MC fantasies, Sobol-initialised + multi-start.
- **FR-F4-004**: Bounds MUST be [0, 0.999999]⁴ with fidelity fixed at 1.0. Best of 4 candidates MUST be selected by highest posterior mean.

#### F5-Specific

- **FR-F5-001**: Surrogate MUST be GP with Matérn 2.5 kernel, ARD (4 lengthscales), outputs log1p-transformed, lengthscale init=0.5, noise init=0.1·Var(y_transformed), jitter=1e-6.
- **FR-F5-002**: MLL training MUST use 15 random restarts.
- **FR-F5-003**: Acquisition MUST be qLogNoisyExpectedImprovement with q=4. Optimiser: 3000 Sobol starts → best 50 → L-BFGS.
- **FR-F5-004**: Candidate selection MUST filter to posterior mean > median, then pick the candidate farthest from existing data (Euclidean distance).
- **FR-F5-005**: Interior penalty MUST be applied with STEEPNESS=1.0, FLOOR=0.01, as post-hoc re-scoring of the acquisition batch.

#### F6-Specific

- **FR-F6-001**: Surrogate MUST be SingleTaskGP with Matérn 1.5 kernel, ScaleKernel wrapper, ARD (5 lengthscales), GaussianLikelihood with noise ≥ 1e-2, noise init=0.2, Standardize(m=1) outcome transform, lengthscale init=0.5, outputscale init=1.0.
- **FR-F6-002**: MLL training MUST use fit_gpytorch_mll with 15 random restarts with manual seed.
- **FR-F6-003**: Acquisition MUST be qLogNoisyExpectedImprovement with q=4, 512 Sobol QMC samples, prune_baseline=True. Optimiser: raw_samples=3000, num_restarts=50.
- **FR-F6-004**: Feasibility bounds MUST constrain x4 (milk) ∈ [0.10, 1.0], all other dimensions ∈ [0.01, 1.0].
- **FR-F6-005**: Distance-based exploration MUST filter candidates to mean > median, then select the farthest from existing data.
- **FR-F6-006**: Interior penalty MUST be applied with STEEPNESS=1.0, FLOOR=0.01. Rank-based scoring MUST be used for negative output values.

#### F7-Specific

- **FR-F7-001**: Surrogate MUST be a neural network with architecture 6→5→5→1, ReLU activations, dropout p=0.1. Inputs and outputs MUST be z-score normalised.
- **FR-F7-002**: Training MUST use Adam optimiser with lr=0.005 for 200 epochs.
- **FR-F7-003**: Acquisition MUST be MC Dropout EI with MC_SAMPLES=50 stochastic forward passes. EI is computed as mean(max(pred_i − y_best, 0)) over MC samples.
- **FR-F7-004**: N_CANDIDATES=20 000 uniformly sampled in [0, 1]⁶.
- **FR-F7-005**: Interior penalty MUST be applied multiplicatively with STEEPNESS=0.1, FLOOR=0.01.

#### F8-Specific

- **FR-F8-001**: Surrogate MUST be SingleTaskGP with Matérn 2.5 kernel, ARD (8 lengthscales), Standardize(m=1) output transform, noise ≥ 1e-7.
- **FR-F8-002**: MLL training MUST use fit_gpytorch_mll (exact marginal log-likelihood).
- **FR-F8-003**: Acquisition MUST be qExpectedImprovement with q=1, 256 MC samples (SobolQMCNormalSampler), fantasisation enabled, xi=0.01 (best_f = y_max + xi).
- **FR-F8-004**: Optimisation MUST use num_restarts=30, raw_samples=4096, bounds [0, 0.999999]⁸.
- **FR-F8-005**: Fallback: if all qEI values are 0, the candidate with the highest posterior mean MUST be selected.

### Key Entities

- **SingleTaskGP (SFGP)**: BoTorch's standard GP surrogate for single-output problems. Used by F2, F3, F6, F8 with varying kernel and noise configurations.
- **Multi-Fidelity GP (MFGP)**: GP with LinearTruncatedFidelityKernel for multi-fidelity modelling. Used by F4 with a constant fidelity column.
- **Standard GP**: BoTorch GP with log-transformed outputs for large dynamic ranges. Used by F5.
- **Neural Network Surrogate**: A compact feedforward network with MC dropout for uncertainty estimation. Used by F7.
- **qLogNEI (qLogNoisyExpectedImprovement)**: Log-space noisy expected improvement for numerical stability. Used by F2, F3, F4, F5, F6.
- **MF-qNEI**: Multi-fidelity batch NEI with cost-awareness. Used by F4.
- **MC Dropout EI**: Expected improvement estimated via Monte Carlo dropout forward passes. Used by F7.
- **qEI**: Batch expected improvement with MC sampling. Used by F8.
- **Interior Penalty**: Boundary suppression via sinusoidal product: $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{d} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$. Applied to F5, F6, F7.
- **Week 8 Data**: Function-specific observation counts ranging from 18 (F2) to 48 (F8), accumulated from initial data plus 8 weekly submissions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 7 notebooks (`f2 - week 8.ipynb` through `f8 - week 8.ipynb`) execute end-to-end without errors, producing all expected outputs.
- **SC-002**: Each notebook's proposed sample point has all coordinates within [0.0, 0.999999] and the submission query is in valid format `0.xxxxxx-0.xxxxxx-...-0.xxxxxx`.
- **SC-003**: For functions with interior penalty (F5, F6, F7), the proposed sample point is not on the boundary of the search space (no coordinate is 0.0 or 0.999999).
- **SC-004**: Each notebook's convergence plot shows clearly whether the latest observation improved upon the previous best.
- **SC-005**: All data points are correctly loaded for each function, matching the expected sample counts (F2: 18, F3: 23, F4: 38, F5: 28, F6: 28, F7: 38, F8: 48).
- **SC-006**: Surrogate function visualisations are legible and correctly annotated with training points and the proposed candidate.
- **SC-007**: All 7 original notebooks (`f2.ipynb` through `f8.ipynb`) remain unmodified.

## Clarifications

### Session 2026-03-01

- Q: Should the spec be updated to match the actual Week 7 code for F7 dropout (0.2→0.1), F7 epochs (500→200), and F2 acquisition (NEI→qLogNEI)? → A: Yes, correct all 3 to match code (code is ground truth).
- Q: F3 acquisition is labelled "NEI" in spec but code uses qLogNoisyExpectedImprovement (qLogNEI). Correct to match code? → A: Yes, correct F3 to qLogNEI and update Key Entities (same rule as F2).
- Q: FR-F3-002 and FR-F5-002 say "10–20 random restarts" but code uses exactly 15. Tighten to match? → A: Yes, tighten both to 15 restarts (code is ground truth).
- Q: FR-F3-001 omits noise floor — all other GP FRs specify one. Code uses GreaterThan(1e-6). Add it? → A: Yes, add noise ≥ 1e-6 to FR-F3-001 (matches code, consistent with other FRs).

## Assumptions

- All 7 data file pairs (`updated_inputs - Week 8.npy` / `updated_outputs - Week 8.npy`) exist in `./data/f2/` through `./data/f8/` with the expected sample counts.
- The F2 problem is 2D, F3 is 3D, F4 is 4D, F5 is 4D, F6 is 5D, F7 is 6D, F8 is 8D — all searching over [0, 1]^d.
- The same hyperparameters from Week 7 remain appropriate for Week 8. No hyperparameter tuning is specified.
- Each notebook follows the constitution convention: a new file `fX - week 8.ipynb` in `./functions/fX/`, fully self-contained.
- The original `fX.ipynb` files are not modified.
- All problems are maximisation tasks (Constitution Principle VII).
- The BoTorch/PyTorch stack is available in the Python environment (except F7 which uses scikit-learn/PyTorch for the NN).
- F4's fidelity column remains constant at 1.0 (single-fidelity operation within the multi-fidelity framework).
- F6's feasibility bounds (x4 ∈ [0.10, 1.0]) reflect a domain constraint from prior weeks and are carried forward.
