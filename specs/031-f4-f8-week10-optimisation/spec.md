# Feature Specification: F4–F8 Week 10 Optimisation Strategy Changes

**Feature Branch**: `031-f4-f8-week10-optimisation`  
**Created**: 2026-03-12  
**Status**: Draft  
**Input**: User description: "Apply week 10 strategy changes to F4 through F8. F4: switch MFGP→SFGP Matérn-2.5 ARD, MF-qNEI→qLogNEI q=4, add Standardize(m=1), noise_lb=1e-3, MLL restarts 30+. F5: raw_samples 5000→8000, relax distance threshold, MLL restarts 50→60, log1p→log, evaluate single-transform. F6: maintain approach, milk constraint 0.10→0.12, noise_lb 1e-2→1e-3, raw_samples→5000. F7: EI weight 30%→50%, STEEPNESS 0.05→0.02, candidates 20k→50k, add MC dropout. F8: XI 0.01→0.05, MC samples 256→512, verify noise_lb=1e-7, raw_samples 4096→8192, qEI→qLogNEI."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — F4 Surrogate & Acquisition Overhaul (Priority: P1)

As a researcher optimising the 4D black-box F4 (40 observations, output range [-4.03, 0.53]), I want to replace the multi-fidelity GP and acquisition function with a simpler, more appropriate single-fidelity pipeline so that the surrogate better models my single-fidelity data and proposes higher-quality candidates.

**Why this priority**: F4's current surrogate is fundamentally mismatched — using a multi-fidelity kernel on single-fidelity data wastes model capacity and adds parameters without benefit. Correcting this has the largest expected improvement impact.

**Independent Test**: Run the F4 week 10 notebook end-to-end from data loading through submission query; verify the GP is a standard SFGP with Matérn-2.5 ARD, Standardize(m=1) is applied, noise_lb=1e-3, MLL optimisation uses ≥30 restarts, qLogNEI with q=4 proposes a valid candidate, and all visualisations render.

**Acceptance Scenarios**:

1. **Given** the F4 week 10 notebook with week 10 data loaded (40 observations, 4D), **When** the GP surrogate cell is executed, **Then** it fits a SingleTaskGP with Matérn-2.5 ARD kernel, Standardize(m=1) outcome transform, and noise constraint lower bound of 1e-3.
2. **Given** a fitted SFGP, **When** the acquisition cell is executed, **Then** qLogNEI with q=4 is optimised using ≥2048 raw samples and produces 4 candidates in [0, 0.999999]⁴.
3. **Given** 4 acquisition candidates, **When** distance-based selection is applied, **Then** the best non-duplicate candidate is selected and formatted as `x1-x2-x3-x4` with 6 decimal places.
4. **Given** the completed optimisation run, **When** the visualisation cells are executed, **Then** 2D contour slices and a convergence plot are rendered showing all observations and the proposed point.

---

### User Story 2 — F5 Exploration Tuning & Transform Simplification (Priority: P1)

As a researcher optimising the 4D black-box F5 (large positive outputs >1000), I want to fine-tune the exploration/exploitation balance and simplify the output transform pipeline so that the GP posterior is cleaner and the acquisition optimiser finds better candidates.

**Why this priority**: F5 shows large jumps when improvement occurs, indicating the acquisition is missing good regions. Increasing raw samples and simplifying the double-transform addresses both issues.

**Independent Test**: Run the F5 week 10 notebook end-to-end; verify raw_samples=8000, acquisition restarts=60, MLL restarts=15, log transform replaces log1p, acquisition produces a valid candidate, and visualisations render.

**Acceptance Scenarios**:

1. **Given** the F5 week 10 notebook with data loaded, **When** the output transform is applied, **Then** a log transform (not log1p) is used, and Standardize interaction is evaluated against a single-transform baseline.
2. **Given** a fitted GP, **When** the acquisition cell runs, **Then** qLogNEI uses raw_samples=8000 and the distance-based selection threshold is relaxed compared to week 9.
3. **Given** the MLL fitting cell, **When** executed, **Then** 15 MLL random restarts are performed for GP fitting, and the acquisition optimiser uses ≥60 num_restarts.
4. **Given** the completed run, **When** the submission query is printed, **Then** it follows the format `x1-x2-x3-x4` with values in [0, 0.999999].

---

### User Story 3 — F6 Incremental Refinement (Priority: P2)

As a researcher optimising the 5D recipe black-box F6 (30 observations, milk constraint), I want to make targeted refinements to the existing working approach so that the search focuses on higher-quality feasible regions without disrupting the positive trajectory.

**Why this priority**: F6's current approach is working; changes should be conservative to avoid regression.

**Independent Test**: Run the F6 week 10 notebook; verify SFGP Matérn-1.5 ARD with rank-based interior penalty is maintained, milk constraint threshold is 0.12, noise_lb=1e-3, raw_samples=5000, and a valid 5D submission is produced.

**Acceptance Scenarios**:

1. **Given** the F6 week 10 notebook with data loaded, **When** the GP and penalty cells are executed, **Then** the surrogate is SFGP Matérn-1.5 ARD with rank-based interior penalty (unchanged from week 9 approach).
2. **Given** the feasibility check, **When** a candidate is evaluated, **Then** the milk constraint threshold is 0.12 (increased from 0.10).
3. **Given** the GP noise configuration, **When** the model is fitted, **Then** noise_lb is 1e-3 (reduced from 1e-2) and raw_samples is 5000.
4. **Given** the completed run, **When** the submission query is printed, **Then** it follows `x1-x2-x3-x4-x5` format with values in [0, 0.999999].

---

### User Story 4 — F7 Exploration Boost with MC Dropout (Priority: P2)

As a researcher optimising the 6D black-box F7 (NN surrogate, 36 observations, 6 consecutive failures), I want to shift the acquisition blend towards exploration and add MC dropout uncertainty so that the search escapes the local optimum it has been stuck in.

**Why this priority**: F7 has failed 6 consecutive submissions — the current exploitative strategy needs a significant exploration boost.

**Independent Test**: Run the F7 week 10 notebook; verify EI weight is 50%, STEEPNESS=0.02, candidate pool is 50k, MC dropout with dropout=0.05 is used for uncertainty, and a valid 6D submission is produced.

**Acceptance Scenarios**:

1. **Given** the F7 week 10 notebook with data loaded, **When** the acquisition blend is computed, **Then** the EI weight is 50% (mean weight 50%) instead of the previous 30%.
2. **Given** the interior penalty configuration, **When** applied, **Then** STEEPNESS=0.02 (reduced from 0.05) for softer boundary penalisation.
3. **Given** the candidate generation step, **When** random candidates are generated, **Then** the pool size is 50,000 (increased from 20,000).
4. **Given** the uncertainty estimation, **When** the NN makes predictions, **Then** MC dropout with dropout=0.05 is used across multiple forward passes to produce uncertainty estimates.
5. **Given** the completed run, **When** the submission query is printed, **Then** it follows `x1-x2-x3-x4-x5-x6` format with values in [0, 0.999999].

---

### User Story 5 — F8 Exploration & Numerical Stability (Priority: P2)

As a researcher optimising the 8D black-box F8 (46 observations), I want to increase exploration and improve numerical stability so that the acquisition function explores more of the vast 8D space and the log-transformed EI avoids numerical issues.

**Why this priority**: The 8D space is under-explored, and switching to qLogNEI improves numerical stability over vanilla qEI.

**Independent Test**: Run the F8 week 10 notebook; verify XI=0.05, MC samples=512, noise_lb=1e-7 stability check passes, raw_samples=8192, qLogNEI replaces qEI, and a valid 8D submission is produced.

**Acceptance Scenarios**:

1. **Given** the F8 week 10 notebook with data loaded, **When** the acquisition function is configured, **Then** qLogNEI replaces qEI using X_baseline for the noisy EI formulation (XI is not applicable to qLogNEI).
2. **Given** the acquisition MC estimation, **When** samples are drawn, **Then** 512 MC samples are used (increased from 256).
3. **Given** the GP noise configuration, **When** the model is fitted, **Then** noise_lb=1e-7 is verified stable (Cholesky decomposition succeeds without errors).
4. **Given** the acquisition optimiser, **When** raw candidates are generated, **Then** 8192 raw samples are used (increased from 4096).
5. **Given** the completed run, **When** the submission query is printed, **Then** it follows `x1-x2-x3-x4-x5-x6-x7-x8` format with values in [0, 0.999999].

---

### Edge Cases

- What happens if the GP Cholesky decomposition fails with noise_lb=1e-7 on F8? The notebook should catch the error, increase noise_lb to 1e-6, re-fit, and log a warning.
- What happens if all q=4 candidates from qLogNEI in F4 are duplicates of existing observations? The notebook should report this and fall back to the next-best non-duplicate candidate, or flag for manual review.
- What happens if log(y) is undefined for F5 outputs? Since outputs are >1000, this cannot occur, but a guard checking all outputs are strictly positive should be present.
- What happens if MC dropout on F7 produces highly variable uncertainty? The number of forward passes should be ≥50 to stabilise estimates.
- What happens if the milk constraint at 0.12 eliminates all candidates in F6? The notebook should report infeasibility and fall back to the previous threshold of 0.10.

## Requirements *(mandatory)*

### Functional Requirements

**F4 — Surrogate & Acquisition Overhaul**:

- **FR-001**: The F4 notebook MUST fit a SingleTaskGP with Matérn-2.5 ARD kernel on the 4D input data.
- **FR-002**: The F4 notebook MUST apply Standardize(m=1) as the GP outcome transform.
- **FR-003**: The F4 notebook MUST set the GP noise constraint lower bound to 1e-3.
- **FR-004**: The F4 notebook MUST perform ≥30 MLL random restarts during hyperparameter optimisation.
- **FR-005**: The F4 notebook MUST use qLogNEI with q=4 as the acquisition function.
- **FR-006**: The F4 notebook MUST apply distance-based selection to choose the best non-duplicate candidate from the q=4 batch.

**F5 — Exploration Tuning & Transform Simplification**:

- **FR-007**: The F5 notebook MUST use a log transform (not log1p) for output preprocessing.
- **FR-008**: The F5 notebook MUST use raw_samples=8000 for acquisition optimisation.
- **FR-009**: The F5 notebook MUST use ≥60 acquisition optimisation restarts (num_restarts) and 15 MLL restarts for GP fitting.
- **FR-010**: The F5 notebook MUST relax the distance-based selection threshold compared to week 9.
- **FR-011**: The F5 notebook MUST evaluate the Standardize(m=1) interaction with the log transform and document whether single or double transform is used with justification.

**F6 — Incremental Refinement**:

- **FR-012**: The F6 notebook MUST maintain the SFGP Matérn-1.5 ARD surrogate with rank-based interior penalty.
- **FR-013**: The F6 notebook MUST set the milk feasibility constraint threshold to 0.12.
- **FR-014**: The F6 notebook MUST set noise_lb to 1e-3.
- **FR-015**: The F6 notebook MUST use raw_samples=5000 for acquisition optimisation.

**F7 — Exploration Boost with MC Dropout**:

- **FR-016**: The F7 notebook MUST use a 50% mean / 50% EI acquisition blend.
- **FR-017**: The F7 notebook MUST set the interior penalty STEEPNESS to 0.02.
- **FR-018**: The F7 notebook MUST generate 50,000 random candidates for acquisition evaluation.
- **FR-019**: The F7 notebook MUST implement MC dropout (dropout=0.05) across ≥50 forward passes for uncertainty estimation.

**F8 — Exploration & Numerical Stability**:

- **FR-020**: The F8 notebook MUST use qLogNEI as the acquisition function (replacing qEI).
- **FR-021**: The F8 notebook MUST document that qLogNEI replaces qEI and that the XI parameter is not applicable; exploration is achieved via increased MC samples (512) and raw_samples (8192).
- **FR-022**: The F8 notebook MUST use 512 MC samples for acquisition estimation.
- **FR-023**: The F8 notebook MUST use raw_samples=8192 for acquisition optimisation.
- **FR-024**: The F8 notebook MUST verify noise_lb=1e-7 stability and log a warning if Cholesky decomposition issues are detected.

**Cross-cutting**:

- **FR-025**: Each notebook (F4–F8) MUST produce a submission query in the format `x1-x2-x3-...-xn` with values in [0, 0.999999] and 6 decimal places.
- **FR-026**: Each notebook MUST provide 2D contour visualisations of the surrogate function across input dimension pairs.
- **FR-027**: Each notebook MUST provide a convergence plot showing the running best objective value with the proposed point marked.
- **FR-028**: Each notebook MUST perform a duplicate check against existing observations before outputting the submission.
- **FR-029**: Each notebook MUST document all hyperparameter values and the rationale for changes from week 9 in a markdown section.

### Key Entities

- **Observation**: A pair (x, y) where x is an n-dimensional input vector in [0, 0.999999]ⁿ and y is the scalar objective value. Key attributes: input dimension (varies by function), observation source (initial vs submission), week number.
- **Surrogate Model**: The fitted probabilistic model used to predict the objective at unobserved locations. Key attributes: model type (SFGP, NN), kernel configuration, noise bounds, number of MLL restarts, output transform.
- **Acquisition Function**: The function used to score candidate points for evaluation. Key attributes: function type (qLogNEI, EI blend), batch size q, raw samples for optimisation, exploration parameter (XI).
- **Submission Query**: The formatted candidate point selected for black-box evaluation. Key attributes: dimensionality, decimal precision (6 places), duplicate status.
- **Interior Penalty**: A soft penalty applied to acquisition values near domain boundaries. Key attributes: steepness, floor, applicable functions (F6, F7).
- **MC Dropout Module**: An uncertainty estimation technique for neural network surrogates using multiple stochastic forward passes. Key attributes: dropout rate, number of forward passes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Each of the 5 notebooks (F4–F8) executes end-to-end without errors and produces a valid submission query.
- **SC-002**: All GP-based surrogates (F4, F5, F6, F8) converge during MLL fitting — at least 50% of restarts converge to within 10% of the best loss value.
- **SC-003**: All submission queries contain values strictly within [0, 0.999999] with exactly 6 decimal places and correct dimensionality (F4: 4D, F5: 4D, F6: 5D, F7: 6D, F8: 8D).
- **SC-004**: Each notebook produces at least one surrogate visualisation (2D contour or equivalent) and one convergence plot.
- **SC-005**: All proposed candidates pass the duplicate check — no candidate is identical to an existing observation within 1e-6 tolerance.
- **SC-006**: All hyperparameter changes from week 9 are documented in markdown with rationale in each notebook.

## Assumptions

- Week 10 data files (`updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy`) exist for all functions F4–F8 in their respective data directories.
- The existing week 10 notebooks for F4–F8 contain review/visualisation sections from the earlier review phase and new optimisation sections should be appended (not replace existing content).
- F5 outputs are all strictly positive (>1000), making log transform safe without guards for non-positive values.
- F6's milk constraint refers to a specific input dimension's value being above a threshold; the dimension index is defined in the existing notebook.
- F7's neural network architecture (2L×5N) from week 9 is retained; only the acquisition blend, penalty parameters, candidate pool, and uncertainty method change.
- The BoTorch, GPyTorch, and PyTorch libraries are available in the `sdd-dev` Python environment.
