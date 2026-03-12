# Feature Specification: F2 Week 10 — SFGP Optimisation Run

**Feature Branch**: `029-f1-f8-week10-review` (same branch — extension)  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "Add optimisation run to existing F2 week 10 notebook. Switch from Matérn-1.5 to Matérn-2.5 ARD, add Standardize(m=1) output transform, increase MLL restarts to 50, widen lengthscale bounds to [0.005, 10.0], add 4,096 Sobol seed points for acquisition multi-start, and provide surrogate visualisation with sampled points in a different colour and proposed point."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

As a student working on the F2 black box optimisation challenge, I need to add a new optimisation section to the existing F2 week 10 notebook that applies the approved strategy changes (SFGP Matérn-2.5 ARD + Standardize(m=1) + wider LS bounds + 50 MLL restarts + Sobol seeding) and proposes the next sample point for submission.

**Why this priority**: The challenge requires a submission each week. Without a proposed sample point, no submission can be made.

**Independent Test**: Open the F2 week 10 notebook, run all cells including the new optimisation section, and verify a properly formatted next sample point is produced.

**Acceptance Scenarios**:

1. **Given** the existing F2 week 10 notebook with 12 cells (review + evaluation), **When** the new optimisation cells are appended, **Then** the notebook runs top-to-bottom without errors and the existing review cells remain unchanged.
2. **Given** the SFGP is fitted with Matérn-2.5 ARD and Standardize(m=1), **When** I run the optimisation cell, **Then** the GP fits successfully and the printed hyperparameters (lengthscales, noise, outputscale) are reasonable values with lengthscales within [0.005, 10.0].
3. **Given** q=4 candidates are proposed by qLogNEI, **When** distance-based selection is applied, **Then** the single best candidate is selected and formatted as `x1-x2` with 6 decimal places, each value in [0.0, 0.999999].

---

### User Story 2 — Visualise New Surrogate and Acquisition Surface (Priority: P2)

As a student, I need to see the GP posterior and acquisition landscape to understand where the model thinks the optimum might be and whether the acquisition function is exploring broadly enough. All sampled points should be visible in a distinct colour from the proposed point.

**Why this priority**: Visualisation is required by the constitution and supports learning objectives. Without it, the surrogate behaviour cannot be verified.

**Independent Test**: After running all cells, verify the 3-panel contour plot renders showing posterior mean, uncertainty, and acquisition surface with colour-coded data point overlays.

**Acceptance Scenarios**:

1. **Given** the GP is fitted, **When** the visualisation cell is executed, **Then** a 3-panel contour plot is displayed showing GP posterior mean, GP posterior uncertainty (std), and qLogNEI acquisition surface.
2. **Given** the contour plots are rendered, **When** I examine the point overlays, **Then** initial samples are shown in blue, submission samples in orange, and the proposed next point as a green star — all clearly distinguishable.
3. **Given** the visualisation uses a 50×50 grid, **When** I inspect the acquisition surface, **Then** the Sobol-seeded acquisition should show broad exploration across [0,1]².

---

### User Story 3 — Display Convergence with Proposed Point (Priority: P3)

As a student, I need an updated convergence plot that includes the proposed next point to see how it fits into the overall trajectory.

**Why this priority**: Convergence context helps verify the proposal makes sense relative to the optimisation history.

**Independent Test**: Verify the convergence plot shows all existing data points plus the proposed point marked distinctly.

**Acceptance Scenarios**:

1. **Given** the optimisation produces a proposed next point, **When** the convergence plot cell is executed, **Then** existing data and the proposed point (marked with a green star) are displayed with the running best trajectory.

---

### Edge Cases

- What happens when the GP fit fails to converge during MLL restarts? → Each restart is independent; the best (lowest loss) model is kept. With 50 restarts, degenerate fits are less likely.
- What happens when all q=4 candidates cluster in the same region? → The distance-based selection step will pick the one farthest from training data, but if all four are close, the result is still the best available candidate.
- What happens when the proposed point is a duplicate of an existing sample? → A duplicate check prints a warning; the point is still valid for submission but noted.
- What happens when Standardize(m=1) encounters constant outputs? → BoTorch's Standardize handles edge cases internally; with 20 diverse samples this is extremely unlikely.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: New cells MUST be appended to the existing `functions/f2/f2 - week 10.ipynb` notebook after the current Step 5 (Proposed Strategy Improvements). Existing cells (1–12) MUST NOT be modified.
- **FR-002**: The notebook MUST include all necessary library imports and hyperparameter constants in a dedicated configuration cell before any optimisation logic.
- **FR-003**: The surrogate MUST be a single-fidelity Gaussian Process (SFGP) with a Matérn-2.5 ARD kernel (one lengthscale per input dimension) and standardised outputs via `Standardize(m=1)`. Inputs are already in [0,1] so no additional input normalisation is required.
- **FR-004**: Lengthscale bounds MUST be set to [0.005, 10.0] via `Interval` constraint, widened from the week 9 bounds of [0.01, 2.0] to allow the GP to capture both fine and coarse-scale structure.
- **FR-005**: The GP MUST be fitted using at least 50 MLL restarts to avoid degenerate hyperparameters (increased from week 9's 15 restarts, matching the aggressive restart count used for F5/F6). The model with the lowest negative log-likelihood is retained.
- **FR-006**: All hyperparameters MUST be declared as named constants at the top of the configuration cell with clear comments explaining their purpose and chosen values, including justification for changes from week 9.
- **FR-007**: The acquisition function MUST be q-Log Noisy Expected Improvement (qLogNEI) with q=4 (proposing 4 candidates per batch), using a quasi-Monte Carlo sampler with 512 MC samples.
- **FR-008**: The acquisition optimisation MUST seed the search with at least 4,096 Sobol-generated initial points (up from week 9's 1,024, matching F8's aggressive count) and use at least 20 multi-start restarts to ensure broad search coverage across the entire [0,1]² domain.
- **FR-009**: From the q=4 candidates, a single submission point MUST be selected using distance-based filtering: keep candidates with posterior mean ≥ median of the batch, then select the one with maximum minimum-distance to all existing training points.
- **FR-010**: The selected point MUST be formatted as `x1-x2` with 6 decimal places, each value clamped to [0.0, 0.999999]. A duplicate check against all existing samples MUST be performed.
- **FR-011**: A 3-panel contour visualisation MUST be produced on a 50×50 grid showing: (1) GP posterior mean, (2) GP posterior standard deviation, (3) acquisition surface — with initial samples (blue), submissions (orange), and proposed point (green star) overlaid.
- **FR-012**: An updated convergence plot MUST be displayed showing the running best with the proposed next point marked distinctly (green star).

### Key Entities

- **Optimisation config**: Named hyperparameter constants (KERNEL_NU=2.5, ARD_NUM_DIMS=2, LS_LOWER=0.005, LS_UPPER=10.0, NOISE_LB=1e-4, N_MLL_RESTARTS=50, MC_SAMPLES=512, Q_BATCH=4, NUM_RESTARTS=20, RAW_SAMPLES=4096, GRID_RES=50) declared with explanatory comments per FR-006, including justification for week 9 → week 10 changes. NOISE_LB reduced from 1e-3 to 1e-4 because Standardize(m=1) rescales outputs to unit variance.
- **SFGP Model**: Single-fidelity Gaussian Process with Matérn-2.5 ARD kernel and `Standardize(m=1)` output transform, trained via multi-restart MLL optimisation.
- **Acquisition Candidates**: q=4 candidates from `optimize_acqf`, reduced to 1 via distance-based selection.
- **Submission Point**: Single 2D point formatted as `x1-x2`, 6 decimal places, in [0.0, 0.999999].

## Assumptions

- The existing F2 week 10 notebook (cells 1–12) is already complete with data loading, convergence plot, pair plots, evaluation, and improvement suggestions. Data variables (`inputs`, `outputs`, `n_total`, `N_INITIAL`, `N_DIMS`, etc.) are available from earlier cells.
- Per the project constitution, BoTorch/GPyTorch is the mandated GP library, installed in the `sdd-dev` environment.
- F2 outputs are in a normal range (approximately [0.25, 0.67]) — no log transform is needed, but `Standardize(m=1)` output transform is applied for better GP conditioning.
- All inputs are in [0, 1] and the GP normalises inputs internally (consistent with other notebooks).
- The notebook follows the constitution convention: new sections are appended, existing cells are never modified.
- F2 has N_DIMS=2, N_INITIAL=10, and 10 weekly submissions (weeks 3–10 inclusive), totalling 20 samples.

## Clarifications

### Session 2026-03-11

- Q: How many MLL restarts should be used? → A: 50 (aggressive, matches F5/F6 patterns)
- Q: What noise lower bound with Standardize(m=1)? → A: 1e-4 (reduced from 1e-3 to account for output rescaling)
- Q: How many Sobol raw samples for acquisition? → A: 4,096 (aggressive, matches F8's count)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes without errors from top to bottom, including all new optimisation cells.
- **SC-002**: The GP fits successfully with reasonable hyperparameters (lengthscales in [0.005, 10.0], noise ≥ 1e-4, outputscale > 0) after multi-restart MLL optimisation.
- **SC-003**: The proposed next sample point is in the valid range [0.0, 0.999999] for both dimensions and is not a duplicate of any existing sample.
- **SC-004**: The 3-panel surrogate visualisation renders correctly with all three surfaces (mean, std, acquisition) and correct point overlays (blue initial, orange submissions, green star proposed).
- **SC-005**: The convergence plot includes the proposed point marked distinctly.
- **SC-006**: All hyperparameters are explicitly documented with values and justifications for changes from week 9 visible in the notebook.
