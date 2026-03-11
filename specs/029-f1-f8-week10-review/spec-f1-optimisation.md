# Feature Specification: F1 Week 10 — SFGP Optimisation Run

**Feature Branch**: `029-f1-f8-week10-review` (same branch — extension)  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "Add optimisation run to existing F1 week 10 notebook. Switch from Hurdle Model to SFGP Matérn-2.5 ARD, use qLogNEI q=4 with distance-based selection, apply log transform to outputs, and seed acquisition with 10,000 Sobol points."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run Optimisation and Propose Next Sample Point (Priority: P1) 🎯 MVP

As a student working on the F1 black box optimisation challenge, I need to add a new optimisation section to the existing F1 week 10 notebook that applies the approved strategy changes (SFGP + qLogNEI + log transform + Sobol seeding) and proposes the next sample point for submission.

**Why this priority**: The challenge requires a submission each week. Without a proposed sample point, no submission can be made.

**Independent Test**: Open the F1 week 10 notebook, run all cells including the new optimisation section, and verify a properly formatted next sample point is produced.

**Acceptance Scenarios**:

1. **Given** the existing F1 week 10 notebook with 12 cells (review + evaluation), **When** the new optimisation cells are appended, **Then** the notebook runs top-to-bottom without errors and the existing review cells remain unchanged.
2. **Given** the SFGP is fitted on log-transformed outputs, **When** I run the optimisation cell, **Then** the GP fits successfully and the printed hyperparameters (lengthscales, noise, outputscale) are reasonable values.
3. **Given** q=4 candidates are proposed by qLogNEI, **When** distance-based selection is applied, **Then** the single best candidate is selected and formatted as `x1-x2` with 6 decimal places, each value in [0.0, 0.999999].

---

### User Story 2 — Visualise New Surrogate and Acquisition Surface (Priority: P2)

As a student, I need to see the GP posterior and acquisition landscape to understand where the model thinks the optimum might be and whether the acquisition function is exploring broadly enough.

**Why this priority**: Visualisation is required by the constitution and supports learning objectives. Without it, the surrogate behaviour cannot be verified.

**Independent Test**: After running all cells, verify the 3-panel contour plot renders showing posterior mean, uncertainty, and acquisition surface.

**Acceptance Scenarios**:

1. **Given** the GP is fitted, **When** the visualisation cell is executed, **Then** a 3-panel contour plot is displayed showing GP posterior mean, GP posterior uncertainty (std), and qLogNEI acquisition surface.
2. **Given** the contour plots are rendered, **When** I examine the point overlays, **Then** initial samples are shown in blue, submission samples in orange, and the proposed next point as a green star.
3. **Given** the visualisation uses a 50×50 grid, **When** I inspect the acquisition surface, **Then** the Sobol-seeded acquisition should show broad exploration across [0,1]².

---

### User Story 3 — Display Convergence with Proposed Point (Priority: P3)

As a student, I need an updated convergence plot that includes the proposed next point to see how it fits into the overall trajectory.

**Why this priority**: Convergence context helps verify the proposal makes sense relative to the optimisation history.

**Independent Test**: Verify the convergence plot shows all existing data points plus the proposed point marked distinctly.

**Acceptance Scenarios**:

1. **Given** the optimisation produces a proposed next point, **When** the convergence plot cell is executed, **Then** existing data and the proposed point (marked with a green star) are displayed with log y-axis.

---

### Edge Cases

- What happens when all outputs are non-positive for the log transform? → Outputs are clamped to ε=1e-300 before log, producing large negative values but valid GP inputs.
- What happens when the GP fit fails to converge during MLL restarts? → Each restart is independent; the best (lowest loss) model is kept. If all fail, BoTorch's default fitting still produces a usable model.
- What happens when all q=4 candidates cluster in the same region? → The distance-based selection step will pick the one farthest from training data, but if all four are close, the result is still the best available candidate.
- What happens when the proposed point is a duplicate of an existing sample? → A duplicate check prints a warning; the point is still valid for submission but noted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: New cells MUST be appended to the existing `functions/f1/f1 - week 10.ipynb` notebook after the current Step 5 (Proposed Strategy Improvements). Existing cells (1–12) MUST NOT be modified.
- **FR-002**: The notebook MUST include all necessary library imports and hyperparameter constants in a dedicated configuration cell before any optimisation logic.
- **FR-003**: Outputs MUST be log-transformed before GP fitting using `log(max(y, ε))` where ε = 1e-300. This transforms the multi-order-of-magnitude range into a manageable GP input space.
- **FR-004**: The surrogate MUST be a single-fidelity Gaussian Process (SFGP) with a Matérn-2.5 ARD kernel (one lengthscale per input dimension), with normalised inputs. The GP is fitted on the manually log-transformed outputs (see FR-003).
- **FR-005**: The GP MUST be fitted using multiple MLL restarts (at least 15) to avoid degenerate hyperparameters. The model with the lowest negative log-likelihood is retained.
- **FR-006**: All hyperparameters MUST be declared as named constants at the top of the configuration cell with clear comments explaining their purpose and chosen values.
- **FR-007**: The acquisition function MUST be q-Log Noisy Expected Improvement (qLogNEI) with q=4 (proposing 4 candidates per batch), using a quasi-Monte Carlo sampler with 512 MC samples.
- **FR-008**: The acquisition optimisation MUST seed the search with at least 10,000 Sobol-generated initial points and use at least 20 multi-start restarts to ensure broad search coverage across the entire [0,1]² domain.
- **FR-009**: From the q=4 candidates, a single submission point MUST be selected using distance-based filtering: keep candidates with posterior mean ≥ median of the batch, then select the one with maximum minimum-distance to all existing training points.
- **FR-010**: The selected point MUST be formatted as `x1-x2` with 6 decimal places, each value clamped to [0.0, 0.999999]. A duplicate check against all existing samples MUST be performed.
- **FR-011**: A 3-panel contour visualisation MUST be produced on a 50×50 grid showing: (1) GP posterior mean, (2) GP posterior standard deviation, (3) acquisition surface — with initial samples (blue), submissions (orange), and proposed point (green star) overlaid.
- **FR-012**: An updated convergence plot MUST be displayed showing the running best with log y-axis, including all existing data points and the proposed next point marked distinctly.

### Key Entities

- **Optimisation config**: 12 named hyperparameter constants (KERNEL_NU, ARD_NUM_DIMS, LS bounds, NOISE_LB, N_MLL_RESTARTS, LOG_EPSILON, MC_SAMPLES, Q_BATCH, NUM_RESTARTS, RAW_SAMPLES, GRID_RES) declared with explanatory comments per FR-006.
- **Log-transformed outputs**: `y_log = log(max(y, 1e-300))` — the GP is fitted in this space; all posterior predictions and comparisons operate on log-scale values.
- **SFGP Model**: Single-fidelity Gaussian Process with Matérn-2.5 ARD kernel, trained via multi-restart MLL optimisation.
- **Acquisition Candidates**: q=4 candidates from `optimize_acqf`, reduced to 1 via distance-based selection.
- **Submission Point**: Single 2D point formatted as `x1-x2`, 6 decimal places, in [0.0, 0.999999].

## Assumptions

- The existing F1 week 10 notebook (cells 1–12) is already complete with data loading, convergence plot, pair plots, evaluation, and improvement suggestions. Data variables (`inputs`, `outputs`, `n_total`, `N_INITIAL`, etc.) are available from earlier cells.
- Per the project constitution, BoTorch/GPyTorch is the mandated GP library, installed in the `sdd-dev` environment.
- The log transform of F1 outputs will produce values in approximately [-690, -35] given the range [1e-300, 7.7e-16].
- All inputs are in [0, 1] and the GP normalises inputs internally (consistent with other notebooks).
- The notebook follows the constitution convention: new sections are appended, existing cells are never modified.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes without errors from top to bottom, including all new optimisation cells.
- **SC-002**: The GP fits successfully with reasonable hyperparameters (lengthscales in [0.01, 2.0], noise > 0, outputscale > 0) after multi-restart MLL optimisation.
- **SC-003**: The proposed next sample point is in the valid range [0.0, 0.999999] for both dimensions and is not a duplicate of any existing sample.
- **SC-004**: The 3-panel surrogate visualisation renders correctly with all three surfaces (mean, std, acquisition) and correct point overlays.
- **SC-005**: The convergence plot includes the proposed point and maintains log y-axis scaling.
- **SC-006**: All hyperparameters are explicitly documented with values and justifications visible in the notebook.
