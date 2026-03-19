# Feature Specification: Week 12 Bayesian Optimisation Loop (F1–F8)

**Feature Branch**: `035-f1-f8-week12-optimisation`  
**Created**: 2026-03-18  
**Status**: Draft  
**Input**: User description: "Implement a new Bayesian optimisation loop for F1 to F8 using the same strategy as week 11 — providing the same performance visualizations as last week."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Generate Week 12 Submission Candidates for All Functions (Priority: P1)

As a student working on the capstone black-box optimisation challenge, I need to run the Bayesian optimisation loop on the latest data for all 8 functions using the same strategies that were active in week 11, so that I can propose next sample points for each function's week 12 submission.

**Why this priority**: Without running the BO loop, there are no submission candidates. This is the primary output of the feature — the remaining submissions depend on having proposed points.

**Independent Test**: Open any single function notebook (e.g., `f1 - week 12.ipynb`), run all cells, and verify it loads the latest data, fits the surrogate, optimises the acquisition function, and outputs a clearly labelled submission query with the proposed next sample point.

**Acceptance Scenarios**:

1. **Given** the week 11 data files exist for a function, **When** I run the week 12 notebook for that function, **Then** it loads the data, fits the surrogate model using the same configuration as the previous optimisation round, and proposes a candidate point.
2. **Given** the notebook is executed, **When** I view the submission query output, **Then** I see the proposed input coordinates, the surrogate configuration summary, fitted hyperparameters, and a duplicate check confirming the point has not been previously observed.
3. **Given** the notebook is executed for any function, **When** I view the proposed point, **Then** the point lies within the valid [0, 1] bounds for each dimension and is not a duplicate of any existing observation.

---

### User Story 2 — Review Performance with Same Visualisations as Previous Weeks (Priority: P1)

As a student, I need each week 12 notebook to include the same performance visualisations as the previous optimisation notebooks (convergence plots, 2D pair plots, performance evaluation, surrogate surface, acquisition surface, and proposed-point convergence), so I can assess the current state before and after proposing the next point.

**Why this priority**: The visualisations provide critical context for understanding whether the optimisation is progressing, stalling, or regressing. Without them, the submission is blind.

**Independent Test**: Run any function's notebook and verify all visualisation sections render correctly: convergence plot, pair plots with green star for best, performance evaluation metrics, surrogate contour panels, and updated convergence showing the proposed point.

**Acceptance Scenarios**:

1. **Given** the notebook is executed, **When** I view the convergence plot, **Then** I see the running best objective value over all iterations, with initial samples in blue and submissions in orange, matching the format from previous weeks.
2. **Given** the notebook is executed, **When** I view the 2D pair plots, **Then** each sample point is coloured by type (blue initial, orange submission), submissions are numbered by week, and the overall best sample is marked with a green star.
3. **Given** the notebook is executed, **When** I view the surrogate visualisation, **Then** I see panels showing the posterior mean surface, uncertainty surface, and acquisition function surface (for functions where this is computationally feasible), with the proposed next point marked.
4. **Given** the notebook is executed, **When** I view the updated convergence, **Then** the proposed point appears at the end of the trajectory with its predicted value.

---

### Edge Cases

- What if the proposed candidate is a duplicate of an existing observation? → The notebook must detect this and flag it clearly. It should not automatically resample — the student decides the next step.
- What if the GP fitting fails to converge on any restart? → The notebook should still complete; the best model from available restarts is used. Zero restarts converging should produce a clear error message.
- What if F1 outputs contain negative values for log-scale display? → Same handling as previous weeks: non-positive outputs are clamped to epsilon before log computation.
- What if the acquisition optimiser returns a boundary point? → For functions using interior penalty, boundary points are naturally down-weighted. For functions without interior penalty, boundary points are valid candidates.
- What if F7's neural network training does not converge? → Training should complete for the configured number of epochs regardless; the student assesses the loss curve.
- What if a high-dimensional function (F7=6D, F8=8D) produces too many pair plot subplots? → Use the same grid layout approach as week 11 with appropriately sized figures.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create 8 new notebooks named `fX - week 12.ipynb` (X = 1..8), each in its corresponding `./functions/fX/` folder, following the constitution convention for weekly iteration notebooks.
- **FR-002**: Each notebook MUST be self-contained with all imports, configuration, data loading, surrogate fitting, acquisition optimisation, candidate selection, visualisation, and submission query — executable independently without running other notebooks.
- **FR-003**: Each notebook MUST load `updated_inputs - Week 11.npy` and `updated_outputs - Week 11.npy` from the corresponding `./data/fX/` folder. Week 11 data is the latest available; the notebooks propose candidates for the week 12 submission.
- **FR-004**: Each notebook MUST use the **same surrogate model, kernel, output transform, acquisition function, and selection method** as used in the week 10 optimisation notebooks (which produced the most recent submissions). The per-function strategy is specified in the Strategy Reference Table below.
- **FR-005**: Each notebook MUST display a **convergence plot** showing the running best (maximum) objective value over all iterations, with initial samples in blue and submissions in orange. F1 MUST use a logarithmic y-axis (non-positive values clamped to epsilon).
- **FR-006**: Each notebook MUST display **2D pair plots** for all unique pairs of input dimensions, with initial samples in blue (unmarked), submissions in orange (numbered by week), and the overall best sample marked with a green star (marker size 500, zorder 5). The legend MUST include "Best" with the star marker.
- **FR-007**: Each notebook MUST contain a **performance evaluation** section computing: best value found, number of improvements over initial, consecutive non-improving stretch, and stalling detection (3+ consecutive submissions without improvement).
- **FR-008**: Each notebook MUST output a clearly labelled **submission query** block containing: the surrogate configuration summary, fitted hyperparameters (lengthscales, noise, output scale), proposed input coordinates clamped to [0, 0.999999], and a duplicate check against all existing observations.
- **FR-009**: For functions where contour visualisation is feasible (2D input), the notebook MUST display a **3-panel surrogate visualisation**: (1) posterior mean surface, (2) posterior uncertainty surface, (3) acquisition function surface, with existing data points and the proposed next point overlaid.
- **FR-010**: Each notebook MUST display an **updated convergence plot** showing the existing trajectory plus the proposed point with its predicted value at the end.
- **FR-011**: Each notebook MUST use the **multi-restart MLL fitting** approach (multiple random seeds with the best negative MLL selected) to guard against degenerate hyperparameter optima, with the restart count matching the previous optimisation round.
- **FR-012**: For functions using **distance-based candidate selection** (batch q > 1), the notebook MUST apply the same two-stage selection: quality gate (candidates with posterior mean at or above the median/percentile threshold) followed by maximum minimum-distance to existing data.
- **FR-013**: For functions using **interior penalty**, the notebook MUST apply the same penalty function and parameters to re-score qualified candidates before final selection.

### Strategy Reference Table

Each function MUST use the strategy from its week 10 optimisation notebook:

| Function | Surrogate | Kernel | Output Transform | Acquisition | Selection | Interior Penalty |
|----------|-----------|--------|-----------------|-------------|-----------|-----------------|
| F1 (2D) | SFGP | Matérn-2.5 ARD | log(y) clamped | qLogNEI q=4 | Median gate + max min-distance | STEEPNESS=0.5, FLOOR=0.01 |
| F2 (2D) | SFGP | Matérn-2.5 ARD | Standardize(m=1) | qLogNEI q=4 | Median gate + max min-distance | STEEPNESS=0.02, FLOOR=0.01 |
| F3 (3D) | SFGP | Matérn-2.5 ARD | Shift (y − y_min) | qLogNEI q=3 | Median gate + max min-distance | None |
| F4 (4D) | SFGP | Matérn-2.5 ARD | Standardize(m=1) | qLogNEI q=4 | Median gate + max min-distance | None |
| F5 (4D) | SFGP | Matérn-1.5 ARD | log(y) + Standardize(m=1) | qLogNEI q=4 | 25th-pctl gate + max min-distance | None |
| F6 (5D) | SFGP | Matérn-1.5 ARD | Standardize(m=1) | qLogNEI q=4 | Rank-based (mean + penalty), milk ≥ 0.12 | Rank-based, STEEPNESS=1.0, FLOOR=0.01 |
| F7 (6D) | Neural Network (6→5→5→1) | N/A | Manual z-score | 50% mean + 50% EI (MC dropout) | Penalised acquisition argmax | STEEPNESS=0.02, FLOOR=0.02 |
| F8 (8D) | SFGP | Matérn-2.5 ARD | Standardize(m=1) | qLogNEI q=1 | No batch selection (q=1) | None |

### Function Data Summary (Week 11 — latest available)

| Function | Input Dims | Initial Samples | Total Samples | Submissions |
|----------|-----------|----------------|---------------|-------------|
| F1 | 2 | 10 | 21 | 11 |
| F2 | 2 | 10 | 21 | 11 |
| F3 | 3 | 15 | 26 | 11 |
| F4 | 4 | 30 | 41 | 11 |
| F5 | 4 | 20 | 31 | 11 |
| F6 | 5 | 20 | 31 | 11 |
| F7 | 6 | 30 | 41 | 11 |
| F8 | 8 | 40 | 51 | 11 |

### Key Entities

- **Function Data**: Input/output numpy arrays for each of the 8 functions, loaded from `./data/fX/` folders. Inputs are in [0, 1], outputs are 1-dimensional real values. All functions are maximisation tasks.
- **Surrogate Model**: The fitted model (GP or NN) that approximates the black-box function from observed data. Hyperparameters are fitted per execution.
- **Acquisition Function**: The criterion used to select the next sample point, balancing exploration (uncertain regions) and exploitation (high-predicted regions).
- **Submission Query**: The final output of each notebook — the proposed input coordinates for the next evaluation of the black-box function.
- **Convergence Plot**: Line chart of running best (max) objective. X-axis = sample number, Y-axis = best-so-far value. Blue for initial, orange for submissions.
- **2D Pair Plots**: Scatter plots for each pair of input dimensions. Colour distinguishes initial (blue) from submission (orange). Submissions numbered by week. Best sample marked with green star.

## Assumptions

- Week 11 data files exist in all `./data/fX/` folders (confirmed present for all 8 functions).
- Week 12 data does NOT yet exist — these notebooks propose candidates for the week 12 submission using week 11 data.
- Sample counts in the table above are confirmed from week 11 data files.
- All 8 functions are maximisation tasks with 1-dimensional output.
- The "same strategy as week 11" means the strategy configurations from the week 10 optimisation notebooks are retained without modification.
- Contour visualisations (3-panel GP surfaces) are feasible for 2D functions (F1, F2) and may be included for higher-dimensional functions using 2D slices at the discretion of the implementer. At minimum, F1 and F2 must have contour visualisations.
- F7 uses a neural network surrogate — its visualisation will differ from GP-based functions (no contour surface; training loss curve may substitute).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 8 notebooks execute from top to bottom without errors and produce all required visualisations and submission queries.
- **SC-002**: Each notebook proposes a candidate point within valid bounds [0, 0.999999] per dimension that is not a duplicate of any existing observation.
- **SC-003**: Convergence plots clearly show the optimisation trajectory for each function, with initial/submission colour coding and (for F1) logarithmic y-axis.
- **SC-004**: 2D pair plots display all sample points with correct weekly numbering, two-colour coding, and a green star marking the overall best output location.
- **SC-005**: Performance evaluation correctly identifies stalling functions (3+ consecutive submissions without improvement) and reports improvement counts.
- **SC-006**: Submission query blocks contain all required information (surrogate config, fitted hyperparameters, proposed coordinates, duplicate check) for direct use in the course submission portal.
