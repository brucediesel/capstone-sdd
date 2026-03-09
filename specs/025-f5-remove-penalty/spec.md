# Feature Specification: F5 Week 9 — Remove Interior Penalty

**Feature Branch**: `025-f5-remove-penalty`  
**Created**: 2026-03-09  
**Status**: Draft  
**Input**: User description: "create a new branch for F5 this week - remove interior penalty from week 9 notebooks - don't make any other changes"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Remove Interior Penalty from Acquisition (Priority: P1)

As a researcher, I want the F5 week 9 notebook to use plain qLogNEI acquisition without the interior penalty wrapper, so that the acquisition function is not artificially suppressing boundary candidates and I can evaluate whether the unpenalised optimiser produces better proposals.

**Why this priority**: The interior penalty is the only change requested. Removing it from the acquisition pipeline is the core deliverable — everything else follows from this.

**Independent Test**: Run the acquisition cell — `optimize_acqf` passes `nei` directly as `acq_function` without any `PenalisedAcquisition` wrapper. No `STEEPNESS`, `FLOOR`, or `EPS_BOUND` variables are referenced. Candidates are optimised over the standard `[0, 1]` bounds.

**Acceptance Scenarios**:

1. **Given** the F5 week 9 notebook with interior penalty code, **When** the penalty is removed, **Then** the acquisition cell uses plain `qLogNoisyExpectedImprovement` passed directly to `optimize_acqf` with standard `[0, 1]^4` bounds
2. **Given** the `PenalisedAcquisition` class and `penalised_nei` variable exist, **When** the penalty is removed, **Then** neither the class definition nor any reference to it remains in the notebook
3. **Given** constants `STEEPNESS`, `FLOOR`, and `EPS_BOUND` exist in the hyperparameters cell, **When** the penalty is removed, **Then** these constants are deleted from the cell

---

### User Story 2 — Remove Penalty Visualisation (Priority: P1)

As a researcher, I want penalty-specific visualisation cells removed so the notebook only shows the standard surrogate and convergence plots without penalty-related panels.

**Why this priority**: The penalty visualisation (Step 6) is meaningless without the penalty. Keeping it would cause runtime errors or confusion.

**Independent Test**: The notebook contains no penalty contour panels. The surrogate visualisation (Step 5) uses the base NEI selected point. The penalty visualisation section (Step 6 markdown + code) is removed entirely.

**Acceptance Scenarios**:

1. **Given** Step 6 contains a 3-panel penalty visualisation, **When** the penalty is removed, **Then** the entire Step 6 section (markdown header + code cell) is removed from the notebook
2. **Given** Step 5 visualisation references `next_x_ip` (the penalty-selected point), **When** the penalty is removed, **Then** Step 5 uses `best_point` (the base NEI distance-selected point) and the plot title no longer mentions "IP"

---

### User Story 3 — Update Documentation and Submission (Priority: P2)

As a researcher, I want the notebook title, hyperparameter table, submission cell, and strategy recommendations updated to reflect that interior penalty has been removed, so the notebook is internally consistent and accurately describes the current strategy.

**Why this priority**: Documentation consistency is important but secondary to the functional code changes. The notebook must be self-consistent for submission.

**Independent Test**: The notebook title does not mention "Interior Penalty". The hyperparameter table has no IP rows. The submission cell shows only the base NEI submission. The strategy section notes the penalty was removed.

**Acceptance Scenarios**:

1. **Given** the title mentions "Interior Penalty", **When** the penalty is removed, **Then** the title reads "GP Matérn-5/2 + qLogNEI (4D)" without penalty reference
2. **Given** the hyperparameter table contains rows 16 (IP STEEPNESS) and 17 (IP FLOOR), **When** the penalty is removed, **Then** those rows are deleted and the table has 15 rows
3. **Given** the submission cell shows both base NEI and IP submissions, **When** the penalty is removed, **Then** only the base NEI submission is shown and no penalty parameters are printed
4. **Given** the strategy section recommends relaxing STEEPNESS, **When** the penalty is removed, **Then** the recommendations note the penalty was evaluated and removed, and focus on other potential improvements

---

### Edge Cases

- What happens if the base NEI still proposes boundary-stuck candidates? The notebook still produces valid submissions — the convergence and exploration metrics flag any issues for the researcher to evaluate.
- What happens if the Step 4 penalty explanation markdown is kept? It is removed entirely since the code it explains no longer exists.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The `PenalisedAcquisition` class definition MUST be removed from the notebook
- **FR-002**: The `penalised_nei` variable and its usage in `optimize_acqf` MUST be removed; `optimize_acqf` MUST pass `nei` directly as `acq_function`
- **FR-003**: Constants `STEEPNESS`, `FLOOR`, and `EPS_BOUND` MUST be removed from the hyperparameters cell
- **FR-004**: The tightened bounds `BOUNDS_IP` MUST be removed; acquisition MUST use the standard `BOUNDS` (`[0, 1]^4`)
- **FR-005**: The Step 4 interior penalty explanation markdown cell MUST be removed entirely
- **FR-006**: The Step 4 interior penalty code cell MUST be removed entirely (the cell containing `PenalisedAcquisition`, `penalised_nei`, and the penalised optimization)
- **FR-007**: The Step 6 penalty visualisation section (markdown header + code cell) MUST be removed entirely
- **FR-008**: The Step 5 surrogate visualisation MUST reference the base NEI `best_point` instead of penalty-selected `next_x_ip`; the plot title MUST not mention "IP"
- **FR-009**: The Step 8 submission cell MUST show only the base NEI selected point; all penalty diagnostic output and parameter references MUST be removed
- **FR-010**: The notebook title, hyperparameter table, and strategy recommendations MUST be updated to remove all penalty references
- **FR-011**: No other changes MUST be made — surrogate model, kernel, output transform (log1p → z-score), acquisition function type (qLogNEI), MC samples (512), q value (4), acquisition restarts (50), raw samples (3000), distance-based selection logic, MLL restarts (15), and all existing non-penalty code MUST remain unchanged

### Key Entities

- **F5 Week 9 Notebook**: `functions/f5/f5 - week 9.ipynb` — the single file modified by this feature
- **Interior Penalty Components**: `PenalisedAcquisition` class, `STEEPNESS`/`FLOOR`/`EPS_BOUND` constants, `penalised_nei` variable, `BOUNDS_IP` tightened bounds, Step 4 explanation and code cells, Step 6 visualisation cells

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes end-to-end without errors, producing a valid submission query in `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` format with all values in [0.0, 0.999999]
- **SC-002**: Zero references to `PenalisedAcquisition`, `penalised_nei`, `STEEPNESS`, `FLOOR`, `EPS_BOUND`, or `BOUNDS_IP` exist anywhere in the notebook
- **SC-003**: The surrogate model configuration is identical to before (GP Matérn-5/2, ARD, log1p → z-score, 15 restarts, outcome_transform=None)
- **SC-004**: The base acquisition configuration is identical to before (qLogNEI, q=4, 50 restarts, 3000 raw samples, distance-based selection)
- **SC-005**: All visualisations render correctly — surrogate plots use base NEI point, convergence plot unchanged, no penalty panels present
- **SC-006**: LOO cross-validation and convergence metrics cells execute unchanged and produce valid results

## Assumptions

- The existing base NEI acquisition (Step 3) already produces valid candidates via distance-based selection; removing the penalty simply means using those candidates directly for submission
- The notebook's output transform (log1p → z-score), GP configuration, and all other hyperparameters are correct and should not be modified
- Step 3 (base NEI) remains fully intact — only Step 4 (penalty wrapper), Step 6 (penalty viz), and penalty references in other cells are removed
- Performance evaluation cells (convergence metrics, exploration spread, LOO) do not reference interior penalty and need no changes
