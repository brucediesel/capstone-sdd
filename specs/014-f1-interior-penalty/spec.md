# Feature Specification: F1 Interior Penalty on Acquisition Function

**Feature Branch**: `014-f1-interior-penalty`  
**Created**: 2026-02-24  
**Status**: Draft  
**Input**: User description: "update F1 by adding an interior penalty to the acquisition function — a soft penalty that decays towards the edges, with a steepness hyperparameter and a floor to ensure the penalty never reaches zero"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add Interior Penalty to F1 Acquisition (Priority: P1)

As a student running Bayesian Optimisation on Function 1 (2D radiation source detection), I want the acquisition function to discourage candidates near the search space boundaries so that proposed sample points are pushed towards the interior where the radiation source is more plausible.

**Why this priority**: F1 candidates have been clustering on the edges of [0, 1]². This is the core change that prevents boundary-hugging proposals.

**Independent Test**: Run the notebook through the new section — the proposed next sample point should have both coordinates comfortably away from 0 and 1 (e.g. ≥ 0.05 and ≤ 0.95), and the penalised acquisition surface plot should visibly show reduced values along all four edges.

**Acceptance Scenarios**:

1. **Given** the F1 notebook has the existing Week 7 hurdle model and weighted UCB code, **When** a new section titled "Week 7 — Interior Penalty" is appended (existing cells unchanged), **Then** the notebook runs end-to-end without errors and produces a modified acquisition landscape.
2. **Given** the `STEEPNESS` hyperparameter is set to a moderate value (e.g. 2.0), **When** the acquisition function is evaluated on a uniform grid, **Then** the penalised acquisition surface clearly shows suppressed values in a visible band along all four boundaries.
3. **Given** the `FLOOR` hyperparameter is set to a positive value (e.g. 0.01), **When** a candidate sits exactly on a corner (0, 0), **Then** the interior penalty factor at that point equals `FLOOR` (never zero), preserving gradient information.

---

### User Story 2 - Explicit Hyperparameter Documentation (Priority: P2)

As a student submitting a capstone notebook, I want the steepness and floor hyperparameters to be clearly defined in a single constants cell with markdown documentation explaining their effect, so that the examiner can understand and modify them.

**Why this priority**: Capstone requirement — all hyperparameters must be explicit and justified.

**Independent Test**: Read the hyperparameter cell and markdown — `STEEPNESS` and `FLOOR` should be defined, printed, and their rationale explained in the preceding markdown.

**Acceptance Scenarios**:

1. **Given** the new section is added, **When** the hyperparameter cell is inspected, **Then** `STEEPNESS` and `FLOOR` are defined as named constants alongside the existing Week 7 hyperparameters.
2. **Given** the markdown table documents the parameters, **When** a reviewer reads the table, **Then** the rationale for the chosen default values is clear and relates to the F1 problem geometry.

---

### User Story 3 - Visualise Interior Penalty Effect (Priority: P2)

As a student, I want the 3-panel surrogate visualisation to include the interior penalty effect so that I can see how the acquisition landscape changes compared to the unpenalised version.

**Why this priority**: Visual confirmation is essential for the capstone submission and for tuning the steepness parameter.

**Independent Test**: The acquisition surface panel should show a visible "moat" of low values along the edges relative to the centre.

**Acceptance Scenarios**:

1. **Given** the notebook is executed, **When** the penalised acquisition surface is plotted, **Then** the colour map clearly shows suppressed values near all four boundaries.
2. **Given** the convergence plot is generated, **When** compared to the Week 7 convergence plot, **Then** both the previous running best and the new proposed point are shown.

---

### Edge Cases

- What happens when `STEEPNESS` is set to 0? The interior penalty becomes a flat constant — effectively disabling the penalty. Documented as valid degenerate case.
- What happens when `FLOOR` is set to 1.0? The penalty has no effect — the acquisition is unchanged. Documented as a degenerate case.
- What happens at exact corners (0,0), (0,1), (1,0), (1,1)? sin(0) = sin(π) = 0, so the raw penalty is 0; the floor ensures the total factor equals `FLOOR` (never zero).
- What happens when steepness is very large (e.g. 50)? The penalty becomes a near-step function — effectively shrinking the search space to a smaller interior box. Documented in hyperparameter table.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST add a new section after the existing Week 7 submission cells. Existing code cells MUST NOT be modified or replaced.
- **FR-002**: The interior penalty MUST be implemented as a multiplicative factor applied to the existing penalised acquisition function, computed as: for each candidate $x = (x_1, x_2)$, the penalty is $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{d} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$. This evaluates to `FLOOR` at any edge and approaches 1.0 at the centre.
- **FR-003**: The `STEEPNESS` hyperparameter MUST be defined as a named constant and control how aggressively boundary regions are penalised. Higher values create a steeper transition from low to high penalty.
- **FR-004**: The `FLOOR` hyperparameter MUST be defined as a named constant and ensure the penalty factor never reaches zero, preserving optimiser gradients at the boundaries.
- **FR-005**: The section MUST include a hyperparameter documentation table explaining `STEEPNESS`, `FLOOR`, and their relationship to the existing Week 7 parameters (`KAPPA`, `PENALTY_RADIUS`).
- **FR-006**: The section MUST produce a 3-panel visualisation matching the Week 7 style: (1) hurdle mean prediction, (2) hurdle uncertainty, (3) penalised acquisition with interior penalty applied — all including the new proposed point.
- **FR-007**: The section MUST produce a convergence plot showing the running best observed values and the new candidate.
- **FR-008**: The section MUST output a formatted submission query string in the `x1-x2` format with 6 decimal places, clipped to [0.0, 0.999999].
- **FR-009**: The section MUST validate that the proposed point satisfies: (a) all coordinates in [0.0, 0.999999], (b) minimum distance ≥ 0.05 from any existing data point.
- **FR-010**: The interior penalty MUST be applied to the full candidate set before selecting the best candidate, not as a post-hoc filter.

### Key Entities

- **Interior Penalty Function**: A smooth, multiplicative weight $w(x) \in [\text{FLOOR}, 1.0]$ applied to the acquisition score at each candidate. Parameterised by `STEEPNESS` and `FLOOR`.
- **STEEPNESS**: Controls the width of the boundary suppression zone. Higher values create a narrower, steeper transition. Lower values create a broader, gentler suppression zone.
- **FLOOR**: The minimum value of the penalty at the boundary. Prevents the penalty from being exactly zero, preserving gradient information for the optimiser.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The proposed next sample point has all coordinates at least 0.05 away from both 0 and 1 (i.e. in [0.05, 0.95]).
- **SC-002**: The penalised acquisition surface visualisation shows at least a 50% reduction in acquisition value within 0.05 of any boundary compared to the interior maximum.
- **SC-003**: The notebook section runs end-to-end without errors when all preceding Week 7 cells have been executed.
- **SC-004**: The submission query is in valid format: `0.xxxxxx-0.xxxxxx` with both values in [0.0, 0.999999].
- **SC-005**: All existing notebook cells remain unchanged — the new section is purely additive.

## Assumptions

- The existing Week 7 code (hurdle model, weighted UCB, local penalization) has already been executed and variables (`X_w7`, `y_w7`, `stage1_clf`, `stage2_rf`, `KAPPA`, `PENALTY_RADIUS`, `N_CANDIDATES`, `GRID_RES`, `FALLBACK_MODE`, `y_binary`) are available in the notebook kernel.
- The search space is [0, 1]² — a 2-dimensional unit square.
- The sin(πx)² formulation naturally handles the [0, 1] range without additional normalisation.
- Default values of `STEEPNESS = 2.0` and `FLOOR = 0.01` are reasonable starting points; the student may adjust based on observed results.
- This feature only modifies F1. No other function notebooks are changed.
