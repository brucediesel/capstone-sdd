# Feature Specification: F5 Interior Penalty on Acquisition Function

**Feature Branch**: `015-f5-interior-penalty`  
**Created**: 2026-02-24  
**Status**: Draft  
**Input**: User description: "update F5 by adding an interior penalty to the acquisition function — a soft penalty that decays towards the edges, with a steepness hyperparameter and a floor to ensure the penalty never reaches zero"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add Interior Penalty to F5 Acquisition (Priority: P1)

As a student running Bayesian Optimisation on Function 5 (4D chemical yield), I want the acquisition function to discourage candidates near the search space boundaries so that proposed sample points are pushed towards the interior where the true optimum is more likely to reside.

**Why this priority**: F5 candidates from the GP + NEI pipeline tend to cluster near edges of the 4D hypercube because GP posterior variance is highest far from data. This is the core change that prevents boundary-hugging proposals.

**Independent Test**: Run the notebook through the new section — the proposed next sample point should have all four coordinates comfortably away from 0 and 1 (e.g. ≥ 0.05 and ≤ 0.95), and the surrogate visualisation should show reduced acquisition values along the edges.

**Acceptance Scenarios**:

1. **Given** the F5 notebook has the existing Week 7 GP + NEI code, **When** a new section titled "Week 7 — Interior Penalty" is appended (existing cells unchanged), **Then** the notebook runs end-to-end without errors and produces a modified acquisition landscape.
2. **Given** the `STEEPNESS` hyperparameter is set to a moderate value (e.g. 2.0), **When** the acquisition function is evaluated, **Then** the penalised acquisition surface clearly shows suppressed values near all boundaries of the 4D space.
3. **Given** the `FLOOR` hyperparameter is set to a positive value (e.g. 0.01), **When** a candidate sits exactly on a corner, **Then** the interior penalty factor at that point equals `FLOOR` (never zero), preserving gradient information for the optimiser.

---

### User Story 2 - Explicit Hyperparameter Documentation (Priority: P2)

As a student submitting a capstone notebook, I want the steepness and floor hyperparameters to be clearly defined in a single constants cell with markdown documentation explaining their effect, so that the examiner can understand and modify them.

**Why this priority**: Capstone requirement — all hyperparameters must be explicit and justified.

**Independent Test**: Read the hyperparameter cell and markdown — `STEEPNESS` and `FLOOR` should be defined, printed, and their rationale explained.

**Acceptance Scenarios**:

1. **Given** the new section is added, **When** the hyperparameter cell is inspected, **Then** `STEEPNESS` and `FLOOR` are defined as named constants alongside the existing Week 7 hyperparameters.
2. **Given** the markdown table documents the parameters, **When** a reviewer reads the table, **Then** the rationale for the chosen default values is clear and relates to the F5 problem geometry (4D chemical yield).

---

### User Story 3 - Visualise Interior Penalty Effect (Priority: P2)

As a student, I want the 3-panel surrogate visualisation to include the interior penalty effect so that I can see how the acquisition landscape changes compared to the unpenalised version.

**Why this priority**: Visual confirmation is essential for the capstone submission and for tuning the steepness parameter.

**Independent Test**: The acquisition surface panel should show visibly reduced values near the edges of the 2D slice.

**Acceptance Scenarios**:

1. **Given** the notebook is executed, **When** the penalised acquisition surface is plotted on a 2D slice, **Then** the colour map shows suppressed values near all four edges of the slice.
2. **Given** the convergence plot is generated, **When** compared to the Week 7 convergence plot, **Then** both the previous running best and the new proposed point are shown.

---

### Edge Cases

- What happens when `STEEPNESS` is set to 0? The interior penalty becomes a flat constant — effectively disabling the penalty. Documented as valid degenerate case.
- What happens when `FLOOR` is set to 1.0? The penalty has no effect — the acquisition is unchanged. Documented as a degenerate case.
- What happens at exact corners (e.g. all coordinates 0 or 1)? sin(0) = sin(π) = 0, so the raw penalty is 0; the floor ensures the total factor equals `FLOOR` (never zero).
- What happens when steepness is very large (e.g. 50)? The penalty becomes a near-step function in each dimension — effectively shrinking the search space to a smaller interior hypercube. Documented in hyperparameter table.
- What happens in 4D where the penalty is a product of 4 sin² terms? Corner suppression is exponentially stronger: a 4D corner receives FLOOR⁴ ≈ 0 (with FLOOR=0.01 and no additive floor the raw product would approach 0, but the additive floor ensures the minimum is exactly FLOOR).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST add a new section after the existing Week 7 submission cells. Existing code cells MUST NOT be modified or replaced.
- **FR-002**: The interior penalty MUST be implemented as a multiplicative factor applied to the acquisition candidates, computed as: for each candidate $x = (x_1, x_2, x_3, x_4)$, the penalty is $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{4} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$. This evaluates to `FLOOR` at any edge and approaches 1.0 at the centre.
- **FR-003**: The `STEEPNESS` hyperparameter MUST be defined as a named constant and control how aggressively boundary regions are penalised. Higher values create a steeper transition from low to high penalty.
- **FR-004**: The `FLOOR` hyperparameter MUST be defined as a named constant and ensure the penalty factor never reaches zero, preserving optimiser gradients at the boundaries.
- **FR-005**: The section MUST include a hyperparameter documentation table explaining `STEEPNESS`, `FLOOR`, and their relationship to the existing Week 7 parameters (kernel, acquisition, distance-based selection).
- **FR-006**: The section MUST produce a 3-panel surrogate visualisation matching the Week 7 style: (1) GP posterior mean, (2) GP posterior std, (3) interior-penalised acquisition — all on the same 2D slice through the top-2 important dimensions.
- **FR-007**: The section MUST produce a convergence plot showing the running best observed values and the new candidate.
- **FR-008**: The section MUST output a formatted submission query string in the `x1-x2-x3-x4` format with 6 decimal places, clipped to [0.0, 0.999999].
- **FR-009**: The section MUST validate that the proposed point satisfies: (a) all coordinates in [0.0, 0.999999], (b) minimum distance ≥ 0.05 from any existing data point.
- **FR-010**: The interior penalty MUST be applied to the acquisition candidates after `optimize_acqf` returns the batch and before the distance-based selection step, by re-scoring each candidate with the penalty-weighted acquisition value.

### Key Entities

- **Interior Penalty Function**: A smooth, multiplicative weight $w(x) \in [\text{FLOOR}, 1.0]$ applied to the acquisition score at each candidate. Parameterised by `STEEPNESS` and `FLOOR`. In 4D, the product of four sin² terms provides stronger corner suppression than in 2D.
- **STEEPNESS**: Controls the width of the boundary suppression zone. Higher values create a narrower, steeper transition. Lower values create a broader, gentler suppression zone.
- **FLOOR**: The minimum value of the penalty at the boundary. Prevents the penalty from being exactly zero, preserving gradient information for the optimiser.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The proposed next sample point has all four coordinates at least 0.05 away from both 0 and 1 (i.e. in [0.05, 0.95]).
- **SC-002**: The penalised acquisition surface visualisation (on the 2D slice) shows at least a 50% reduction in acquisition value within 0.05 of any edge compared to the interior maximum.
- **SC-003**: The notebook section runs end-to-end without errors when all preceding Week 7 cells have been executed.
- **SC-004**: The submission query is in valid format: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` with all four values in [0.0, 0.999999].
- **SC-005**: All existing notebook cells remain unchanged — the new section is purely additive.

## Assumptions

- The existing Week 7 code (GP Matérn-5/2, qLogNEI, distance-based selection) has already been executed and variables (`X_raw`, `y_raw`, `X_train`, `Y_train`, `best_model`, `ls`, `y_mean`, `y_std_val`, `candidates`, `pred_means_orig`, `dists`, `best_point`, `nei`) are available in the notebook kernel.
- The search space is [0, 1]⁴ — a 4-dimensional unit hypercube.
- The sin(πx)² formulation naturally handles the [0, 1] range without additional normalisation.
- Default values of `STEEPNESS = 2.0` and `FLOOR = 0.01` are reasonable starting points; the student may adjust based on observed results.
- This feature only modifies F5. No other function notebooks are changed.
- The interior penalty is applied as a post-hoc re-scoring of the `optimize_acqf` batch candidates, not as a modification to the BoTorch acquisition function object itself.
