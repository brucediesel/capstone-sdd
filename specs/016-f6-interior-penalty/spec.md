# Feature Specification: F6 Interior Penalty on Acquisition Function

**Feature Branch**: `016-f6-interior-penalty`  
**Created**: 2026-02-24  
**Status**: Draft  
**Input**: User description: "update F6 by adding an interior penalty to the acquisition function — a soft penalty that decays towards the edges, with a steepness hyperparameter and a floor to ensure the penalty never reaches zero"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add Interior Penalty to F6 Acquisition (Priority: P1)

As a student running Bayesian Optimisation on Function 6 (5D food recipe — flour, sugar, eggs, butter, milk), I want the acquisition function to discourage candidates near the search space boundaries so that proposed sample points are pushed towards the interior where the true optimum is more likely to reside.

**Why this priority**: F6 candidates from the GP + NEI pipeline cluster near edges and corners of the 5D hypercube, particularly at the x4 (milk) boundary. The GP learned a strong anti-correlation with milk (ρ=−0.758), driving proposals to the x4=0 boundary. While feasibility bounds (x4 ≥ 0.10) partially address this, the interior penalty provides a smoother, more general solution across all 5 dimensions.

**Independent Test**: Run the notebook through the new section — the proposed next sample point should have all five coordinates comfortably away from the boundaries (e.g. ≥ 0.05 for x0–x3, ≥ 0.15 for x4 given its 0.10 lower bound), and the surrogate visualisation should show reduced desirability along the edges.

**Acceptance Scenarios**:

1. **Given** the F6 notebook has the existing Week 7 SFGP + NEI code (cells 0–59 unchanged), **When** a new section titled "Week 7 — Interior Penalty" is appended, **Then** the notebook runs end-to-end without errors and produces a modified candidate selection.
2. **Given** the `STEEPNESS` hyperparameter is set to a moderate value, **When** the penalty is applied to the batch of 4 candidates from `optimize_acqf`, **Then** boundary-proximate candidates are demoted in the selection ranking relative to interior candidates.
3. **Given** the `FLOOR` hyperparameter is set to a positive value (e.g. 0.01), **When** a candidate sits exactly on a corner, **Then** the interior penalty factor at that point equals `FLOOR` (never zero), preserving gradient information for the optimiser.

---

### User Story 2 - Explicit Hyperparameter Documentation (Priority: P2)

As a student submitting a capstone notebook, I want the steepness and floor hyperparameters to be clearly defined in a single constants cell with markdown documentation explaining their effect, so that the examiner can understand and modify them.

**Why this priority**: Capstone requirement — all hyperparameters must be explicit and justified.

**Independent Test**: Read the hyperparameter cell and markdown — `STEEPNESS` and `FLOOR` should be defined, printed, and their rationale explained including the 5D-specific considerations.

**Acceptance Scenarios**:

1. **Given** the new section is added, **When** the hyperparameter cell is inspected, **Then** `STEEPNESS` and `FLOOR` are defined as named constants alongside the existing Week 7 hyperparameters.
2. **Given** the markdown table documents the parameters, **When** a reviewer reads the table, **Then** the rationale for the chosen default values is clear and relates to the F6 problem geometry (5D food recipe, all-negative outputs).

---

### User Story 3 - Visualise Interior Penalty Effect (Priority: P2)

As a student, I want the 3-panel surrogate visualisation to include the interior penalty effect so that I can see how the candidate landscape changes compared to the unpenalised version.

**Why this priority**: Visual confirmation is essential for the capstone submission and for tuning the steepness parameter.

**Independent Test**: The penalised surface panel should show visibly reduced desirability near the edges of the 2D slice.

**Acceptance Scenarios**:

1. **Given** the notebook is executed, **When** the penalised surface is plotted on a 2D slice through the top-2 important dimensions, **Then** the colour map shows suppressed desirability near all four edges of the slice.
2. **Given** the convergence plot is generated, **When** compared to the Week 7 convergence plot, **Then** both the previous running best and the new proposed point are shown.

---

### Edge Cases

- What happens when `STEEPNESS` is set to 0? The interior penalty becomes a flat constant — effectively disabling the penalty. Documented as a valid degenerate case.
- What happens when `FLOOR` is set to 1.0? The penalty has no effect — the acquisition is unchanged. Documented as a degenerate case.
- What happens at exact corners (e.g. all coordinates 0 or 1)? sin(0) = sin(π) = 0, so the raw penalty is 0; the floor ensures the total factor equals `FLOOR` (never zero).
- What happens when steepness is very large (e.g. 50)? The penalty becomes a near-step function in each dimension — effectively shrinking the search space to a smaller interior hypercube. Documented in hyperparameter table.
- What happens in 5D where the penalty is a product of 5 sin² terms? Corner suppression is exponentially stronger than in 2D or 4D: a 5D corner receives a raw penalty of approximately sin(π·0.05)^(2·S) raised to the 5th power — extremely small. The additive floor ensures the minimum is exactly FLOOR.
- What happens with F6's all-negative outputs? The penalty scoring must account for the fact that all posterior means are negative. Simply multiplying a negative value by w(x) < 1 makes it less negative (higher in maximisation sense), which would promote boundary candidates instead of penalising them. The re-scoring mechanism must correctly reduce the desirability of boundary candidates regardless of output sign.
- What happens with the feasibility constraint x4 ≥ 0.10? The interior penalty is applied after `optimize_acqf` already respects the feasibility bounds. The penalty provides additional boundary avoidance on top of the existing bounds.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The notebook MUST add a new section after the existing Week 7 cells (cells 0–59 unchanged). Existing code cells MUST NOT be modified or replaced.
- **FR-002**: The interior penalty MUST be computed using the formula: for each candidate $x = (x_1, x_2, x_3, x_4, x_5)$, the penalty weight is $w(x) = \text{FLOOR} + (1 - \text{FLOOR}) \cdot \prod_{i=1}^{5} \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$. This evaluates to `FLOOR` at any edge and approaches 1.0 at the centre.
- **FR-003**: The `STEEPNESS` hyperparameter MUST be defined as a named constant and control how aggressively boundary regions are penalised. Higher values create a steeper transition from low to high penalty.
- **FR-004**: The `FLOOR` hyperparameter MUST be defined as a named constant and ensure the penalty factor never reaches zero, preserving optimiser gradients at the boundaries.
- **FR-005**: The section MUST include a hyperparameter documentation table explaining `STEEPNESS`, `FLOOR`, and their relationship to the existing Week 7 parameters (kernel, acquisition, distance-based selection).
- **FR-006**: The section MUST produce a 3-panel surrogate visualisation: (1) GP posterior mean, (2) GP posterior std, (3) interior-penalised desirability surface — all on the same 2D slice through the top-2 important dimensions.
- **FR-007**: The section MUST produce a convergence plot showing the running best observed values and the Week 6→7 boundary.
- **FR-008**: The section MUST output a formatted submission query string in the `x1-x2-x3-x4-x5` format with 6 decimal places, clipped to [0.0, 0.999999].
- **FR-009**: The section MUST validate that the proposed point satisfies: (a) all coordinates in [0.0, 0.999999], (b) minimum distance ≥ 0.05 from any existing data point, (c) x4 (milk) ≥ 0.10.
- **FR-010**: The interior penalty MUST be applied to the acquisition candidates after `optimize_acqf` returns the batch and before the distance-based selection step, by re-scoring each candidate's desirability with the penalty weight.
- **FR-011**: The re-scoring MUST correctly reduce the desirability of boundary candidates regardless of the sign of the posterior mean. F6's all-negative output space means the re-scoring cannot simply multiply posterior means by the penalty weight.

### Key Entities

- **Interior Penalty Function**: A smooth, multiplicative weight $w(x) \in [\text{FLOOR}, 1.0]$ computed per candidate. Parameterised by `STEEPNESS` and `FLOOR`. In 5D, the product of five sin² terms provides stronger corner suppression than in 2D or 4D.
- **STEEPNESS**: Controls the width of the boundary suppression zone. Higher values create a narrower, steeper transition. For 5D, a lower default value than 4D (F5) is appropriate due to the multiplicative effect across more dimensions.
- **FLOOR**: The minimum value of the penalty at the boundary. Prevents the penalty from being exactly zero, preserving gradient information.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The proposed next sample point has all five coordinates at least 0.05 away from their respective boundary (for x0–x3: in [0.05, 0.95]; for x4: in [0.15, 0.95] given the 0.10 lower bound).
- **SC-002**: The penalised desirability surface visualisation (on the 2D slice) shows near-zero values within 0.05 of any edge.
- **SC-003**: The notebook section runs end-to-end without errors when all preceding Week 7 cells have been executed.
- **SC-004**: The submission query is in valid format: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` with all five values in [0.0, 0.999999] and x4 ≥ 0.10.
- **SC-005**: All existing notebook cells (0–59) remain unchanged — the new section is purely additive.

## Assumptions

- The existing Week 7 code (SFGP Matérn-1.5 ARD, qLogNoisyExpectedImprovement q=4, Standardize(m=1), distance-based selection) has already been executed and kernel variables (`X_raw`, `y_raw`, `X_train`, `Y_train`, `best_model`, `ls`, `os_val`, `noise`, `candidates`, `pred_means`, `dists`, `best_point`, `nei`, `BOUNDS`) are available in the notebook kernel.
- The search space is effectively [0.01, 1.0]⁴ × [0.10, 1.0] due to the feasibility-constrained bounds in `optimize_acqf`.
- The sin(πx)² formulation naturally handles the [0, 1] range without additional normalisation.
- F6 outputs are all negative (confirmed by `assert (y_raw < 0).all()`); the re-scoring mechanism will account for this.
- The output transform is Standardize(m=1) (BoTorch default), which auto-untransforms posterior predictions to original (negative) scale — no manual inverse transform is needed.
- This feature only modifies F6. No other function notebooks are changed.
- The interior penalty is applied as a post-hoc re-scoring of the `optimize_acqf` batch candidates, not as a modification to the BoTorch acquisition function object itself.
