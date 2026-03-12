# Feature Specification: F1, F2 & F5 Interior Penalty

**Feature Branch**: `031-f4-f8-week10-optimisation` (same branch — no new branch)  
**Created**: 2026-03-12  
**Status**: Draft  
**Input**: User description: "Stay on the same code branch. Implement an interior penalty for F1, F2 and F5. Steepness should be very shallow for the penalty."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — F1 Interior Penalty (Priority: P1) 🎯 MVP

As the optimiser operator for F1 (2D), I want to add a shallow interior penalty to the acquisition function so that proposed sample points are softly discouraged from landing on domain boundaries, improving the chance of finding interior optima.

**Why this priority**: F1 has shown persistent stalling with near-zero outputs. Boundary-trapped candidates waste evaluations. F1 is the simplest function (2D) and serves as a fast validation of the interior penalty integration with the existing qLogNEI + distance-based selection pipeline.

**Independent Test**: Run the F1 week 10 notebook end-to-end; verify the interior penalty is applied to acquisition candidates, the selected candidate has all coordinates away from exact boundaries (0.0 or 1.0), and the submission format remains valid.

**Acceptance Scenarios**:

1. **Given** the F1 week 10 notebook with existing qLogNEI acquisition, **When** the interior penalty cell is appended and executed, **Then** the penalty weight for each candidate is computed using the sin-based formula with STEEPNESS = 0.02 and FLOOR = 0.01.
2. **Given** q=4 candidates from optimize_acqf, **When** the interior penalty is applied as a post-hoc re-scoring, **Then** the candidate with the highest penalised acquisition value is selected and the submission is in valid format (2D, [0, 0.999999]).
3. **Given** the interior penalty is applied, **When** the penalty weight is printed for each candidate, **Then** all weights are in [FLOOR, 1.0] and the penalty effect (whether selection changed) is documented.

---

### User Story 2 — F2 Interior Penalty (Priority: P1)

As the optimiser operator for F2 (2D), I want to add a shallow interior penalty to discourage boundary-hugging candidates and escape the local optimum the function has been trapped in.

**Why this priority**: F2 has been stuck in a local optimum. The interior penalty provides an additional mechanism to redirect the search toward unexplored interior regions. F2 shares the same 2D structure as F1, making the implementation pattern identical.

**Independent Test**: Run the F2 week 10 notebook end-to-end; verify interior penalty applied, candidate re-scored, valid 2D submission produced.

**Acceptance Scenarios**:

1. **Given** the F2 week 10 notebook with existing qLogNEI acquisition, **When** the interior penalty cell is appended, **Then** the penalty is applied with the same sin-based formula, STEEPNESS = 0.02, FLOOR = 0.01.
2. **Given** q=4 candidates, **When** the penalised re-scoring is applied, **Then** the selected candidate and submission format (2D, [0, 0.999999]) are valid and not a duplicate of existing observations.

---

### User Story 3 — F5 Interior Penalty (Priority: P1)

As the optimiser operator for F5 (4D), I want to add a shallow interior penalty to discourage boundary candidates in the higher-dimensional search space.

**Why this priority**: F5 operates in 4D where boundary effects are more pronounced — candidates can land on boundaries across any of 4 dimensions. The interior penalty ensures exploration is biased toward the interior, which is particularly valuable given F5's log-transformed output space.

**Independent Test**: Run the F5 week 10 notebook end-to-end; verify interior penalty applied across all 4 dimensions, valid 4D submission.

**Acceptance Scenarios**:

1. **Given** the F5 week 10 notebook with existing qLogNEI acquisition, **When** the interior penalty is added, **Then** the sin-based penalty operates over all 4 input dimensions with STEEPNESS = 0.02 and FLOOR = 0.01.
2. **Given** q=4 candidates in 4D, **When** the penalised re-scoring selects the best candidate, **Then** the submission is in valid format (4D, [0, 0.999999]) and not a duplicate.

---

### Edge Cases

- What happens when all candidates have nearly identical penalty weights? The penalty has minimal effect, and selection defaults to the unpenalised best — this is acceptable for very shallow steepness.
- What happens when a candidate lies exactly on a boundary (0.0 or 1.0)? sin(0) = 0 and sin(π) = 0, so the penalty weight drops to FLOOR, heavily penalising boundary points even with shallow steepness.
- What happens when the penalty changes the selected candidate to one with a lower acquisition value? This is the intended behavior — the penalty trades a small amount of acquisition quality for better interior positioning.

## Clarifications

### Session 2026-03-12

- Q: How should the interior penalty compose with the distance-based selection? → A: Apply distance filter first, then select by highest penalised acquisition value among survivors.
- Q: What exact STEEPNESS value should be used? → A: STEEPNESS = 0.02 (matches F7 week 10).
- Q: What exact FLOOR value should be used? → A: FLOOR = 0.01 (matches F6, maximum penalty dynamic range).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each notebook (F1, F2, F5) MUST compute an interior penalty weight for every candidate returned by optimize_acqf using the formula: `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`.
- **FR-002**: The STEEPNESS parameter MUST be set to 0.02 so the penalty is a near-no-op in the interior and only suppresses candidates very close to exact boundaries.
- **FR-003**: The FLOOR parameter MUST be set to 0.01 to ensure no candidate's acquisition value is completely zeroed out while maximising the penalty's dynamic range.
- **FR-004**: The interior penalty MUST be applied as a multiplicative post-hoc re-scoring of the acquisition values — it MUST NOT modify the GP, the acquisition function construction, or the optimize_acqf call.
- **FR-005**: After the existing distance-based filter removes near-duplicate candidates, the system MUST select the survivor with the highest penalised acquisition value (acquisition × interior_weight).
- **FR-006**: The system MUST print the penalty weight for each candidate and whether the penalty changed the selected candidate relative to the unpenalised selection.
- **FR-007**: The existing distance-based selection logic MUST be preserved and applied first — candidates that fail the distance filter are discarded before the interior penalty re-ranks the survivors.
- **FR-008**: The final submission MUST remain clamped to [0.0, 0.999999] with correct dimensionality (F1: 2D, F2: 2D, F5: 4D).
- **FR-009**: The STEEPNESS and FLOOR values MUST be defined as named constants in the imports/configuration cell of each notebook for visibility and tunability.

### Key Entities

- **Interior Penalty Weight**: A scalar in [FLOOR, 1.0] computed per candidate point, equal to FLOOR at domain boundaries and approaching 1.0 in the domain interior.
- **Penalised Acquisition Value**: The product of the original acquisition value and the interior penalty weight; used for final candidate selection.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 3 notebooks (F1, F2, F5) execute end-to-end without errors and produce a valid submission query after interior penalty application.
- **SC-002**: All penalty weights are within [FLOOR, 1.0] for every candidate in every notebook.
- **SC-003**: All submissions contain values strictly within [0, 0.999999] with correct dimensionality (F1: 2D, F2: 2D, F5: 4D) and are not duplicates of existing observations.
- **SC-004**: The penalty effect is documented in each notebook's output — whether the penalty changed the selected candidate is printed.
- **SC-005**: STEEPNESS = 0.02 and FLOOR = 0.01 in all 3 notebooks.

## Assumptions

- The existing week 10 optimisation cells (surrogate fitting, acquisition optimisation, distance-based selection) are already in place and working for F1, F2, and F5. The interior penalty is an incremental addition, not a replacement.
- The sin-based interior penalty formula is the established project pattern (used in F6, F7, and prior specs 014/015). No alternative penalty function is needed.
- "Very shallow steepness" means STEEPNESS = 0.02. At this value, the exponent is 0.04, making sin(πx)^0.04 ≈ 1.0 everywhere except right at x = 0 or x = 1. This is consistent with the F7 week 10 usage.
- The interior penalty is applied after optimize_acqf returns candidates — it does not modify the BoTorch acquisition function or optimisation loop.
- All 3 implementations share the same penalty formula and hyperparameter values; no per-function customisation is required.
