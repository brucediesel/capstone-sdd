# Feature Specification: F1 Log-Scale Convergence Plot

**Feature Branch**: `028-f1-log-convergence`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "change the process_results.ipynb notebook to present the convergence plot of F1 in log scale to better see outputs."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — View F1 convergence on log scale (Priority: P1)

As a researcher reviewing weekly optimisation results, I want the F1 convergence subplot to use a logarithmic y-axis so that I can distinguish between output values that span several orders of magnitude.

**Why this priority**: F1 outputs include values very close to zero and values orders of magnitude larger. A linear scale compresses the smaller values, making it impossible to see week-over-week improvements. This is the sole purpose of the change.

**Independent Test**: Run all cells of `process_results.ipynb` for any week that includes F1 data. Verify the F1 subplot y-axis displays a log scale while all other function subplots remain on a linear scale.

**Acceptance Scenarios**:

1. **Given** the notebook is executed with valid weekly data, **When** the convergence graph renders, **Then** the F1 subplot y-axis is logarithmic and labelled accordingly.
2. **Given** F1 output data contains both small and large positive values, **When** the log-scale subplot renders, **Then** the differences between small values are visually distinguishable.
3. **Given** F2–F8 data is present, **When** the convergence graph renders, **Then** the F2–F8 subplots retain their existing linear y-axis scale.

---

### Edge Cases

- If F1 contains zero or negative output values, the log scale must handle them gracefully (e.g., by using `symlog` or by clipping to a small positive floor) so the plot does not error out.
- If F1 data is missing for a given week, the existing behaviour (empty subplot) should be preserved unchanged.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The convergence graph cell MUST set the F1 subplot y-axis to logarithmic scale.
- **FR-002**: The F2–F8 subplots MUST remain on a linear y-axis scale (no change from current behaviour).
- **FR-003**: The F1 subplot MUST continue to display initial samples, BO submission markers, running best line, and the initial/BO boundary exactly as the other subplots do.
- **FR-004**: If F1 output data includes zero or negative values, the subplot MUST still render without errors.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The F1 convergence subplot y-axis tick labels are logarithmically spaced when the notebook is executed.
- **SC-002**: All 8 convergence subplots render without errors across any valid weekly dataset.
- **SC-003**: F2–F8 subplots are visually unchanged compared to the current notebook output.

## Assumptions

- All F1 outputs collected to date are positive, so a standard log scale is expected to work without special handling. A safety guard for zero/negative values is included as a precaution.
- No changes are needed to any cells other than the convergence graph cell (Cell 13 of the notebook).
- The notebook's existing plotting library (matplotlib) supports log-scale axes natively.
