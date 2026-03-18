# Feature Specification: ARD Relevance Visualisation

**Feature Branch**: `034-ard-relevance-visualisation`  
**Created**: 2025-07-18  
**Status**: Draft  
**Input**: User description: "For F1-F8 calculate and visualise the ARD individual relevance parameters for each of the features. Add these to each of the existing week 11 notebooks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - ARD Relevance Calculation and Bar Chart (Priority: P1)

As a researcher reviewing week 11 results, I want to see the Automatic Relevance Determination (ARD) lengthscale parameters extracted from a fitted Gaussian Process for each function, displayed as a horizontal bar chart showing the relative importance of each input dimension, so that I can understand which input features have the most influence on the objective.

For each of the 8 functions (F1–F8), a new section is added to the existing week 11 review notebook that:
1. Fits a SingleTaskGP surrogate on the latest available data (week 11) using the same kernel configuration as the function's optimisation notebooks.
2. Extracts the per-dimension ARD lengthscale parameters from the fitted kernel.
3. Computes a normalised relevance score (inverse lengthscale, normalised to sum to 1).
4. Displays a horizontal bar chart with one bar per input dimension, labelled with the dimension name/index, showing relative relevance.

**Why this priority**: The core value of this feature is giving the researcher a visual understanding of which input dimensions matter most. Without the calculation and visualisation, the feature has no value.

**Independent Test**: Can be fully tested by running each week 11 notebook end-to-end and verifying that the ARD bar chart cell executes without error and produces a readable horizontal bar chart with the correct number of bars matching the function's input dimensionality.

**Acceptance Scenarios**:

1. **Given** the week 11 notebook for F1 (2 input dimensions) is opened, **When** all cells are executed, **Then** a horizontal bar chart appears showing 2 bars representing the relative relevance of each input dimension, with bars labelled appropriately.
2. **Given** the week 11 notebook for F8 (8 input dimensions) is opened, **When** all cells are executed, **Then** a horizontal bar chart appears showing 8 bars representing the relative relevance of each input dimension.
3. **Given** the week 11 notebook for F7 (neural network surrogate, 6 input dimensions) is opened, **When** all cells are executed, **Then** a diagnostic GP is fitted on the same data and a horizontal bar chart appears showing 6 bars, with a note that this is a diagnostic GP (not the production NN surrogate).

---

### User Story 2 - Consistent Presentation Across All 8 Notebooks (Priority: P2)

As a researcher, I want the ARD visualisation to follow a consistent format across all 8 function notebooks, so that I can easily compare relevance patterns between functions during my review.

**Why this priority**: Consistency aids comparison and understanding but is secondary to having the visualisation at all.

**Independent Test**: Can be verified by visually inspecting all 8 notebooks side-by-side and confirming consistent chart style, labelling conventions, and section placement.

**Acceptance Scenarios**:

1. **Given** all 8 week 11 notebooks have been executed, **When** the ARD charts are compared, **Then** they all use the same chart type (horizontal bar), colour scheme, axis labels, and title format.
2. **Given** any week 11 notebook is opened, **When** the user scrolls to the ARD section, **Then** it appears in the same relative position within the notebook (after the existing evaluation/strategy sections).

### Edge Cases

- **F7 uses a Neural Network surrogate (not a GP)**: F7's production surrogate is a neural network with no kernel or lengthscale parameters. A separate diagnostic GP must be fitted on the same data purely to extract ARD relevance. The chart must clearly indicate this is a diagnostic analysis rather than the production surrogate's parameters.
- **Low-dimensional functions (F1, F2 have only 2 dimensions)**: The bar chart still works correctly but shows only 2 bars. The visualisation must remain readable and not appear distorted with very few bars.
- **All dimensions have similar relevance**: When all lengthscales are nearly equal, the bar chart should still render correctly, showing approximately equal bars — this is itself an informative result.
- **GP fitting fails to converge**: If the GP fitting process encounters numerical issues (e.g., singular matrix), the notebook cell should produce a clear error message rather than silently failing.

> **Accepted Risk**: No explicit error handling is added for GP convergence failure. BoTorch's `fit_gpytorch_mll` raises informative exceptions by default (e.g., `NotPSDError`), which will surface naturally as cell errors during manual notebook execution. Per Constitution Principle I (simplicity), wrapping this in try/except is unnecessary.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each week 11 notebook (F1–F8) MUST contain a new section that fits a Gaussian Process surrogate on the latest available data and extracts the per-dimension ARD lengthscale parameters.
- **FR-002**: Each notebook MUST compute a normalised relevance score from the ARD lengthscales (inverse lengthscale, normalised to sum to 1) for each input dimension.
- **FR-003**: Each notebook MUST display a horizontal bar chart showing the relative relevance of each input dimension, with one bar per dimension.
- **FR-004**: The bar chart MUST label each bar with the corresponding input dimension name or index.
- **FR-005**: The bar chart MUST include a descriptive title, axis labels, and consistent formatting across all 8 notebooks.
- **FR-006**: For F7 (neural network surrogate), the notebook MUST fit a separate diagnostic GP on the same data for ARD extraction and clearly annotate that the ARD analysis is diagnostic (not from the production surrogate).
- **FR-007**: The GP fitted for ARD extraction MUST use the same kernel family, configuration (Matérn kernel with ARD), and output transform as used in the function's optimisation notebooks (e.g., log for F1, log1p for F5, Standardize for others), except for F7 which requires a new diagnostic GP.
- **FR-008**: The ARD section MUST appear after the existing evaluation and strategy sections in each week 11 notebook, preserving the existing notebook structure.
- **FR-009**: Each notebook MUST print a small table of the raw ARD lengthscale values (one per dimension) alongside the normalised relevance bar chart, so that both absolute and relative scales are visible.

### Key Entities

- **ARD Lengthscale**: A per-dimension parameter from the GP kernel. Smaller values indicate the model is more sensitive to that dimension (higher relevance). One lengthscale per input dimension.
- **Normalised Relevance Score**: The inverse of each lengthscale, divided by the sum of all inverse lengthscales, yielding a value between 0 and 1 per dimension that sums to 1 across all dimensions.
- **Function Dimensionality**: F1=2, F2=2, F3=3, F4=4, F5=4, F6=5, F7=6, F8=8 input dimensions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 8 week 11 notebooks execute end-to-end without errors, including the new ARD section.
- **SC-002**: Each notebook's ARD bar chart displays the correct number of bars matching the function's input dimensionality (F1:2, F2:2, F3:3, F4:4, F5:4, F6:5, F7:6, F8:8).
- **SC-003**: The relevance scores shown in each chart sum to 1 (within floating-point tolerance).
- **SC-004**: All 8 charts use a consistent visual format (same chart type, colour scheme, labelling conventions).
- **SC-005**: The F7 notebook clearly indicates that the ARD analysis uses a diagnostic GP rather than the production neural network surrogate.
- **SC-006**: Each notebook displays a printed table of raw lengthscale values with the correct number of entries matching the function's input dimensionality.

## Assumptions

- The latest available data for each function includes all updates up to and including week 11 (loaded by the week 11 notebooks as `updated_inputs - Week 11.npy` / `updated_outputs - Week 11.npy`).
- The kernel configuration for each function's diagnostic GP matches the kernel used in its optimisation notebooks (Matérn-2.5 for F1–F4, F8; Matérn-1.5 for F5, F6; new Matérn-2.5 for F7 diagnostic).
- Week 11 notebooks may be modified to add new cells, consistent with the Constitution's rule that the current iteration's notebook may be updated until finalised.
- The BoTorch/GPyTorch libraries are available in the project's Python environment.

## Clarifications

### Session 2026-03-18

- Q: Should the ARD diagnostic GP replicate each function's specific output transform from its optimisation notebook, or use a uniform Standardize across all 8? → A: Match each function's optimisation notebook output transform (log for F1, log then Standardize for F5, Standardize for others).
- Q: Should the ARD section display just the normalised relevance bar chart, or also include raw lengthscale values? → A: Show both: normalised relevance bar chart AND a small printed table of raw lengthscale values.
