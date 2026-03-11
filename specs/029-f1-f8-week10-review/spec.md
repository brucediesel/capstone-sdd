# Feature Specification: Week 10 Performance Review & Visualisation

**Feature Branch**: `029-f1-f8-week10-review`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "Create new branch for week 10. Create new notebooks for F1-F8 in their appropriate folders. Visualise performance of last submission output before deciding strategy. Provide convergence plots (log scale for F1, zero negative outputs) and 2D plots of outputs vs pairs of inputs with numbered sample points. Evaluate optimisation performance and suggest improvements. Stop after proposing improvements."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Review Week 10 Data Across All Functions (Priority: P1)

As a student working on the capstone black box optimisation challenge, I need to load the latest (Week 10) data for all 8 functions and visualise the performance history before deciding on a strategy for the next submission.

**Why this priority**: Without visualising current performance, I cannot make informed decisions about which functions need strategy changes and which are performing well.

**Independent Test**: Open any single function notebook (e.g., `f1 - week 10.ipynb`), run all cells, and verify that convergence and 2D pair plots render correctly with all data points.

**Acceptance Scenarios**:

1. **Given** the Week 10 data files exist in the data folders, **When** I run the notebook for any function, **Then** the data loads successfully and the number of samples matches expectations.
2. **Given** the notebook is executed, **When** I view the convergence plot, **Then** I see the running best objective over all iterations, with initial samples and submissions visually distinguished.
3. **Given** the notebook is executed, **When** I view the 2D pair plots, **Then** each sample point is numbered in sampling order, with initial values in a different colour from submission values.

---

### User Story 2 — Identify Performance Issues & Propose Improvements (Priority: P2)

As a student, I need an evaluation of each function's optimisation trajectory so I can identify which functions are stalling, converging well, or need a change of strategy.

**Why this priority**: The challenge has limited remaining submissions; understanding where effort should be focused is critical for maximising final scores.

**Independent Test**: After running any function's notebook, verify that a markdown section at the end evaluates performance and proposes specific strategy improvements.

**Acceptance Scenarios**:

1. **Given** all cells are executed, **When** I read the performance evaluation markdown, **Then** I see quantitative metrics (best value trajectory, improvement per submission, stalling detection).
2. **Given** the performance evaluation identifies stalling, **When** I read the improvement suggestions, **Then** the suggestions are specific, actionable, and justified by the data patterns.

---

### Edge Cases

- What happens when F1 outputs contain negative values for log-scale display? → Set negative outputs to zero before computing log; do not use symlog.
- What happens when all submission points cluster in a small region? → The 2D pair plots with numbered points will reveal this visually; the evaluation markdown should flag it as insufficient exploration.
- What happens when a function shows no improvement across all 10 submissions? → The evaluation should explicitly flag persistent stalling and propose a substantial strategy change.
- What happens with high-dimensional functions (F7=6D, F8=8D) producing many pair plots? → Arrange subplots in a readable grid layout with appropriately sized figures.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create 8 new notebooks named `fX - week 10.ipynb` (X = 1..8), each in its corresponding `./functions/fX/` folder, following the constitution convention for weekly iteration notebooks.
- **FR-002**: Each notebook MUST be self-contained with all imports, data loading, visualisation, and evaluation — executable independently without running other notebooks.
- **FR-003**: Each notebook MUST load `updated_inputs - Week 10.npy` and `updated_outputs - Week 10.npy` from the corresponding `./data/fX/` folder.
- **FR-004**: Each notebook MUST display a convergence plot showing the running best (maximum) objective value over all iterations, with initial samples visually distinguished from weekly submissions (blue for initial, orange for submissions).
- **FR-005**: For F1 specifically, the convergence plot MUST use a logarithmic y-axis scale. Negative output values MUST be set to zero before computing the log. symlog MUST NOT be used.
- **FR-006**: Each notebook MUST display 2D scatter plots of outputs versus each unique pair of input dimensions, producing $\binom{d}{2}$ subplots where $d$ is the input dimensionality.
- **FR-007**: In the 2D pair plots, each sample point MUST be annotated with its sampling order number (1, 2, 3, ... N).
- **FR-008**: Initial sample points MUST be displayed in a distinct colour (blue) from submission points (orange) in all 2D pair plots.
- **FR-009**: Each notebook MUST contain a markdown section evaluating the optimisation performance, including: best value found, number of improvements observed, whether the optimisation is stalling (no new best in 3+ consecutive submissions), and the spatial spread of sample points.
- **FR-010**: Each notebook MUST contain a markdown section proposing specific strategy improvements based on the performance evaluation. Suggestions must be specific and actionable (e.g., "switch kernel from Matérn-2.5 to Matérn-1.5" rather than "try a different kernel").
- **FR-011**: Notebooks MUST NOT propose a next sample point or run any optimisation loop — they stop after proposing improvements. Strategy changes will be specified in a follow-up after review.

### Function Data Summary

| Function | Input Dims | Initial Samples | Total Samples (Week 10) |
|----------|-----------|----------------|------------------------|
| F1       | 2         | 10             | 20                     |
| F2       | 2         | 10             | 20                     |
| F3       | 3         | 15             | 25                     |
| F4       | 4         | 30             | 40                     |
| F5       | 4         | 20             | 30                     |
| F6       | 5         | 20             | 30                     |
| F7       | 6         | 30             | 40                     |
| F8       | 8         | 40             | 50                     |

### Key Entities

- **Function Data**: Input/output numpy arrays for each of the 8 functions, loaded from `./data/fX/` folders. Inputs are in range [0, 1], outputs are 1-dimensional real values.
- **Convergence Plot**: Line chart of running best (maximum) objective value. X-axis = sample number, Y-axis = best-so-far value. Vertical line or colour change at the boundary between initial and submission samples.
- **2D Pair Plots**: Scatter plots where each subplot shows two input dimensions on the axes, with output value encoded as marker colour. Points annotated with their sample index number. Initial points in blue, submissions in orange.

## Assumptions

- Week 10 data files already exist in all `./data/fX/` folders (confirmed present).
- Sample counts in the table above assume 10 weekly submissions have been incorporated. The notebook code will dynamically determine the actual count from the loaded data.
- All outputs are 1-dimensional for all 8 functions.
- F1 outputs may contain negative values (near-zero radiation readings far from source); these are set to zero before log transform.
- The 2D pair plots for high-dimensional functions (F7: 15 subplots, F8: 28 subplots) should use large figure sizes to remain readable.
- All 8 functions are maximisation tasks.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 8 notebooks execute without errors and produce all required visualisations when run from top to bottom.
- **SC-002**: Convergence plots clearly show whether each function's optimisation is improving, stalling, or deteriorating over the 10 submission rounds.
- **SC-003**: 2D pair plots display all sample points with correct sequential numbering and two-colour coding, enabling visual identification of clustering or gaps in the explored input space.
- **SC-004**: Performance evaluations correctly identify functions that have stalled (no improvement in 3+ consecutive submissions) versus those still improving.
- **SC-005**: Improvement suggestions are specific enough to be directly implementable in a follow-up specification (e.g., "increase κ to 2.0 to encourage exploration" rather than "try more exploration").
