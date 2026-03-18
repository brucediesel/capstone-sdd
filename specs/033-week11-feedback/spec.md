# Feature Specification: Week 11 Performance Review & Feedback

**Feature Branch**: `033-week11-feedback`
**Created**: 2026-03-17
**Status**: Draft
**Input**: User description: "Start a new code branch for week 11. Create new notebooks for F1 to F8. Review the outputs from the week 11 sample and provide feedback in each of the notebooks - showing convergence graph and proposals for next sample strategy."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Review Week 11 Data Across All Functions (Priority: P1)

As a student working on the capstone black-box optimisation challenge, I need to load the latest (Week 11) data for all 8 functions and visualise the performance history before deciding on a strategy for the next submission.

**Why this priority**: Without visualising current performance, I cannot make informed decisions about which functions need strategy changes and which are performing well.

**Independent Test**: Open any single function notebook (e.g., `f1 - week 11.ipynb`), run all cells, and verify that convergence and 2D pair plots render correctly with all data points.

**Acceptance Scenarios**:

1. **Given** the Week 11 data files exist in the data folders, **When** I run the notebook for any function, **Then** the data loads successfully and the number of samples matches expectations.
2. **Given** the notebook is executed, **When** I view the convergence plot, **Then** I see the running best objective over all iterations, with initial samples and submissions visually distinguished.
3. **Given** the notebook is executed, **When** I view the 2D pair plots, **Then** each sample point is numbered in sampling order, with initial values in a different colour from submission values, and the overall best output is marked with a green star.

---

### User Story 2 — Identify Performance Issues & Propose Improvements (Priority: P1)

As a student, I need an evaluation of each function's optimisation trajectory so I can identify which functions are stalling, converging well, or need a change of strategy.

**Why this priority**: The challenge has limited remaining submissions; understanding where effort should be focused is critical for maximising final scores.

**Independent Test**: After running any function's notebook, verify that a markdown section at the end evaluates performance and proposes specific strategy improvements.

**Acceptance Scenarios**:

1. **Given** all cells are executed, **When** I read the performance evaluation markdown, **Then** I see quantitative metrics (best value trajectory, improvement per submission, stalling detection).
2. **Given** the performance evaluation identifies stalling, **When** I read the improvement suggestions, **Then** the suggestions are specific, actionable, and justified by the data patterns.

---

### User Story 3 — Best Output Location on Pair Plots (Priority: P1)

As a student, I need to see where the best-performing sample is located in the input space so I can understand which region of the design space produces the highest objective.

**Why this priority**: Knowing the spatial location of the best output informs exploitation strategies and helps verify the surrogate model is exploring the right regions.

**Independent Test**: Run any notebook and verify the pair plots show a green star marker at the input coordinates of the overall best (highest output) sample.

**Acceptance Scenarios**:

1. **Given** a notebook is executed, **When** I view any 2D pair plot, **Then** the sample with the highest overall output value is marked with a green star (no value annotation).
2. **Given** a notebook is executed, **When** I view the legend, **Then** the green star marker is included as "Best" alongside the Initial (blue) and Submissions (orange) entries.

---

### Edge Cases

- What happens when F1 outputs contain negative values for log-scale display? → Set negative outputs to zero before computing log; do not use symlog.
- What happens when all submission points cluster in a small region? → The 2D pair plots with numbered points will reveal this visually; the evaluation markdown should flag it as insufficient exploration.
- What happens when a function shows no improvement across all 11 submissions? → The evaluation should explicitly flag persistent stalling and propose a substantial strategy change.
- What happens with high-dimensional functions (F7=6D, F8=8D) producing many pair plots? → Arrange subplots in a readable grid layout with appropriately sized figures.
- What happens when the best sample is an initial sample (not a submission)? → The green star still marks it; this indicates the optimisation has not yet surpassed the initial sample quality.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create 8 new notebooks named `fX - week 11.ipynb` (X = 1..8), each in its corresponding `./functions/fX/` folder, following the constitution convention for weekly iteration notebooks.
- **FR-002**: Each notebook MUST be self-contained with all imports, data loading, visualisation, and evaluation — executable independently without running other notebooks.
- **FR-003**: Each notebook MUST load `updated_inputs - Week 11.npy` and `updated_outputs - Week 11.npy` from the corresponding `./data/fX/` folder.
- **FR-004**: Each notebook MUST display a convergence plot showing the running best (maximum) objective value over all iterations, with initial samples visually distinguished from weekly submissions (blue for initial, orange for submissions).
- **FR-005**: For F1 specifically, the convergence plot MUST use a logarithmic y-axis scale. Any non-positive output values MUST be clamped to a small epsilon (1e-300) before computing the log so that the plot remains valid. symlog MUST NOT be used.
- **FR-006**: Each notebook MUST display 2D scatter plots showing each unique pair of input dimensions, producing $\binom{d}{2}$ subplots where $d$ is the input dimensionality.
- **FR-007**: In the 2D pair plots, submission sample points MUST be annotated with their submission week number (3, 4, 5, ..., 11). Initial sample points are not numbered.
- **FR-008**: Initial sample points MUST be displayed in blue, submission points in orange. No output value encoding via colour — the pair plots show spatial coverage only.
- **FR-PAIR-BEST**: Each 2D pair plot MUST mark the overall best output sample with a green star marker (`marker='*'`, green, s=500, zorder=5). No numerical annotation — marker only. The legend MUST include a "Best" entry with the star marker.
- **FR-009**: Each notebook MUST contain a markdown section that summarises the current (Week 10) surrogate model and acquisition function used for that function, then evaluates the optimisation performance including: best value found, number of improvements observed, whether the optimisation is stalling (no new best in 3+ consecutive submissions), and the spatial spread of sample points.
- **FR-010**: Each notebook MUST contain a markdown section proposing specific strategy improvements relative to the current surrogate and acquisition function. Suggestions must name the current configuration and propose concrete changes.
- **FR-011**: Notebooks MUST NOT propose a next sample point or run any optimisation loop — they stop after proposing improvements.

### Function Data Summary

| Function | Input Dims | Initial Samples | Total Samples (Week 11) |
|----------|-----------|----------------|------------------------|
| F1       | 2         | 10             | 21                     |
| F2       | 2         | 10             | 21                     |
| F3       | 3         | 15             | 26                     |
| F4       | 4         | 30             | 41                     |
| F5       | 4         | 20             | 31                     |
| F6       | 5         | 20             | 31                     |
| F7       | 6         | 30             | 41                     |
| F8       | 8         | 40             | 51                     |

### Key Entities

- **Function Data**: Input/output numpy arrays for each of the 8 functions, loaded from `./data/fX/` folders. Inputs are in range [0, 1], outputs are 1-dimensional real values.
- **Convergence Plot**: Line chart of running best (maximum) objective value. X-axis = sample number, Y-axis = best-so-far value. Vertical line or colour change at the boundary between initial and submission samples.
- **2D Pair Plots**: Scatter plots where each subplot shows two input dimensions on the axes. Colour distinguishes initial (blue) from submission (orange) points — no output value encoding. Submission points annotated with their week number (3–11); initial points unmarked. Overall best sample marked with a green star.

## Clarifications

### Session 2026-03-17

- Q: "Best output" = overall best across all samples, or best among submissions only? → A: Overall best across all samples (initial + submissions).
- Q: Marker style for best output on pair plots? → A: Red star, no value annotation. Legend includes "Best" entry.
- Q: Red star not visible on all functions — how to enhance? → A: Increase marker size from s=200 to s=350.
- Q: Red star still not visible at s=350 — further enhancement? → A: Change colour from red to green for better contrast against blue/orange. Increase size from s=350 to s=500.

## Assumptions

- Week 11 data files already exist in all `./data/fX/` folders (confirmed present).
- Sample counts in the table above are confirmed from loaded data.
- All outputs are 1-dimensional for all 8 functions.
- F1 outputs may contain negative values (near-zero radiation readings far from source); these are set to zero before log transform.
- The 2D pair plots for high-dimensional functions (F7: 15 subplots, F8: 28 subplots) use large figure sizes to remain readable.
- All 8 functions are maximisation tasks.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 8 notebooks execute without errors and produce all required visualisations when run from top to bottom.
- **SC-002**: Convergence plots clearly show whether each function's optimisation is improving, stalling, or deteriorating over the 11 submission rounds.
- **SC-003**: 2D pair plots display all sample points with correct sequential numbering, two-colour coding, and a green star marking the overall best output location.
- **SC-004**: Performance evaluations correctly identify functions that have stalled (no improvement in 3+ consecutive submissions) versus those still improving.
- **SC-005**: Improvement suggestions are specific enough to be directly implementable in a follow-up specification.
