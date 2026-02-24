# Feature Specification: F7 Week 7 — Neural Network Surrogate with NEI & Interior Penalty

**Feature Branch**: `017-f7-nn-interior-penalty`  
**Created**: 2025-02-24  
**Status**: Draft  
**Input**: User description: "process week 7 outputs for F7 - Use a Neural Network as a surrogate function with the following hyperparameters: 2L×5N, lr=0.005 - Use NEI as acquisition function with interior penalty - evaluate the problem and data and select appropriate other appropriate hyperparameters - Provide the same visualisations as previous weeks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Train NN Surrogate and Propose Next Sample via NEI with Interior Penalty (Priority: P1)

A capstone student loads the Week 7 cumulative data (37 samples, 6D) for Function 7, trains a compact neural network surrogate (2 hidden layers × 5 nodes), computes Noisy Expected Improvement from MC Dropout predictions, applies an interior penalty to discourage boundary-hugging candidates, and obtains a submission-ready query point.

**Why this priority**: This is the core deliverable — without the trained surrogate and penalised acquisition, no submission can be made for Week 7.

**Independent Test**: Run all new cells top-to-bottom after executing Week 6 cells. The submission query is printed in the required format (`x1-x2-x3-x4-x5-x6`, 6 decimal places, all values in [0, 0.999999]).

**Acceptance Scenarios**:

1. **Given** Week 7 data files exist (`updated_inputs - Week 7.npy`, `updated_outputs - Week 7.npy`), **When** the data loading cell executes, **Then** 37 samples × 6 dimensions are loaded and summary statistics are printed.
2. **Given** the data is loaded and normalised, **When** the NN training cell executes, **Then** a 6→5→5→1 network trains to convergence with a training loss curve displayed and training R² reported.
3. **Given** a trained NN surrogate, **When** the acquisition cell executes, **Then** MC Dropout-based Expected Improvement is computed for candidate points, interior penalty weights are applied, and the best penalised candidate is selected.
4. **Given** a selected candidate, **When** the submission cell executes, **Then** the query is printed in `x1-x2-...-x6` format with feasibility validation passing (all coordinates in [0, 0.999999]).

---

### User Story 2 — Document Hyperparameters with Rationale (Priority: P2)

The student (and capstone examiner) can read a markdown cell that lists all hyperparameters — architecture, learning rate, dropout, epochs, MC samples, interior penalty steepness and floor — with clear rationale for each choice.

**Why this priority**: Capstone assessment requires explicit hyperparameter documentation and justification.

**Independent Test**: Read the markdown cell — all hyperparameters are listed in a table with values and rationale.

**Acceptance Scenarios**:

1. **Given** the new Week 7 section exists, **When** an examiner reads the markdown header, **Then** a hyperparameter table documents architecture (2L×5N), learning rate (0.005), dropout rate, epochs, MC samples, STEEPNESS, and FLOOR with rationale for each.

---

### User Story 3 — Visualise Surrogate, Penalty, and Convergence (Priority: P2)

The student can view the same style of visualisations as Weeks 5 and 6 — a 3-panel surrogate plot (NN mean, MC dropout uncertainty, interior penalty heatmap) and a convergence plot — to assess model quality and optimisation progress.

**Why this priority**: Consistent visualisation across weeks enables comparison and demonstrates iterative improvement to the examiner.

**Independent Test**: Three-panel figure renders with the NN mean heatmap, uncertainty heatmap, and interior penalty heatmap. Convergence plot renders with running best and IP-selected predicted mean.

**Acceptance Scenarios**:

1. **Given** a trained NN and computed interior weights, **When** the 3-panel visualisation cell executes, **Then** Panel 1 shows the NN mean prediction, Panel 2 shows MC Dropout uncertainty, and Panel 3 shows the interior penalty heatmap — all projected onto the two most important input dimensions.
2. **Given** the full Week 7 dataset and a selected candidate, **When** the convergence plot cell executes, **Then** the running best line is displayed across all 37 observations with the IP-selected candidate's predicted mean marked.

---

### Edge Cases

- **Minimum at boundary**: If all high-performing candidates concentrate at input space boundaries, the interior penalty should steer selection toward nearby interior alternatives while still favouring high predicted performance.
- **MC Dropout variance collapse**: With only 5 nodes per layer and dropout, variance estimates may be very small — the system should still produce valid EI values even when uncertainty is near-zero.
- **Training instability with small network**: A 6→5→5→1 architecture has very few parameters (~46). If training fails to converge, the training loss curve should make this visible to the user.
- **All EI values near zero**: If the NN predicts no improvement over the current best for any candidate, the system should still select a point (fallback to maximum interior weight, providing an exploratory point).
- **Candidate at exact boundary (0.0 or 1.0)**: Interior penalty weight should equal FLOOR (not zero), ensuring no candidate is completely eliminated.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load Week 7 cumulative data from `data/f7/updated_inputs - Week 7.npy` and `data/f7/updated_outputs - Week 7.npy`, producing arrays of shape (37, 6) and (37,) respectively.
- **FR-002**: System MUST normalise inputs (z-score per dimension) and outputs (z-score) before training for network stability.
- **FR-003**: System MUST define a neural network with architecture 6→5→5→1 (2 hidden layers × 5 nodes each), using ReLU activations and Dropout.
- **FR-004**: System MUST train the network using Adam optimiser with learning rate 0.005, displaying a training loss curve and reporting training R².
- **FR-005**: System MUST generate candidate points uniformly in [0, 1]⁶ for acquisition evaluation.
- **FR-006**: System MUST compute Expected Improvement via MC Dropout — running multiple stochastic forward passes with dropout enabled, computing EI(x) = mean(max(prediction_i − y_best, 0)) for each candidate.
- **FR-007**: System MUST compute interior penalty weights using the formula w(x) = FLOOR + (1 − FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS) for each candidate.
- **FR-008**: System MUST combine EI and interior penalty via multiplication: penalised_EI(x) = EI(x) · w(x), selecting the candidate with the highest penalised EI.
- **FR-009**: System MUST produce a 3-panel visualisation: Panel 1 (NN mean prediction heatmap), Panel 2 (MC Dropout uncertainty heatmap), Panel 3 (interior penalty heatmap) — projected onto the two most important input dimensions as determined by gradient-based feature importance.
- **FR-010**: System MUST produce a convergence plot showing the running best across all 37 observations, with the IP-selected candidate's predicted mean indicated.
- **FR-011**: System MUST format the selected point as a submission query in `x1-x2-...-x6` format with 6 decimal places, all values clipped to [0, 0.999999], and validate the format.
- **FR-012**: System MUST append all new cells after existing cell 49 (the last cell in the notebook) without modifying any existing cells (constitution rule: weekly sections, no cell replacement).

### Key Entities

- **Training Data**: 37 cumulative observations (6D inputs, 1D outputs) from Weeks 3–7 of the black-box optimisation challenge. All outputs are positive (range [0.003, 2.305]).
- **Neural Network Surrogate**: A compact fully-connected network (6→5→5→1) trained on normalised data, providing predictions and MC Dropout-based uncertainty estimates.
- **Interior Penalty Weight**: A per-candidate scalar in [FLOOR, 1.0] that suppresses boundary-hugging candidates, computed from the product-of-sines formula.
- **Penalised EI**: The product of MC-based Expected Improvement and interior penalty weight, used to rank and select the next query point.
- **Submission Query**: The final 6D point formatted for challenge submission.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The neural network trains without errors and achieves a positive training R² (indicating the model captures some signal from the data).
- **SC-002**: The submission query contains exactly 6 coordinates, each in [0, 0.999999] with 6 decimal places, matching the required format.
- **SC-003**: The interior penalty demonstrably influences candidate selection — the output should indicate whether the penalty changed the selection compared to raw EI.
- **SC-004**: All three visualisation panels render correctly — NN mean, MC uncertainty, and interior penalty heatmaps are visible and informative.
- **SC-005**: The convergence plot shows the running best value across all 37 observations, enabling comparison with previous weeks' progress.
- **SC-006**: No existing cells (1–49) are modified — only new cells are appended to the notebook.

## Assumptions

- **Dropout rate**: 0.2 is used (consistent with Weeks 5–6 for this function) unless the user specifies otherwise.
- **Epochs**: Selected based on the smaller network size; a reasonable default (500–1000) will be used, tuned for convergence.
- **MC Samples**: 50 stochastic forward passes (consistent with Weeks 5–6) provide sufficient variance estimation.
- **Number of candidates**: 20,000 random candidates in [0, 1]⁶ (consistent with Week 6).
- **Interior penalty hyperparameters**: STEEPNESS=1.0 and FLOOR=0.01 (consistent with F5 and F6 implementations).
- **Multiplicative penalty**: Since all F7 outputs are positive, EI values are non-negative, and w(x) ∈ [FLOOR, 1.0], the multiplicative approach (penalised_EI = EI × w) works correctly without requiring rank-based scoring.
- **Activation function**: ReLU (consistent with previous weeks' NN implementations for F7).
- **Feature importance**: Gradient-based input importance (same method as Weeks 5–6) determines which 2 dimensions to use for the 2D slice visualisations.
- **Candidate generation**: Uniform random sampling in [0, 1]⁶ (consistent with previous weeks).
