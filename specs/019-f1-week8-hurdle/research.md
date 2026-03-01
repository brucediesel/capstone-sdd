# Research: F1 Week 8 — Hurdle Model Bayesian Optimisation

**Feature**: `019-f1-week8-hurdle`  
**Date**: 2026-03-01

## Research Questions

### RQ-1: Data shape and characteristics of Week 8 data

**Task**: Verify the updated Week 8 data files exist and have the expected shape for F1.

**Findings**:
- `updated_inputs - Week 8.npy`: shape **(18, 2)** — 18 observations, 2 input dimensions ✓
- `updated_outputs - Week 8.npy`: shape **(18,)** — 18 scalar outputs ✓
- Input range: [0.000330, 0.999460] — within [0.0, 1.0] ✓
- Output range: **[-0.003606, 0.000000]** — all values are zero or negative
- Best observed value: **0.000000** at index 2, location ≈ (0.731, 0.733)

**Decision**: Data is valid and matches expectations (10 initial + 8 weekly = 18 points). No NaN values, no out-of-range inputs.

**Impact on Hurdle Model**: The `y > 0` threshold used by the Stage 1 classifier may result in very few or zero "positive" samples, depending on floating-point precision. The existing fallback logic (`FALLBACK_MODE` when `n_positive < MIN_POSITIVE = 3`) is critical here. If triggered, the notebook degrades to pure exploration, which is appropriate given the lack of improvement.

---

### RQ-2: Suitability of reusing Week 7 hurdle model strategy

**Task**: Confirm the Week 7 strategy (hurdle model + weighted UCB + local penalization + interior penalty) remains appropriate for Week 8.

**Findings**:
- The hurdle model was chosen specifically because F1 has a sparse positive landscape (radiation source detection). Most observations return zero/near-zero.
- Week 7 already used KAPPA = 3.0 (exploration-focused) because no improvement had been observed.
- Week 8 adds one more data point but the fundamental challenge remains — the model needs to explore more of the 2D space.
- The interior penalty (STEEPNESS = 0.1, FLOOR = 0.01) prevents boundary clustering, which was the key issue fixed in Week 7.
- Local penalization (PENALTY_RADIUS = 0.15) encourages diversity in proposals by suppressing regions near existing observations.

**Decision**: Reuse the identical strategy. No hyperparameter changes justified — the exploration-heavy configuration remains correct for a problem with no observed improvement after 8 iterations.

**Rationale**: Changing the surrogate or hyperparameters without evidence of benefit would introduce unnecessary risk. The current setup correctly handles the zero-dominated output landscape.

**Alternatives considered**:
- Increase KAPPA further — rejected: 3.0 is already strongly exploration-focused; higher values would essentially ignore the classifier signal.
- Change surrogate to GP — rejected: Standard GP assumes continuous smooth response; the zero-dominated landscape violates this assumption. The hurdle model explicitly handles the binary nature.
- Reduce PENALTY_RADIUS — rejected: With 18 points in [0,1]², keeping 0.15 ensures adequate spacing.

---

### RQ-3: Self-contained notebook implementation pattern

**Task**: Confirm the notebook structure follows the constitution's per-iteration convention.

**Findings**:
- Constitution principle III requires `f1 - week 8.ipynb` in `./functions/f1/`.
- The notebook must be fully self-contained: imports, data loading, surrogate fitting, acquisition, visualisation, submission output.
- The original `f1.ipynb` must NOT be modified.
- No dependencies on kernel state from other notebooks.

**Decision**: Implement as a single notebook with the following cell structure:

| Cell | Type | Purpose |
|------|------|---------|
| 1 | Markdown | Title, week 8 overview, link to previous work |
| 2 | Code | Imports (numpy, matplotlib, sklearn) |
| 3 | Markdown | Hyperparameter documentation table |
| 4 | Code | Define all hyperparameter constants |
| 5 | Code | Load & validate Week 8 data, derive binary labels |
| 6 | Markdown | Data summary explanation |
| 7 | Code | Display data in tabular format |
| 8 | Markdown | Stage 1 explanation |
| 9 | Code | Fit Stage 1 — calibrated logistic classifier |
| 10 | Markdown | Stage 2 explanation |
| 11 | Code | Fit Stage 2 — random forest regressor (or fallback) |
| 12 | Markdown | Acquisition function explanation |
| 13 | Code | Weighted UCB + local penalization + interior penalty |
| 14 | Code | 3-panel contour plot |
| 15 | Code | Convergence plot |
| 16 | Markdown | Submission section |
| 17 | Code | Format and validate submission query |

**Rationale**: Mirrors the Week 7 cell structure but consolidated into a single self-contained notebook rather than cells appended to the existing notebook.

---

### RQ-4: Best practices for sklearn hurdle model with sparse positive data

**Task**: Confirm the logistic regression + random forest hurdle approach is appropriate for 18 points with near-zero outputs.

**Findings**:
- With 18 data points, `CalibratedClassifierCV(cv=3)` performs 3-fold cross-validation for probability calibration. Each fold has ~12 training points — adequate for logistic regression.
- `class_weight='balanced'` handles the likely class imbalance (few or no true positives).
- If `n_positive < 3`: Stage 2 RF cannot be meaningfully trained. The fallback to pure exploration (uniform acquisition weighted only by local penalization and interior penalty) is the correct strategy.
- `RandomForestRegressor(n_estimators=100, max_depth=3)` with only a few positive samples would overfit. The fallback protects against this.

**Decision**: No changes needed. The existing fallback logic is well-designed for this scenario.

---

## Summary

All research questions resolved. No NEEDS CLARIFICATION items remain.

| Question | Status | Decision |
|----------|--------|----------|
| RQ-1: Data shape | Resolved | 18×2 inputs, 18 outputs, all valid |
| RQ-2: Strategy reuse | Resolved | Keep identical Week 7 strategy |
| RQ-3: Notebook structure | Resolved | Self-contained ~17-cell notebook |
| RQ-4: Hurdle model suitability | Resolved | Appropriate; fallback logic handles sparse positives |
