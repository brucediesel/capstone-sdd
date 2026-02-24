# Research: F5 Interior Penalty Design Decisions

**Feature**: 015-f5-interior-penalty  
**Date**: 2026-02-24

## 1. 4D Penalty Strength Analysis

### Decision
Use `STEEPNESS = 1.0` (reduced from F1's 2.0) as the default for 4D.

### Analysis

The interior penalty is a product of D independent sin² terms. In 4D, corner/edge/face suppression is exponentially stronger than in 2D:

**At STEEPNESS = 2.0 (F1's default)**:

| Location | Coordinates | Per-dim sin²ˢ | w(x) (product) | Effective suppression |
|----------|-------------|:--------------:|:---------------:|:--------------------:|
| 4D corner | all = 0.05 | 0.024 | 0.024⁴ ≈ 3.5×10⁻⁷ | 99.99997% |
| 4D edge | 2 @ 0.05, 2 @ 0.5 | 0.024 / 1.0 | 0.024² ≈ 5.9×10⁻⁴ | 99.94% |
| 4D face | 1 @ 0.05, 3 @ 0.5 | 0.024 / 1.0 | 0.024 | 97.6% |
| Centre | all = 0.5 | 1.0 | 1.0 | 0% |

**At STEEPNESS = 1.0 (proposed for 4D)**:

| Location | Coordinates | Per-dim sin²ˢ | w(x) (product) | Effective suppression |
|----------|-------------|:--------------:|:---------------:|:--------------------:|
| 4D corner | all = 0.05 | 0.156 | 0.156⁴ ≈ 5.9×10⁻⁴ | 99.94% |
| 4D edge | 2 @ 0.05, 2 @ 0.5 | 0.156 / 1.0 | 0.156² ≈ 0.024 | 97.6% |
| 4D face | 1 @ 0.05, 3 @ 0.5 | 0.156 / 1.0 | 0.156 | 84.4% |
| Centre | all = 0.5 | 1.0 | 1.0 | 0% |

**Key insight**: S=1.0 in 4D provides the same face-suppression (~84.4%) as S=1.0 in 2D per dimension, but the multiplicative effect across 4 dimensions gives much stronger corner (99.94%) and edge (97.6%) suppression. Meanwhile, the 50%-contour sits at approximately x=0.37 per dimension, preserving about 63% of each dimension's range → ~16% of the 4D volume has w ≥ 0.5.

S=2.0 in 4D is excessively aggressive: only ~0.6% of the hypercube has w ≥ 0.5.

### Rationale
- Corner suppression of 99.94% at S=1.0 is already overwhelming for boundary avoidance
- The 4D multiplicative effect provides sufficient edge protection without over-concentrating
- Student can increase to 1.5 or 2.0 if boundary proposals persist; documented in hyperparameter table

### Alternatives Considered
- S=2.0 (F1 default): Too aggressive in 4D — effective search space shrinks to ~0.6% of the hypercube
- S=0.5: Too permissive — face suppression only 39.5%, insufficient to prevent single-face boundary attraction

## 2. Post-Hoc Re-Scoring Strategy

### Decision
Re-score candidates using `pred_means_orig * w(x)` (posterior mean × penalty), NOT the joint NEI acquisition value.

### Rationale
- BoTorch's `qLogNoisyExpectedImprovement` returns a **single joint acquisition value for the entire q-point batch**, not per-candidate scores
- The `acq_value` from `optimize_acqf` is a scalar — it cannot be decomposed into per-candidate contributions
- `pred_means_orig` is already computed per-candidate (shape `(4,)`) in the existing Week 7 code
- Multiplying `pred_means_orig * w(x)` gives a per-candidate "penalty-weighted expected yield" that naturally penalises boundary candidates
- This is simple, interpretable, and uses only existing variables

### Alternatives Considered
- Multiply joint NEI value by w(x): Impossible — NEI is a single scalar for the batch, not per-candidate
- Re-run optimize_acqf with penalty-aware objective: Complex — requires custom BoTorch acquisition class, violates constitution simplicity principle
- Evaluate single-point LogEI at each candidate: Possible but adds complexity; posterior mean is simpler and captures the same boundary-avoidance goal

## 3. Integration with Distance-Based Selection

### Decision
Replace `pred_means_orig` with `pred_means_orig * w(x)` in the existing median filter, then continue with distance-based selection unchanged.

### Rationale
- The current pipeline: filter to `pred_means_orig >= median(pred_means_orig)`, then select farthest from data
- Replacing the filter input with penalty-weighted means naturally demotes boundary candidates below the median threshold
- The distance-based second stage (exploration) remains unchanged — still picks farthest from data among qualifying candidates
- This is the minimal-change integration: one line of code changes the filter variable

### Alternatives Considered
- Use penalty as tiebreaker only: Too weak — won't prevent boundary candidates from passing the median filter
- Composite score (mean × distance × penalty): Over-engineers the selection; mixing distance and prediction in a single score requires scale calibration
- Apply penalty after distance selection: Defeats the purpose — the selected point might still be boundary-adjacent

## 4. Visualising Penalised Acquisition on 2D Slice

### Decision
Use `posterior_mean_orig * w(x)` on the 80×80 grid as a proxy for penalised acquisition, shown in Panel 3.

### Rationale
- Computing batch `qLogNEI` on a grid is infeasible: the q-batch acquisition function evaluates a joint batch, not a single-point surface
- `posterior_mean_orig * w(x)` on the grid directly mirrors the actual selection scoring used in the re-scoring cell
- The penalty effect is clearly visible: near-zero values at boundaries, high values in the interior
- Consistent with the F1 approach (which visualised the penalised acquisition, not raw NEI)

### Alternatives Considered
- Analytic ExpectedImprovement per grid point: Feasible but adds complexity (needs best_f, different acqf class); posterior mean × penalty is simpler
- Plot the penalty function alone (no mean): Doesn't show the interaction between surrogate prediction and boundary suppression
- Keep the dimension-relevance bar chart: Doesn't fulfil FR-006 (penalised acquisition panel)

## 5. Batch Size (q parameter)

### Decision
Keep `q=4` — do not increase.

### Rationale
- The penalty re-ranks candidates, it doesn't eliminate them (FLOOR > 0 ensures all remain viable)
- `optimize_acqf` with q=4 already produces a spatially diverse joint batch (BoTorch optimises the joint q-EI)
- Increasing q doubles the `optimize_acqf` cost (more L-BFGS restarts × larger batch) with minimal benefit
- With only 4 candidates and a soft penalty, the selection pipeline has enough variety

### Alternatives Considered
- q=8: Doubles acquisition optimisation time; BoTorch already produces diverse batches via joint optimisation
- q=2: Reduces diversity too much — fewer candidates to filter from
