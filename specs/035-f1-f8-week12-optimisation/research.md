# Research: Week 12 Bayesian Optimisation Loop (F1–F8)

**Feature**: 035-f1-f8-week12-optimisation
**Date**: 2026-03-18
**Purpose**: Resolve all technical unknowns and document per-function configuration decisions for the week 12 optimisation notebooks.

## Research Task 1: Source Template Strategy

**Question**: What does "same strategy as week 11" mean in practice?

**Finding**: The week 11 notebooks are **review-only** notebooks (convergence plots, pair plots, performance evaluation, strategy proposals, ARD analysis). They do NOT contain optimisation loops. The actual BO strategies were implemented in the **week 10 optimisation notebooks** (`fX - week 10.ipynb`). The week 12 notebooks must replicate the week 10 optimisation code with the data week reference updated from 10 to 11.

**Decision**: Use the week 10 optimisation notebooks as the source template for each function's week 12 notebook. The review sections (Steps 1–4) come from the week 11 review pattern; the optimisation sections (Steps 5+) come from the week 10 optimisation code.

**Alternatives considered**:
- Use week 11 review notebooks as template → Rejected: they contain no optimisation code
- Modify week 10 strategies based on week 11 proposals → Rejected: spec says "same strategy"

## Research Task 2: Data Availability

**Question**: Which data files exist for the week 12 notebooks to load?

**Finding**: Week 11 data files (`updated_inputs - Week 11.npy` and `updated_outputs - Week 11.npy`) exist for all 8 functions. Week 12 data does NOT exist yet — the notebooks propose candidates to submit.

**Decision**: All notebooks load week 11 data. WEEK config parameter is set to 11 (data week), with notebook title referencing "Week 12" (submission week).

## Research Task 3: Per-Function Hyperparameter Configuration

All values extracted directly from the week 10 optimisation notebook code cells.

### F1 (2D) — SFGP + Log Transform + Interior Penalty

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 2.5 |
| ARD_NUM_DIMS | 2 |
| LS_LOWER | 0.01 (Interval) |
| LS_UPPER | 2.0 (Interval) |
| NOISE_LB | 1e-4 |
| N_MLL_RESTARTS | 15 |
| LOG_EPSILON | 1e-300 |
| MC_SAMPLES | 512 |
| Q_BATCH | 4 |
| NUM_RESTARTS | 20 |
| RAW_SAMPLES | 10000 |
| STEEPNESS | 0.5 |
| FLOOR | 0.01 |
| GRID_RES | 50 |
| Output transform | `log(max(y, 1e-300))` pre-applied |
| Outcome transform | None (log is manual) |
| Selection | Median gate + max min-distance + IP re-scoring |

### F2 (2D) — SFGP + Standardize + Interior Penalty

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 2.5 |
| ARD_NUM_DIMS | 2 |
| LS_LOWER | 0.005 (Interval) |
| LS_UPPER | 10.0 (Interval) |
| NOISE_LB | 1e-4 |
| N_MLL_RESTARTS | 50 |
| MC_SAMPLES | 512 |
| Q_BATCH | 4 |
| NUM_RESTARTS | 20 |
| RAW_SAMPLES | 4096 |
| STEEPNESS | 0.02 |
| FLOOR | 0.01 |
| GRID_RES | 50 |
| Output transform | None |
| Outcome transform | `Standardize(m=1)` |
| Selection | Median gate + max min-distance + IP re-scoring |

### F3 (3D) — SFGP + Shift Transform

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 2.5 |
| ARD_NUM_DIMS | 3 |
| LS bounds | Unconstrained (randomised 0.1–0.6 init) |
| NOISE_LB | 1e-4 |
| N_MLL_RESTARTS | 40 |
| MC_SAMPLES | 512 |
| Q_BATCH | 3 |
| NUM_RESTARTS | 20 |
| RAW_SAMPLES | 2048 |
| GRID_RES | 50 |
| Output transform | `y - y_min` (shift to non-negative) |
| Outcome transform | None |
| Interior penalty | None |
| Selection | Median gate + max min-distance |

### F4 (4D) — SFGP + Standardize

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 2.5 |
| ARD_NUM_DIMS | 4 |
| LS bounds | Unconstrained (randomised 0.1–0.6 init) |
| NOISE_LB | 1e-3 |
| N_MLL_RESTARTS | 30 |
| MC_SAMPLES | 512 |
| Q_BATCH | 4 |
| NUM_RESTARTS | 20 |
| RAW_SAMPLES | 2048 |
| GRID_RES | 50 |
| Output transform | None |
| Outcome transform | `Standardize(m=1)` |
| Interior penalty | None |
| Selection | Median gate + max min-distance |

### F5 (4D) — SFGP + Log + Standardize

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 1.5 |
| ARD_NUM_DIMS | 4 |
| LS bounds | Unconstrained |
| NOISE_LB | 1e-6 |
| N_MLL_RESTARTS | 15 |
| MC_SAMPLES | 512 |
| Q_BATCH | 4 |
| NUM_RESTARTS | 60 |
| RAW_SAMPLES | 8000 |
| STEEPNESS | 0.02 |
| FLOOR | 0.01 |
| Output transform | `np.log(outputs)` (all positive) |
| Outcome transform | `Standardize(m=1)` |
| Selection | 25th-percentile gate + max min-distance + IP |

### F6 (5D) — SFGP + Standardize + Rank-Based IP + Milk Feasibility

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 1.5 |
| ARD_NUM_DIMS | 5 |
| LS bounds | Unconstrained |
| NOISE_LB | 1e-3 |
| N_MLL_RESTARTS | 15 |
| MC_SAMPLES | 512 |
| Q_BATCH | 4 |
| NUM_RESTARTS | 50 |
| RAW_SAMPLES | 5000 |
| STEEPNESS | 1.0 |
| FLOOR | 0.01 |
| GRID_RES | 80 |
| MILK_THRESHOLD | 0.12 |
| MILK_FALLBACK | 0.10 |
| dim_names | ['flour', 'sugar', 'eggs', 'butter', 'milk'] |
| Output transform | None |
| Outcome transform | `Standardize(m=1)` |
| Selection | Rank-based (rank_mean + rank_penalty), milk ≥ 0.12 filter, distance tiebreak |
| IP formula | `w(x) = FLOOR + (1−FLOOR) · ∏ sin(πxᵢ)^(2·STEEPNESS)` |

### F7 (6D) — Neural Network + MC Dropout

| Parameter | Value |
|-----------|-------|
| Architecture | 6→5→5→1 (71 params) |
| LEARNING_RATE | 0.005 |
| EPOCHS | 200 |
| DROPOUT | 0.05 |
| MC_SAMPLES | 50 |
| N_CANDIDATES | 50000 |
| STEEPNESS | 0.02 |
| FLOOR | 0.02 |
| EXPLOITATION_WEIGHT | 0.5 |
| dim_names | ['learning_rate', 'reg_strength', 'n_layers', 'dropout', 'batch_size', 'optimizer'] |
| Transform | Manual z-score (X and y) |
| Acquisition | 0.5 × μ_norm + 0.5 × EI_norm |
| Selection | argmax(combined_acq × interior_weight) |

### F8 (8D) — SFGP + Standardize + qLogNEI q=1

| Parameter | Value |
|-----------|-------|
| KERNEL_NU | 2.5 |
| ARD_NUM_DIMS | 8 |
| LS bounds | Unconstrained |
| NOISE_LB | 1e-7 |
| N_MLL_RESTARTS | 30 |
| MC_SAMPLES | 512 |
| Q_BATCH | 1 |
| NUM_RESTARTS | 30 |
| RAW_SAMPLES | 8192 |
| Output transform | None |
| Outcome transform | `Standardize(m=1)` |
| Interior penalty | None |
| Selection | No batch selection (q=1, single candidate) |
| Cholesky check | Yes (stability validation post-fit) |

## Research Task 4: Notebook Structure Pattern

**Question**: What is the common cell structure for optimisation notebooks?

**Finding**: The week 10 optimisation notebooks follow a consistent structure across all GP-based functions:

1. **Title** (Markdown) — Week + function + strategy summary
2. **Imports & Config** (Code) — libraries + config constants
3. **Load Data** (Markdown + Code) — load `.npy` files, display summary table
4. **Convergence Plot** (Markdown + Code) — running best, blue/orange split
5. **2D Pair Plots** (Markdown + Code) — green star for best
6. **Performance Evaluation** (Markdown + Code) — metrics, stalling detection
7. **Strategy Description** (Markdown) — current config, evaluation, proposals
8. **Optimisation Config** (Markdown + Code) — hyperparameter constants
9. **Data Preparation** (Markdown + Code) — transforms (log, shift, standardize)
10. **Surrogate Fitting** (Markdown + Code) — multi-restart MLL
11. **Acquisition Optimisation** (Markdown + Code) — qLogNEI + distance selection
12. **Interior Penalty Re-scoring** (Markdown + Code) — if applicable
13. **Submission Query** (Code) — formatted output block
14. **Surrogate Visualisation** (Markdown + Code) — 3-panel contour (2D only)
15. **Updated Convergence** (Markdown + Code) — with predicted proposed point

F7 (NN) differs: replaces steps 8–12 with NN training, MC dropout uncertainty, and blended acquisition scoring.

**Decision**: Week 12 notebooks follow the same cell structure. The review cells (1–7) use updated week 11 data. The optimisation cells (8–15) replicate week 10 code verbatim with only the WEEK constant changed.

## Research Task 5: F1 Standardize vs Log Transform

**Question**: The conversation summary mentions F1 was changed from `log(y+ε)` to `Standardize(m=1)` in feature 034 because data has negative outputs. Which transform should week 12 use?

**Finding**: The feature 034 change applied only to the **ARD analysis GP fitting** (diagnostic cells added to the week 11 review notebook). The actual **optimisation GP** in week 10 uses `log(max(y, 1e-300))` as the pre-applied transform. The ARD cells used `Standardize(m=1)` because it was a separate diagnostic fit that needed to handle negative outputs differently.

**Decision**: Week 12 F1 optimisation uses `log(max(y, 1e-300))` — same as week 10. The Standardize(m=1) is only for ARD diagnostic purposes (not part of this feature).

## Research Task 6: Visualisation Requirements

**Question**: Which visualisations are needed per function?

**Finding**:

| Visualisation | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| Convergence (log for F1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2D pair plots + green star | ✓ (1) | ✓ (1) | ✓ (3) | ✓ (6) | ✓ (6) | ✓ (10) | ✓ (15) | ✓ (28) |
| Performance evaluation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3-panel contour | ✓ | ✓ | — | — | — | — | — | — |
| NN gradient importance | — | — | — | — | — | — | ✓ | — |
| Updated convergence | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Decision**: F1 and F2 get full 3-panel contour visualisation. F7 gets the NN-specific gradient importance 2D slice. All others get convergence + pair plots + performance evaluation + updated convergence.

## Conclusions

All NEEDS CLARIFICATION items resolved. The implementation is straightforward: clone each function's week 10 optimisation notebook, update the WEEK constant from 10 to 11 (data source) and title from "Week 10" to "Week 12" (submission target), and verify execution. No strategy changes are required.
