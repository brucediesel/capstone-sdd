# Cell Contracts: F5 Interior Penalty Section

**Feature**: 015-f5-interior-penalty  
**Date**: 2026-02-24

Each new cell appended to `functions/f5/f5.ipynb` is documented below with its inputs, outputs, and contract.

**Cell numbering**: Cells 58–63 (appended after existing cell 57, the Week 7 submission query).

---

## Cell 58: Section Header (Markdown)

**Type**: Markdown  
**Cell ID**: `IP-00`

**Content contract**:
- Section title: `## Week 7 — Interior Penalty on Acquisition Function`
- Motivation paragraph explaining boundary-clustering problem in 4D (GP posterior variance highest at hypercube faces/edges)
- Note that 4D penalty is exponentially stronger than 2D → reduced STEEPNESS
- Hyperparameter table with columns: Parameter, Value, Rationale
- Must include `STEEPNESS`, `FLOOR`
- Mathematical formula: `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`
- Reference to existing Week 7 GP and NEI parameters

---

## Cell 59: Hyperparameter Constants (Code)

**Type**: Code  
**Cell ID**: `IP-01`

**Inputs**: None (standalone constants)  
**Outputs (kernel variables)**:
| Variable | Type | Value |
|----------|------|-------|
| `STEEPNESS` | `float` | `1.0` |
| `FLOOR` | `float` | `0.01` |

**Contract**:
- Define both values as named constants
- Print both values for audit trail
- Both values reused by all subsequent cells
- No imports needed (already imported in Cell 51)

---

## Cell 60: Interior Penalty Computation + Candidate Re-Scoring + Selection (Code)

**Type**: Code  
**Cell ID**: `IP-02`

**Inputs (from existing Week 7 cells)**:
| Variable | Source Cell | Type |
|----------|-----------|------|
| `candidates` | Cell 54 | `torch.Tensor` shape `(4, 4)` |
| `pred_means_orig` | Cell 54 | `np.ndarray` shape `(4,)` |
| `dists` | Cell 54 | `torch.Tensor` shape `(4,)` |
| `X_raw` | Cell 51 | `np.ndarray` shape `(27, 4)` |
| `STEEPNESS` | Cell 59 | `float` |
| `FLOOR` | Cell 59 | `float` |

**Outputs (kernel variables)**:
| Variable | Type | Description |
|----------|------|-------------|
| `interior_weight` | `np.ndarray` shape `(4,)` | Penalty factor per candidate, ∈ [FLOOR, 1.0] |
| `weighted_means` | `np.ndarray` shape `(4,)` | `pred_means_orig * interior_weight` |
| `next_x_ip` | `np.ndarray` shape `(4,)` | Selected point, clipped to [0, 0.999999] |
| `min_dist_ip` | `float` | Min distance from `next_x_ip` to any point in `X_raw` |

**Contract**:
1. Convert candidates to numpy: `cand_np = candidates.cpu().numpy()`
2. Compute `raw_penalty = np.prod(np.sin(np.pi * cand_np) ** (2 * STEEPNESS), axis=1)`
3. Compute `interior_weight = FLOOR + (1 - FLOOR) * raw_penalty`
4. Compute `weighted_means = pred_means_orig * interior_weight`
5. Apply median filter: `above_median = weighted_means >= np.median(weighted_means)`
6. Among above-median candidates, select farthest from training data (using `dists`)
7. Clip selected point to [0, 0.999999]
8. Print table showing all 4 candidates: coords, raw mean, penalty weight, weighted mean, distance, selected flag
9. Print selected point coordinates, weighted mean, min distance
10. Warn (do not assert) if min distance < 0.05

---

## Cell 61: 3-Panel Surrogate + Penalised Mean Visualisation (Code)

**Type**: Code  
**Cell ID**: `IP-03`

**Inputs**:
| Variable | Source |
|----------|--------|
| `X_raw`, `y_raw` | Cell 51 |
| `best_model` | Cell 53 |
| `ls` (lengthscales) | Cell 53 |
| `y_mean`, `y_std_val` | Cell 53 |
| `STEEPNESS`, `FLOOR` | Cell 59 |
| `next_x_ip` | Cell 60 |
| `best_point` | Cell 54 (used for slice anchor) |

**Outputs**: matplotlib figure (3 panels), displayed inline

**Contract**:
- Identify top-2 important dims (shortest lengthscales from `ls`)
- Fix remaining 2 dims at `next_x_ip` values (the interior-penalty-selected point)
- Build 80×80 grid over top-2 dims
- Panel 1: GP posterior mean (original scale via expm1 inverse transform) — contourf "viridis"
- Panel 2: GP posterior std (original scale) — contourf "magma"
- Panel 3: Penalised mean = `mean_orig * w(x)` where w(x) computed on grid — contourf "plasma"
- All panels: red scatter for training points (projected to top-2 dims); magenta star for `next_x_ip`
- Suptitle includes "Interior Penalty (S={STEEPNESS}, F={FLOOR})"
- Panel 3 title includes "Penalised Mean" to distinguish from raw mean

---

## Cell 62: Convergence Plot (Code)

**Type**: Code  
**Cell ID**: `IP-04`

**Inputs**:
| Variable | Source |
|----------|--------|
| `y_raw` | Cell 51 |
| `next_x_ip` | Cell 60 |

**Outputs**: matplotlib figure, displayed inline

**Contract**:
- Plot `np.maximum.accumulate(y_raw)` as running best (line + markers)
- Scatter individual observations
- Mark Week 6 → 7 boundary with vertical dashed line at x=26.5
- Print best observed value and its observation index
- Same style as Cell 56 (Week 7 convergence)

---

## Cell 63: Submission Query (Code)

**Type**: Code  
**Cell ID**: `IP-05`

**Inputs**:
| Variable | Source |
|----------|--------|
| `next_x_ip` | Cell 60 |
| `STEEPNESS`, `FLOOR` | Cell 59 |

**Outputs**: Formatted string printed to stdout

**Contract**:
- Clip to [0.0, 0.999999]
- Format as `"0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx"` (4 dimensions, 6 decimal places)
- Validate: 4 dimensions, each in [0.0, 0.999999]
- Print summary block:
  - Surrogate: GP Matérn-5/2 ARD
  - Acquisition: qLogNEI (q=4) + interior penalty re-scoring
  - Selection: Penalty-weighted median filter → farthest from data
  - Interior penalty parameters: STEEPNESS, FLOOR
  - Fitted lengthscales (from `ls`)
  - Selected candidate index and posterior mean
