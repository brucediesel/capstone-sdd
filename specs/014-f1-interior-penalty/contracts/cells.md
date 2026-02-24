# Cell Contracts: F1 Interior Penalty Section

**Feature**: 014-f1-interior-penalty  
**Date**: 2026-02-24

Each new cell appended to `functions/f1/f1.ipynb` is documented below with its inputs, outputs, and contract.

---

## Cell 1: Section Header (Markdown)

**Type**: Markdown  
**Title**: `## Week 7 — Interior Penalty on Acquisition Function`

**Content contract**:
- Motivation paragraph explaining why the interior penalty is needed (boundary clustering)
- Hyperparameter table with columns: Parameter, Value, Rationale
- Must include `STEEPNESS`, `FLOOR` and reference to existing `KAPPA`, `PENALTY_RADIUS`
- Mathematical formula for the penalty function

---

## Cell 2: Hyperparameter Constants (Code)

**Inputs**: None (standalone constants)  
**Outputs (kernel variables)**:
| Variable | Type | Value |
|----------|------|-------|
| `STEEPNESS` | `float` | `2.0` |
| `FLOOR` | `float` | `0.01` |

**Contract**:
- Print both values for audit
- Both values reused by all subsequent cells
- No imports needed

---

## Cell 3: Interior Penalty Computation + Candidate Selection (Code)

**Inputs (from existing W7 cells)**:
| Variable | Source | Type |
|----------|--------|------|
| `X_cand` | W7-07 | `np.ndarray` shape `(20000, 2)` |
| `acq_penalized` | W7-07 | `np.ndarray` shape `(20000,)` |
| `X_w7` | W7-02 | `np.ndarray` shape `(17, 2)` |
| `STEEPNESS` | Cell 2 | `float` |
| `FLOOR` | Cell 2 | `float` |

**Outputs (kernel variables)**:
| Variable | Type | Description |
|----------|------|-------------|
| `interior_weight` | `np.ndarray` shape `(20000,)` | Penalty factor per candidate |
| `acq_with_interior` | `np.ndarray` shape `(20000,)` | Final acquisition scores |
| `next_x_ip` | `np.ndarray` shape `(2,)` | Selected point, clipped to [0, 0.999999] |
| `min_dist_ip` | `float` | Min distance from `next_x_ip` to any point in `X_w7` |

**Contract**:
- Compute `interior_weight = FLOOR + (1 - FLOOR) * np.prod(np.sin(np.pi * X_cand) ** (2 * STEEPNESS), axis=1)`
- Compute `acq_with_interior = acq_penalized * interior_weight`
- Select argmax, clip to [0, 0.999999]
- Print selected point coordinates, acquisition value, min distance
- Assert min distance ≥ 0.05 (warn if not)

---

## Cell 4: 3-Panel Surrogate + Acquisition Visualisation (Code)

**Inputs**:
| Variable | Source |
|----------|--------|
| `X_w7`, `y_w7`, `y_binary` | W7-02 |
| `stage1_clf` | W7-05 |
| `stage2_rf` | W7-06 (or None if FALLBACK_MODE) |
| `KAPPA`, `PENALTY_RADIUS`, `GRID_RES`, `FALLBACK_MODE` | W7-04 |
| `STEEPNESS`, `FLOOR` | Cell 2 |
| `next_x_ip` | Cell 3 |

**Outputs**: matplotlib figure (3 panels), displayed inline

**Contract**:
- Panel 1: Hurdle mean prediction `p(x)·μ(x)` on GRID_RES×GRID_RES grid
- Panel 2: Hurdle uncertainty `p(x)·σ_RF(x)` on same grid
- Panel 3: Penalised acquisition WITH interior penalty — `(acq_raw · local_penalty · interior_weight)` on same grid
- All panels: red/blue scatter for positive/non-positive training points; yellow star for `next_x_ip`
- Title includes "Interior Penalty (S={STEEPNESS}, F={FLOOR})"

---

## Cell 5: Convergence Plot (Code)

**Inputs**:
| Variable | Source |
|----------|--------|
| `y_w7` | W7-02 |
| `next_x_ip` | Cell 3 |

**Outputs**: matplotlib figure, displayed inline

**Contract**:
- Plot `np.maximum.accumulate(y_w7)` as running best
- Scatter individual observations
- Mark week boundaries with vertical dashed lines
- Print best observed value and its observation index

---

## Cell 6: Submission Query (Code)

**Inputs**:
| Variable | Source |
|----------|--------|
| `next_x_ip` | Cell 3 |
| `STEEPNESS`, `FLOOR` | Cell 2 |

**Outputs**: Formatted string printed to stdout

**Contract**:
- Clip to [0.0, 0.999999]
- Format as `"0.xxxxxx-0.xxxxxx"` (6 decimal places)
- Validate: 2 dimensions, each in [0.0, 0.999999]
- Print summary: surrogate type, acquisition type, interior penalty parameters, selected point
