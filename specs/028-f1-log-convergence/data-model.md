# Data Model: F1 Log-Scale Convergence Plot

**Feature**: 028-f1-log-convergence
**Date**: 2026-03-11 (updated after user clarification)

## Entities

This feature has no new data entities. It modifies the visualisation of existing data.

### Existing Entity: F1 Output Array

| Field | Type | Description |
|-------|------|-------------|
| `updated_outputs['f1']` | numpy array (19,) | F1 black-box output values |
| Values | float64 | Range: -3.6e-003 to 7.7e-016 (includes zeros) |
| `n_initial` | int | 10 — number of initial samples |
| `running_max` | numpy array (19,) | Cumulative maximum of clipped outputs |

### Transformation: Clip Negatives to Zero

| Step | Code | Purpose |
|------|------|---------|
| Clip | `out = np.maximum(out, 0)` | Remove negative values so log scale works |
| Effect | Zero/negative → 0 | matplotlib omits non-positive values on log scale |
| Applied to | F1 only (`if fn == 'f1'`) | FR-001 + FR-002 compliance |

### Modification: Y-Axis Scale

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Scale type | `log` | Plain logarithmic scale |
| Applied to | F1 subplot only (index 0) | FR-001 + FR-002 compliance |

### Validation Rules

- Clipping and log scale must only be applied when `fn == 'f1'`
- F2–F8 subplots must retain default linear scale
- The existing scatter points, running max line, and boundary marker must render on the log-scaled axis (zero-valued points omitted by design)
- `running_max` is computed from clipped values — consistent with plotted data
