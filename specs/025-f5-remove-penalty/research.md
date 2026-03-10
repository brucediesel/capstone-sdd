# Research: F5 Week 9 — Kernel, Standardize & Raw Samples

**Branch**: `026-f5-kernel-standardize` | **Date**: 2026-03-09

## R1 — Matérn-1.5 vs Matérn-5/2 Kernel

**Decision**: Replace Matérn-5/2 (`nu=2.5`) with Matérn-1.5 (`nu=1.5`)

**Rationale**: The current GP with Matérn-5/2 produces boundary-stuck candidates (all coords near 0.999999). Matérn-5/2 assumes 2× differentiable (smooth) sample paths, which with only 29 data points in 4D can lead to oversmoothing — the surrogate underestimates local roughness and overconfidently assigns high acquisition values near boundaries. Matérn-1.5 (1× differentiable, rougher) better handles the small-data regime and heterogeneous lengthscales.

**API**: `MaternKernel(nu=1.5, ard_num_dims=4)` — single parameter change, no other code modifications.

**Alternatives Considered**:
| Approach | Status | Why |
|----------|--------|-----|
| Keep Matérn-5/2 + tighter bounds | Rejected | Band-aid, doesn't address oversmoothing root cause |
| RBF kernel | Rejected | Infinite smoothness; worse for rough landscapes |
| Matérn-0.5 | Rejected | Too rough/discontinuous; overkill for this problem |
| Interior penalty wrapper | Rejected | Already tried and removed (spec 025) — suppresses boundaries artificially |

## R2 — BoTorch Standardize(m=1) Replacing Manual Z-Score

**Decision**: Replace manual z-score with `Standardize(m=1)` as `outcome_transform`; keep manual `log1p`.

**Rationale**: The current pipeline applies `log1p` then manual z-score, requiring manual inverse transforms (`expm1(pred * std + mean)`) in every downstream cell. `Standardize(m=1)` handles z-scoring internally and auto-applies the inverse when calling `model.posterior()`, simplifying downstream code. This also eliminates the need for manual z-score recomputation per LOO fold.

**Import**: `from botorch.models.transforms.outcome import Standardize`

**Pipeline change**:
- **Before**: `y_raw → log1p → manual z-score → Y_train` | `outcome_transform=None` | inverse: `expm1(pred * y_std_val + y_mean)`
- **After**: `y_raw → log1p → Y_train` | `outcome_transform=Standardize(m=1)` | inverse: `expm1(posterior.mean)`

**Impact on cells**:

| Cell # | Current | After Standardize |
|--------|---------|-------------------|
| 4 (constants) | Computes `y_mean`, `y_std_val`, applies z-score | Applies `log1p` only; remove z-score code |
| 8 (GP training) | `outcome_transform=None` | `outcome_transform=Standardize(m=1)` |
| 10 (acquisition) | Inverse: `expm1(pred * y_std_val + y_mean)` | Inverse: `expm1(posterior.mean)` |
| 12 (viz) | Inverse: `expm1(grid_mu * y_std_val + y_mean)` | Inverse: `expm1(grid_mu)` for mean; sigma simplified |
| 16 (submission) | References `outcome_transform=None` | References `outcome_transform=Standardize(m=1)` |
| 22 (LOO) | Manual z-score per fold, manual inverse | Each fold GP auto-standardizes; inverse: `expm1(pred)` |

**Alternatives Considered**:
| Approach | Status | Why |
|----------|--------|-----|
| Keep manual z-score | Rejected | More code, error-prone per-fold recomputation, no API benefit |
| Drop log1p too, use only Standardize | Rejected | log1p compresses heavy-tailed F5 outputs (range 0–3400); Standardize alone would struggle with skew |

## R3 — Increase raw_samples 3000 → 5000

**Decision**: Increase `raw_samples` from 3000 to 5000 in `optimize_acqf`

**Rationale**: In 4D space, Sobol quasi-random initial coverage with 3000 raw samples provides limited density. Increasing to 5000 gives ~67% better initial coverage, improving the chance of finding good starting points for L-BFGS-B restarts, especially in areas away from boundaries.

**Code change**: Single parameter: `raw_samples=3000` → `raw_samples=5000`

**Alternatives Considered**:
| Approach | Status | Why |
|----------|--------|-----|
| 3000 (unchanged) | Rejected | Boundary-stuck results suggest inadequate initial coverage |
| 10000 | Rejected | Diminishing returns; 5000 is a reasonable step up without excessive compute |
| Increase num_restarts instead | Rejected | num_restarts=50 already high; raw_samples is the bottleneck |

## R4 — Complete Cell Action Map (23 cells)

The notebook currently has 23 cells after penalty removal. The 3 changes affect:

| Cell # | Type | Action | Change Description |
|--------|------|--------|--------------------|
| 1 | md | EDIT | Update title: "Matérn-1.5" instead of "Matérn-5/2" |
| 2 | code | EDIT | Add `Standardize` import |
| 3 | md | EDIT | Update hyperparams table: kernel nu, raw_samples, outcome_transform |
| 4 | code | EDIT | Remove z-score code, pass `y_log` directly as Y_train |
| 5 | md | — | Unchanged |
| 6 | code | — | Unchanged (data loading) |
| 7 | md | — | Unchanged |
| 8 | code | EDIT | `nu=1.5`, `outcome_transform=Standardize(m=1)`, update prints |
| 9 | md | — | Unchanged |
| 10 | code | EDIT | `raw_samples=5000`, simplify inverse transform |
| 11 | md | — | Unchanged |
| 12 | code | EDIT | Simplify inverse for grid, update suptitle |
| 13 | md | — | Unchanged |
| 14 | code | — | Unchanged (convergence plot) |
| 15 | md | — | Unchanged |
| 16 | code | EDIT | Update surrogate description in print |
| 17 | md | — | Unchanged |
| 18 | code | — | Unchanged (convergence metrics) |
| 19 | md | — | Unchanged |
| 20 | code | — | Unchanged (exploration spread) |
| 21 | md | — | Unchanged |
| 22 | code | EDIT | LOO: `nu=1.5`, `Standardize(m=1)`, simplified inverse |
| 23 | md | EDIT | Update strategy: document kernel/transform changes |

**Summary**: 10 cells EDIT, 13 cells unchanged.
