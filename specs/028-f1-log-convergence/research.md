# Research: F1 Log-Scale Convergence Plot

**Feature**: 028-f1-log-convergence
**Date**: 2026-03-11 (updated after user clarification)

## R1: Log-Scale Choice for F1 Data

### Context

F1 output data has extreme characteristics:
- 19 values total (10 initial + 9 BO submissions)
- Positive values span ~230 orders of magnitude (1e-245 to 7.7e-016)
- Contains exact zero values
- Contains small negative values (~-3.6e-003, ~-1.1e-003)

### Decision: Clip negatives to zero + plain `log` scale

Per user clarification: set any negative outputs to zero, then use `ax.set_yscale('log')`. Do not use `symlog`.

### Rationale

1. **User-directed** — explicit instruction to avoid symlog and clip negatives to zero
2. **Simplicity** — `np.maximum(out, 0)` + `ax.set_yscale('log')` is two lines, straightforward
3. **Effective** — positive values spanning ~230 orders of magnitude are well-served by log scale
4. **Acceptable trade-off** — zero-valued and negative points are omitted from the plot (matplotlib skips non-positive values on log scale), which is acceptable given user direction

### Alternatives Considered

| Scale | Zeros | Negatives | Extreme Spread | Verdict |
|-------|-------|-----------|----------------|---------|
| `log` + clip | Omitted | Clipped to 0 (omitted) | Excellent | **SELECTED** (user directed) |
| `symlog` (1e-4) | Yes | Yes | Excellent | REJECTED — user explicitly rejected |
| `asinh` | Yes | Yes | Good | Not requested |
| Custom clipping to ε | Displayed | Displayed as ε | Good | Over-engineered |

### Gotchas

- Points with value 0 or negative will not appear on the plot — this is by design per user instruction
- `running_max` is computed after clipping, so the green "Best Found" line also reflects clipped values
- Only applied to F1 subplot; F2–F8 remain unaffected

## R2: Conditional Application to F1 Only

### Decision

Add the clipping and `set_yscale` calls inside the existing `for idx, fn in enumerate(FUNCTIONS)` loop, guarded by `if fn == 'f1':`.

### Rationale

- Minimal change — no restructuring of the loop
- Clear intent — the condition documents exactly which subplot is affected
- F2–F8 subplots are untouched (FR-002 satisfied by default)
- Clipping is applied before `running_max` computation so the running best line is consistent with plotted points
