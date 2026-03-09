# Research: F1 Week 9 — log Transform (No Penalties)

**Date**: 2026-03-09  
**Feature**: 023-f1-week9-log

## Research Task 1: log1p vs log effectiveness for F1's output range

### Context

F1's positive outputs span magnitudes from 1e-245 to 7.7e-16 (13 positive values out of 19 total observations in Week 9 data).

### Findings

**log1p is ineffective for F1's output range**:

| Metric | log1p | log |
|--------|-------|-----|
| Output range | [1.87e-245, 7.71e-16] | [-563.5, -34.8] |
| Spread | ~0 (values ≈ inputs) | ~529 units |
| RF signal quality | No useful variation | Strong variation |

- `log1p(x) = log(1 + x)`. For x << 1 (Taylor expansion), `log1p(x) ≈ x`.
- All 13 positive F1 outputs are far below 1 (max is 7.7e-16), so `log1p` passes values through unchanged.
- `log(x)` maps these to the range [-563.5, -34.8], providing ~529 units of spread for the RF to model.

**Decision**: Use `log(y)` for Stage 2 RF training targets.
**Rationale**: Provides interpretable, well-spread training targets; RF splits can differentiate observations.
**Alternatives considered**: log1p (current — ineffective), log10 (would also work but log is standard), Box-Cox (adds complexity for marginal gain).

## Research Task 2: RF regressor handling of large-magnitude negative targets

### Context

With `log` transform, RF training targets range from -563.5 to -34.8. Need to confirm scikit-learn's RandomForestRegressor handles this correctly.

### Findings

- RandomForestRegressor is scale-invariant by construction — it splits on thresholds, not magnitudes.
- Large negative values (e.g., -563) are handled identically to small values in terms of tree splitting.
- Per-tree uncertainty (std across tree predictions) remains valid in log-space.
- **No issues expected**. The RF regressor will function correctly with these training targets.

**Decision**: No special handling needed for large-magnitude negative log values.
**Rationale**: RF is inherently scale-invariant; splits operate on thresholds.

## Research Task 3: Back-transformation and acquisition function interaction

### Context

The acquisition function uses weighted UCB: `a(x) = p(x)·mu(x) + kappa·p(x)·sigma_RF(x)`. Need to determine whether mu and sigma should be in log-space or back-transformed for acquisition.

### Findings

- The acquisition function ranks candidates — absolute values don't matter, only relative ordering.
- In log-space, `mu` values range from -563 to -35. Higher (less negative) = better, which aligns with maximisation.
- The UCB formula `p(x)·mu + kappa·p(x)·sigma` functions correctly in log-space:
  - mu in [-563, -35]: higher is better ✓
  - sigma > 0: exploration bonus adds to mu ✓
  - p(x) in [0, 1]: weights both terms ✓
- Back-transformation via `exp(mu)` would collapse values to near-zero, destroying ranking signal.

**Decision**: Compute acquisition function entirely in log-space. Only use `exp()` for final display (if needed).
**Rationale**: Log-space preserves ranking information needed for candidate selection.
**Note**: The contour visualisation will also display in log-space (per clarification).

## Research Task 4: log(0) safety guard

### Context

The log transform is only applied to positive outputs. Need to confirm the data pipeline prevents log(0).

### Findings

- F1 Week 9 data: 13 positive, 1 zero, 5 negative out of 19 observations.
- Stage 1 classifier produces binary labels `y > 0` (strict inequality).
- Stage 2 only trains on `y[y > 0]` — the zero value is excluded.
- `np.log()` on any positive float, no matter how small (e.g., 1e-245), produces a valid finite result.
- `np.log(0)` = -inf — but this case is prevented by the `y > 0` filter.
- Subnormal floats (e.g., 1e-245 near the minimum representable float) produce valid log values (~-563).

**Decision**: Existing `y > 0` filter is sufficient. No additional guards needed.
**Rationale**: Strict inequality excludes zero; all positive floats produce valid log values.

## Research Task 5: Removing Local Penalization from Weighted UCB

### Context

Local penalization applied a multiplicative Gaussian mask (PENALTY_RADIUS=0.15) around existing data points to suppress re-sampling near observations. User requests removal.

### Findings

- With only 19 points in [0,1]², coverage is sparse — penalizing proximity may over-restrict the search
- KAPPA=3.0 already provides strong exploration pressure via the uncertainty term
- Without penalization, the acquisition is purely data-driven: high probability × high predicted value or high uncertainty
- Simpler acquisition function aligns with constitution Principle I (Simplicity)
- If the proposed point happens to be close to an existing observation, the proximity warning (distance < 0.05) still alerts the user

**Decision**: Remove local penalization entirely from the acquisition function.
**Rationale**: Sparse data + strong KAPPA exploration already prevents clustering; user explicitly requested removal.
**Alternatives considered**: Reduce PENALTY_RADIUS (rejected — user specified full removal), keep with radius 0.05 (also rejected).

## Research Task 6: Removing Interior Penalty from Weighted UCB

### Context

Interior penalty was a sinusoidal function (STEEPNESS=0.1, FLOOR=0.01) that suppressed candidates near the [0,1]² boundary. User requests removal.

### Findings

- The penalty prevented edge-clustering, but with only 19 samples the boundary region may contain viable optima
- Removal allows the acquisition function to freely explore the full [0,1]² domain
- The submission format clips to [0.0, 0.999999] regardless, so boundary proposals are still valid
- Simpler acquisition aligns with constitution Principle I
- If edge proposals are suboptimal, the performance evaluation section will detect stalling

**Decision**: Remove interior penalty entirely from the acquisition function.
**Rationale**: Full domain exploration allowed; boundary candidates are valid submissions; simplicity.
**Alternatives considered**: Reduce STEEPNESS (rejected — user specified full removal).

## Research Task 7: Simplified Acquisition Function Behaviour

### Context

With both penalties removed, the acquisition function simplifies to: `a(x) = p(x)·mu(x) + KAPPA·p(x)·sigma_RF(x)`, with KAPPA=3.0 unchanged.

### Findings

- The function naturally balances exploitation (p·mu term) and exploration (p·sigma term)
- p(x) weights both terms by the classifier's probability of a positive output
- KAPPA=3.0 keeps exploration emphasis, appropriate given F1 has been stalling
- Isolating the effect of removing penalties (vs. also changing KAPPA) allows clean comparison with previous iterations
- The 3-panel contour Panel 3 shows the raw weighted UCB surface (no penalty mask)

**Decision**: Use raw weighted UCB with KAPPA=0.5 (exploitation-focused). See Research Task 8 below for updated rationale.
**Rationale**: Log transform gives the surrogate meaningful signal (R² ≈ 0.90); only 4 budget submissions remain.

## Research Task 8: KAPPA Reduction from 3.0 to 0.5 (Exploitation Focus)

### Context

The clarification session determined that F1 should prioritise exploitation over exploration. The log transform gives the surrogate meaningful signal (R² ≈ 0.90 on log-scale targets), and only 4 budget submissions remain — insufficient for broad exploration.

### Findings

- In UCB acquisition a(x) = μ(x) + κ·σ(x), standard BO uses κ ∈ [1.96, 2.5] for balanced exploration-exploitation
- κ=0.5 heavily favours predicted high-value regions (exploitation) while retaining slight exploration pressure
- With R² ≈ 0.90, the surrogate can meaningfully differentiate between regions — trusting its predictions is justified
- The proposed candidate at (0.729, 0.780) is 0.047 from the best-ever observation at (0.731, 0.733) — exploitation is working as intended
- Risk: potential to miss better optima in unexplored high-σ areas if surrogate is locally miscalibrated
- Mitigation: Week 10 can adjust to κ=0.1 (near-greedy) or back to κ=1.0+ if this submission shows no improvement

**Decision**: KAPPA=0.5 (exploitation-focused)
**Rationale**: Log transform provides meaningful surrogate signal; only 4 submissions remain; aggressive exploitation is defensible given the surrogate's accuracy.
**Alternatives Considered**: κ=1.96 (standard balanced — too exploratory for 4 remaining submissions), κ=0.1 (near-greedy — available as Week 10 fallback), κ=3.0 (previous value — pure exploration, inappropriate given improved surrogate).

## Summary

All unknowns resolved. No NEEDS CLARIFICATION items remain.

| Item | Resolution |
|------|-----------|
| log1p ineffectiveness | Confirmed — log1p ≈ identity for F1's range |
| RF with large negatives | No issues — RF is scale-invariant |
| Acquisition in log-space | Use log-space throughout; preserves ranking |
| log(0) safety | Existing y > 0 filter is sufficient |
| Local penalization removal | Removed — sparse data + strong classifier gating suffices |
| Interior penalty removal | Removed — full domain exploration allowed |
| Simplified acquisition | Raw weighted UCB with KAPPA=0.5 (exploitation-focused) |
| KAPPA reduction | 3.0 → 0.5 — log transform gives meaningful signal, 4 submissions remain |
