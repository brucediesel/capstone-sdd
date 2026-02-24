# Research: F1 Interior Penalty Design Decisions

**Feature**: 014-f1-interior-penalty  
**Date**: 2026-02-24

## 1. Penalty Shape Analysis

### Decision
Use `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(π·xᵢ)^(2·STEEPNESS)` as the interior penalty.

### Analysis by STEEPNESS value

For a single dimension, `sin(π·x)^(2·S)` at key distances from the boundary:

| STEEPNESS (S) | w(0.01) | w(0.05) | w(0.10) | w(0.20) | w(0.50) | 50% contour (δ from edge) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 0.176 | 0.395 | 0.553 | 0.724 | 1.000 | ~0.083 |
| 1.0 | 0.031 | 0.156 | 0.305 | 0.524 | 1.000 | ~0.196 |
| 2.0 | 0.001 | 0.024 | 0.093 | 0.274 | 1.000 | ~0.318 |
| 5.0 | ~0.0 | 0.001 | 0.008 | 0.052 | 1.000 | ~0.424 |
| 10.0 | ~0.0 | ~0.0 | ~0.0 | 0.003 | 1.000 | ~0.461 |

Note: In 2D, the penalty is the product across both dimensions. At a corner: `w ≈ w₁ · w₂`, so suppression is even stronger.

### Rationale
- S=2.0 gives ~97.6% suppression at x=0.05 (single dim), strong enough to prevent boundary-hugging
- The 50% contour at δ≈0.318 from each edge leaves the central region [0.32, 0.68]² fully available
- S=5.0+ is too aggressive — suppresses 95% of the search space

## 2. Interaction with Local Penalization

### Decision
The interior penalty is safe to combine multiplicatively with the existing local Gaussian penalization.

### Rationale
- **Local penalization** targets re-sampling: `∏ᵢ[1 - exp(-‖x-xᵢ‖²/(2·0.15²))]` suppresses within ~0.15 of each existing point
- **Interior penalty** targets boundary attraction: suppresses within ~0.05–0.32 of each edge (depending on steepness)
- These address orthogonal failure modes — boundary variance inflation vs. re-sampling near known points
- Combined dead zones occur only at boundary-AND-near-data regions, exactly where we least want to sample
- Both are multiplicative, bounded in [0, 1], and smooth — no pathological interaction

### Alternatives Considered
- Additive combination: Rejected — changes the acquisition scale and requires tuning λ relative to UCB values
- Max-based combination: Rejected — loses the dual suppression benefit

## 3. STEEPNESS Default Value

### Decision
`STEEPNESS = 2.0`

### Rationale
- At S=2.0, a candidate at x=0.05 from any edge receives ~2.4% of interior acquisition value (97.6% suppression per dimension)
- Still allows boundary proposals if raw acquisition is overwhelmingly large there (via FLOOR=0.01, minimum is 1% not 0%)
- With 17 existing samples in [0,1]², the remaining unexplored interior is ample
- S=1.0 is a fallback if S=2.0 over-concentrates: only 84.4% suppression at x=0.05

### Alternatives Considered
- S=1.0: Too permissive — candidates 0.10 from edge still get ~30% of interior value
- S=5.0: Too aggressive — effective search space shrinks to ~[0.42, 0.58]² central island

## 4. FLOOR Default Value

### Decision
`FLOOR = 0.01`

### Rationale
- 100× suppression ratio (1.0 at centre vs 0.01 at boundary)
- F1 uses random candidate selection (20K uniform samples + argmax), NOT gradient-based — gradient death is not a concern
- Numerically distinguishable from 0 in float64
- If a boundary region truly has ~100× better raw acquisition than interior, the 0.01 floor still allows it to win

### Alternatives Considered
- FLOOR=0.001: Unnecessarily aggressive — 0.01 already provides 100× suppression
- FLOOR=0.1: Too lenient — only 10× suppression, may not prevent boundary clustering
- FLOOR=0.0: Creates exact zeros — mathematically degenerate

## 5. Penalty Shape Choice

### Decision
sin(πx)² formulation over alternatives.

### Rationale

| Shape | C∞ smooth | 0 at edges | 1 at centre | Separable | 1 parameter | Pure NumPy |
|-------|:---------:|:----------:|:-----------:|:---------:|:-----------:|:----------:|
| sin(πx)^(2S) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linear ramp | ✗ (kink) | ✗ | ✓ | ✓ | ✓ | ✓ |
| Beta CDF | ✓ | ✗ | ✗ | ✓ | ✗ (2 params) | ✓ |
| Clipped parabola | ✓ | ✓ | ✓ | ✓ | ✗ (fixed) | ✓ |

### Alternatives Considered
- **Linear ramp** `min(x, 1-x, δ)/δ`: Non-differentiable kinks. Flat interior provides no signal about edge distance.
- **Beta CDF**: Over-parameterised (2 parameters), never exactly reaches 0 or 1.
- **Clipped parabola** `(4x(1-x))^S`: Viable but sin² is the standard in BO literature and more recognisable to examiners.
