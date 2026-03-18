# Research: Week 11 Performance Review & Feedback

**Date**: 2026-03-17 | **Branch**: `033-week11-feedback`

## Research Context

This is a review-only feature — no surrogate fitting, no acquisition optimisation. The research focuses on documenting the current state of each function's strategy (used in Week 10) so the review notebooks can accurately evaluate performance and propose improvements.

No NEEDS CLARIFICATION items existed in the Technical Context. Research below consolidates the per-function strategies and data characteristics for reference during implementation.

## Per-Function Strategy Summary (Week 10)

### F1 — 2D, 21 samples (10 initial + 11 submissions)

- **Decision**: SFGP Matérn-2.5 ARD + qLogNEI q=4 + log transform + distance-based selection + 10,000 Sobol seeds + interior penalty (shallow) + aggressive exploitation
- **Rationale**: Log transform handles large output range. Interior penalty avoids boundary samples. Distance-based selection from q=4 candidates promotes spatial diversity.
- **Alternatives considered**: Hurdle Model + Weighted UCB (used weeks 7-8, dropped due to stalling); standard GP without log transform (poor fit on orders-of-magnitude variation)

### F2 — 2D, 21 samples (10 initial + 11 submissions)

- **Decision**: SFGP Matérn-2.5 ARD + Standardize(m=1) + qLogNEI q=4 + 50 MLL restarts + LS bounds [0.005, 10.0] + 4,096 Sobol seeds + interior penalty (shallow) + aggressive exploitation
- **Rationale**: Standardize provides stable training. Tight lengthscale bounds prevent over-smoothing. 50 MLL restarts ensure global optimum of marginal likelihood.
- **Alternatives considered**: Matérn-1.5 (too rough for smooth F2 landscape); fewer MLL restarts (inconsistent fits)

### F3 — 3D, 26 samples (15 initial + 11 submissions)

- **Decision**: SFGP Matérn-2.5 ARD + shift transform (y - y_min) + qLogNEI q=3 + 40 MLL restarts + noise_lb=1e-4 + 2048 raw samples
- **Rationale**: Shift transform ensures all outputs positive for qLogNEI. Lower noise bound for tight fit. q=3 balances exploration budget with 3D space.
- **Alternatives considered**: BART (explored in week 6, abandoned due to poor uncertainty calibration); MFGP (no multi-fidelity data available)

### F4 — 4D, 41 samples (30 initial + 11 submissions)

- **Decision**: SFGP Matérn-2.5 ARD + Standardize(m=1) + qLogNEI q=4 + noise_lb=1e-3 + 30 MLL restarts + 512 MC samples + distance-based selection
- **Rationale**: Larger initial sample (30) supports GP in 4D. Standardize and moderate noise floor for stable training. Distance-based selection from q=4 batch.
- **Alternatives considered**: MFGP (used weeks 8-9, switched to SFGP for simplicity); higher q values (diminishing returns with 4D budget)

### F5 — 4D, 31 samples (20 initial + 11 submissions)

- **Decision**: GP Matérn-1.5 ARD + log transform + Standardize(m=1) + qLogNEI q=4 + 60 MLL restarts + 8000 raw samples
- **Rationale**: Matérn-1.5 for rougher landscape. Log transform handles output range. 60 MLL restarts for reliable convergence in 4D.
- **Alternatives considered**: Matérn-2.5 (too smooth for F5 landscape); interior penalty (removed — was over-constraining the search space)

### F6 — 5D, 31 samples (20 initial + 11 submissions)

- **Decision**: SFGP Matérn-1.5 ARD + rank-based interior penalty + qLogNEI q=4 + noise_lb=1e-3 + 5000 raw samples + milk≥0.12 constraint
- **Rationale**: F6 is the best-performing function (4/10 improvements, not stalling). Rank-based penalty preserves relative ordering while avoiding boundaries. Milk constraint from domain knowledge.
- **Alternatives considered**: Matérn-2.5 (Matérn-1.5 better captures F6's roughness); removing interior penalty (boundary samples were problematic)

### F7 — 6D, 41 samples (30 initial + 11 submissions)

- **Decision**: Neural Network (6→5→5→1, dropout=0.05) + 50%mean/50%EI blend + interior penalty (STEEPNESS=0.02) + 50k candidates
- **Rationale**: 6D space is challenging for GP. Small NN avoids overfitting with 41 samples. Blended acquisition balances exploitation (mean) with exploration (EI). 50k random candidates for high-D search.
- **Alternatives considered**: GP (poor scaling in 6D with 41 samples); pure EI (too exploratory in later iterations)

### F8 — 8D, 51 samples (40 initial + 11 submissions)

- **Decision**: SFGP Matérn-2.5 ARD + Standardize(m=1) + qLogNEI + 512 MC samples + 8192 raw samples + ≥30 MLL restarts
- **Rationale**: 8D is the highest-dimensional problem. SFGP with Standardize for stable training at scale. High MC sample count for reliable acquisition evaluation. 8192 raw samples for broad initial search.
- **Alternatives considered**: Neural network surrogate (considered but GP still viable with 51 samples in 8D); dimensionality reduction (no clear redundant dimensions identified)

## Visualisation Best Practices for Review Notebooks

- **Decision**: Convergence plot + 2D pair plots with green star best marker
- **Rationale**: Convergence shows temporal progression of optimisation. Pair plots reveal spatial distribution in input space. Green star on best output enables quick visual identification of the optimal region.
- **Alternatives considered**: 3D surface plots (impractical for d>3); parallel coordinates (less intuitive for pair-wise relationships)

## Notebook Structure Pattern

- **Decision**: 10-cell template — title/config, imports, data loading, summary table, convergence plot, pair plots, best value highlight, evaluation markdown, strategy proposal markdown
- **Rationale**: Matches Week 10 review notebook structure for consistency. Self-contained per constitution.
- **Alternatives considered**: Shared utility module (violates constitution simplicity principle)

## Red Star Marker Visibility (Clarification 2026-03-17)

- **Decision**: Increase scatter `s` parameter from 200 to 350 for the red star best-output marker in all 8 notebooks.
- **Rationale**: At `s=200`, the star is ~14pt diameter. In notebooks with many subplots (F7: 15 subplots in 5-column grid, F8: 28 subplots in 7-column grid), individual axes are small enough that the star can overlap with underlying blue/orange dots and become hard to distinguish. At `s=350` (~18.7pt diameter), the star is ~34% larger in area, providing sufficient visibility even in the densest grid layouts.
- **Impact**: Single parameter change in cell index 7 (pair plot code cell) of all 8 notebooks. The `Line2D` legend element uses `markersize=15` (independent of scatter `s`) and remains unchanged.
- **Alternatives considered**: Adding black edge outline (rejected by user); both size increase and edge outline (rejected); different marker type such as crosshair or arrow (rejected).
- **Outcome**: Implemented, but user reports the star is still not visible at s=350 with red colour. See next section.

## Green Star Marker & Size Increase (Clarification 2026-03-17, Session 2)

- **Decision**: Change marker colour from red to green and increase marker size from s=350 to s=500.
- **Rationale**:
  - **Colour**: Red (`c='red'`) has poor contrast against the orange submission points (both warm hues). Green provides strong complementary contrast against both blue (initial) and orange (submissions), making the best-output marker immediately distinguishable. In colour theory, green sits between blue and yellow on the visible spectrum, maximising perceptual distance from both blue (~470nm) and orange/red (~600-700nm).
  - **Size**: s=350 (~18.7pt diameter) was insufficient for high-dimensional subplot grids. s=500 (~22.4pt diameter) provides a ~43% area increase over s=350 and a 2.5× area increase over the original s=200. This ensures the star dominates even on F8's 28-subplot grid (7×4 layout) where each subplot is ~2 inches wide. The star at s=500 will subtend roughly 8-10% of subplot axis area, well above the 5% visibility threshold for annotation markers.
  - **Legend**: The `Line2D` legend element uses `color='green'` (was `color='red'`) and `markersize=15` (unchanged). The label remains "Best".
- **Impact**: Two parameter changes in cell index 7, source line 38 of all 8 notebooks: `c='red'` → `c='green'` and `s=350` → `s=500`. One parameter change in the legend construction: `color='red'` → `color='green'`.
- **Alternatives considered**:
  - s=400 (+14% area over 350): Too modest, unlikely to resolve the visibility issue given s=350 already failed
  - s=450 (+29% area over 350): Moderate but still conservative
  - s=500 (+43% area over 350): **Selected** — substantial increase provides high confidence of visibility without overwhelming lower-dimensional plots (F1/F2 have only 1 subplot)
  - s=600 (+71% area over 350): Excessive for F1/F2's single-subplot layout; star would dominate the plot
  - Keeping red with larger size only: Red-on-orange contrast is fundamentally poor regardless of size
  - Black edgecolor with green fill: Unnecessary complexity — green alone provides sufficient contrast
