# Research: Week 10 Performance Review & Visualisation

**Feature**: 029-f1-f8-week10-review  
**Date**: 2026-03-11

## Research Questions

### R1: What are the current (week 9) strategies for each function?

**Decision**: Document each function's week 9 surrogate and acquisition for the performance evaluation markdown.

| Function | Surrogate | Acquisition | Key Hyperparameters |
|----------|-----------|-------------|---------------------|
| F1 | Hurdle Model (LogReg classifier + RF regressor on log(y)) | Weighted UCB, κ=0.5 | C=1.0, n_estimators=100, max_depth=3, 20k candidates |
| F2 | SFGP Matérn-1.5 ARD | qLogNEI q=4 with distance selection | noise_lb=1e-3, LS bounds [0.01, 2.0], 15 MLL restarts |
| F3 | SFGP Matérn-2.5 ARD, Standardize(m=1) | qLogNEI q=1 | 20 restarts, 512 raw samples, noise_lb=1e-6 |
| F4 | MFGP Matérn-2.5 + LinearTruncatedFidelityKernel | MF-qNEI q=4, 64 fantasies | noise_lb=1e-4, fidelity fixed at 1.0 |
| F5 | GP Matérn-1.5 ARD, log1p + Standardize(m=1) | qLogNEI q=4, distance-based | 50 restarts, 5000 raw samples |
| F6 | SFGP Matérn-1.5 ARD, noise_lb=1e-2 | qLogNEI q=4, rank-based interior penalty | 50 restarts, 3000 raw samples, milk ≥ 0.10 constraint |
| F7 | Neural Network (6→5→5→1, dropout=0.05) | 70% mean + 30% EI blend, interior penalty | lr=0.005, 200 epochs, 20k candidates, STEEPNESS=0.05 |
| F8 | SFGP Matérn-2.5 ARD, Standardize(m=1) | qEI, XI=0.01 | noise_lb=1e-7, 30 restarts, 4096 raw samples, 256 MC |

**Rationale**: These details are needed for each notebook's strategy summary (FR-009) and improvement suggestions (FR-010).

### R2: Which functions are stalling?

**Decision**: 4 functions stalling, 4 showing progress.

| Function | Stalling? | Improvements (out of 9 submissions) | Notes |
|----------|-----------|-------------------------------------|-------|
| F1 | YES | 0/9 | All submissions produced near-zero outputs; no improvement across all 9 rounds |
| F2 | YES | 1/9 | Trapped in local optimum; degenerate GP history from week 8 |
| F3 | NO | 2/9 | Slow but steady improvement; Standardize(m=1) helped |
| F4 | YES | 5/9 | MFGP on single-fidelity data; early improvements stalled in later weeks |
| F5 | NO | 9/9 | Consistent improvement; log1p + Matérn-1.5 working well |
| F6 | NO | 9/9 | All-negative outputs trending towards zero (improvement); rank-based IP effective |
| F7 | YES | 4/9 | NN exploitation shift helped initially but recent submissions show no gains |
| F8 | NO | 8/9 | Near-continuous improvement; qEI has numerical warnings but results are good |

**Rationale**: Stalling detection is critical for FR-009 evaluation and FR-010 improvement suggestions.

### R3: F1 log-scale visualisation approach

**Decision**: Use `np.log(np.maximum(outputs, 1e-300))` to handle the log scale safely.

- F1 outputs span ~230 orders of magnitude (7.71e-16 to 1.87e-245)
- Prequential Evaluation confirmed log(y) produces meaningful range [-565, -35]
- Negative outputs: spec says set to zero — but F1 outputs are all positive (18/19 non-zero)
- For the convergence plot: use `np.log10` for the y-axis with `plt.yscale('log')`, clamping any zeros or negatives to a small epsilon before log

**Alternatives considered**: 
- symlog — rejected per user requirement
- log1p — produces near-zero values for tiny exponentials; insufficient dynamic range

### R4: 2D pair plot layout for high-dimensional functions

**Decision**: Use `itertools.combinations(range(d), 2)` to generate all pairs. Layout in grid with `math.ceil(n_pairs / cols)` rows.

| Function | Dims | Pairs | Suggested Grid |
|----------|------|-------|----------------|
| F1 | 2 | 1 | 1×1 |
| F2 | 2 | 1 | 1×1 |
| F3 | 3 | 3 | 1×3 |
| F4 | 4 | 6 | 2×3 |
| F5 | 4 | 6 | 2×3 |
| F6 | 5 | 10 | 2×5 or 3×4 |
| F7 | 6 | 15 | 3×5 |
| F8 | 8 | 28 | 4×7 |

**Rationale**: Grid layout keeps pair plots readable. Figure size scaled proportionally to number of subplots.

### R5: Best practices for matplotlib convergence + pair plots in notebooks

**Decision**: Use standard matplotlib with consistent styling across all 8 notebooks.

- Convergence: `plt.plot` with running max, blue section for initial, orange for submissions, vertical dashed line at boundary
- Pair plots: `plt.scatter` with blue for initial points, orange for submissions, annotate submission points with week number using `plt.annotate`
- Font size adjusted for subplot count to remain readable
- All figures use `fig.tight_layout()` for clean spacing

**Alternatives considered**: Plotly (interactive) — rejected for simplicity per constitution; seaborn — unnecessary dependency.
