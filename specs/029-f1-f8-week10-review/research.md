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

---

## F1 Optimisation Run — Research (Phase 0)

**Spec**: [spec-f1-optimisation.md](spec-f1-optimisation.md)  
**Date**: 2026-03-11

### R6: Matérn-2.5 vs Matérn-1.5 for F1

**Decision**: Use Matérn-2.5 (nu=2.5) for F1.

- F1 outputs span ~230 orders of magnitude — the posterior surface benefits from the twice-differentiability of Matérn-2.5 for smoother interpolation between sparse positive observations
- Project precedent: F3, F4 (MFGP), and F8 all use Matérn-2.5 successfully; F2, F5, F6 use Matérn-1.5
- Prequential evaluation on F8 showed nu=2.5 outperformed nu=1.5 in grid search
- For 2D functions with very sparse data (20 points), the smoother kernel provides better extrapolation

**Alternatives considered**: Matérn-1.5 (used by F2) — rejected because F1's extremely sparse near-zero observations need smoother interpolation; RBF (too smooth, may underfit noise).

**Rationale**: The smoothness assumption of Matérn-2.5 is appropriate because F1 outputs suggest a continuous underlying function with steep but smooth gradients near the optimum.

### R7: Log-transform implementation for F1

**Decision**: Use `y_log = torch.log(torch.clamp(y, min=1e-300))` as manual transform before GP fitting.

- F1 week 9 already uses `np.log(y_pos)` for the RF regressor — same principle, different framework
- F5 uses `log1p` — rejected for F1 because outputs are < 1e-15, making log1p(y) ≈ y (no compression)
- BoTorch's `Standardize(m=1)` can be applied AFTER the manual log transform for additional normalisation
- Expected range after log: approximately [-690, -35] from the current 20 data points
- No BoTorch output transform needed — manual log is applied once to the tensor before constructing the GP

**Alternatives considered**:
- BoTorch `Log` outcome transform — exists but applies inverse transform on predictions, adding complexity
- `log1p` — produces near-zero values for F1's tiny outputs; insufficient dynamic range
- No transform — GP fitting on raw values spanning 230 orders of magnitude fails numerically

**Rationale**: Manual log before GP fitting is the simplest approach (per constitution Principle I) and matches the F1 week 9 pattern.

### R8: Sobol seeding with 10,000 points — performance impact

**Decision**: Use `raw_samples=10000` for optimize_acqf.

- Project precedent: F1 week 9 uses 20,000 random candidates for acquisition search (even heavier)
- F2 uses raw_samples=1024 (2D); F5 uses 5000 (4D); F8 uses 4096 (8D)
- For 2D, 10,000 Sobol points provide dense coverage of [0,1]² (100×100 equivalent)
- BoTorch's optimize_acqf generates Sobol points internally — no custom code needed
- Performance: 10k Sobol in 2D is fast (<1s); the q=4 L-BFGS optimisation from 20 restarts is the bottleneck

**Alternatives considered**: raw_samples=1024 (F2 default) — rejected because F1 has 0/10 improvements, indicating the acquisition landscape needs denser initial coverage; raw_samples=20000 — unnecessary given BoTorch's optimiser uses L-BFGS from best Sobol points.

**Rationale**: 10,000 provides good coverage without computational overhead. Matches the user's explicit request.

### R9: Distance-based candidate selection from q=4

**Decision**: Reuse the F2 week 9 two-stage selection pattern verbatim.

**Pattern** (from F2 week 9):
1. Get posterior means for all q=4 candidates
2. Quality gate: keep candidates with mean ≥ median of the batch
3. Exploration bonus: from qualified set, select the one with maximum minimum-distance to all training points

- This pattern is proven in F2 and balances quality (promising predictions) with exploration (spatial diversity)
- For F1 with 20 training points in 2D, the distance computation is trivial
- Uses `torch.cdist` for efficient Euclidean distance calculation

**Alternatives considered**: Pure exploitation (pick highest mean) — rejected because F1 needs exploration; pure exploration (pick farthest from data) — rejected because it ignores model predictions entirely.

**Rationale**: Two-stage filter ensures the proposed point is both promising AND spatially diverse.

### R10: GP hyperparameter bounds and constraints

**Decision**: Use the following constraints for F1's SFGP:

| Parameter | Constraint | Justification |
|-----------|-----------|---------------|
| Lengthscale (per dim) | Interval(0.01, 2.0) | Prevents collapse to zero or infinity; matches F2 pattern |
| Noise variance | GreaterThan(1e-4) | F1 log-outputs are noisy; 1e-4 prevents numerical issues |
| Outputscale | Default (unconstrained) | Let MLL determine appropriate scale for log-transformed data |
| MLL restarts | 15 | Matches F2; sufficient to avoid degenerate solutions for 2D |

- F2 uses noise_lb=1e-3, but F1's log-transformed outputs may have less observation noise → use 1e-4
- Lengthscale bounds [0.01, 2.0] prevent degenerate fits: too small = overfitting, too large = constant model
- Each restart uses a different `torch.manual_seed(seed)` for reproducible but diverse initialisation

**Alternatives considered**: noise_lb=1e-3 (F2 value) — potentially too aggressive for log-space where values span [-690, -35]; Unconstrained lengthscales — risk of degenerate solutions with only 20 points.

### R11: Existing notebook variables in scope

**Decision**: The new cells can reuse variables from the existing 12 cells.

Variables available after running existing cells 1–12:
- `inputs` (ndarray, shape [20, 2]) — all input data
- `outputs` (ndarray, shape [20]) — all output data
- `N_INITIAL` = 10
- `FUNC_NUM` = 1
- `DATA_DIR` = '../../data/f1/'
- `USE_LOG_SCALE` = True
- `WEEK` = 10
- `N_DIMS` = 2
- `n_total` = 20
- `n_submissions` = 10
- `running_best` (ndarray from `np.maximum.accumulate(outputs)`)

New cells will import BoTorch/GPyTorch, convert data to tensors, and proceed with the GP pipeline. No re-loading of data is needed.

---

## F2 Optimisation Run — Research (Phase 0)

**Spec**: [spec-f2-optimisation.md](spec-f2-optimisation.md)  
**Date**: 2026-03-11

### R12: Matérn-2.5 vs Matérn-1.5 for F2

**Decision**: Switch to Matérn-2.5 (nu=2.5) for F2.

- F2 outputs are in [0.25, 0.67] — narrow range where Matérn-1.5's rougher (C¹) posterior creates multiple local modes trapping the acquisition function
- Project precedent: F3 (3D, narrow output range) and F8 (8D) both use Matérn-2.5 + Standardize(m=1) and are NOT stalling
- Matérn-2.5 (C²) produces smoother posteriors enabling qLogNEI to find better optima
- F2 week 10 review itself identifies the Matérn-1.5 kernel as creating "a rough posterior that traps the acquisition in local modes"

**Alternatives considered**: Keep Matérn-1.5 + increase restarts only — addresses symptoms not root cause; RBF — too smooth, less standard in BO practice.

**Rationale**: The twice-differentiability of Matérn-2.5 is appropriate for F2's narrow-range function where smoother interpolation between 20 samples will improve acquisition landscape quality.

### R13: Standardize(m=1) interaction with GP fitting

**Decision**: Add Standardize(m=1) as outcome_transform in SingleTaskGP.

- Applied as `SingleTaskGP(..., outcome_transform=Standardize(m=1))` — transforms during construction, before MLL fitting
- Rescales outputs to zero-mean, unit-variance internally; model.posterior() auto-untransforms predictions
- Noise constraint (NOISE_LB=1e-4) operates in standardised space — becomes slightly looser relative to data scale, reducing overfitting risk
- Improves MLL numerical stability for F2's narrow output range (~0.42 span)
- Proven pattern: F3 and F8 both use Standardize(m=1) successfully

**Alternatives considered**: Manual z-score — adds complexity, error-prone; No standardisation — risks ill-conditioned MLL with narrow output range.

**Rationale**: BoTorch's built-in Standardize is simpler (constitution Principle I) and numerically superior to both manual z-score and no standardisation.

### R14: Lengthscale bounds [0.005, 10.0] — impact with 20 samples in 2D

**Decision**: Approve widened bounds [0.005, 10.0].

- LS=0.005 captures structure on ~0.5% of input range (fine-grained)
- LS=10.0 is effectively a flat prior over [0,1]² (coarse baseline) — MLL will avoid this if data has structure
- With 20 points, average spacing ≈ √(1/20) ≈ 0.224; useful LS range is roughly [0.05, 5.0]
- The wider bounds [0.005, 10.0] contain this range plus margin for MLL to explore
- Week 9's [0.01, 2.0] may force MLL into degenerate local optima; wider bounds allow escape
- F3 and F8 use effectively wide bounds and are non-stalling

**Alternatives considered**: Keep [0.01, 2.0] — perpetuates stalling; [0.001, 100.0] — overly permissive for 2D.

**Rationale**: [0.005, 10.0] is a practical middle ground giving MLL freedom to find the optimal lengthscale without extreme values.

### R15: Existing F2 week 10 notebook variables in scope

**Decision**: Reuse existing variables from cells 1–12 directly.

Variables available after running existing cells:
- `inputs` (ndarray, shape [20, 2]) — all input data
- `outputs` (ndarray, shape [20]) — all output data
- `N_INITIAL` = 10
- `FUNC_NUM` = 2
- `DATA_DIR` = '../../data/f2/'
- `USE_LOG_SCALE` = False
- `WEEK` = 10
- `N_DIMS` = 2
- `n_total` = 20
- `n_submissions` = 10
- `running_best` (ndarray from `np.maximum.accumulate(outputs)`)
- `init_best` (numpy.float64, ~0.6112) — best from initial samples
- `stalling` = True — confirmed by evaluation cell

New cells will import BoTorch/GPyTorch, convert data to tensors, and proceed with the GP pipeline. No re-loading of data is needed. No log transform required — Standardize(m=1) handles output conditioning.
