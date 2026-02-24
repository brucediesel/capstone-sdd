# Research: F8 Week 7 — SFGP + qEI Acquisition

**Feature**: 018-f8-sfgp-qei
**Date**: 2025-02-24

---

## Decision 1: GP Configuration — Matern 2.5 + ARD + Standardize

**Decision**: Use `SingleTaskGP` with explicit `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-7))`, and the default `Standardize(m=1)` outcome transform.

**Rationale**: The user specified Matern 2.5, standardise, and noise >= 1e-07. BoTorch's `SingleTaskGP` applies `Standardize(m=1)` by default when no custom `outcome_transform` is passed. The default kernel is already Matern 2.5 + ARD, but passing it explicitly ensures the configuration is visible and documented. With 47 samples in 8D, the GP kernel matrix is 47x47 — computationally trivial. ARD gives one lengthscale per dimension, enabling feature importance analysis.

**Alternatives considered**:
- Default `SingleTaskGP` (no explicit kernel): Works the same but hyperparameters are implicit. Explicit is better for a capstone project that requires clear documentation.
- Manual z-score normalisation + `outcome_transform=None`: Adds complexity. The built-in `Standardize` handles this automatically and is the recommended BoTorch pattern.
- Matern 1.5 or RBF: User specified Matern 2.5. No reason to deviate.

---

## Decision 2: qEI MC Sample Count — 256

**Decision**: Start with 256 MC samples via `SobolQMCNormalSampler(sample_shape=torch.Size([256]))`.

**Rationale**: The user specified "256 (512 if stable)". With q=1 (single candidate), 256 Sobol quasi-random samples provide low-discrepancy coverage of the GP posterior. The SobolQMCNormalSampler produces more uniform samples than IID normal draws, so 256 is equivalent to ~500 IID samples in variance reduction terms. For a single 8D candidate, memory is not a concern. If execution completes without numerical warnings, 512 could be used, but 256 is the safe default.

**Alternatives considered**:
- 512: Marginal improvement in EI estimate accuracy. User allows it "if stable". Start with 256 for safety.
- 128: Insufficient for reliable qEI estimates with a Sobol sampler.
- IIDNormalSampler: Higher variance than Sobol. No reason to prefer it.

**API note**: BoTorch 0.16.1+ uses `sample_shape=torch.Size([N])`, not the deprecated `num_samples=N` parameter.

---

## Decision 3: Improvement Threshold xi = 0.01

**Decision**: Set `best_f = Y_train.max().item() + 0.01` when constructing `qExpectedImprovement`.

**Rationale**: `qExpectedImprovement` computes E[max(f(x) - best_f, 0)]. There is no explicit `xi` parameter in the API. The standard approach to incorporate an improvement threshold is to inflate `best_f` by `xi`, so that only improvements exceeding xi contribute positively. With y_max = 9.953 and xi = 0.01, best_f = 9.963. This is a very modest threshold (0.1% of the output range), primarily serving to filter out noise rather than aggressively penalise small gains.

**Alternatives considered**:
- No xi (best_f = y_max exactly): Risk of proposing points with negligible expected improvement.
- xi = 0.1: More aggressive; could miss moderate-improvement candidates. User specified 0.01.

---

## Decision 4: Fantasisation

**Decision**: Enable fantasisation by not passing `X_pending=None` (the default). For q=1 with no pending points, fantasisation is a no-op.

**Rationale**: The user specified "Fantasization: enabled for pending points". BoTorch's MC acquisition functions support fantasisation natively when `q > 1` (each point in the batch fantasises about the others' outcomes). For q=1 with no pending evaluations, there is nothing to fantasise about. The capability is "enabled" in the sense that the infrastructure supports it — if pending points existed, they could be passed via `X_pending`.

**Alternatives considered**:
- q=2 with fantasisation: Would propose two points simultaneously but the challenge only submits one per week.
- Explicitly setting `X_pending=[]`: Unnecessary; the default (None) is equivalent.

---

## Decision 5: Acquisition Optimisation — 30 Restarts, 4096 Raw Samples

**Decision**: Use `optimize_acqf(acq_fn, bounds=BOUNDS, q=1, num_restarts=30, raw_samples=4096)`.

**Rationale**: The F8 initial submission used exactly these settings for 8D optimisation and they worked reliably. 4096 raw Sobol samples provide good initial coverage of [0,1]^8, and 30 L-BFGS-B restarts from the best raw samples ensure thorough local optimisation. With a smooth GP posterior and smooth qEI surface, this is more than sufficient. The GP kernel matrix inversion (47x47) is fast, so each acquisition evaluation is cheap.

**Alternatives considered**:
- 50 restarts, 8192 raw: More thorough but unnecessary for 47-sample GP. Would double computation time for marginal benefit.
- 10 restarts, 2048 raw: Risky in 8D. Prior F8 cells already validated 30/4096 as reliable.

---

## Decision 6: Feature Importance — GP Lengthscale Inversion

**Decision**: Extract ARD lengthscales from `model.covar_module.base_kernel.lengthscale` and display importance as 1/lengthscale (normalised). Shorter lengthscale = more important dimension.

**Rationale**: This is the standard GP feature importance method and was used in the F8 initial submission (cell 12). It provides a direct comparison with the initial GP-based analysis. The Weeks 5-6 NN sections used gradient-based importance; returning to lengthscale-based importance for Week 7 (which uses a GP surrogate) is natural and simpler.

**Alternatives considered**:
- Gradient-based importance (Weeks 5-6 pattern): Only applicable to NN surrogates.
- Sensitivity analysis: More thorough but adds code complexity. Lengthscale inversion is the standard for GPs.

---

## Decision 7: 3-Panel Visualisation — Mean, Std, EI

**Decision**: Create a 3-panel figure showing GP posterior mean, GP posterior std, and EI acquisition surface on a 2D grid through the top-2 most important dimensions (by lengthscale), fixing remaining 6 dimensions at the best observed point's values.

**Rationale**: Matches the Week 5-6 visualisation pattern (3-panel: mean, uncertainty, acquisition). Weeks 5-6 used NN mean, MC std, and UCB. Week 7 naturally maps to GP mean, GP std, and EI. The top-2 dimensions are selected by smallest ARD lengthscale (highest feature importance). Grid resolution of 50x50 = 2500 points is sufficient for smooth contour plots.

**Visualisation EI note**: For the 2D grid, use analytic `ExpectedImprovement` (not qEI) to evaluate 2500 grid points efficiently. Analytic EI is equivalent to qEI with infinite MC samples for q=1, and runs in milliseconds vs minutes. The qEI is used for the actual candidate selection (with Sobol MC samples) while analytic EI is used for plotting.

**Alternatives considered**:
- Only 2-panel (mean + std): Misses the acquisition function visualisation. User wants "same visualisations as previous weeks".
- 4-panel: Adds complexity without user request.

---

## Decision 8: Convergence Plot — Weekly Boundaries

**Decision**: Plot running best across all 47 observations with vertical dashed lines at sample indices 40 (initial), 45 (Week 5), 46 (Week 6), and 47 (Week 7).

**Rationale**: The Week 6 convergence plot marked initial->Week 5 boundary at 40.5. Week 7 extends this with additional boundaries. Running best (cumulative maximum) is the standard convergence metric for maximisation.

**Alternatives considered**:
- Include the predicted mean of proposed point: Adds complexity. Keep it simple.
- Simple scatter of all y values: Less informative than running best.

---

## Decision 9: Fallback for All-Zero qEI

**Decision**: If `acq_value <= 0` after optimisation, fall back to selecting the candidate with the highest GP posterior mean (pure exploitation).

**Rationale**: With xi=0.01, the GP must predict improvement of at least 0.01 over y_max = 9.953 for qEI to be positive. If the GP is overconfident (low posterior variance near y_max), qEI may be zero. In this case, the best strategy is exploitation — select the point where the GP mean is highest. This is simpler than the F7 approach (which used interior penalty weight as fallback) because F8 does not use an interior penalty.

**Alternatives considered**:
- Random exploration: Wasteful when the GP provides a good mean estimate.
- Reduce xi to 0: Would make qEI non-zero for any predicted improvement, defeating the purpose of xi.
