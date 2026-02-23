# Research: F5 Week 7 — GP Matérn-5/2 + NEI

**Feature**: 011-f5-gp-nei
**Date**: 2026-02-23
**Purpose**: Resolve all technical unknowns from the Technical Context before implementation.

---

## RES-001: Log Transform Strategy

**Decision**: Use manual `log1p` / `expm1` transform, NOT BoTorch's built-in `outcome_transform=Log()`. Apply `torch.log1p(Y)` before fitting, then z-score standardise the log-transformed outputs via `Standardize(m=1)`. Use `np.expm1()` to invert for reporting and visualisation.

**Rationale**: BoTorch's `Log()` outcome transform uses plain `torch.log()` / `torch.exp()`, not `log1p`. With outputs as low as ~0.1, `log1p` provides better numerical behaviour near zero. Manual transform + z-score matches the established F3/F4 codebase pattern, keeping the pipeline consistent and explicit.

**Alternatives Considered**:
- `outcome_transform=Log()` — uses `log` not `log1p`; doesn't match codebase convention
- `ChainedOutcomeTransform(Log(), Standardize())` — same `log` vs `log1p` issue
- Custom `OutcomeTransform` subclass — over-engineered for this use case

---

## RES-002: Lengthscale Initialisation

**Decision**: Construct kernel as `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))`, then set `model.covar_module.base_kernel.lengthscale = 0.25` after construction. The scalar broadcasts to all 4 ARD dimensions.

**Rationale**: This is the exact pattern used in F3's Week 7 implementation. Setting after construction is clear, explicit, and places the initialisation visibly next to the output scale and noise initialisations.

**Alternatives Considered**:
- `MaternKernel(lengthscale=torch.full((1,4), 0.25))` — works but more verbose
- `model.initialize(**{'covar_module.base_kernel.lengthscale': 0.25})` — less readable

---

## RES-003: Noise Initialisation

**Decision**: Create `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`, then set `model.likelihood.noise = 0.03`. This initialises learnable noise at 3% of the standardised output variance (which is 1.0 by construction after z-scoring).

**Rationale**: The `train_Yvar` parameter is for fixed heteroscedastic noise (per-observation), not learnable noise initialisation. Setting `model.likelihood.noise` directly initialises it as a learnable parameter that MLL optimises. After z-score standardisation, `Var(y_std) = 1.0`, so `0.03 * Var(y_log) / Var(y_log) = 0.03` in standardised space. The `GreaterThan(1e-6)` constraint provides jitter floor matching the spec's requirement.

**Alternatives Considered**:
- `train_Yvar=torch.full_like(Y, noise)` — makes noise fixed (wrong semantics)
- `noise_constraint=Interval(1e-6, 0.1)` — constraining upper bound is unnecessary

---

## RES-004: NEI Exploration Parameter (ξ)

**Decision**: BoTorch's `qLogNoisyExpectedImprovement` has no direct ξ parameter. Use default configuration with `prune_baseline=True`. The user's ξ=0.01 intent is inherently satisfied by NEI's formulation.

**Rationale**: NEI's exploration-exploitation tradeoff is implicit — it computes expected improvement over the noisy best-observed value, naturally balancing exploitation (high mean) with exploration (high uncertainty). The `eta` parameter is for constraint smoothing only, not exploration. `tau_max=0.01` and `tau_relu=1e-6` are numerical approximation parameters for the smooth max/ReLU operators.

**Alternatives Considered**:
- Misuse `eta` as ξ — wrong semantics (eta is for constraints)
- Manual offset to baseline — fragile and non-standard
- Analytic `ExpectedImprovement` with `best_f - 0.01` — only supports q=1

---

## RES-005: NEI Acquisition Optimisation

**Decision**: Use `optimize_acqf` with `raw_samples=3000` and `num_restarts=50`. This directly implements "3000 Sobol starts → best 50 → L-BFGS".

**Rationale**: `optimize_acqf` works in two stages: (1) evaluate `raw_samples` Sobol points, select best `num_restarts`, (2) L-BFGS-B from those starting points. This is the exact strategy requested. F3 used `num_restarts=10, raw_samples=512`; F4 used `num_restarts=20, raw_samples=512`. For F5 with q=2 and the large output range, the user specified the more aggressive 3000/50 configuration.

**Alternatives Considered**:
- `batch_initial_conditions` — bypasses Sobol; unnecessarily complex
- `sequential=True` — greedy q=2; joint optimisation (default) is standard

---

## RES-006: Multiple MLL Restarts

**Decision**: Use manual restart loop with `N_RESTARTS=15`, following the F3/F4 pattern. Construct fresh model per restart with different `torch.manual_seed(seed)`, fit via `fit_gpytorch_mll`, score by neg-MLL, keep best via `copy.deepcopy`.

**Rationale**: `fit_gpytorch_mll` performs a single L-BFGS run with no built-in restart mechanism. Both F3 and F4 use the same manual restart pattern. 15 restarts is the sweet spot (middle of the 10–20 range recommended in the F5 research cell).

**Alternatives Considered**:
- Single `fit_gpytorch_mll` — risks local MLL optimum
- More L-BFGS iterations per restart — doesn't address multi-modality
- `fit_gpytorch_scipy` — lower-level; `fit_gpytorch_mll` works fine

---

## RES-007: Week 6 Visualisation Adaptation

**Decision**: Replicate F5 Week 6's 3-panel layout: (1) GP posterior mean contour, (2) GP posterior std contour, (3) ARD lengthscale-based dimension relevance bar chart using `1/ℓ` (inverse lengthscale, normalised). Identify top-2 important dimensions via shortest lengthscales; fix other 2 at proposed point values.

**Rationale**: Week 6 used 3 panels: mean, std, feature importance. For GBT, importance came from `feature_importances_`. For GP, the natural analog is inverse ARD lengthscale: shorter ℓ = more rapid variation = more important. Using `1/ℓ` ensures taller bars = more important, matching Week 6's visual convention. The 2D slice approach (top-2 dims by importance, fix others) matches F4 Week 7's strategy.

**Alternatives Considered**:
- F3-style pairwise slices — yields 6 pairs for 4D, too many panels
- F4-style 2-panel (mean + std only) — loses feature importance panel
- Raw lengthscale bars — shorter bars = more important is counter-intuitive
