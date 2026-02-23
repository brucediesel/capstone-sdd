# Research: F5 Week 7 — GP Matérn-5/2 + NEI

**Feature**: 011-f5-gp-nei
**Date**: 2026-02-23
**Purpose**: Resolve all technical unknowns from the Technical Context before implementation.

---

## RES-001: Output Transform Strategy

**Decision**: Use manual `log1p` / `expm1` transform followed by manual z-score standardisation. Pass `outcome_transform=None` to `SingleTaskGP` to disable the default `Standardize(m=1)`. Use `np.expm1()` to invert for reporting and visualisation.

**Rationale**: BoTorch's `Log()` outcome transform uses plain `torch.log()` / `torch.exp()`, not `log1p`. With outputs as low as ~0.1, `log1p` provides better numerical behaviour near zero. Manual log1p + z-score matches the established F3/F4 codebase pattern. Passing `outcome_transform=None` prevents double-standardization (manual z-score + default Standardize would corrupt the posterior). BoTorch will emit an `InputDataWarning` since data is pre-standardized with mean≈0, std≈1 — this is expected and harmless.

**Alternatives Considered**:
- Default `outcome_transform` (omit parameter) — would apply `Standardize(m=1)` on top of manual z-score, causing double-standardization
- `outcome_transform=Log()` — uses `log` not `log1p`; doesn't match codebase convention
- `ChainedOutcomeTransform(Log(), Standardize())` — same `log` vs `log1p` issue

---

## RES-002: Lengthscale Initialisation

**Decision**: Construct kernel as `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))`, then set `model.covar_module.base_kernel.lengthscale = 0.5` after construction. The scalar broadcasts to all 4 ARD dimensions.

**Rationale**: ℓ=0.5 (per spec FR-007) encourages a smoother GP prior with broader uncertainty in unexplored regions, supporting exploration. This is double the default BoTorch init (~0.25). The pattern of setting lengthscale after construction matches F3/F4 codebase convention — clear, explicit, and placed next to noise and outputscale inits.

**Alternatives Considered**:
- `MaternKernel(lengthscale=torch.full((1,4), 0.5))` — works but more verbose
- `model.initialize(**{'covar_module.base_kernel.lengthscale': 0.5})` — less readable

---

## RES-003: Noise Initialisation

**Decision**: Create `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`, then set `model.likelihood.noise = 0.1 * Y_train.var().item()`. Since Y_train is z-scored (variance ≈ 1.0), this effectively initialises noise at ~0.1.

**Rationale**: Per spec FR-008, noise = 0.1·Var(y_transformed). The higher init (vs. older 0.03 value) prevents the GP from interpolating too tightly through observations near the current best, creating a smoother posterior that supports exploration. Setting `model.likelihood.noise` directly initialises it as a learnable parameter that MLL optimises. The `GreaterThan(1e-6)` constraint provides a jitter floor.

**Alternatives Considered**:
- `train_Yvar=torch.full_like(Y, noise)` — makes noise fixed/heteroscedastic (wrong semantics)
- `noise_constraint=Interval(1e-6, 0.5)` — constraining upper bound is unnecessary; MLL should be free to optimise

---

## RES-004: NEI Exploration Mechanism (no ξ)

**Decision**: BoTorch's `qLogNoisyExpectedImprovement` has NO ξ parameter. The `eta` parameter (default 0.001) only controls constraint indicator smoothing — zero effect when `constraints=None`. Do NOT pass `eta` for exploration. The user's ξ=0.3 exploration intent is achieved through: q=4 batch diversity, distance-based candidate selection, ℓ=0.5 lengthscale init, and 0.1·Var(y) noise init.

**Rationale**: Confirmed via API inspection. The constructor accepts: `model, X_baseline, sampler, objective, posterior_transform, constraints, X_pending, prune_baseline, fat, tau_max, tau_relu, cache_root, eta`. The `eta` key only multiplies inside the smooth sigmoid for constraint satisfaction. Passing it as ξ would have no exploration effect and would mislead future readers.

**Alternatives Considered**:
- Misuse `eta` as ξ — wrong semantics; only affects constraints
- Manual offset to baseline (fake `best_f - ξ`) — fragile, non-standard
- Analytic `ExpectedImprovement` with `best_f - 0.3` — only supports q=1, loses noisy-EI benefits

---

## RES-005: Acquisition Optimisation Setup

**Decision**: Use `optimize_acqf` with `q=4, raw_samples=3000, num_restarts=50`. This directly implements "3000 Sobol starts → best 50 → L-BFGS".

**Rationale**: `optimize_acqf` operates in two stages: (1) evaluate 3000 Sobol points within bounds, select best 50 via Boltzmann sampling on acquisition values, (2) L-BFGS-B from those 50 starting points. With q=4, each optimisation produces 4 candidates jointly. The 3000/50 configuration is more aggressive than F3 (512/10) and F4 (512/20), matching the spec's emphasis on exploration for the stuck F5 function.

**Alternatives Considered**:
- `batch_initial_conditions` — bypasses Sobol; unnecessarily complex
- `sequential=True` — greedy one-at-a-time; joint optimisation (default `False`) is standard for batch NEI

---

## RES-006: Multiple MLL Restarts

**Decision**: Use manual restart loop with `N_RESTARTS=15`, following the F3/F4 pattern. Construct fresh model per restart with different `torch.manual_seed(seed)`, fit via `fit_gpytorch_mll`, score by neg-MLL, keep best via `copy.deepcopy`.

**Rationale**: `fit_gpytorch_mll` performs a single L-BFGS run with no built-in restart mechanism. Both F3 and F4 use the same manual restart pattern. 15 restarts is the sweet spot (middle of the 10–20 range from FR-009). Each restart: construct → init HPs (ℓ=0.5, noise=0.1·Var, outputscale=1.0) → fit → eval → score → deepcopy if best.

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
