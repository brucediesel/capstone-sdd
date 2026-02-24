# Research: F6 Week 7 — SFGP Matérn-1.5 + NEI

**Feature**: 012-f6-sfgp-nei
**Date**: 2026-02-24
**Purpose**: Resolve all technical unknowns from the Technical Context before implementation.

---

## RES-001: Matérn ν=1.5 vs ν=2.5

**Decision**: Use Matérn ν=1.5 as explicitly specified by the user. Construct kernel as `ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5))`.

**Rationale**: The Matérn-ν kernel is ⌈ν⌉−1 times differentiable. ν=1.5 (C¹, once-differentiable) produces a rougher posterior than ν=2.5 (C², twice-differentiable). With only 27 samples in 5D (~5.4 samples per dimension), there is insufficient data to distinguish between the two — the choice is primarily an inductive bias about function roughness. Key advantages of ν=1.5 for this problem:

- **Wider posterior uncertainty** in gaps between observations (variance decays as ~e^{-√3·r/ℓ} vs ~e^{-√5·r/ℓ}), amplifying exploration signals for NEI.
- **Less overconfident interpolation** in sparse 5D space — conservative assumption when function smoothness is unknown.
- **Synergy with distance-based selection** (FR-013): candidates in under-explored regions get higher acquisition values.
- **Zero-cost API change**: `nu=1.5` vs `nu=2.5` — all other API (SingleTaskGP, fit_gpytorch_mll, qLogNEI, optimize_acqf) is identical.

The 15-restart MLL fitting (FR-005) adequately handles the slightly rougher MLL landscape that ν=1.5 may introduce.

**Alternatives Considered**:
- ν=2.5 (agent initial recommendation) — default choice, already used in F5/F7/F8; user chose differently
- ν=0.5 (exponential) — too rough (C⁰), jagged posteriors, unstable acquisition optimisation
- RBF (ν→∞) — too smooth for sparse data, would over-smooth and under-explore

---

## RES-002: Standardize(m=1) Default Outcome Transform

**Decision**: Use BoTorch's default `Standardize(m=1)` outcome transform. Do NOT pass `outcome_transform=None` or apply any manual transform. This is the key difference from F5.

**Rationale**: `SingleTaskGP` defaults to `Standardize(m=1)` when `outcome_transform` is not specified. This z-scores the training targets internally: `z = (y - mean) / std`. For F6's data (mean ≈ -1.36, std ≈ 0.68), the standardized range becomes approximately [-1.45, +1.70].

Key behaviours verified:
1. **All-negative outputs handled correctly**: z-scoring is sign-agnostic.
2. **Posterior in original space**: `model.posterior(X_test).mean` and `.variance` are automatically untransformed back to original output space. No manual inverse transform needed for visualisation.
3. **Acquisition function compatibility**: `qLogNoisyExpectedImprovement` calls `model.posterior()` which auto-untransforms — improvement is computed in original space.
4. **No double-standardization risk**: Unlike F5 (which used manual log1p + z-score and therefore required `outcome_transform=None`), F6 has no manual transform, so the default is correct.

**Contrast with F5**: F5 required manual `log1p` + z-score because its range was 30,000x (0.1 to 3395). F6's range is only 12.5x — the default z-score alone is sufficient.

**Alternatives Considered**:
- `outcome_transform=None` + manual z-score — unnecessary complexity for 12.5x range
- `outcome_transform=Log()` — outputs are all-negative, `log()` would fail
- Leaving default and also manually transforming — double-standardization, would corrupt posteriors

---

## RES-003: Noise Constraint and Initialisation

**Decision**: Use `GaussianLikelihood(noise_constraint=GreaterThan(1e-8))` and initialise noise to **0.1** (not `0.1 * y_raw.var()`).

**Rationale**: The `Standardize(m=1)` transform guarantees that the GP's internal training targets have variance ≈ 1.0. Therefore, `0.1 · Var(y_standardized)` = `0.1 · 1.0` = `0.1`. Using `0.1 * Var(y_raw)` = `0.1 * 0.33` = `0.033` would only give 3.3% noise-to-signal ratio instead of the intended 10%.

Noise constraint details:
- `GreaterThan(1e-8)` is a hard floor on the learnable σ²_n parameter.
- Numerically safe for 27 samples in 5D — MLL optimizer naturally settles around 1e-6 to 1e-5 even with 1e-8 floor.
- GPyTorch adds independent Cholesky jitter (1e-8 for float64) as a second safety net.
- The constraint and jitter are independent mechanisms — do not interact.

**Spec clarification**: FR-008 states "0.1 · Var(y_train) (where y_train are the raw outputs)". Since `Standardize(m=1)` operates internally, the noise init should reflect the standardized variance (≈1.0), not raw variance (0.33). The implementation uses `model.likelihood.noise = 0.1` with a code comment explaining why.

**Alternatives Considered**:
- `GreaterThan(1e-6)` (F5 pattern) — user explicitly specified 1e-8
- `0.1 * y_raw.var()` = 0.033 — only 3.3% noise ratio, would over-interpolate
- `train_Yvar=torch.full_like(Y, noise)` — makes noise fixed/heteroscedastic (wrong semantics)

---

## RES-004: NEI Exploration Mechanism

**Decision**: BoTorch's `qLogNoisyExpectedImprovement` has NO ξ parameter. Exploration is achieved through: q=4 batch diversity, distance-based candidate selection, ℓ=0.5 lengthscale init, Matérn-1.5's wider posterior uncertainty, and 0.1 noise init.

**Rationale**: Confirmed via API inspection. The constructor accepts: `model, X_baseline, sampler, objective, posterior_transform, constraints, X_pending, prune_baseline, fat, tau_max, tau_relu, cache_root, eta`. The `eta` parameter only controls smooth sigmoid for constraint satisfaction — zero effect when `constraints=None`. The exploration intent is encoded in the surrounding hyperparameter choices and post-hoc selection strategy.

**Alternatives Considered**:
- Misuse `eta` as ξ — wrong semantics, only affects constraints
- Analytic `ExpectedImprovement` with manual `best_f - ξ` — only supports q=1, loses noisy-EI benefits

---

## RES-005: Acquisition Optimisation Setup

**Decision**: Use `optimize_acqf` with `q=4, raw_samples=3000, num_restarts=50`. Bounds = `[0,1]⁵`. This directly implements "3000 Sobol starts → best 50 → L-BFGS".

**Rationale**: `optimize_acqf` operates in two stages: (1) evaluate 3000 Sobol points within bounds, select best 50 via Boltzmann sampling on acquisition values, (2) L-BFGS-B from those 50 starting points. With q=4, each optimisation produces 4 candidates jointly. The 3000/50 configuration matches F5's approach and the spec's emphasis on exploration.

**Alternatives Considered**:
- `batch_initial_conditions` — bypasses Sobol; unnecessarily complex
- `sequential=True` — greedy one-at-a-time; joint optimisation (default `False`) is standard for batch NEI

---

## RES-006: Multiple MLL Restarts

**Decision**: Use manual restart loop with `N_RESTARTS=15`, following the F3/F4/F5 pattern. Construct fresh model per restart with different `torch.manual_seed(seed)`, fit via `fit_gpytorch_mll`, score by neg-MLL, keep best via `copy.deepcopy`.

**Rationale**: `fit_gpytorch_mll` performs a single L-BFGS run with no built-in restart mechanism. 15 restarts is the middle of the 10–20 range. The Matérn-1.5 kernel may have a slightly rougher MLL landscape than Matérn-2.5, making multiple restarts slightly more valuable. Each restart: construct → init HPs → fit → eval → score → deepcopy if best.

**Important**: When initialising hyperparameters, the noise init is set to `0.1` (not `0.1 * Y_train.var().item()`) because `Standardize(m=1)` ensures internal Y_train has variance ≈ 1.0. See RES-003.

**Alternatives Considered**:
- Single `fit_gpytorch_mll` — risks local MLL optimum
- `fit_gpytorch_scipy` — lower-level; `fit_gpytorch_mll` works fine

---

## RES-007: Week 6 Visualisation Adaptation

**Decision**: Replicate the Week 6 3-panel layout: (1) GP posterior mean contour, (2) GP posterior std contour, (3) dimension relevance bar chart using `1/ℓ` (inverse lengthscale, normalised). Plus a separate convergence plot with Week 6→7 boundary.

**Rationale**: Week 6 used NN + MC Dropout with 3 panels: NN mean, MC uncertainty, gradient importance. For SFGP:
- Panel 1: GP posterior mean replaces NN mean — semantically equivalent
- Panel 2: GP posterior std replaces MC Dropout uncertainty — semantically equivalent
- Panel 3: 1/ℓ (inverse ARD lengthscale, normalised) replaces gradient magnitude — shorter ℓ = more important, same interpretive purpose

The 2D slice approach selects the top-2 important dimensions (shortest lengthscales) and fixes the other 3 at the proposed best point's values. An 80×80 evaluation grid provides sufficient resolution.

**Key difference from F5**: No manual inverse transform needed for visualisation — `model.posterior(X_test).mean` returns values in original space automatically (see RES-002).

**Alternatives Considered**:
- Pairwise slices — yields 10 pairs for 5D, too many panels
- Raw lengthscale bars — shorter bars = more important is counter-intuitive
- F5's `expm1` inverse transform — not needed since Standardize auto-untransforms
