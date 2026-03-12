# Research: F4–F8 Week 10 Optimisation Strategy Changes

**Date**: 2026-03-12 | **Branch**: `031-f4-f8-week10-optimisation`

## R1: MFGP → SFGP Migration (F4)

**Decision**: Use `SingleTaskGP` with custom `covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))` and `outcome_transform=Standardize(m=1)`.

**Rationale**: The `SingleTaskMultiFidelityGP` with `LinearTruncatedFidelityKernel` adds unnecessary parameters (power parameter, fidelity kernel) on single-fidelity data. F4 only has single-fidelity observations — the synthetic fidelity column (all 1.0) provides no information. A standard SFGP with Matérn-2.5 ARD matches the data structure properly.

**Alternatives considered**:
- Keep MFGP: Rejected — 7 consecutive stalling submissions indicate the model is fundamentally mismatched.
- Use default RBF kernel: Rejected — Matérn-2.5 is more appropriate for black-box functions that may have finite smoothness.

**Implementation detail**: Drop the synthetic fidelity column from `X_train` (use only 4D inputs). Pass custom `likelihood` and `covar_module` to `SingleTaskGP`.

## R2: Noise Constraint Implementation (F4, F6, F8)

**Decision**: Use `GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))` for all GP-based functions.

**Rationale**: Standard gpytorch pattern. No simpler alternative exists — `SingleTaskGP` doesn't have a `noise_lb` parameter.

**Note**: Passing a custom likelihood disables BoTorch's default LogNormal prior on noise. This is acceptable when explicitly constraining noise.

| Function | noise_lb | Rationale |
|----------|----------|-----------|
| F4 | 1e-3 | 40 samples in 4D; prevents overfitting |
| F6 | 1e-3 | Reduce from 1e-2 to allow finer posterior resolution |
| F8 | 1e-7 | Keep current; verify Cholesky stability |

## R3: MLL Restart Loop Pattern

**Decision**: Manual restart loop — construct fresh model per restart, randomise hyperparameters, fit, track best loss and `state_dict`.

**Rationale**: BoTorch's `fit_gpytorch_mll` retries on *failure*, not for multi-start optimisation of the MLL surface. The manual loop explores the multi-modal likelihood landscape. This pattern is well-validated in the codebase (F3 week 10, all PE notebooks).

**Hyperparameter randomisation ranges** (for unit-cube inputs):
- Lengthscale: `torch.rand(1, d) * 0.5 + 0.1` → [0.1, 0.6]
- Outputscale: `torch.rand(1) * 2.0` → [0.0, 2.0]
- Noise: `max(noise_lb * 10, torch.rand(1) * 0.01)` → ensures above floor

| Function | MLL Restarts | Dims | Total Obs |
|----------|-------------|------|-----------|
| F4 | ≥30 | 4 | 40 |
| F5 | 15 (MLL) + 60 (acq) | 4 | 30 |
| F6 | 15 | 5 | 30 |
| F8 | ≥30 | 8 | 50 |

## R4: qLogNEI Configuration

**Decision**: Use `qLogNoisyExpectedImprovement` from `botorch.acquisition.logei` for F4, F5, F6, F8. The `q` parameter is passed to `optimize_acqf`, NOT to the acquisition function constructor.

**Rationale**: Log-transformed EI is more numerically stable than vanilla qEI, especially for large output ranges and high dimensions. BoTorch explicitly warns against using `qExpectedImprovement` and recommends `qLogExpectedImprovement` or `qLogNoisyExpectedImprovement`.

**MC samples**: 512 via `SobolQMCNormalSampler(sample_shape=torch.Size([512]))` — provides low-variance quasi-MC estimates.

| Function | q | raw_samples | num_restarts | MC samples |
|----------|---|-------------|-------------|------------|
| F4 | 4 | 2048 | 20 | 512 |
| F5 | 4 | 8000 | 60 | 512 |
| F6 | 4 | 5000 | 50 | 512 |
| F8 | 1 | 8192 | 30 | 512 |

## R5: Standardize(m=1) with Negative Outputs (F4)

**Decision**: `Standardize(m=1)` handles negative outputs correctly. No shift transform needed for F4.

**Rationale**: `Standardize` computes pure z-score standardisation: `(y - mean) / std`. This works correctly regardless of output sign. For F4's range [-4.03, 0.53], the transformed outputs centre at 0 with unit variance — exactly what the GP hyperparameter priors expect.

**Contrast with F3**: F3 used a shift transform because its outputs are deeply negative with a specific structure. F4's output range is narrower and better suited to z-score.

## R6: log vs log1p Transform (F5)

**Decision**: Switch from `log1p` to `log` transform. Keep `Standardize(m=1)` as double transform.

**Rationale**: For F5's outputs (>1000 in majority), `log1p ≈ log` numerically. The `log` inverse (`exp`) is simpler than `expm1`. However, the raw data shows some values near ~1.11 where `log(1.11) = 0.104` vs `log1p(1.11) = 0.747` — a meaningful difference. A guard should verify all outputs are strictly positive before applying `log`.

**Implementation**: Change `np.log1p(y_raw)` → `np.log(y_raw)` and `np.expm1(...)` → `np.exp(...)`. Keep `Standardize(m=1)` — it normalises the log-space outputs to zero mean/unit variance, which helps GP hyperparameter fitting.

**Alternatives considered**:
- Single transform (log only, no Standardize): Rejected — Standardize is free and improves conditioning.
- Keep log1p: Rejected — the spec explicitly requests log, and for values >1000 the difference is negligible.

## R7: Distance-Based Selection Relaxation (F5)

**Decision**: Relax the quality gate from `>= median` to `>= 25th percentile` or accept all candidates.

**Rationale**: The current code filters q=4 candidates to those with posterior mean ≥ median, then picks the farthest from training data. With only 4 candidates and a median gate, exactly 2 pass the filter. Lowering the gate to 25th percentile (or removing it) allows more candidates through, giving the distance criterion more options. The spec notes the distance criterion may be "too aggressive, causing some candidates to be placed in suboptimal locations just for diversity." Relaxing the quality gate is the simplest control.

## R8: F6 Milk Constraint

**Decision**: Milk dimension is x4 (0-indexed, the 5th ingredient). Threshold changes from 0.10 → 0.12.

**Rationale**: The current bounds tensor has `[0.10, 1.00]` for dimension 4. Increasing to 0.12 focuses the search on higher-quality feasible regions where milk content is above a minimum. The ingredients are: flour (0), sugar (1), eggs (2), butter (3), milk (4).

**Fallback**: If no candidates satisfy milk ≥ 0.12, fall back to 0.10 and log a warning.

## R9: MC Dropout for F7

**Decision**: Keep existing NN architecture (6→5→5→1 with `DROPOUT=0.05`). Increase `MC_SAMPLES` from 30 to ≥50.

**Rationale**: The MC dropout mechanism is already correctly implemented — `model.train()` enables dropout during forward passes, `torch.no_grad()` prevents gradient computation. The week 9 reduction from 50→30 was premature. ≥50 forward passes provides more stable uncertainty estimates.

**Other changes**:
- `EXPLOITATION_WEIGHT`: 0.7 → 0.5 (50/50 mean/EI blend)
- `STEEPNESS`: 0.05 → 0.02 (near-no-op penalty — exponent 0.04 makes sin^0.04 ≈ 1 everywhere except right at boundaries)
- `N_CANDIDATES`: 20,000 → 50,000

## R10: qEI → qLogNEI Migration (F8)

**Decision**: Replace `qExpectedImprovement` with `qLogNoisyExpectedImprovement`. The `XI` parameter cannot be directly replicated — qLogNEI infers the baseline from `X_baseline`.

**Rationale**: BoTorch explicitly warns: "qExpectedImprovement has known numerical issues… strongly recommended to replace [with] qLogExpectedImprovement." For 8D with 50 observations, numerical stability is critical.

**Implementation**: 
```python
# Old: qExpectedImprovement(model=model, best_f=y_max + XI, sampler=sampler)
# New: qLogNoisyExpectedImprovement(model=model, X_baseline=X_train, sampler=sampler, prune_baseline=True)
```

**XI equivalent**: qLogNEI doesn't have a direct XI parameter. The exploration behaviour is inherent in the noisy EI formulation (which accounts for observation noise). The `XI=0.05` exploratory intent is partially achieved by the switch from qEI to qLogNEI itself (better numerical behaviour in exploration regions) and by increasing MC samples (512) and raw_samples (8192).

## R11: F8 noise_lb=1e-7 Stability

**Decision**: Keep noise_lb=1e-7 but add a post-fit diagnostic to verify Cholesky stability.

**Rationale**: The fitted noise consistently hits the floor (0.00000010), suggesting true observation noise is very low. However, in 8D with 50 observations, the kernel matrix K + 1e-7*I may be near-singular. BoTorch's internal `_psd_safe_cholesky` adds jitter (1e-6 to 1e-4) as a safety net, which may mask issues.

**Diagnostic**: After fitting, check `model.likelihood.noise.item()` and attempt Cholesky on `K + noise*I`. If condition number exceeds 1e10, log a warning and consider increasing to 1e-6.
