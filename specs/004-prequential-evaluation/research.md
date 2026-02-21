# Research: F6 SFGP + NN Grid Update

**Feature**: 004-prequential-evaluation (F6 extension)
**Date**: 2025-07-15

## R1: SingleTaskGP Kernel Configuration

**Decision**: Use `ScaleKernel(base_kernel(ard_num_dims=5))` pattern with `covar_module` parameter.

**Rationale**: BoTorch 0.16.1's `SingleTaskGP.__init__` accepts `covar_module` and `likelihood` as optional kwargs. Passing a custom `ScaleKernel(MaternKernel(nu=X, ard_num_dims=5))` overrides the default kernel while preserving ARD (Automatic Relevance Determination) for all 5 input dimensions.

**Alternatives considered**:
- BoTorch's default kernel (no `covar_module`): Does not allow switching between Matérn/RBF — rejected because we need 4 kernel types
- `outcome_transform=Standardize(m=1)`: BoTorch's built-in standardisation — rejected to maintain consistency with MFGP's manual z-score pattern

**Verification**: Smoke-tested all 4 kernel types (Matérn 2.5, 1.5, 0.5, RBF) with `ard_num_dims=5` and `GaussianLikelihood(noise_constraint=GreaterThan(1e-8))`. All fitted and produced valid posteriors.

## R2: Output Transform Strategy (SFGP)

**Decision**: Use manual z-score standardisation, matching the existing MFGP pattern.

**Rationale**: The MFGP function already implements manual z-score:
```python
if transform_type == 'standardise':
    train_mean = y_all[:n_train].mean()
    train_std  = y_all[:n_train].std() + 1e-10
    y_work = (y_all - train_mean) / train_std
```
Reusing this pattern ensures consistency and makes 3-way comparison fair — all families handle transforms identically.

**Alternatives considered**:
- `outcome_transform=Standardize(m=1)`: BoTorch's built-in — would differ from MFGP's manual approach, making comparison unfair
- Log transform: Not applicable — F6 outputs are negative [−2.571, −0.219]

## R3: NN Grid Change Impact

**Decision**: Change from `layers_grid=[2,3], nodes_grid=[3,4,5,6]` (40 configs) to `layers_grid=[1,2,3], nodes_grid=[4,5,6]` (45 configs).

**Rationale**: The spec requests testing shallow architectures (1 hidden layer) and slightly wider nodes (4–6 vs 3–6). Removing 3 nodes isn't a concern — with only 26 samples and 5D input, 3 nodes was borderline too small. Adding 1-layer networks tests whether the function is simple enough for a shallow architecture.

**Alternatives considered**:
- Keep 3 nodes in the grid: Rejected per spec — 3 nodes with 1 layer gives only 5→3→1 = very limited capacity
- Wider range (up to 8+ nodes): Rejected — overfitting risk with 20 training points

## R4: Noise Floor for SFGP (1e-8)

**Decision**: Include `noise_lb=1e-8` in SFGP grid (MFGP only goes to 1e-7).

**Rationale**: SingleTaskGP is simpler than MFGP (no fidelity kernel), so Cholesky decomposition may remain stable at lower noise floors. The spec explicitly includes 5 noise levels for SFGP: {1e-4, 1e-5, 1e-6, 1e-7, 1e-8}.

**Verification**: Smoke test with `GreaterThan(1e-8)` on Matérn 0.5 (roughest kernel, most likely to cause numerical issues) succeeded without errors.

**Alternatives considered**:
- Limit to 1e-7 like MFGP: Rejected — SFGP is simpler and can likely handle lower noise floors
- Add jitter fallback: Already handled by `try/except` pattern in the eval loop

## R5: Import Requirements

**Decision**: Add 4 new imports to the existing import cell (cell 3).

**New imports needed**:
```python
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
```

**Rationale**: `SingleTaskGP` is needed for the SFGP model. `MaternKernel`, `RBFKernel`, and `ScaleKernel` are needed to build custom covariance modules with ARD. The existing cell already imports `SingleTaskMultiFidelityGP`, `GaussianLikelihood`, `GreaterThan`, `ExactMarginalLogLikelihood`, and `fit_gpytorch_mll`.

**Alternatives considered**:
- Separate import cell for SFGP: Rejected — constitution says simple code; one consolidated import cell is cleaner

## R6: SFGP Cell Placement

**Decision**: Insert 6 new cells (3 markdown + 3 code) after cell 21 (Best MFGP), before cell 22 (comparison).

**Rationale**: The notebook flow is: NN → MFGP → Comparison → Viz. Adding SFGP between MFGP and Comparison maintains the pattern of evaluating all families before comparing them. This matches the data flow diagram in the spec.

**Alternatives considered**:
- Insert SFGP before MFGP: Would break the existing notebook flow since MFGP is already implemented
- Insert SFGP after comparison: Illogical — SFGP must be evaluated before the 3-way comparison can use its results

## Dependencies Verified

| Dependency | Version | Status |
|------------|---------|--------|
| BoTorch | 0.16.1 | ✅ Installed, SingleTaskGP API verified |
| GPyTorch | 1.15.1 | ✅ Installed, kernel/likelihood API verified |
| PyTorch | (via BoTorch) | ✅ Working |
| Python | 3.14.2 | ✅ pyenv sdd-dev |
