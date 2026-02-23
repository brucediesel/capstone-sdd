# Research: F3 Week 7 – SFGP with Matérn-5/2 ARD and NEI Acquisition

**Feature**: 009-f3-sfgp-nei  
**Phase**: 0  
**Date**: 2026-02-23  

All NEEDS CLARIFICATION items from Technical Context are resolved below.

---

## RES-001: SFGP Construction with Matérn-5/2 ARD

**Decision**: Use `SingleTaskGP` with a manually-supplied `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))` and a `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`.

**Rationale**: `SingleTaskGP.__init__` in BoTorch 0.16.1 (installed) accepts `covar_module` and `likelihood` as keyword arguments that directly override the defaults. This is the cleanest, least-invasive way to customise the kernel without subclassing. `MaternKernel(nu=2.5, ard_num_dims=3)` gives one lengthscale per input dimension (ARD). Wrapped in `ScaleKernel` for the signal variance parameter.

**Alternatives considered**:
- Subclassing `SingleTaskGP` — rejected: unnecessary complexity; constitution requires simplicity.
- Using `BoTorchModel` (Ax integration) — rejected: Ax is overkill for a single notebook cell.

**Exact API**:
```python
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP

base_kernel = MaternKernel(nu=2.5, ard_num_dims=3)
covar_module = ScaleKernel(base_kernel)
likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
model = SingleTaskGP(X_train, Y_train, covar_module=covar_module, likelihood=likelihood)
```

**Verified**: BoTorch 0.16.1 / GPyTorch 1.15.1 in the `sdd-dev` pyenv environment.

---

## RES-002: z-Score Standardisation

**Decision**: Manually standardise outputs before passing to the model (subtract mean, divide by std). Predictions are then manually de-standardised back to the original scale for display and submission.

**Rationale**: The existing workspace pattern (f5 prequential eval, f2 week 7) uses manual z-score. While BoTorch provides `Standardize` outcome transform, it adds API surface area that is harder to explain step-by-step. Manual standardisation is simpler (constitution: "as simple as possible"), and keeping it explicit makes the notebook more educational.

**Exact steps**:
```python
y_mean = Y_raw.mean()
y_std = Y_raw.std()
Y_train = (Y_raw - y_mean) / y_std       # shape (n, 1), standardised
# After prediction:
mu_original = mu_standardised * y_std + y_mean
```

**Inputs**: Inputs in the Week 7 file are already in [0, 1] (cumulative file, validated on load). No additional scaling is required beyond clamping for submission.

---

## RES-003: MLL Multi-Restart Fitting

**Decision**: Implement a manual restart loop (10–20 iterations) that re-constructs the model with a new `torch.manual_seed(i)` each iteration, calls `fit_gpytorch_mll`, scores the result by evaluating `–MLL`, and keeps the deepcopy of the best model.

**Rationale**: No existing notebook implements multi-restart MLL; all use a single `fit_gpytorch_mll` call. The manual pattern is straightforward and easy to explain in a notebook cell. `fit_gpytorch_mll` internally runs L-BFGS-B to a local optimum; re-seeding the construction step changes the initialisation, exploring different basins.

**Constant**: `N_RESTARTS = 15` — middle of the 10–20 range specified; balances thoroughness with cell runtime.

**Exact pattern**: See RES-001 for construction. Scoring:
```python
model.train()
with torch.no_grad():
    out = model(X_train)
    loss = -mll(out, Y_train.squeeze(-1)).item()
if loss < best_loss:
    best_loss = loss
    best_model = copy.deepcopy(model)
best_model.eval()
```

---

## RES-004: NEI Acquisition Function

**Decision**: Use `qLogNoisyExpectedImprovement` from `botorch.acquisition.logei`. Single candidate (q=1); `X_baseline = X_train`; `prune_baseline=True`.

**Rationale**: `qLogNoisyExpectedImprovement` is the current recommended class in BoTorch 0.16.1 — it uses the log-sum-exp trick for numerical stability and handles noisy observations via the baseline approach (no need for a hand-specified `best_f`). Already in active use in `functions/f2/f2.ipynb` (week 7 section), confirming compatibility with the workspace environment. The older `qNoisyExpectedImprovement` is less numerically stable; analytic `ExpectedImprovement` is inappropriate for noisy problems.

**Alternatives considered**:
- `ExpectedImprovement` (analytic) — rejected: assumes noiseless observations; F3 has drug-discovery noise.
- `qNoisyExpectedImprovement` — rejected: superseded by log variant.

**Exact API**:
```python
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
nei = qLogNoisyExpectedImprovement(model=model, X_baseline=X_train, prune_baseline=True)
```

---

## RES-005: Surrogate Visualisation Strategy

**Decision**: Three pairwise 2D contour plots (x1 vs x2, x1 vs x3, x2 vs x3), with the remaining dimension held at the best-observed coordinate. Mirror the existing Week 5/6 visualisation structure in f3.ipynb (3-panel figure), except all three dimension pairs are shown rather than only the two most important.

**Rationale**: The spec requires pairwise 2D slices (FR-011). Three panels on a 1×3 figure (18×5 inches) matches the Week 5/6 layout and is readable in a notebook. Fixing the third dimension at the best-observed coordinate is the standard slice-plot convention and was already used in Week 5/6.

**Uncertainty bands**: Use GP posterior standard deviation (±2σ as contour overlaid on mean heatmap).

---

## RES-006: Import Consolidation

**Decision**: All imports required for the Week 7 section go in the first new code cell (after the markdown header), following the existing per-section import pattern established in f3.ipynb (each of Weeks 5, 6 re-imports `numpy`, `matplotlib`, and their surrogate library at the top of their respective sections).

**Libraries added for Week 7 not already in scope**: `copy` (stdlib), `gpytorch.kernels`, `gpytorch.constraints`, `gpytorch.likelihoods`, `botorch.acquisition.logei`.

**Versions confirmed**: BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch (workspace default), scikit-learn not needed.
