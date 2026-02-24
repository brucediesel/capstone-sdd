# Quickstart: F6 Week 7 — SFGP Matérn-1.5 + NEI

**Feature**: 012-f6-sfgp-nei
**Notebook**: `functions/f6/f6.ipynb`

---

## Prerequisites

1. Branch `012-f6-sfgp-nei` checked out
2. Python environment `sdd-dev` activated (`pyenv shell sdd-dev`)
3. Data files present:
   - `data/f6/updated_inputs - Week 7.npy` — shape (27, 5)
   - `data/f6/updated_outputs - Week 7.npy` — shape (27,)
4. Existing notebook has 58 cells (49 original + 8 Week 7 cells from first attempt + 1 user-added markdown). Week 7 cells 50–57 will be replaced.

## Implementation Order

### Phase 1: Setup (Cells 50–51)

1. **Cell 50** (Markdown): Week 7 header with strategy rationale
   - Title: `## Week 7 — SFGP Matérn-1.5 + NEI`
   - Include comparison table: Week 6 (NN+MCDropout+UCB κ=0.5) vs Week 7 (SFGP+NEI q=4)

2. **Cell 51** (Code): Imports + data loading
   ```python
   import numpy as np, torch, copy, matplotlib.pyplot as plt
   from botorch.models import SingleTaskGP
   from botorch.fit import fit_gpytorch_mll
   from botorch.acquisition.logei import qLogNoisyExpectedImprovement
   from botorch.optim import optimize_acqf
   from botorch.sampling.normal import SobolQMCNormalSampler
   from gpytorch.mlls import ExactMarginalLogLikelihood
   from gpytorch.kernels import ScaleKernel, MaternKernel
   from gpytorch.likelihoods import GaussianLikelihood
   from gpytorch.constraints import GreaterThan

   X_raw = np.load('../../data/f6/updated_inputs - Week 7.npy')
   y_raw = np.load('../../data/f6/updated_outputs - Week 7.npy')
   # Print stats...
   ```

### Phase 2: Documentation (Cell 52)

3. **Cell 52** (Markdown): Hyperparameter table with 14 entries (incl. Standardize(m=1), noise=0.2, noise floor 1e-2, feasibility-constrained bounds, distance-based selection)

### Phase 3: Model Training (Cell 53)

4. **Cell 53** (Code): GP training pipeline
   ```python
   # No manual transform — Standardize(m=1) handles z-scoring internally
   X_train = torch.tensor(X_raw, dtype=torch.double)
   Y_train = torch.tensor(y_raw, dtype=torch.double).unsqueeze(-1)

   # 15-restart MLL fitting
   N_RESTARTS = 15
   best_loss, best_model = float('inf'), None
   for seed in range(N_RESTARTS):
       torch.manual_seed(seed)
       likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-2))
       covar = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5))
       model = SingleTaskGP(X_train, Y_train, covar_module=covar, likelihood=likelihood)
       # Standardize(m=1) applied automatically — no outcome_transform arg
       model.covar_module.base_kernel.lengthscale = 0.5
       model.likelihood.noise = 0.2  # 20% of standardised Var(y)≈1.0; aggressive exploration init
       model.covar_module.outputscale = 1.0
       mll = ExactMarginalLogLikelihood(model.likelihood, model)
       fit_gpytorch_mll(mll)
       model.eval(); likelihood.eval()
       with torch.no_grad():
           output = model(X_train)
           loss = -mll(output, model.train_targets).item()
       if loss < best_loss:
           best_loss, best_model = loss, copy.deepcopy(model)
   ```

   **Critical notes**:
   - `model.train_targets` returns standardised targets (mean≈0, std≈1)
   - `model(X_train)` operates in standardised space (for MLL scoring)
   - `model.posterior(X)` returns values in **original** space (for viz/acquisition)

### Phase 4: Acquisition & Selection (Cell 54)

5. **Cell 54** (Code): NEI with q=4 + distance-based selection
   ```python
   best_model.eval()
   sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
   nei = qLogNoisyExpectedImprovement(
       model=best_model, X_baseline=X_train, sampler=sampler, prune_baseline=True
   )
   BOUNDS = torch.tensor([[0.01, 0.01, 0.01, 0.01, 0.10], [1.0]*5], dtype=torch.double)
   candidates, acq_value = optimize_acqf(
       acq_function=nei, bounds=BOUNDS, q=4,
       num_restarts=50, raw_samples=3000
   )
   candidates = torch.clamp(candidates, 0.0, 0.999999)

   # Distance-based selection (posterior means in original space — automatic)
   with torch.no_grad():
       post = best_model.posterior(candidates)
       means = post.mean.squeeze(-1)  # (4,) — in original space, all negative
   median_mean = means.median()
   above_median = means >= median_mean
   dists = torch.cdist(candidates, X_train).min(dim=1).values  # (4,)
   mask = above_median
   if mask.sum() == 0:
       best_idx = means.argmax()  # fallback: best predicted mean
   else:
       best_idx = dists[mask].argmax()
       best_idx = torch.where(mask)[0][best_idx]
   best_point = candidates[best_idx]
   ```

### Phase 5: Visualisation (Cells 55–56)

6. **Cell 55** (Code): 3-panel surrogate plot (mean, std, 1/ℓ relevance)
   - No manual inverse transform — `model.posterior()` returns original-space values
   - 5 bars in relevance chart (not 4 as in F5)

7. **Cell 56** (Code): Convergence plot with boundary at 26.5

### Phase 6: Submission (Cell 57)

8. **Cell 57** (Code): Format and validate submission query + HP summary

## Key Patterns

- **No manual transform**: `y_raw → Y_train → SingleTaskGP(Standardize(m=1) default) → posterior in original space`
- **Default outcome transform**: Do NOT pass `outcome_transform` arg — BoTorch's default `Standardize(m=1)` handles everything
- **Noise init = 0.2**: Because Standardize makes internal Var(y)≈1.0; `0.1 * y_raw.var()` = 0.033 would be wrong; 0.2 discourages exact interpolation
- **MLL scoring**: Use `model.train_targets` (standardised) not `Y_train.squeeze(-1)` (raw)
- **Posterior for viz/selection**: Use `model.posterior(X)` which auto-untransforms to original space
- **Restart loop**: F3/F4/F5 pattern — fresh model per seed, deepcopy best
- **Dim selection for viz**: `top2 = np.argsort(lengthscales)[:2]` (shortest = most important)
- **Distance-based selection**: Filter by mean ≥ median, then select farthest from training data
- **Clamping**: `torch.clamp(candidates, 0.0, 0.999999)` before formatting

## Key Differences from F5 (011-f5-gp-nei)

| Aspect | F5 | F6 |
|--------|----|----|
| Kernel | Matérn-2.5 | **Matérn-1.5** |
| Dimensions | 4D | **5D** |
| Output range | [0.1, 3395] (30,000x) | [-2.571, -0.206] (12.5x) |
| Manual transform | log1p + z-score | **None** |
| outcome_transform | `None` (disabled) | **Default Standardize(m=1)** |
| Noise constraint | GreaterThan(1e-6) | **GreaterThan(1e-2)** |
| Noise init | 0.1 * Y_train.var() ≈ 0.1 | **0.2** (aggressive exploration init) |
| Posterior for viz | Manual expm1 untransform | **Automatic** (Standardize auto-untransforms) |
| MLL target | `Y_train.squeeze(-1)` | **`model.train_targets`** |
| Relevance bars | 4 (x0–x3) | **5 (x0–x4)** |
| Submission format | x1-x2-x3-x4 | **x1-x2-x3-x4-x5** |

## Verification

After all 8 cells execute:
- 58 total cells (49 original + 8 corrected Week 7 + 1 user-added markdown)
- Submission query: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` with x4 ≥ 0.10
- All plots render without errors
- No existing cells modified
