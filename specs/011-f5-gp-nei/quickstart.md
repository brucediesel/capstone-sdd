# Quickstart: F5 Week 7 — GP Matérn-5/2 + NEI

**Feature**: 011-f5-gp-nei
**Notebook**: `functions/f5/f5.ipynb`

---

## Prerequisites

1. Branch `011-f5-gp-nei` checked out
2. Python environment `sdd-dev` activated (`pyenv shell sdd-dev`)
3. Data files present:
   - `data/f5/updated_inputs - Week 7.npy` — shape (27, 4)
   - `data/f5/updated_outputs - Week 7.npy` — shape (27,)
4. Existing notebook has 50 cells (last cell: `#VSC-8f8ac8b4`)

## Implementation Order

### Phase 1: Setup (Cells 51–52)

1. **Cell 51** (Markdown): Week 7 header with strategy rationale
   - Title: `## Week 7 — GP Matérn-5/2 + NEI`
   - Include comparison table: Week 6 (GBT, UCB, κ=0.5) vs Week 7 (GP, NEI, q=2)

2. **Cell 52** (Code): Imports + data loading
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

   X_raw = np.load('../../data/f5/updated_inputs - Week 7.npy')
   y_raw = np.load('../../data/f5/updated_outputs - Week 7.npy')
   # Print stats...
   ```

### Phase 2: Documentation (Cell 53)

3. **Cell 53** (Markdown): Hyperparameter table with 12+ entries

### Phase 3: Model Training (Cell 54)

4. **Cell 54** (Code): GP training pipeline
   ```python
   # Transform: log1p → z-score
   y_log = np.log1p(y_raw)
   y_mean, y_std_val = y_log.mean(), y_log.std()
   y_std = (y_log - y_mean) / y_std_val

   # Tensors
   X_train = torch.tensor(X_raw, dtype=torch.double)
   Y_train = torch.tensor(y_std, dtype=torch.double).unsqueeze(-1)

   # 15-restart MLL fitting
   N_RESTARTS = 15
   best_loss, best_model = float('inf'), None
   for seed in range(N_RESTARTS):
       torch.manual_seed(seed)
       likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
       covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))
       model = SingleTaskGP(X_train, Y_train, covar_module=covar, likelihood=likelihood)
       model.covar_module.base_kernel.lengthscale = 0.25
       model.likelihood.noise = 0.03
       model.covar_module.outputscale = 1.0
       mll = ExactMarginalLogLikelihood(model.likelihood, model)
       fit_gpytorch_mll(mll)
       model.eval(); likelihood.eval()
       with torch.no_grad():
           output = model(X_train)
           loss = -mll(output, Y_train.squeeze(-1)).item()
       if loss < best_loss:
           best_loss, best_model = loss, copy.deepcopy(model)
   ```

### Phase 4: Acquisition (Cell 55)

5. **Cell 55** (Code): NEI with q=2
   ```python
   best_model.eval()
   sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
   nei = qLogNoisyExpectedImprovement(
       model=best_model, X_baseline=X_train, sampler=sampler, prune_baseline=True
   )
   BOUNDS = torch.tensor([[0.0]*4, [1.0]*4], dtype=torch.double)
   candidates, acq_value = optimize_acqf(
       acq_function=nei, bounds=BOUNDS, q=2,
       num_restarts=50, raw_samples=3000
   )
   ```

### Phase 5: Visualisation (Cells 56–57)

6. **Cell 56** (Code): 3-panel surrogate plot (mean, std, 1/ℓ relevance)
7. **Cell 57** (Code): Convergence plot with boundary at 26.5

### Phase 6: Submission (Cell 58)

8. **Cell 58** (Code): Format and validate submission query

## Key Patterns

- **Transform pipeline**: `y_raw → log1p → z-score → GP → z-score⁻¹ → expm1 → y_pred`
- **Restart loop**: F3/F4 pattern — fresh model per seed, deepcopy best
- **Dim selection for viz**: `top2 = np.argsort(lengthscales)[:2]` (shortest = most important)
- **Inverse transform for viz**: `y_orig = np.expm1(pred_std * y_std_val + y_mean)`

## Verification

After all 8 cells execute:
- 58 total cells (50 original + 8 new)
- Submission query: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`
- All plots render without errors
- No existing cells modified
