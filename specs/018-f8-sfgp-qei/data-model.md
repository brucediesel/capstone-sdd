# Data Model: F8 Week 7 — SFGP + qEI Acquisition

**Feature**: 018-f8-sfgp-qei
**Date**: 2025-02-24

---

## Entities

### Training Data
- **X_raw**: `np.ndarray` shape (47, 8) — cumulative input observations from initial + Weeks 5-7
- **y_raw**: `np.ndarray` shape (47,) — cumulative output observations, all positive, range [5.59, 9.95]
- **X_train**: `torch.Tensor` shape (47, 8), dtype float64 — BoTorch-ready inputs
- **Y_train**: `torch.Tensor` shape (47, 1), dtype float64 — BoTorch-ready outputs (2D column)

### SFGP Surrogate
- **model**: `SingleTaskGP` instance with:
  - Kernel: `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))`
  - Likelihood: `GaussianLikelihood(noise_constraint=GreaterThan(1e-7))`
  - Outcome transform: `Standardize(m=1)` (default, implicit)
- **mll**: `ExactMarginalLogLikelihood` — training objective
- **lengthscales**: `np.ndarray` shape (8,) — ARD lengthscales after fitting (extracted via `.detach().numpy().ravel()`)
- **best_f**: `float` — Y_train.max() + 0.01 (xi offset)

### Acquisition Function
- **sampler**: `SobolQMCNormalSampler(sample_shape=torch.Size([256]))` — MC posterior sampler
- **acq_fn**: `qExpectedImprovement` instance with best_f and sampler
- **candidate**: `torch.Tensor` shape (1, 8) — raw optimisation output
- **acq_value**: `torch.Tensor` scalar — qEI value at candidate

### Selection
- **next_point**: `np.ndarray` shape (8,) — candidate clamped to [0, 1]
- **submission_query**: `str` — formatted "x1-x2-...-x8" with 6 decimal places

### Visualisation
- **importance**: `np.ndarray` shape (8,) — 1/lengthscale, normalised to sum to 1
- **top2**: tuple of 2 ints — indices of two most important dimensions (smallest lengthscale)
- **grid_mu**: `np.ndarray` shape (50, 50) — GP posterior mean on 2D grid
- **grid_sigma**: `np.ndarray` shape (50, 50) — GP posterior std on 2D grid
- **grid_ei**: `np.ndarray` shape (50, 50) — analytic EI values on 2D grid (faster than qEI for 2500 points)
- **running_best**: `np.ndarray` shape (47,) — cumulative maximum of observations

---

## Comparison: F8 Week 6 vs Week 7

| Aspect | Week 6 (NN) | Week 7 (SFGP) |
|--------|-------------|----------------|
| Surrogate | NN 8→64→32→1 (2,753 params) | SingleTaskGP Matern 2.5 + ARD |
| Uncertainty | MC Dropout (50 passes) | Exact GP posterior variance |
| Acquisition | UCB (κ=0.5) | qEI (256 MC, xi=0.01) |
| Feature importance | Gradient-based | ARD lengthscale inversion |
| Standardisation | Manual z-score | BoTorch Standardize(m=1) |
| Optimisation | Random 20k candidates + argmax | L-BFGS-B (30 restarts, 4096 raw) |
| Samples | 46 | 47 |
| Panel 3 | UCB surface | qEI surface |

---

## Data Flow

```
data/f8/updated_inputs - Week 7.npy  →  X_raw (47,8)  →  X_train (47,8) float64
data/f8/updated_outputs - Week 7.npy →  y_raw (47,)   →  Y_train (47,1) float64
                                                                |
                                                     SingleTaskGP.fit()
                                                                |
                                                     qExpectedImprovement
                                                                |
                                                     optimize_acqf (30 restarts)
                                                                |
                                                     candidate (1,8) → next_point (8,)
                                                                |
                                                     format_query → submission string
```
