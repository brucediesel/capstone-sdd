# Quickstart: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Date**: 2026-03-09  
**Feature**: 024-f3-week9-standardize

## Prerequisites

1. Week 9 data files must exist:
   - `./data/f3/updated_inputs - Week 9.npy`
   - `./data/f3/updated_outputs - Week 9.npy`
   - If missing, run the results processing notebook at `./functions/results/` first.

2. Python environment with:
   - numpy
   - torch
   - botorch (SingleTaskGP, Standardize, qLogNoisyExpectedImprovement, optimize_acqf)
   - gpytorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood)
   - matplotlib
   - scipy

## Implementation Steps

### Step 1: Copy existing `f3 - week 9.ipynb` as base

Start from the existing notebook and apply two targeted changes.

### Step 2: Remove manual z-score standardisation

In the GP training cell, **remove**:
```python
# REMOVE these lines:
y_mean = y_raw.mean()
y_std  = y_raw.std()
y_std_safe = max(y_std, 1e-8)
y_standardised = (y_raw - y_mean) / y_std_safe
```

**Replace** `Y_train` construction with raw values:
```python
# BEFORE:
Y_train = torch.tensor(y_standardised, dtype=torch.float64).unsqueeze(-1)

# AFTER:
Y_train = torch.tensor(y_raw, dtype=torch.float64).unsqueeze(-1)
```

### Step 3: Add Standardize import and pass to SingleTaskGP

Add import:
```python
from botorch.models.transforms.outcome import Standardize
```

Pass to model construction:
```python
# BEFORE:
m = SingleTaskGP(train_X=X_train, train_Y=Y_train, covar_module=covar, likelihood=lik)

# AFTER:
m = SingleTaskGP(train_X=X_train, train_Y=Y_train, covar_module=covar,
                 likelihood=lik, outcome_transform=Standardize(m=1))
```

### Step 4: Remove manual un-standardisation in visualisation

In contour plot cell, **remove**:
```python
# REMOVE these lines:
mean_raw = mean_std * y_std_safe + y_mean
std_raw  = std_std * y_std_safe
```

Use posterior values directly (they are already in original scale):
```python
mean_vals = posterior.mean.squeeze().numpy()
std_vals  = posterior.variance.squeeze().sqrt().numpy()
```

### Step 5: Change acquisition restarts from 10 to 20

In hyperparameters cell:
```python
# BEFORE:
ACQ_RESTARTS = 10

# AFTER:
NUM_RESTARTS_ACQ = 20
```

Update `optimize_acqf` call:
```python
candidate, acq_value = optimize_acqf(
    acq_function=nei,
    bounds=bounds_t,
    q=1,
    num_restarts=NUM_RESTARTS_ACQ,  # 20
    raw_samples=ACQ_RAW,            # 512
)
```

### Step 6: Update LOO cross-validation

Remove per-fold z-score recomputation. Each fold now simply constructs `SingleTaskGP(..., outcome_transform=Standardize(m=1))` with the reduced training set.

Remove manual un-standardisation of LOO predictions:
```python
# REMOVE:
pred_raw = pred_std * y_loo_std + y_loo_mean

# REPLACE WITH:
pred_raw = best_loo_model.posterior(x_held).mean.item()
```

### Step 7: Verify and run

Execute all cells. Confirm:
- [ ] Data loads successfully (24 samples, 3D)
- [ ] GP trains without manual z-score code
- [ ] Posterior returns original-scale values (no manual un-standardisation)
- [ ] Acquisition uses 20 restarts (NUM_RESTARTS_ACQ = 20)
- [ ] Proposed candidate is within bounds
- [ ] Submission query formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx`
- [ ] All values clipped to [0.0, 0.999999]
- [ ] LOO completes without manual z-score per fold
- [ ] Contour plots rendered correctly with three-colour scheme

## Key Differences from Previous Strategy

| Aspect | Before | After |
|--------|--------|-------|
| Output normalisation | Manual z-score | BoTorch Standardize(m=1) — automatic |
| Un-standardisation | Manual in viz + LOO cells | Automatic (posterior returns original scale) |
| Acquisition restarts | 10 | 20 |
