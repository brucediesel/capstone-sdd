# Data Model: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Feature**: 024-f3-week9-standardize  
**Date**: 2026-03-09

## Entities

### Observation
A single black-box evaluation of Function 3.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| x | float64[3] | Each $x_d \in [0.0, 1.0]$ | Compounds A, B, C |
| y | float64 | No NaN/Inf | Objective value (maximise) |
| source | enum | `initial` or `week_N` | Origin of the observation |

**Cardinality**: 24 total (15 initial + 9 submissions)  
**Storage**: `updated_inputs - Week 9.npy` (24×3) and `updated_outputs - Week 9.npy` (24,)

### GP Model (SingleTaskGP with Standardize)

| Component | Value | Notes |
|-----------|-------|-------|
| Surrogate | SingleTaskGP | BoTorch default |
| Outcome Transform | Standardize(m=1) | Auto z-score + auto untransform |
| Kernel | ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3)) | ARD per compound |
| Likelihood | GaussianLikelihood(noise_constraint=GreaterThan(1e-6)) | Noise floor |
| Lengthscale init | 0.25 per dim | Prior for 3D [0,1] space |
| Outputscale init | 1.0 | Signal variance |
| Noise init | 0.1 | Observation noise |
| MLL restarts | 15 | Multi-start hyperparameter optimisation |

### Acquisition Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Function | qLogNoisyExpectedImprovement | Log-space NEI |
| NUM_RESTARTS_ACQ | 20 | Increased from 10 |
| RAW_SAMPLES | 512 | Sobol candidate pool |

### Submission

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| query | string | `x1-x2-x3` format, 6 decimal places | Each $x_d \in [0.0, 0.999999]$ |
| candidate | float64[3] | Clipped to [0.0, 0.999999] | Raw optimiser output, post-clip |

## State Transitions

```
Load Data (24 samples)
  → Validate (shape, range, NaN)
    → Build GP (SingleTaskGP + Standardize)
      → Fit MLL (15 restarts, best model selection)
        → Construct qLogNEI
          → optimize_acqf (20 restarts, 512 raw)
            → Clip & format submission
```

## Changes from Previous Week (Week 8 / existing Week 9)

| Aspect | Before | After |
|--------|--------|-------|
| Output normalisation | Manual z-score (y_mean, y_std, y_standardised) | Standardize(m=1) — automatic |
| Posterior un-standardisation | Manual: `mean_raw = mean_std * y_std_safe + y_mean` | Automatic — posterior returns original scale |
| LOO z-score | Manual recomputation per fold | Automatic — Standardize handles per fold |
| Acquisition restarts | ACQ_RESTARTS = 10 | NUM_RESTARTS_ACQ = 20 |
| Interior penalty | None | None (evaluated and removed per clarification) |
