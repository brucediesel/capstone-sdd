# Data Model: F5 Week 9 — Remove Interior Penalty

**Feature**: 025-f5-remove-penalty  
**Date**: 2026-03-09

## Overview

This feature does not introduce or modify any data entities. It only removes interior penalty code from an existing notebook. The data model below documents the existing entities that remain **unchanged** for reference during implementation.

## Existing Entities (Unchanged)

### Training Data
- **X_raw**: `np.ndarray` shape `(29, 4)` — 29 samples × 4 dimensions, loaded from `data/f5/updated_inputs - Week 9.npy`
- **y_raw**: `np.ndarray` shape `(29,)` — 29 output values, loaded from `data/f5/updated_outputs - Week 9.npy`
- **X_train**: `torch.Tensor` shape `(29, 4)` dtype `float64` — input tensor
- **Y_train**: `torch.Tensor` shape `(29, 1)` dtype `float64` — z-scored log1p-transformed outputs

### GP Model
- **best_model**: `SingleTaskGP` with `MaternKernel(nu=2.5, ard_num_dims=4)`, `ScaleKernel`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`, `outcome_transform=None`
- Fitted via 15-restart MLL optimisation

### Acquisition
- **nei**: `qLogNoisyExpectedImprovement` with `SobolQMCNormalSampler(512)`, `X_baseline=X_train`
- **candidates**: `torch.Tensor` shape `(4, 4)` — q=4 batch candidates from `optimize_acqf` with 50 restarts, 3000 raw samples
- **best_point**: `np.ndarray` shape `(4,)` — selected via distance-based filtering (above-median mean, farthest from data)

### Constants (After Removal)
The following constants remain in the hyperparameters cell:

| Constant | Value | Purpose |
|----------|-------|---------|
| N_INITIAL | 20 | Initial training samples |
| N_TOTAL | 29 | Total samples (20 initial + 9 submissions) |
| N_DIMS | 4 | Input dimensionality |
| N_SUBMISSIONS | 9 | Weekly submissions |
| STALLING_WINDOW | 3 | Consecutive non-improving threshold |
| STALLING_REL_THRESHOLD | 0.05 | 5% relative improvement threshold |
| N_RESTARTS | 15 | MLL fitting restarts |
| DIM | 4 | Alias for N_DIMS |

### Constants Removed
| Constant | Former Value | Reason for Removal |
|----------|-------------|-------------------|
| STEEPNESS | 1.0 | Interior penalty parameter — no longer used |
| FLOOR | 0.01 | Interior penalty minimum — no longer used |
| EPS_BOUND | 0.005 | Tightened bounds epsilon — no longer used |

## Removed Entities

### PenalisedAcquisition (Class — Removed)
- Was a `torch.nn.Module` wrapping `qLogNoisyExpectedImprovement` with additive log-space 4x(1-x) penalty
- Properties: `acq_fn`, `steepness`, `floor`
- Forward: `raw_acq + sum(log(penalty_weight))` over q candidates

### penalised_nei (Variable — Removed)
- Was `PenalisedAcquisition(nei, STEEPNESS, FLOOR)` — the wrapped acquisition function

### BOUNDS_IP (Variable — Removed)
- Was `torch.tensor([[EPS_BOUND]*4, [1-EPS_BOUND]*4])` — tightened bounds for penalty optimisation

### next_x_ip (Variable — Removed)
- Was the penalty-selected candidate point — replaced by `best_point` in all downstream references

## Validation Rules

- All submission coordinates must be in [0.0, 0.999999]
- Submission format: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` (4 dimensions, 6 decimal places)
- Data assertions: `X_raw.shape == (29, 4)`, `y_raw.shape == (29,)`, inputs in [0, 1]
