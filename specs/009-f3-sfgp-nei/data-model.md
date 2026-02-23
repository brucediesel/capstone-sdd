# Data Model: F3 Week 7 – SFGP with Matérn-5/2 ARD and NEI Acquisition

**Feature**: 009-f3-sfgp-nei  
**Phase**: 1  

This document describes every data entity the Week 7 notebook cells read, transform, produce, or display. No new persistent files are created; all entities live in memory within the kernel session.

---

## Input Entities

### E-01: Raw Input Array — `X_raw`

| Attribute | Value |
|-----------|-------|
| Source | `data/f3/updated_inputs - Week 7.npy` |
| Shape | `(n, 3)` — n samples, 3 compound dimensions (A, B, C) |
| dtype | float64 (numpy) |
| Value range | [0.0, 1.0] per component (validated on load) |
| Meaning | Each row is one experiment: [dose_A, dose_B, dose_C] |
| Cumulative? | Yes — contains all observations from initial set through Week 7 |

### E-02: Raw Output Array — `Y_raw`

| Attribute | Value |
|-----------|-------|
| Source | `data/f3/updated_outputs - Week 7.npy` |
| Shape | `(n,)` 1D array |
| dtype | float64 (numpy) |
| Meaning | Objective value (maximisation; transformed negative adverse reactions) |
| Cumulative? | Yes — same row order as E-01 |

---

## Derived Entities (in-memory, training scope)

### E-03: Training Inputs Tensor — `X_train`

| Attribute | Value |
|-----------|-------|
| Derived from | E-01 |
| Shape | `(n, 3)`, torch.float64 |
| Transform | `torch.tensor(X_raw, dtype=torch.float64)` |
| Notes | Inputs are already in [0, 1]; no further scaling applied |

### E-04: Standardisation Parameters — `y_mean`, `y_std`

| Attribute | Value |
|-----------|-------|
| Derived from | E-02 |
| `y_mean` | Scalar float — `Y_raw.mean()` |
| `y_std` | Scalar float — `Y_raw.std()` |
| Purpose | Used to standardise targets before training and de-standardise predictions |

### E-05: Standardised Training Targets — `Y_train`

| Attribute | Value |
|-----------|-------|
| Derived from | E-02, E-04 |
| Shape | `(n, 1)`, torch.float64 |
| Transform | `(Y_raw - y_mean) / y_std`, then `.unsqueeze(-1)` |
| Value range | Mean ≈ 0, std ≈ 1 after transform |

---

## Model Hyperparameters

### E-06: SFGP Configuration Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `N_RESTARTS` | 15 | Middle of spec range 10–20; balances quality with runtime |
| `LENGTHSCALE_INIT` | 0.25 | Reasonable prior for [0,1]-scaled 3D function |
| `SIGNAL_VAR_INIT` | 1.0 | Matches z-scored output variance by construction |
| `NOISE_VAR_INIT` | 0.1 | Conservative 10% noise-to-signal for unknown black-box |
| `JITTER` | 1e-6 | Noise constraint floor for numerical stability |

### E-07: Fitted SFGP Hyperparameters (post-training)

| Parameter | Access path | Notes |
|-----------|-------------|-------|
| Lengthscale ℓ_A | `best_model.covar_module.base_kernel.lengthscale[0, 0]` | Per-dimension ARD |
| Lengthscale ℓ_B | `best_model.covar_module.base_kernel.lengthscale[0, 1]` | Per-dimension ARD |
| Lengthscale ℓ_C | `best_model.covar_module.base_kernel.lengthscale[0, 2]` | Per-dimension ARD |
| Signal variance σ²_f | `best_model.covar_module.outputscale.item()` | ScaleKernel output scale |
| Noise variance σ²_n | `best_model.likelihood.noise.item()` | GaussianLikelihood |

---

## Output Entities

### E-08: NEI Acquisition Candidate — `next_x_raw`

| Attribute | Value |
|-----------|-------|
| Derived from | NEI optimisation over fitted model |
| Shape | `(3,)` numpy array |
| Value range | [0.0, 0.999999] per component (BOUNDS applied during `optimize_acqf`) |
| Meaning | Proposed next compound combination to test |

### E-09: Submission Query String — `submission_query`

| Attribute | Value |
|-----------|-------|
| Derived from | E-08 |
| Format | `"0.xxxxxx-0.yyyyyy-0.zzzzzz"` |
| Construction | `"-".join([f"{x:.6f}" for x in next_x_raw])` |
| Constraints | All components start with `0`, expressed to 6dp, in [0.000000, 0.999999] |

---

## Visualisation Entities

### E-10: Surrogate Prediction Grid

| Attribute | Value |
|-----------|-------|
| Purpose | Input to the three pairwise 2D slice plots |
| Grid size | 50×50 per slice = 2500 points per panel |
| Fixed dimension | Third dimension held at `X_raw[Y_raw.argmax()]` for each pair |
| Predicted values | GP posterior mean (in original scale) and std (σ) |
| Panels | x1 vs x2 (fix x3), x1 vs x3 (fix x2), x2 vs x3 (fix x1) |

### E-11: Convergence Series

| Attribute | Value |
|-----------|-------|
| Source | E-02 (all n observations) |
| Series | `np.maximum.accumulate(Y_raw)` — running best observed value |
| x-axis | Sample number (1 to n) |
| Annotation | Vertical line at x = 15.5 separating initial from weekly samples |

---

## State Transitions

```
E-01 (file) → validate → E-03 (tensor)
E-02 (file) → compute stats → E-04 → standardise → E-05 (tensor)
(E-03, E-05) → restart loop → best_model (E-06/E-07)
best_model + E-03 → NEI → E-08 → format → E-09
best_model + grid → E-10 → plots
E-02 → accumulate → E-11 → convergence plot
```
