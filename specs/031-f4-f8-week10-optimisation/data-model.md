# Data Model: F4–F8 Week 10 Optimisation Strategy Changes

**Date**: 2026-03-12 | **Branch**: `031-f4-f8-week10-optimisation`

## Entities

### Observation

A recorded evaluation of the black-box function.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| x | float[n] | Each xᵢ ∈ [0, 0.999999] | n-dimensional input vector |
| y | float | Unbounded | Scalar objective value (maximise) |
| source | enum | {initial, submission} | Whether from initial dataset or weekly submission |
| week | int | ≥ 0 | Week number when observation was added |

Observation counts per function:

| Function | n_dims | n_initial | n_submitted | n_total |
|----------|--------|-----------|-------------|---------|
| F4 | 4 | 30 | 10 | 40 |
| F5 | 4 | 20 | 10 | 30 |
| F6 | 5 | 20 | 10 | 30 |
| F7 | 6 | 30 | 10 | 40 |
| F8 | 8 | 40 | 10 | 50 |

### SurrogateConfig

Configuration for the surrogate model fitted in each notebook.

| Field | Type | Default | F4 | F5 | F6 | F7 | F8 |
|-------|------|---------|----|----|----|----|-----|
| model_type | str | — | SingleTaskGP | SingleTaskGP | SingleTaskGP | SurrogateNN | SingleTaskGP |
| kernel_nu | float | 2.5 | 2.5 | 1.5 | 1.5 | N/A | 2.5 |
| ard_num_dims | int | n_dims | 4 | 4 | 5 | N/A | 8 |
| noise_lb | float | — | 1e-3 | 1e-6 | 1e-3 | N/A | 1e-7 |
| n_mll_restarts | int | — | ≥30 | 15 | 15 | N/A | ≥30 |
| outcome_transform | str | Standardize(m=1) | Standardize(m=1) | Standardize(m=1) | Standardize(m=1) | manual z-score | Standardize(m=1) |
| output_preprocess | str | None | None | log | None | None | None |
| nn_layers | str | N/A | N/A | N/A | N/A | 6→5→5→1 | N/A |
| nn_dropout | float | N/A | N/A | N/A | N/A | 0.05 | N/A |
| nn_lr | float | N/A | N/A | N/A | N/A | 0.005 | N/A |
| nn_epochs | int | N/A | N/A | N/A | N/A | 200 | N/A |

### AcquisitionConfig

Configuration for the acquisition function and candidate selection.

| Field | Type | F4 | F5 | F6 | F7 | F8 |
|-------|------|----|----|----|----|-----|
| acq_type | str | qLogNEI | qLogNEI | qLogNEI | mean+EI blend | qLogNEI |
| q | int | 4 | 4 | 4 | 1 (argmax) | 1 |
| mc_samples | int | 512 | 512 | 512 | N/A | 512 |
| raw_samples | int | 2048 | 8000 | 5000 | N/A | 8192 |
| num_restarts | int | 20 | 60 | 50 | N/A | 30 |
| xi | float | N/A | N/A | N/A | N/A | N/A (qLogNEI) |
| exploitation_weight | float | N/A | N/A | N/A | 0.5 | N/A |
| n_candidates | int | N/A | N/A | N/A | 50000 | N/A |
| mc_dropout_passes | int | N/A | N/A | N/A | ≥50 | N/A |

### InteriorPenalty

Soft boundary penalty applied to acquisition values.

| Field | Type | F6 | F7 |
|-------|------|----|----|
| steepness | float | 1.0 | 0.02 |
| floor | float | 0.01 | 0.02 |
| scoring | str | rank-based additive | multiplicative |
| feasibility_bounds | dict | x4(milk) ≥ 0.12 | all ≥ 0 |

Formula: $w(\mathbf{x}) = \text{FLOOR} + (1-\text{FLOOR}) \cdot \prod_i \sin(\pi x_i)^{2 \cdot \text{STEEPNESS}}$

### SubmissionQuery

The formatted candidate point for black-box evaluation.

| Field | Type | Constraints |
|-------|------|-------------|
| values | float[n] | Each xᵢ ∈ [0, 0.999999], 6 decimal places |
| format | str | `x1-x2-...-xn` dash-separated |
| is_duplicate | bool | Must be False — checked against all existing observations with atol=1e-6 |

## State Transitions

```
Data Loaded → GP/NN Fitted → Acquisition Optimised → Candidate Selected → Submission Formatted
                                                          ↓ (if duplicate)
                                                   Next-best candidate
                                                          ↓ (if all duplicate)
                                                   Manual review required
```

## Relationships

- Each **Observation** belongs to one function (F4–F8) and one week
- Each function has exactly one **SurrogateConfig** and one **AcquisitionConfig** per week
- **InteriorPenalty** is optional — only F6 and F7 use it
- **SubmissionQuery** is derived from the best candidate after **AcquisitionConfig** optimisation
- **SurrogateConfig.output_preprocess** is applied before GP fitting; **SurrogateConfig.outcome_transform** is applied by BoTorch internally

## Validation Rules

- All input values clamped to [0.0, 0.999999]
- Submission must not duplicate any existing observation (atol=1e-6 per dimension)
- F5: All outputs must be strictly positive before log transform
- F6: Milk dimension (x4) must be ≥ 0.12 in selected candidate; fallback to 0.10 if infeasible
- F7: MC dropout requires ≥50 forward passes for stable uncertainty
- F8: Cholesky decomposition must succeed; if not, increase noise_lb
