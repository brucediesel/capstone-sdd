# Cell Contracts: F8 Week 7 — SFGP + qEI Acquisition

**Feature**: 018-f8-sfgp-qei
**Date**: 2025-02-24
**Target**: `functions/f8/f8.ipynb` — append cells 50–57 after existing cell 49

---

## Cell 50 (Markdown) — Section Header

**Type**: Markdown
**Purpose**: Week 7 section header with hyperparameter documentation table

**Content requirements**:
- Title: "## Week 7 — SFGP + qEI Acquisition"
- Brief explanation: returning to GP surrogate (BoTorch SingleTaskGP) after Weeks 5-6 used NN; now with qEI acquisition
- Hyperparameter table with columns: Parameter, Value, Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Kernel | Matern 2.5 + ARD | User-specified; standard for smooth black-box |
| Noise floor | 1e-07 | User-specified; prevents singular kernel matrix |
| Standardise | Yes (default) | BoTorch Standardize(m=1) on outputs |
| MC samples | 256 | User-specified; Sobol quasi-random for low discrepancy |
| xi | 0.01 | User-specified; modest improvement threshold |
| Restarts | 30 | Matches initial 8D settings (cell 10) |
| Raw samples | 4096 | Matches initial 8D settings (cell 10) |

---

## Cell 51 (Code) — Load Data & Setup

**Type**: Code
**Depends on**: Data files in `data/f8/`

**Inputs**:
- `../../data/f8/updated_inputs - Week 7.npy`
- `../../data/f8/updated_outputs - Week 7.npy`

**Outputs (kernel variables)**:
- `X_raw`: ndarray (47, 8)
- `y_raw`: ndarray (47,)
- `X_train`: torch.Tensor (47, 8, float64)
- `Y_train`: torch.Tensor (47, 1, float64)
- `BOUNDS`: torch.Tensor (2, 8, float64)
- `param_names`: list of 8 strings

**Imports**:
- `numpy`, `torch`, `matplotlib.pyplot`
- `botorch.models.SingleTaskGP`, `botorch.fit.fit_gpytorch_mll`
- `botorch.acquisition.monte_carlo.qExpectedImprovement`
- `botorch.sampling.normal.SobolQMCNormalSampler`
- `botorch.optim.optimize_acqf`
- `gpytorch.mlls.ExactMarginalLogLikelihood`
- `gpytorch.kernels.ScaleKernel`, `gpytorch.kernels.MaternKernel`
- `gpytorch.likelihoods.GaussianLikelihood`
- `gpytorch.constraints.GreaterThan`

**Validation**:
- Print: sample count, dimensions, output range, best observed
- Assert: X_raw.shape == (47, 8), y_raw.shape == (47,)

---

## Cell 52 (Code) — Fit SFGP Surrogate

**Type**: Code
**Depends on**: Cell 51 (X_train, Y_train)

**Constants defined**:
- `XI = 0.01`
- `MC_SAMPLES = 256`
- `NUM_RESTARTS = 30`
- `RAW_SAMPLES = 4096`

**Outputs (kernel variables)**:
- `model`: SingleTaskGP instance (trained)
- `mll`: ExactMarginalLogLikelihood
- `lengthscales`: ndarray (8,) — ARD lengthscales
- `best_f`: float — Y_train.max() + XI

**Behaviour**:
1. Create `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))`
2. Create `GaussianLikelihood(noise_constraint=GreaterThan(1e-7))`
3. Create `SingleTaskGP(X_train, Y_train, covar_module=..., likelihood=...)`
4. Fit MLL via `fit_gpytorch_mll(mll)`
5. Extract and print ARD lengthscales
6. Print best_f = y_max + xi

---

## Cell 53 (Code) — qEI Acquisition & Candidate Selection

**Type**: Code
**Depends on**: Cell 52 (model, best_f, MC_SAMPLES, NUM_RESTARTS, RAW_SAMPLES, BOUNDS)

**Outputs (kernel variables)**:
- `sampler`: SobolQMCNormalSampler
- `acq_fn`: qExpectedImprovement instance
- `candidate`: torch.Tensor (1, 8)
- `acq_value`: torch.Tensor scalar
- `next_point`: ndarray (8,) — clamped to [0, 1]

**Behaviour**:
1. Create `SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))`
2. Create `qExpectedImprovement(model=model, best_f=best_f, sampler=sampler)`
3. Optimise: `optimize_acqf(acq_fn, bounds=BOUNDS, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)`
4. Extract and clamp candidate to [0, 1]
5. Detect fallback: if acq_value <= 0, print warning, use GP posterior mean to select by evaluating model on Sobol candidates
6. Print: candidate coordinates, acquisition value, comparison to current best

---

## Cell 54 (Code) — Feature Importance (Lengthscale)

**Type**: Code
**Depends on**: Cell 52 (model, lengthscales)

**Outputs (kernel variables)**:
- `importance`: ndarray (8,) — normalised 1/lengthscale
- `top2`: tuple of 2 ints — indices of two most important dimensions

**Behaviour**:
1. Compute `importance = 1.0 / lengthscales`; normalise to sum to 1
2. Identify top-2 dimensions (smallest lengthscale / highest importance)
3. Print importance for all 8 dimensions
4. Plot horizontal bar chart of importance

---

## Cell 55 (Code) — 3-Panel 2D Surrogate Visualisation

**Type**: Code
**Depends on**: Cell 52 (model), Cell 53 (acq_fn, next_point), Cell 54 (top2)

**Outputs (kernel variables)**:
- `grid_mu`: ndarray (50, 50)
- `grid_sigma`: ndarray (50, 50)
- `grid_ei`: ndarray (50, 50) — analytic EI for visualisation speed

**Behaviour**:
1. Create 50×50 grid over [0,1]² for the top-2 dimensions
2. Fix remaining 6 dimensions at best observed point values
3. Evaluate GP posterior mean and std on the grid
4. Evaluate analytic EI on the grid (faster than qEI for 2500 points)
5. Plot 3-panel figure: [Mean | Std | EI] with contourf
6. Mark best observed point and proposed candidate with symbols

**Note**: Use analytic `ExpectedImprovement` for the grid visualisation (2500 evaluations). This is faster than qEI and visually equivalent for the 2D slice.

---

## Cell 56 (Code) — Convergence Plot

**Type**: Code
**Depends on**: Cell 51 (y_raw)

**Outputs (kernel variables)**:
- `running_best`: ndarray (47,)

**Behaviour**:
1. Compute `running_best = np.maximum.accumulate(y_raw)`
2. Plot running best vs sample index
3. Add vertical dashed lines at sample boundaries:
   - 40.5 (initial → Week 5)
   - 45.5 (Week 5 → Week 6)
   - 46.5 (Week 6 → Week 7)
4. Print running best at each week boundary

---

## Cell 57 (Code) — Format & Submit Query

**Type**: Code
**Depends on**: Cell 53 (next_point, acq_value)

**Outputs (kernel variables)**:
- `submission_query`: str

**Behaviour**:
1. Define `format_query(point)`: clamp to [0, 0.999999], format 6 decimal places, dash-separated
2. Format next_point as submission string
3. Validate: 8 parts, each parseable as float, each in [0, 0.999999]
4. Print formatted submission with header/footer
5. Print coordinate breakdown and acquisition value summary
