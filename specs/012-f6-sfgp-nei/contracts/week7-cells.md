# Cell Acceptance Contract: F6 Week 7

**Feature**: 012-f6-sfgp-nei
**Notebook**: `functions/f6/f6.ipynb`
**Existing cells**: 47 (cells 1–47, last cell id: `a52b2e42`)
**New cells**: 8 (cells 48–55)

---

## Cell 48: Week 7 Markdown Header

- **Type**: Markdown
- **Insert after**: cell 47 (id `a52b2e42`)
- **Content**: `## Week 7 — SFGP Matérn-1.5 + NEI` with strategy summary explaining the switch from NN + MC Dropout + UCB κ=0.5 (Week 6) to SFGP + NEI q=4 (Week 7), including a comparison table
- **Acceptance**: Renders as formatted markdown with week title, rationale, and comparison table

---

## Cell 49: Imports & Data Loading

- **Type**: Code (Python)
- **Insert after**: Cell 48
- **Imports**: torch, gpytorch, botorch (SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf, SobolQMCNormalSampler), numpy, matplotlib, copy
- **Data**: Load `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy` via relative path `../../data/f6/`
- **Output**: Print shape (27, 5), (27,), input range, output range, best observed value and index
- **Acceptance**:
  - `X_raw.shape == (27, 5)`
  - `y_raw.shape == (27,)`
  - `X_raw.min() >= 0.0` and `X_raw.max() <= 1.0`
  - Best value and index printed
  - All values negative confirmed

---

## Cell 50: Hyperparameter Documentation

- **Type**: Markdown
- **Insert after**: Cell 49
- **Content**: Table of all hyperparameters with columns: Parameter, Value, Rationale
- **Required entries** (minimum 14):
  1. Kernel: Matérn-1.5 (once-differentiable, rougher than 2.5)
  2. ARD: True (5 lengthscales, one per dimension)
  3. Lengthscale init: 0.5 (broader uncertainty for exploration)
  4. Output scale init: 1.0 (matches standardised variance)
  5. Noise init: 0.1 (10% of standardised Var(y)≈1.0; see RES-003)
  6. Noise floor: 1e-8 (tighter than 1e-6; user-specified)
  7. Outcome transform: Standardize(m=1) — BoTorch default z-score
  8. MLL restarts: 15
  9. Acquisition: qLogNoisyExpectedImprovement (NEI)
  10. q: 4 (batch size for diversity)
  11. raw_samples: 3000 (Sobol initial points)
  12. num_restarts: 50 (L-BFGS starting points)
  13. Selection: distance-based (farthest from data, mean ≥ median)
  14. Clamping: [0, 0.999999] before formatting
- **Acceptance**: All 14 entries present with non-empty rationale

---

## Cell 51: GP Training with MLL Restarts

- **Type**: Code (Python)
- **Insert after**: Cell 50
- **Logic**:
  1. Convert raw data to torch double tensors: `X_train (27,5)`, `Y_train (27,1)`
  2. Loop 15 restarts: construct SingleTaskGP (**no** `outcome_transform` arg — uses default `Standardize(m=1)`), init HPs (ℓ=0.5, noise=0.1, outputscale=1.0), fit MLL, score, keep best
  3. Print per-restart neg_MLL
  4. Print fitted HPs: ℓ₁–ℓ₅, σ²_f, σ²_n
- **Key differences from F5 Cell 54**:
  - No `log1p` / z-score transform — raw Y passed directly
  - No `outcome_transform=None` — default `Standardize(m=1)` used
  - `MaternKernel(nu=1.5, ard_num_dims=5)` (not `nu=2.5, ard_num_dims=4`)
  - `noise_constraint=GreaterThan(1e-8)` (not `1e-6`)
  - `noise = 0.1` (not `0.1 * Y_train.var().item()`)
- **Output**:
  - 15 restart scores printed
  - Best neg_MLL value
  - 7 hyperparameter values (5 lengthscales + output scale + noise)
- **Acceptance**:
  - No runtime errors
  - All 15 restarts produce finite neg_MLL values
  - Fitted noise ≥ 1e-8
  - All lengthscales > 0
  - 5 distinct lengthscale values (ARD active)

---

## Cell 52: NEI Acquisition & Distance-Based Selection

- **Type**: Code (Python)
- **Insert after**: Cell 51
- **Logic**:
  1. Construct `qLogNoisyExpectedImprovement` with fitted model, q=4, prune_baseline=True
  2. Call `optimize_acqf` with bounds=[[0]*5, [1]*5], num_restarts=50, raw_samples=3000
  3. Extract 4 candidate points; clamp to [0, 0.999999]
  4. Compute posterior means — these are in **original space** automatically (no manual untransform)
  5. Distance-based selection: filter candidates with mean ≥ median(means), select farthest from X_train
- **Key difference from F5**: No `expm1` inverse transform — `model.posterior()` returns original-space values via `Standardize(m=1)` auto-untransform.
- **Output**:
  - 4 candidate points with coordinates
  - Posterior means (original scale, all negative)
  - Min-distance from each candidate to training data
  - Selected candidate index, coordinates, and rationale
- **Acceptance**:
  - `candidates.shape == (4, 5)`
  - All candidate values in [0, 0.999999]
  - Posterior means all negative (matching F6 output range)
  - Distance-based selection clearly shown
  - Selected candidate identified with distance and predicted value

---

## Cell 53: Surrogate Visualisation (3-Panel)

- **Type**: Code (Python)
- **Insert after**: Cell 52
- **Logic**:
  1. Identify top-2 important dims (shortest ARD lengthscales)
  2. Build 80×80 grid over top-2 dims, fixing other 3 at best_point values
  3. Compute GP posterior mean and std on grid — values in original space automatically
  4. Panel 1: Mean contour with observed points (red) and proposed point (magenta star)
  5. Panel 2: Std contour with observed points and proposed point
  6. Panel 3: Dimension relevance bar chart (1/ℓ normalised, 5 bars for x0–x4)
- **Key difference from F5**: No `expm1` inverse transform for surface values; 5 bars (not 4)
- **Output**: `plt.show()` renders 3-panel figure (18×5 inches)
- **Acceptance**:
  - Figure has exactly 3 subplots
  - Colorbars present on panels 1 and 2
  - Observed points (red dots) and proposed point (star) visible
  - Bar chart shows 5 bars labelled x0–x4
  - Title includes "Week 7" and "SFGP"

---

## Cell 54: Convergence Plot

- **Type**: Code (Python)
- **Insert after**: Cell 53
- **Logic**:
  1. Compute `running_best = np.maximum.accumulate(y_raw)`
  2. Plot running best vs observation number
  3. Add vertical line at x=26.5 (Week 6→7 boundary)
  4. Print running best at end of Week 6 (sample 26) and Week 7 (sample 27)
- **Output**: Convergence plot + printed boundary values
- **Acceptance**:
  - Plot shows non-decreasing curve
  - Vertical dashed red line at x=26.5
  - Legend present
  - Title includes "Week 7"

---

## Cell 55: Submission Query & Summary

- **Type**: Code (Python)
- **Insert after**: Cell 54
- **Logic**:
  1. Format best_point as `x1-x2-x3-x4-x5` with 6 decimal places
  2. Validate: 5 parts, all parseable as float, all in [0, 0.999999]
  3. Print formatted query prominently
  4. Print summary: surrogate type, kernel, acquisition function, q, selection strategy, fitted lengthscales, fitted output scale, fitted noise, and the selected candidate's posterior mean
- **Output**: Submission query string + validation + HP summary
- **Acceptance**:
  - Query matches pattern `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`
  - All 5 values in [0, 0.999999]
  - Validation print: "✓ Submission format validated"
  - Summary table with all key hyperparameters
