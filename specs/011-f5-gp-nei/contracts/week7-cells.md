# Cell Acceptance Contract: F5 Week 7

**Feature**: 011-f5-gp-nei
**Notebook**: `functions/f5/f5.ipynb`
**Existing cells**: 50 (cells 1–50, last = `#VSC-8f8ac8b4`)
**New cells**: 8 (cells 51–58)

---

## Cell 51: Week 7 Markdown Header

- **Type**: Markdown
- **Insert after**: `#VSC-8f8ac8b4` (cell 50)
- **Content**: `## Week 7 — GP Matérn-5/2 + NEI` with strategy summary explaining the switch from GBT to GP, and a comparison table (Week 6 vs Week 7 parameters)
- **Acceptance**: Renders as formatted markdown with week title and rationale

---

## Cell 52: Imports & Data Loading

- **Type**: Code (Python)
- **Insert after**: Cell 51
- **Imports**: torch, gpytorch, botorch (SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf, SobolQMCNormalSampler), numpy, matplotlib, copy
- **Data**: Load `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy`
- **Output**: Print shape (27, 4), (27,), input range, output range, best observed value and index
- **Acceptance**:
  - `X_raw.shape == (27, 4)`
  - `y_raw.shape == (27,)`
  - `X_raw.min() >= 0.0` and `X_raw.max() <= 1.0`
  - Best value and index printed

---

## Cell 53: Hyperparameter Documentation

- **Type**: Markdown
- **Insert after**: Cell 52
- **Content**: Table of all hyperparameters with columns: Parameter, Value, Rationale
- **Required entries** (minimum 12):
  1. Kernel: Matérn-5/2
  2. ARD: True (4 lengthscales)
  3. Lengthscale init: 0.25
  4. Output scale init: ~1.0 (Var of standardised data)
  5. Noise init: 0.03 (3% of standardised variance)
  6. Noise floor: 1e-6 (jitter)
  7. Output transform: log1p → z-score
  8. MLL restarts: 15
  9. Acquisition: qLogNoisyExpectedImprovement (NEI)
  10. q: 2 (batch size)
  11. raw_samples: 3000 (Sobol initial points)
  12. num_restarts: 50 (L-BFGS starting points)
- **Acceptance**: All 12+ entries present with non-empty rationale

---

## Cell 54: GP Training with MLL Restarts

- **Type**: Code (Python)
- **Insert after**: Cell 53
- **Logic**:
  1. Compute `y_log = np.log1p(y_raw)`, z-score → `y_std`
  2. Convert to torch tensors (double precision)
  3. Loop 15 restarts: construct SingleTaskGP, init HPs (ls=0.25, noise=0.03, outputscale=1.0), fit MLL, score, keep best
  4. Print per-restart neg_MLL
  5. Print fitted HPs: ℓ₁–ℓ₄, σ²_f, σ²_n
- **Output**:
  - 15 restart scores printed
  - Best neg_MLL value
  - 6 hyperparameter values (4 lengthscales + output scale + noise)
- **Acceptance**:
  - No runtime errors
  - All 15 restarts produce finite neg_MLL values
  - Fitted noise ≥ 1e-6
  - All lengthscales > 0

---

## Cell 55: NEI Acquisition

- **Type**: Code (Python)
- **Insert after**: Cell 54
- **Logic**:
  1. Construct `qLogNoisyExpectedImprovement` with fitted model, q=2, prune_baseline=True
  2. Call `optimize_acqf` with bounds=[[0]*4, [1]*4], num_restarts=50, raw_samples=3000
  3. Extract 2 candidate points
  4. Compute posterior means (standardised + inverse-transformed to original scale)
  5. Select best candidate by highest posterior mean
- **Output**:
  - 2 candidate points with coordinates
  - Posterior means (standardised and original scale)
  - Best candidate index and coordinates
- **Acceptance**:
  - `candidates.shape == (2, 4)`
  - All candidate values in [0, 1]
  - Posterior means printed for both candidates
  - Best candidate clearly identified

---

## Cell 56: Surrogate Visualisation (3-Panel)

- **Type**: Code (Python)
- **Insert after**: Cell 55
- **Logic**:
  1. Identify top-2 important dims (shortest ARD lengthscales)
  2. Build 80×80 grid over top-2 dims, fixing other 2 at best_point values
  3. Compute GP posterior mean and std on grid (inverse-transformed to original scale)
  4. Panel 1: Mean contour with observed points (red) and proposed point (magenta star)
  5. Panel 2: Std contour with observed points and proposed point
  6. Panel 3: Dimension relevance bar chart (1/ℓ normalised)
- **Output**: `plt.show()` renders 3-panel figure (18×5 inches)
- **Acceptance**:
  - Figure has exactly 3 subplots
  - Colorbars present on panels 1 and 2
  - Observed points (red dots) and proposed point (star) visible
  - Bar chart shows 4 bars labelled x0–x3
  - Title includes "Week 7" and "GP"

---

## Cell 57: Convergence Plot

- **Type**: Code (Python)
- **Insert after**: Cell 56
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

## Cell 58: Submission Query

- **Type**: Code (Python)
- **Insert after**: Cell 57
- **Logic**:
  1. Format best_point as `x1-x2-x3-x4` with 6 decimal places
  2. Validate: 4 parts, all start with "0.", all in [0, 1]
  3. Print formatted query prominently
- **Output**: Submission query string + validation confirmation
- **Acceptance**:
  - Query matches pattern `\d\.\d{6}-\d\.\d{6}-\d\.\d{6}-\d\.\d{6}`
  - All 4 values in [0, 1]
  - Validation print: "✓ Submission format validated"
