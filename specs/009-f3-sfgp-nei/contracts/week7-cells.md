# Cell Contract: F3 Week 7 Notebook Additions

**Feature**: 009-f3-sfgp-nei  
**Target file**: `functions/f3/f3.ipynb`  
**Rule**: All 8 cells below are appended after the last existing cell. Zero existing cells are modified.

---

## Cell 1 — Section Header (Markdown)

**Type**: Markdown  
**Position**: Immediately after last current cell  

**Required content**:
```markdown
## Week 7 — SFGP with Matérn-5/2 ARD

This section fits a **Single-Fidelity Gaussian Process (SFGP)** surrogate on all cumulative
Week 7 data and proposes the next compound combination using **Noisy Expected Improvement (NEI)**.

**Why SFGP for Week 7?**
- GPs with Matérn-5/2 kernels capture smooth-to-moderately-rough functions — appropriate for drug
  dose–response surfaces where the relationship is expected to be continuous but not infinitely smooth.
- **ARD (Automatic Relevance Determination)** gives each compound dimension its own lengthscale,
  letting the model learn which compounds drive the most variation.
- NEI handles the observation noise present in experimental drug assays, unlike analytic EI.
```

**Acceptance**: Cell is markdown, heading level 2 begins with "Week 7".

---

## Cell 2 — Imports and Data Loading (Code)

**Type**: Code  
**Step label**: "Step 1: Load Week 7 Data"  

**Required behaviour**:
1. Import `copy`, `warnings`, `numpy as np`, `torch`, `matplotlib.pyplot as plt`.
2. Import `SingleTaskGP`, `fit_gpytorch_mll`, `ExactMarginalLogLikelihood`.
3. Import `MaternKernel`, `ScaleKernel`, `GreaterThan`, `GaussianLikelihood`.
4. Import `qLogNoisyExpectedImprovement` from `botorch.acquisition.logei`.
5. Import `optimize_acqf` from `botorch.optim`.
6. Load `X_raw = np.load('../../data/f3/updated_inputs - Week 7.npy')`.
7. Load `Y_raw = np.load('../../data/f3/updated_outputs - Week 7.npy')`.
8. Print sample count, input range, output range, best observed value and its index.
9. Validate all inputs are in [0.0, 1.0]; print a warning for any out-of-range row.

**Acceptance**: Cell runs without error; prints at least: sample count, input range, output range, best value.

---

## Cell 3 — Hyperparameter Explanation (Markdown)

**Type**: Markdown  
**Step label**: "Step 2: SFGP Hyperparameters and Justifications"  

**Required content** (must cover all items):
| Hyperparameter | Value | Must be documented |
|---|---|---|
| `N_RESTARTS` | 15 | Why multi-restart is needed |
| `LENGTHSCALE_INIT` | 0.25 | Why ~0.2–0.3 for [0,1]-scaled inputs |
| `SIGNAL_VAR_INIT` | 1.0 | Why = 1 after z-score |
| `NOISE_VAR_INIT` | 0.1 | Conservative 10% noise-to-signal for drug assay noise |
| `JITTER` | 1e-6 | Numerical stability floor |
| Mean function | Constant learned | Why constant mean for unknown function |
| Kernel | Matérn-5/2 ARD | Why Matérn-5/2 vs RBF; what ARD provides |
| Likelihood | Gaussian noise | When to consider Student-t instead |
| NEI | `qLogNoisyExpectedImprovement` | Why NEI over analytic EI |
| Acquisition bounds | [0, 0.999999]³ | Challenge submission constraint |

**Acceptance**: Cell is markdown with a section covering every row in the table above.

---

## Cell 4 — Model Training with Restarts (Code)

**Type**: Code  
**Step label**: "Step 3: Train SFGP with 15 Random Restarts"  

**Required behaviour**:
1. Convert `X_raw` → `X_train` (torch.float64).
2. Compute `y_mean`, `y_std` from `Y_raw`; create `Y_train` (z-scored, shape `(n,1)`, torch.float64).
3. Define constants: `N_RESTARTS=15`, `LENGTHSCALE_INIT=0.25`, `SIGNAL_VAR_INIT=1.0`, `NOISE_VAR_INIT=0.1`.
4. Loop `N_RESTARTS` times with `torch.manual_seed(seed)`:
   - Construct `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))` and `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`.
   - Set initial hyperparameter values.
   - Call `fit_gpytorch_mll(mll)`.
   - Score via –MLL; keep deepcopy of best.
5. Call `best_model.eval()`.
6. Print fitted ℓ_A, ℓ_B, ℓ_C (with dimension labels), σ²_f, σ²_n.
7. Print best restart's –MLL loss.

**Acceptance**: Cell runs without unhandled exceptions; prints five labelled hyperparameter values; prints "Best model" summary line.

---

## Cell 5 — NEI Acquisition (Code)

**Type**: Code  
**Step label**: "Step 4: NEI Acquisition — Propose Next Sample"  

**Required behaviour**:
1. Set `BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [0.999999, 0.999999, 0.999999]], dtype=torch.double)`.
2. Construct `qLogNoisyExpectedImprovement(model=best_model, X_baseline=X_train, prune_baseline=True)`.
3. Call `optimize_acqf(nei, bounds=BOUNDS, q=1, num_restarts=10, raw_samples=512)`.
4. Extract `next_x_raw = candidate.detach().squeeze(0).cpu().numpy()`.
5. Print the proposed point components and the NEI value.

**Acceptance**: Cell runs without error; a `(3,)` numpy array `next_x_raw` is defined; NEI value is printed.

---

## Cell 6 — Surrogate Slice Plots (Code)

**Type**: Code  
**Step label**: "Step 5: Surrogate Visualisation — Pairwise 2D Slices"  

**Required behaviour**:
1. Identify `best_idx = Y_raw.argmax()` and `best_point = X_raw[best_idx]`.
2. Create a 3-panel figure (1×3, figsize=(18,5)).
3. For each panel (pairs: (0,1), (0,2), (1,2)):
   - Build 50×50 grid; fix the third dimension at `best_point[third_dim]`.
   - Compute GP posterior mean and std on the grid (in **original output scale**: multiply std by `y_std`).
   - Plot mean as filled contour (cmap='viridis') with colourbar.
   - Overlay 2σ uncertainty contour lines (colour='white', alpha=0.4).
   - Mark observed points (red dots).
   - Mark proposed next point `next_x_raw` as yellow star.
   - Label axes as 'Compound A / B / C' (matching dimension letter).
   - Title: `"F3 Week 7 Surrogate — x{i+1} vs x{j+1} (x{k+1}={fixed:.3f})"`.
4. Call `plt.tight_layout(); plt.show()`.

**Acceptance**: Three distinct plots render without error; proposed point is marked; axes are labelled; colourbar present.

---

## Cell 7 — Convergence Plot (Code)

**Type**: Code  
**Step label**: "Step 6: Convergence Plot"  

**Required behaviour**:
1. Compute `running_max = np.maximum.accumulate(Y_raw)`.
2. Plot `running_max` as blue line with markers; scatter individual observations in gray.
3. Vertical dashed red line at x=15.5 (initial → weekly boundary).
4. x-axis label: `"Sample Number"`.
5. y-axis label: `"Best Observed Output"`.
6. Title: `"Function 3 — Convergence Plot (Week 7)"`.
7. Print best observed value and at which sample number it was achieved.

**Acceptance**: Plot renders; x-axis labelled "Sample Number"; y-axis labelled "Best Observed Output"; boundary line present.

---

## Cell 8 — Submission Query (Code)

**Type**: Code  
**Step label**: "Step 7: Format Submission Query"  

**Required behaviour**:
1. Clamp `next_x_raw` components to [0.0, 0.999999].
2. Format as `"-".join([f"{x:.6f}" for x in clamped])`.
3. Print a summary block with surrogate type, acquisition type, fitted hyperparameters, proposed raw point, and the formatted submission string.
4. Print the final query string on its own line clearly marked for copy/paste.

**Acceptance**: Output contains a string matching `r"0\.\d{6}-0\.\d{6}-0\.\d{6}"` (i.e., three 6dp components separated by hyphens, all starting with 0).

---

## Summary Counts

| Item | Count |
|------|-------|
| Markdown cells | 2 (cells 1, 3) |
| Code cells | 6 (cells 2, 4, 5, 6, 7, 8) |
| Total new cells | 8 |
| Existing cells modified | 0 |
