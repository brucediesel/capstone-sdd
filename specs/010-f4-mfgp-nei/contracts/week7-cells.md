# Cell Contract: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Date**: 2026-02-23 | **Branch**: `010-f4-mfgp-nei`

This contract defines the 8 new cells to append after the existing last cell (`#VSC-21b0ced4`, Research markdown, cell 52).

## Cell 53: Week 7 Section Header (Markdown)

**Type**: Markdown
**Insert after**: `#VSC-21b0ced4` (cell 52)

**Content requirements**:
- `## Week 7 — Multi-Fidelity GP (Matérn-5/2 ARD + LinTrunc) + MF-qNEI` as the heading
- Brief description: surrogate choice rationale (PE winner), acquisition strategy, weekly context

**Acceptance**: Cell renders as a visible section break with the week number.

---

## Cell 54: Imports & Data Loading (Code)

**Type**: Code (Python)
**Insert after**: Cell 53

**Inputs**: `data/f4/updated_inputs - Week 7.npy`, `data/f4/updated_outputs - Week 7.npy`
**Outputs printed**:
- Number of samples (expected: 37)
- Input shape (expected: (37, 4))
- Input range validation (all values in [0, 1])
- Best observed value and its sample index
- Output mean and std (for z-score reference)

**Imports required**:
- `numpy`, `torch`, `copy`, `warnings`
- `matplotlib.pyplot`
- `botorch.models.SingleTaskMultiFidelityGP`
- `botorch.fit.fit_gpytorch_mll`
- `botorch.acquisition.logei.qLogNoisyExpectedImprovement`
- `botorch.optim.optimize_acqf`
- `botorch.sampling.normal.SobolQMCNormalSampler`
- `gpytorch.mlls.ExactMarginalLogLikelihood`
- `gpytorch.constraints.GreaterThan`
- `gpytorch.likelihoods.GaussianLikelihood`

**Acceptance**: Prints sample count, input range, best value/index. No errors.

---

## Cell 55: Hyperparameter Documentation (Markdown)

**Type**: Markdown
**Insert after**: Cell 54

**Content requirements**: Table documenting all hyperparameters with justifications:
- nu=2.5 (Matérn-5/2)
- linear_truncated=True (LinearTruncatedFidelityKernel)
- noise floor ≥ 1e-4
- z-score standardisation
- 15 MLL restarts
- q=4 batch, 64 MC samples
- Sobol initialisation (512 raw samples, 20 restarts)
- fixed_features={4: 1.0}

**Acceptance**: Renders as a readable table with ≥ 10 hyperparameters documented.

---

## Cell 56: MFGP Training with Multi-Restart MLL (Code)

**Type**: Code (Python)
**Insert after**: Cell 55

**Inputs**: X_raw, y_raw from Cell 54
**Processing**:
1. Z-score standardise outputs: `y_std = (y_raw - y_mean) / y_std_val`
2. Convert to torch tensors, append fidelity column (all 1.0)
3. Loop 15 restarts: seed → create model → fit MLL → score → deepcopy if best
4. Load best model state

**Outputs printed**:
- Per-restart: seed, negative MLL (or "FAILED" if NaN)
- Best restart summary: seed, neg_mll
- Fitted hyperparameters: ℓ₁–ℓ₄, σ²_f, σ²_n, fidelity power

**Acceptance**: Completes without error; noise ≥ 1e-4; all HPs printed.

---

## Cell 57: MF-qNEI Acquisition (Code)

**Type**: Code (Python)
**Insert after**: Cell 56

**Inputs**: Fitted MFGP model, X_mf training tensor
**Processing**:
1. Create `SobolQMCNormalSampler(sample_shape=torch.Size([64]))`
2. Create `qLogNoisyExpectedImprovement(model, X_baseline=X_mf, sampler, prune_baseline=True)`
3. Set bounds (5D): spatial [0, 0.999999]⁴, fidelity [1.0, 1.0]
4. `optimize_acqf(nei, bounds, q=4, num_restarts=20, raw_samples=512, fixed_features={4: 1.0})`
5. Extract spatial coordinates: `candidates[:, :4]`
6. Evaluate posterior mean for each candidate; select best

**Outputs printed**:
- All 4 candidates with coordinates
- Joint acquisition value
- Selected best candidate (by posterior mean) with index

**Acceptance**: 4 candidates returned, all coordinates in [0, 0.999999]. Best candidate identified.

---

## Cell 58: Surrogate Visualisation (Code)

**Type**: Code (Python)
**Insert after**: Cell 57

**Inputs**: Fitted MFGP, training data, proposed point
**Processing**:
1. Identify top-2 dims by shortest ARD lengthscales
2. Build 80×80 grid over top-2 dims, fix other 2 at proposed point values
3. Append fidelity=1.0 column to grid
4. Get posterior mean and std on grid (de-standardise)
5. Plot 2-panel figure: predicted mean contour + posterior std contour
6. Overlay observed points (red dots) and proposed point (yellow star)

**Outputs**: 2-panel figure with colorbars, axis labels, title.

**Acceptance**: Figure renders with no errors; contours visible; markers present.

---

## Cell 59: Convergence Plot (Code)

**Type**: Code (Python)
**Insert after**: Cell 58

**Inputs**: y_raw (original scale)
**Processing**:
1. Compute `running_best = np.maximum.accumulate(y_raw)`
2. Plot running best vs observation number
3. Add vertical line at x=30.5 (initial→weekly boundary)
4. Label axes, add title, legend, grid

**Outputs**: Line chart with running best and boundary marker.

**Acceptance**: Plot renders; boundary line at 30.5; best values printed.

---

## Cell 60: Submission Query (Code)

**Type**: Code (Python)
**Insert after**: Cell 59

**Inputs**: best_point from Cell 57
**Processing**:
1. Clip to [0, 0.999999]
2. Format as dash-separated string, 6 decimal places
3. Validate format (4 parts, each in [0, 1])
4. Print summary: surrogate type, acquisition, fitted HPs, query

**Outputs printed**:
- Submission summary header
- Surrogate: MFGP (Matérn-5/2 ARD + LinTrunc)
- Acquisition: MF-qNEI (q=4, 64 MC samples)
- Fitted hyperparameters
- `>>> SUBMISSION QUERY: 0.XXXXXX-0.YYYYYY-0.ZZZZZZ-0.WWWWWW`
- Format validation result

**Acceptance**: Valid submission string in correct format. All 4 coordinates in [0, 0.999999].
