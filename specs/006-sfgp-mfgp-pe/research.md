# Research: 006-sfgp-mfgp-pe

**Branch**: `006-sfgp-mfgp-pe`  
**Date**: 2026-02-22  
**Status**: Complete — no NEEDS CLARIFICATION items remain

---

## Decision 1: MFGP Library

**Decision**: BoTorch `MultiTaskGP` (already installed at v0.16.1)  
**Rationale**: EmuKit and GPy are not installed; the project constitution mandates BoTorch as the default GP library; `MultiTaskGP` is production-grade, uses the same `fit_gpytorch_mll` fitting pattern as the existing notebook's SFGP, and requires no new dependencies.  
**Alternatives considered**:  
- **EmuKit** — has a dedicated `NonLinearMultiFidelityModel`; cleaner API for multi-fidelity; rejected because it is not installed and adding a dependency conflicts with the constitution's BoTorch-first rule  
- **GPy** — classic multi-fidelity AR1 model; rejected because not installed and uses an incompatible NumPy-only API that cannot share the existing notebook's torch/BoTorch pattern  
- **Manual AR1 (two BoTorch GPs)** — could implement the Kennedy-O'Hagan autoregressive model by hand with two `SingleTaskGP` models; rejected because `MultiTaskGP` in BoTorch provides the same ICM (Intrinsic Coregionalization Model) semantics with a mature, tested implementation

---

## Decision 2: MFGP Fidelity Assignment for Prequential Loop

**Decision**: Low-fidelity set = indices 0–9 (initial 10 samples, always fixed); high-fidelity set = indices 10–16 (7 weekly update points, revealed one at a time during the prequential loop)  
**Rationale**: The initial batch and weekly submissions represent a natural observation hierarchy — the initial DOE (Design of Experiments) samples are lower-commitment; the sequential BO-guided submissions are the high-value, targeted observations. This mirrors canonical multi-fidelity BO practice.  
**Alternatives considered**:  
- **All points = same fidelity** — would reduce MFGP to a standard GP identical to SFGP; rejected because it defeats the purpose of including MFGP  
- **Random fidelity assignment** — introduces irreproducibility and loses the temporal/procedural meaning of the split; rejected

**Edge case at step 0**: At prequential step 0, there are 0 high-fidelity training observations. `MultiTaskGP` requires at least 1 observation per task to compute the inter-task correlation. Resolution: step 0 uses a plain `SingleTaskGP` fitted on the 10 LF points to predict X[10] (the first HF observation). This fallback is applied consistently across all 50 MFGP configurations and is documented in a comments block in the code.

---

## Decision 3: MultiTaskGP Tensor Format

**Decision**: Augment each training input with a task-index column. LF rows → task 0; HF rows → task 1. Use `task_feature=-1` (default: last column). Predict at the test point with task index 1 (high-fidelity).  
**Rationale**: This is the canonical BoTorch `MultiTaskGP` input format as documented in BoTorch 0.16. It requires no model subclassing.  
**Code pattern**:
```python
# Build augmented training set at prequential step t (t >= 1)
X_lf = torch.tensor(X_all[:10], dtype=torch.float64)
y_lf = torch.tensor(y_all[:10], dtype=torch.float64).unsqueeze(-1)
X_hf = torch.tensor(X_all[10:10+t], dtype=torch.float64)
y_hf = torch.tensor(y_all[10:10+t], dtype=torch.float64).unsqueeze(-1)

# Task columns (0 = LF, 1 = HF)
task_lf = torch.zeros(len(X_lf), 1, dtype=torch.float64)
task_hf = torch.ones(len(X_hf), 1, dtype=torch.float64)

# Concatenate along data dimension
X_train = torch.cat([
    torch.cat([X_lf, task_lf], dim=-1),
    torch.cat([X_hf, task_hf], dim=-1)
], dim=0)
y_train = torch.cat([y_lf, y_hf], dim=0)

model = MultiTaskGP(X_train, y_train, task_feature=-1, rank=config['rank'])
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Predict at HF task
X_test_aug = torch.cat([
    torch.tensor(X_all[10+t:10+t+1], dtype=torch.float64),
    torch.ones(1, 1, dtype=torch.float64)  # task 1 = HF
], dim=-1)
model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_aug)
    mean = posterior.mean.squeeze().item()
    std = posterior.variance.sqrt().squeeze().item()
```

---

## Decision 4: SFGP Hyperparameter Space (50 Configurations)

**Decision**: 50 configurations spanning 4 kernel types × 4 noise lower bounds × 2 ARD settings + 18 additional configurations varying log-transform and input normalisation.  

**Core 32**: `{kernel: [matern05, matern15, matern25, rbf]} × {noise_lb: [1e-6, 1e-5, 1e-4, 1e-3]} × {ard: [True, False]}` — input normalisation on, no log-transform  
**Additional 18**: 3 kernels (matern15, matern25, rbf) × 3 noise bounds (1e-6, 1e-5, 1e-4) × log-transform on × ARD on; + kernel matern25 × noise 1e-5 × input_normalize off (×ARD True/False = 2) = 18 total  

**Axes covered**:
- `kernel_type`: `'matern05'`, `'matern15'`, `'matern25'`, `'rbf'`
- `noise_lb`: `1e-6`, `1e-5`, `1e-4`, `1e-3`
- `ard`: `True`, `False` (ARD = per-dimension lengthscale)
- `log_transform`: `True`, `False` (apply `log(|y| + eps)` to outputs before fitting)
- `input_normalize`: `True`, `False` (prepend a `Normalize` input transform)

---

## Decision 5: MFGP Hyperparameter Space (50 Configurations)

**Decision**: 50 configurations spanning 3 kernel types × 2 rank values × 4 noise lower bounds × 2 output-standardise settings + 2 extra rank-1 configs with tighter noise.  

**Core 48**: `{kernel: [matern15, matern25, rbf]} × {rank: [1, 2]} × {noise_lb: [1e-6, 1e-5, 1e-4, 1e-3]} × {output_standardize: [True, False]}`  
**Additional 2**: matern25 × rank=1 × noise_lb {5e-6, 5e-5} × output_standardize True  

**Axes covered**:  
- `kernel_type`: `'matern15'`, `'matern25'`, `'rbf'`
- `rank`: 1 or 2 (ICM rank; 1 = intrinsic coregionalization, 2 = richer inter-task structure)
- `noise_lb`: `1e-6`, `1e-5`, `1e-4`, `1e-3`
- `output_standardize`: `True` (apply `Standardize` output transform), `False`
- `step0_fallback`: always `'lf_sfgp'` — fixed strategy (not varied)

**Why rank ≤ 2**: With only 2 tasks (LF and HF), rank=1 is the standard ICM; rank=2 is the maximum meaningful rank before the model overfits the inter-task correlation on small data.

---

## Decision 6: SFGP Naming and Backward Compatibility

**Decision**: Name the surrogate "SFGP" throughout the new notebook, but build it using the identical `SingleTaskGP` + `ExactMarginalLogLikelihood` pattern already in the notebook. The existing `compute_metrics()` and `plot_prequential_results()` functions are moved to a "Utility Functions" section and reused without modification by both SFGP and MFGP evaluation loops.  
**Rationale**: Renaming to SFGP clearly contrasts with MFGP; it does not require any code change to the model or fitting logic.

---

## Decision 7: Visualisation Preservation

**Decision**: All three existing chart types are retained:  
1. Per-model per-step prediction chart (`plot_prequential_results()`) — called for best SFGP config and best MFGP config  
2. 3-panel bar chart (MAE / NLP / Coverage) — 2 bars (SFGP vs MFGP); same colour convention  
3. Horizontal ranked bar chart — 100 configs colour-coded (SFGP = blue `#2196F3`, MFGP = orange `#FF9800`)  
4. **New**: winner-detail section showing the overall best model's per-step predictions prominently with a title declaring the winner  

The final ranked table uses the same `pd.DataFrame` style, sorted by NLP, with a `Model` column set to `'SFGP'` or `'MFGP'`.

---

## Decision 8: Week 7 Data File Verification

**Decision**: The data-loading cell validates existence of `data/f2/updated_inputs - Week 7.npy` and `data/f2/updated_outputs - Week 7.npy` with an `assert os.path.exists(...)` guard before loading, to fail fast with a clear message if the files are missing.  
**Rationale**: Prevents silent dimension errors downstream; consistent with the constitution's "clearly explained" code requirement.

---

## Summary Table

| # | Decision | Chosen | Key Rejection |
|---|----------|--------|---------------|
| 1 | MFGP library | BoTorch `MultiTaskGP` | EmuKit/GPy not installed |
| 2 | Fidelity split | Index 0–9 = LF, 10–16 = HF | Random split would lose temporal meaning |
| 3 | Tensor format | Task-index augmented, `task_feature=-1` | Canonical BoTorch pattern |
| 4 | SFGP configs | 50: kernel × noise × ARD × log-xform | Wider grid than original 10-config baseline |
| 5 | MFGP configs | 50: kernel × rank × noise × standardize | rank ≤ 2 (max for 2-task problem) |
| 6 | SFGP naming | `SingleTaskGP` renamed "SFGP" | No code change to model |
| 7 | Visualisation | All 4 existing chart types preserved | Consistency with existing notebook |
| 8 | Data guard | `assert os.path.exists` | Early fail with clear message |
