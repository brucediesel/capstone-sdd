# Cell Contracts: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Feature**: 017-f7-nn-interior-penalty  
**Date**: 2025-02-24  
**Target**: `functions/f7/f7.ipynb` — append cells 50–57 after existing cell 49

---

## Cell 50 (Markdown) — Section Header

**Type**: Markdown  
**Purpose**: Week 7 section header with hyperparameter documentation table

**Content requirements**:
- Title: "## Week 7 — Neural Network + NEI with Interior Penalty"
- Brief explanation of the approach: compact NN surrogate, MC Dropout EI, multiplicative interior penalty
- Hyperparameter table with columns: Parameter, Value, Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | 6→5→5→1 (2L×5N) | User-specified; compact for 37 samples |
| Learning rate | 0.005 | User-specified; moderate for Adam |
| Dropout | 0.1 | Reduced from 0.2; 5-node layers need minimal dropout |
| Epochs | 200 | 71 params converge fast; avoids memorisation |
| MC samples | 50 | Sufficient for EI ranking |
| Candidates | 20,000 | Nearest-neighbour ≈ 0.19 in 6D |
| STEEPNESS | 1.0 | Consistent with F5/F6 |
| FLOOR | 0.01 | Prevents zero-weight at boundaries |

---

## Cell 51 (Code) — Load Data & Set Up

**Type**: Code  
**Depends on**: Data files in `data/f7/`

**Inputs**:
- `../../data/f7/updated_inputs - Week 7.npy`
- `../../data/f7/updated_outputs - Week 7.npy`

**Outputs (kernel variables)**:
- `X_raw`: ndarray (37, 6)
- `y_raw`: ndarray (37,)
- `X_mean`, `X_std`: ndarray (6,) each
- `y_mean`, `y_std`: float
- `X_norm`: ndarray (37, 6) — z-score normalised
- `y_norm`: ndarray (37,) — z-score normalised
- `X_tensor`: torch.Tensor (37, 6, float32)
- `y_tensor`: torch.Tensor (37, 1, float32)
- `hyperparam_names`: list of 6 strings

**Validation**:
- Print: sample count, dimensions, output range, best observed
- Assert: X_raw.shape == (37, 6), y_raw.shape == (37,)
- Assert: all outputs positive

---

## Cell 52 (Code) — Define & Train Neural Network

**Type**: Code  
**Depends on**: Cell 51 (X_tensor, y_tensor)

**Constants defined**:
- `LEARNING_RATE = 0.005`
- `EPOCHS = 200`
- `DROPOUT = 0.1`

**Outputs (kernel variables)**:
- `model`: SurrogateNN instance (trained)
- `losses`: list of training losses
- `train_r2`: float (R² on training data)

**Behaviour**:
- Define `SurrogateNN(nn.Module)`: 6→5→5→1 with ReLU + Dropout(0.1)
- Train with Adam (lr=0.005), MSE loss, 200 epochs
- Print progress every 40 epochs
- Plot training loss curve (log scale)
- Compute and print training R² (original scale)

---

## Cell 53 (Code) — MC Dropout EI + Interior Penalty

**Type**: Code  
**Depends on**: Cell 52 (model), Cell 51 (X_mean, X_std, y_mean, y_std, y_raw)

**Constants defined**:
- `MC_SAMPLES = 50`
- `N_CANDIDATES = 20_000`
- `STEEPNESS = 1.0`
- `FLOOR = 0.01`

**Outputs (kernel variables)**:
- `candidates`: ndarray (20000, 6) — uniform in [0,1]⁶
- `ei`: ndarray (20000,) — raw MC Dropout EI
- `interior_weight`: ndarray (20000,) — penalty weights
- `penalised_ei`: ndarray (20000,) — EI × w(x)
- `best_idx`: int — index of best penalised candidate
- `best_point`: ndarray (6,) — selected candidate coordinates
- `raw_best_idx`: int — index of best raw EI candidate
- `mu`: ndarray (20000,) — MC mean predictions (original scale)
- `sigma`: ndarray (20000,) — MC std (original scale)

**Behaviour**:
1. Generate 20,000 random candidates in [0,1]⁶
2. Normalise candidates, run 50 MC Dropout forward passes
3. Un-normalise predictions: `mc_preds_orig = mc * y_std + y_mean`
4. Compute EI: `ei = mean(max(mc_preds_orig - y_best, 0), axis=0)`
5. Compute interior weight: `w = FLOOR + (1-FLOOR) * prod(sin(π*x)^(2*S), axis=1)`
6. Compute penalised EI: `penalised_ei = ei * interior_weight`
7. Select best: `best_idx = argmax(penalised_ei)`
8. Fallback: if max(penalised_ei) == 0, select `argmax(interior_weight)` (exploratory)
9. Print: comparison table (raw EI rank, penalty, penalised EI), penalty effect

**Validation**:
- Assert: interior_weight in [FLOOR, 1.0]
- Assert: best_point coords in [0, 1]
- Print whether penalty changed the selection

---

## Cell 54 (Code) — Feature Importance via Gradients

**Type**: Code  
**Depends on**: Cell 52 (model), Cell 51 (X_norm)

**Outputs (kernel variables)**:
- `grad_importance`: ndarray (6,) — normalised to sum to 1
- `top2`: ndarray (2,) — indices of two most important dims

**Behaviour**:
- Compute mean absolute gradient of model output w.r.t. each input dimension
- Normalise to sum to 1
- Print feature importance with bar indicators

---

## Cell 55 (Code) — 3-Panel Visualisation

**Type**: Code  
**Depends on**: Cell 52 (model), Cell 53 (best_point, penalised_ei, interior_weight, mu, sigma), Cell 54 (top2, grad_importance), Cell 51 (X_raw, y_raw, X_mean, X_std, y_mean, y_std)

**Outputs**: Figure (3 panels), no new kernel variables

**Behaviour**:
1. Build 50×50 grid on top-2 dimensions, fix other dims at best-observed values
2. MC Dropout predictions on grid → grid_mu, grid_sigma
3. Compute interior penalty on grid → grid_penalty
4. Panel 1: NN mean heatmap (viridis), observed data scatter, best_point star
5. Panel 2: MC Dropout uncertainty heatmap (YlOrRd), observed data scatter
6. Panel 3: Interior penalty heatmap (RdYlGn), best_point star
7. Suptitle: "Function 7 — NN Surrogate (Week 7, NEI + Interior Penalty)"

---

## Cell 56 (Code) — Convergence Plot

**Type**: Code  
**Depends on**: Cell 51 (y_raw), Cell 53 (best_point, mu, best_idx)

**Outputs**: Figure, no new kernel variables

**Behaviour**:
- Plot running best across all 37 observations
- Add horizontal line for IP-selected candidate's predicted mean
- Add dashed line for raw-EI best candidate's predicted mean (comparison)
- Boundary markers for initial/weekly data boundaries
- Print: running best, IP-selected mean, whether penalty changed selection

---

## Cell 57 (Code) — Submission Query

**Type**: Code  
**Depends on**: Cell 53 (best_point, best_idx, raw_best_idx, penalised_ei, ei, interior_weight, mu)

**Outputs**: Printed submission query + validation

**Behaviour**:
1. Clip best_point to [0, 0.999999]
2. Format as `x1-x2-...-x6` with 6 decimal places
3. Validate: 6 parts, each parseable as float in [0, 0.999999]
4. Print: submission query, hyperparameter metadata (architecture, lr, dropout, epochs, STEEPNESS, FLOOR)
5. Print: penalty effect (raw vs penalised selection)
6. Print: per-dimension coordinates with names
