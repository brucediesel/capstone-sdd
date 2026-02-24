# Data Model: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Feature**: 017-f7-nn-interior-penalty  
**Date**: 2025-02-24

---

## Entities

### Training Data
- **X_raw**: `np.ndarray` shape (37, 6) — cumulative input observations from Weeks 3–7
- **y_raw**: `np.ndarray` shape (37,) — cumulative output observations, all positive, range [0.003, 2.305]
- **X_mean, X_std**: `np.ndarray` shape (6,) each — per-dimension normalisation statistics
- **y_mean, y_std**: `float` — output normalisation statistics
- **X_norm**: `np.ndarray` shape (37, 6) — z-score normalised inputs
- **y_norm**: `np.ndarray` shape (37,) — z-score normalised outputs

### Neural Network Surrogate
- **model**: `SurrogateNN(nn.Module)` — architecture 6→5→5→1
  - Layer 1: `nn.Linear(6, 5)` + `nn.ReLU()` + `nn.Dropout(0.1)`
  - Layer 2: `nn.Linear(5, 5)` + `nn.ReLU()` + `nn.Dropout(0.1)`
  - Output: `nn.Linear(5, 1)`
  - Parameters: ~71 total (30+5 + 25+5 + 5+1)
- **losses**: `list[float]` — training loss per epoch (200 entries)
- **train_r2**: `float` — R² on training data (original scale)

### Candidate Evaluation
- **candidates**: `np.ndarray` shape (20000, 6) — uniform random in [0, 1]⁶
- **mc_preds_orig**: `np.ndarray` shape (50, 20000) — MC Dropout predictions (original scale)
- **ei**: `np.ndarray` shape (20000,) — Expected Improvement per candidate
- **interior_weight**: `np.ndarray` shape (20000,) — penalty weight per candidate, range [0.01, 1.0]
- **penalised_ei**: `np.ndarray` shape (20000,) — EI × w(x) per candidate

### Selection
- **best_idx**: `int` — index of selected candidate in candidates array
- **best_point**: `np.ndarray` shape (6,) — selected candidate coordinates
- **raw_best_idx**: `int` — index of best candidate by raw EI (for comparison)

### Visualisation
- **grad_importance**: `np.ndarray` shape (6,) — gradient-based feature importance (normalised to sum to 1)
- **top2**: `np.ndarray` shape (2,) — indices of two most important dimensions
- **grid_mu**: `np.ndarray` shape (50, 50) — NN mean prediction on 2D grid
- **grid_sigma**: `np.ndarray` shape (50, 50) — MC Dropout std on 2D grid
- **grid_penalty**: `np.ndarray` shape (50, 50) — interior penalty weight on 2D grid

---

## Comparison: F7 Week 6 vs Week 7

| Aspect | Week 6 | Week 7 |
|--------|--------|--------|
| Architecture | 6→64→32→1 (2,209 params) | 6→5→5→1 (71 params) |
| Dropout | 0.2 | 0.1 |
| Learning rate | 0.01 | 0.005 |
| Epochs | 500 | 200 |
| Acquisition | UCB (κ=0.5) | MC Dropout EI × interior penalty |
| Candidates | 20,000 | 20,000 |
| Interior penalty | None | w(x) = 0.01 + 0.99·∏sin(πxᵢ)^2 |
| Panel 3 | Feature importance bar chart | Interior penalty heatmap |

---

## Data Flow

```text
Week 7 .npy files
    │
    ▼
[Cell 50] Load & normalise → X_raw, y_raw, X_norm, y_norm
    │
    ▼
[Cell 51] Define & train NN → model, losses, train_r2
    │
    ▼
[Cell 52] MC Dropout EI + penalty → candidates, ei, interior_weight, penalised_ei, best_point
    │
    ▼
[Cell 53] Feature importance → grad_importance, top2
    │
    ▼
[Cell 54] 3-panel visualisation → grid_mu, grid_sigma, grid_penalty (figure)
    │
    ▼
[Cell 55] Convergence plot → running_best (figure)
    │
    ▼
[Cell 56] Submission query → formatted string + validation
```
