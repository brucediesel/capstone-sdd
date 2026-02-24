# Quickstart: F8 Week 7 — SFGP + qEI Acquisition

**Feature**: 018-f8-sfgp-qei
**Date**: 2025-02-24

---

## Prerequisites

1. **Python 3.11+** with: `torch`, `botorch`, `gpytorch`, `numpy`, `matplotlib`
2. **Data files**: `data/f8/updated_inputs - Week 7.npy` and `data/f8/updated_outputs - Week 7.npy` present
3. **No new dependencies** — uses BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib (all pre-installed)

## Running

1. Open `functions/f8/f8.ipynb` in VS Code or Jupyter.
2. Run all cells top-to-bottom (Ctrl+Shift+Enter or "Run All").
3. New cells 50–57 execute after existing Week 6 cells.

## Hyperparameters

| Parameter | Value | Cell | Tunable |
|-----------|-------|------|---------|
| Kernel | Matern 2.5 + ARD | 52 | Yes — change nu to 1.5 for rougher functions |
| Noise floor | 1e-07 | 52 | Yes — increase if GP fitting is unstable |
| `XI` | 0.01 | 52 | Yes — increase to require larger improvements |
| `MC_SAMPLES` | 256 | 52 | Yes — increase to 512 for smoother qEI |
| `NUM_RESTARTS` | 30 | 52 | Yes — increase for more thorough search |
| `RAW_SAMPLES` | 4096 | 52 | Yes — increase for better initial candidates |

## Verification Checklist

After running all cells, confirm:

- [ ] **Cell 51**: 47 samples × 8 dims loaded, all outputs positive
- [ ] **Cell 52**: GP fitted, 8 ARD lengthscales printed (all positive)
- [ ] **Cell 53**: qEI optimised, candidate with 8 coords in [0, 1], acq_value printed
- [ ] **Cell 53**: If acq_value = 0, fallback to posterior mean reported
- [ ] **Cell 54**: Feature importance bar chart for all 8 dimensions
- [ ] **Cell 55**: 3-panel figure — GP mean, GP std, EI surface
- [ ] **Cell 55**: Best point and candidate marked on panels
- [ ] **Cell 56**: Convergence plot with running best + weekly boundaries
- [ ] **Cell 57**: Submission query printed in x1-x2-...-x8 format
- [ ] **Cell 57**: Format validation passes (8 parts, each in [0, 0.999999])
- [ ] **No existing cells modified** — cells 1–49 unchanged

## Key Differences from Week 6

| Aspect | Week 6 (NN) | Week 7 (SFGP) |
|--------|-------------|----------------|
| Surrogate | NN 8→64→32→1 | SingleTaskGP Matern 2.5 + ARD |
| Uncertainty | MC Dropout (50 passes) | Exact GP posterior |
| Acquisition | UCB (κ=0.5) | qEI (256 MC, xi=0.01) |
| Feature importance | Gradient-based | ARD lengthscale inversion |
| Panel 3 | UCB surface | EI surface |
| Data | 46 samples | 47 samples |

## Troubleshooting

- **GP fitting warning / non-convergence**: Increase noise floor from 1e-07 to 1e-04 in cell 52. Also try increasing the number of fitting iterations.
- **qEI = 0 everywhere**: The GP is overconfident and predicts no improvement over best_f + xi = 9.963. The fallback selects the point with the highest posterior mean. Consider reducing xi to 0.0.
- **Memory error with 512 MC samples**: Stay at 256 (the default). This is sufficient for q=1 single-candidate optimisation.
- **optimize_acqf very slow**: The GP has a 47×47 kernel matrix — each posterior evaluation requires a solve. With 4096 raw + 30 restarts, expect 10–30 seconds. If too slow, reduce to 15 restarts and 2048 raw.
- **Candidate at boundary**: The GP genuinely predicts the best region is at the boundary. Unlike F5/F6/F7, there is no interior penalty for F8. Accept the prediction.
