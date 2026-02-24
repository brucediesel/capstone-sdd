# Quickstart: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Feature**: 017-f7-nn-interior-penalty  
**Date**: 2025-02-24

---

## Prerequisites

1. **Python 3.11+** with: `torch`, `numpy`, `matplotlib`
2. **Data files**: `data/f7/updated_inputs - Week 7.npy` and `data/f7/updated_outputs - Week 7.npy` present
3. **No new dependencies** — uses only PyTorch, NumPy, Matplotlib (already installed)

## Running

1. Open `functions/f7/f7.ipynb` in VS Code or Jupyter.
2. Run all cells top-to-bottom (Ctrl+Shift+Enter or "Run All").
3. New cells 50–57 execute after existing Week 6 cells.

## Hyperparameters

| Parameter | Value | Cell | Tunable |
|-----------|-------|------|---------|
| Architecture | 6→5→5→1 | 52 | Yes — adjust node count per layer |
| `LEARNING_RATE` | 0.005 | 52 | Yes — decrease if loss oscillates |
| `DROPOUT` | 0.1 | 52 | Yes — increase for more MC variance (max 0.3 for 5-node layers) |
| `EPOCHS` | 200 | 52 | Yes — increase if loss hasn't plateaued |
| `MC_SAMPLES` | 50 | 53 | Yes — increase for smoother EI estimates |
| `N_CANDIDATES` | 20,000 | 53 | Yes — increase for finer search |
| `STEEPNESS` | 1.0 | 53 | Yes — increase for narrower interior band |
| `FLOOR` | 0.01 | 53 | Yes — increase to soften boundary penalty |

## Verification Checklist

After running all cells, confirm:

- [ ] **Cell 51**: 37 samples × 6 dims loaded, all outputs positive
- [ ] **Cell 52**: Training loss curve displayed (log scale), R² > 0 printed
- [ ] **Cell 53**: EI values computed, interior penalty printed, best candidate selected
- [ ] **Cell 53**: Penalty effect reported (whether selection changed vs raw EI)
- [ ] **Cell 54**: Feature importance printed for all 6 dimensions
- [ ] **Cell 55**: 3-panel figure — NN mean, MC uncertainty, interior penalty heatmap
- [ ] **Cell 55**: Best point marked with star on all panels
- [ ] **Cell 56**: Convergence plot with running best + IP-selected predicted mean
- [ ] **Cell 57**: Submission query printed in `x1-x2-...-x6` format
- [ ] **Cell 57**: Format validation passes (6 parts, each in [0, 0.999999])
- [ ] **No existing cells modified** — cells 1–49 unchanged

## Key Differences from Week 6

| Aspect | Week 6 | Week 7 |
|--------|--------|--------|
| Architecture | 6→64→32→1 (2,209 params) | 6→5→5→1 (71 params) |
| Dropout | 0.2 | 0.1 |
| Learning rate | 0.01 | 0.005 |
| Epochs | 500 | 200 |
| Acquisition | UCB (κ=0.5) | MC Dropout EI × interior penalty |
| Panel 3 | Feature importance bar chart | Interior penalty heatmap |
| Data | 36 samples | 37 samples |

## Troubleshooting

- **Low R² (< 0.0)**: The 71-parameter network may be too small for 6D data. Check the training loss curve — if it hasn't converged, increase epochs to 500 or increase learning rate to 0.01.
- **All EI = 0**: The NN predicts no improvement anywhere. The fallback selects the most interior candidate. Consider reducing STEEPNESS to 0.5 or increasing MC_SAMPLES to 100.
- **MC variance very small**: With dropout=0.1, uncertainty estimates are naturally small. This is expected. EI still differentiates candidates via the mean prediction.
- **Best point at boundary despite penalty**: The EI at that boundary point is so much higher than interior alternatives that even after penalty it wins. This means the model genuinely believes this is the best region.
