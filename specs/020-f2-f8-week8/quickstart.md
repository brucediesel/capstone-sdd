# Quickstart: F2–F8 Week 8 Notebooks

**Feature**: `020-f2-f8-week8` | **Branch**: `020-f2-f8-week8`

## Prerequisites

1. **Python environment**: `sdd-dev` conda environment with BoTorch, GPyTorch, PyTorch, numpy, matplotlib, scikit-learn
2. **Data files**: All 7 pairs of `updated_inputs - Week 8.npy` / `updated_outputs - Week 8.npy` in `./data/f2/` through `./data/f8/`
3. **Git branch**: `020-f2-f8-week8` checked out

## Running

Each notebook is self-contained. Open and run all cells sequentially:

```bash
# From repo root
cd functions/fX
# Open fX - week 8.ipynb in VS Code / Jupyter
# Select kernel: sdd-dev (Python 3.14.2)
# Run All Cells
```

## Expected Outputs Per Notebook

| Output | Description |
|--------|-------------|
| Data table | All observations with best highlighted |
| Model summary | Fitted surrogate diagnostics (lengthscales, noise, etc.) |
| Submission query | Formatted `0.xxxxxx-0.xxxxxx-...-0.xxxxxx` |
| Surrogate plot | 2D/3D contour or slice visualisation |
| Convergence plot | Running maximum with weekly boundaries |

## Per-Function Hyperparameter Tables

### F2 — SFGP + qLogNEI (2D)

| Parameter | Value |
|-----------|-------|
| Kernel | Matérn ν=1.5, ARD |
| Noise floor | 1e-3 |
| Input transform | Normalize(d=2) |
| Acquisition | qLogNEI, q=1 |
| num_restarts | 10 |
| raw_samples | 512 |
| Interior penalty | None |

### F3 — SFGP + qLogNEI (3D)

| Parameter | Value |
|-----------|-------|
| Kernel | Matérn ν=2.5, ARD (3 dims) |
| Noise floor | 1e-6 |
| Output transform | Manual z-score |
| Lengthscale init | 0.25 |
| Noise init | 0.1 |
| Outputscale init | 1.0 |
| MLL restarts | 15 |
| Acquisition | qLogNEI, q=1 |
| num_restarts | 10 |
| raw_samples | 512 |
| Interior penalty | None |

### F4 — MFGP + qLogNEI (4D + fidelity)

| Parameter | Value |
|-----------|-------|
| Model | SingleTaskMultiFidelityGP |
| Kernel | Matérn ν=2.5, ARD, LinearTruncated |
| Noise floor | 1e-4 |
| Output transform | Manual z-score |
| Fidelity column | Index 4, fixed at 1.0 |
| MLL restarts | 15 |
| Acquisition | qLogNEI, q=4, 64 MC fantasies |
| num_restarts | 20 |
| raw_samples | 512 |
| Selection | Highest posterior mean of 4 |

### F5 — GP + qLogNEI + Interior Penalty (4D)

| Parameter | Value |
|-----------|-------|
| Kernel | Matérn ν=2.5, ARD (4 dims) |
| Noise floor | 1e-6 |
| Output transform | log1p → z-score (manual) |
| Lengthscale init | 0.5 |
| Noise init | 0.1·Var(y_transformed) |
| Outputscale init | 1.0 |
| MLL restarts | 15 |
| Acquisition | qLogNEI, q=4, 512 QMC |
| num_restarts | 50 |
| raw_samples | 3000 |
| Selection | Mean > median, farthest from data |
| STEEPNESS | 1.0 |
| FLOOR | 0.01 |

### F6 — SFGP + qLogNEI + Interior Penalty (5D)

| Parameter | Value |
|-----------|-------|
| Kernel | Matérn ν=1.5, ARD (5 dims), ScaleKernel |
| Noise floor | 1e-2 |
| Noise init | 0.2 |
| Lengthscale init | 0.5 |
| Outputscale init | 1.0 |
| Output transform | Standardize(m=1) (auto) |
| MLL restarts | 15 |
| Acquisition | qLogNEI, q=4, 512 QMC |
| num_restarts | 50 |
| raw_samples | 3000 |
| Feasibility | x4 ∈ [0.10, 1.0], others ∈ [0.01, 1.0] |
| Selection | Rank-based scoring, farthest from data |
| STEEPNESS | 1.0 |
| FLOOR | 0.01 |

### F7 — Neural Network + MC Dropout EI + Interior Penalty (6D)

| Parameter | Value |
|-----------|-------|
| Architecture | 6→5→5→1 (71 params) |
| Activation | ReLU |
| Dropout | 0.1 |
| Optimiser | Adam, lr=0.005 |
| Epochs | 200 |
| Loss | MSE |
| Normalisation | z-score (X and y) |
| MC_SAMPLES | 50 |
| N_CANDIDATES | 20 000 |
| STEEPNESS | 0.1 |
| FLOOR | 0.01 |

### F8 — SFGP + qEI (8D)

| Parameter | Value |
|-----------|-------|
| Kernel | Matérn ν=2.5, ARD (8 dims) |
| Noise floor | 1e-7 |
| Output transform | Standardize(m=1) (auto) |
| XI | 0.01 |
| MC_SAMPLES | 256 |
| Acquisition | qEI, q=1 |
| num_restarts | 30 |
| raw_samples | 4096 |
| Fallback | Highest posterior mean if qEI=0 |

## Validation

After execution, verify each notebook:

1. ✅ All cells execute without error
2. ✅ Correct sample count loaded (see data-model.md)
3. ✅ Submission query in valid format
4. ✅ Convergence plot shows running maximum
5. ✅ Surrogate visualisation is legible
6. ✅ Original `fX.ipynb` is not modified (`git diff --stat functions/fX/fX.ipynb`)
