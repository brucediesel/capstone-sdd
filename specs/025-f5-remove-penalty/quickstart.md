# Quickstart: F5 Week 9 — Kernel, Standardize & Raw Samples

**Branch**: `026-f5-kernel-standardize` | **Date**: 2026-03-09

## Prerequisites

- Branch `025-f5-remove-penalty` merged or available as base
- Python 3.14 (pyenv `sdd-dev`)
- BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib installed
- Data files present: `data/f5/updated_inputs - Week 9.npy`, `data/f5/updated_outputs - Week 9.npy`, `data/f5/initial_inputs.npy`, `data/f5/initial_outputs.npy`

## Setup

```bash
cd /Users/brucediesel/Development/capstone-sdd
git checkout 025-f5-remove-penalty
git checkout -b 026-f5-kernel-standardize
```

## Implementation Steps

### 1. Add Standardize import (cell 2)
Add `from botorch.models.transforms.outcome import Standardize` to import block.

### 2. Update hyperparameter table (cell 3)
- Change kernel row: `nu=2.5` → `nu=1.5`
- Change raw_samples row: `3000` → `5000`  
- Change outcome_transform row: `None` → `Standardize(m=1)`

### 3. Simplify transform code (cell 4)
- Remove manual z-score computation (`y_mean`, `y_std_val`, `y_std`)
- Pass `y_log` directly: `Y_train = torch.tensor(y_log, ...).unsqueeze(-1)`

### 4. Update GP training (cell 8)
- Change `MaternKernel(nu=2.5, ...)` → `MaternKernel(nu=1.5, ...)`
- Change `outcome_transform=None` → `outcome_transform=Standardize(m=1)`
- Update print statements to reflect new kernel/transform

### 5. Update acquisition (cell 10)
- Change `raw_samples=3000` → `raw_samples=5000`
- Simplify inverse transform: `np.expm1(posterior.mean.cpu().numpy())` (no manual z-score inverse)

### 6. Update visualisation (cell 12)
- Simplify inverse for grid mean/sigma
- Update suptitle: "Matérn-1.5" instead of "Matérn-5/2"

### 7. Update submission print (cell 16)
- Change surrogate description to reflect Matérn-1.5 and Standardize

### 8. Update LOO (cell 22)
- Change `MaternKernel(nu=2.5, ...)` → `MaternKernel(nu=1.5, ...)`
- Add `outcome_transform=Standardize(m=1)` to each LOO fold GP
- Remove manual z-score per fold
- Simplify inverse: `np.expm1(pred)`

### 9. Update title (cell 1)
- "Matérn-5/2" → "Matérn-1.5"

### 10. Update strategy (cell 23)
- Document the kernel change, Standardize adoption, and raw_samples increase

## Verification Checklist

| # | Check | Pass? |
|---|-------|-------|
| 1 | Branch `026-f5-kernel-standardize` created from `025-f5-remove-penalty` | |
| 2 | Notebook has 23 cells (unchanged count) | |
| 3 | `Standardize` import present in cell 2 | |
| 4 | No `y_mean`, `y_std_val`, `y_std` variables in cell 4 | |
| 5 | `MaternKernel(nu=1.5, ...)` in cells 8 and 22 | |
| 6 | `outcome_transform=Standardize(m=1)` in cells 8 and 22 | |
| 7 | `raw_samples=5000` in cell 10 | |
| 8 | All inverse transforms use `expm1(posterior.mean)` (no manual z-score inverse) | |
| 9 | All code cells execute without errors | |
| 10 | Submission query format: `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` with all values in [0, 0.999999] | |
| 11 | LOO MAE produces valid result | |
| 12 | Suptitle and prints reference "Matérn-1.5" and "Standardize(m=1)" | |
| 13 | Zero references to manual z-score variables (`y_mean`, `y_std_val`, `y_std`) except in `y_std` as LOO internal | |
| 14 | Results reviewed and further improvements suggested | |
