# Quickstart: ARD Relevance Visualisation

**Feature**: 034-ard-relevance-visualisation
**Branch**: `034-ard-relevance-visualisation`

## What This Feature Does

Adds an ARD (Automatic Relevance Determination) feature relevance section to each of the 8 week 11 review notebooks (F1–F8). Each section:
1. Fits a Gaussian Process on the function's data
2. Extracts per-dimension lengthscale parameters
3. Prints a raw lengthscale table
4. Displays a normalised relevance bar chart

## Prerequisites

- Python environment: `sdd-dev` (pyenv)
- Required packages: `botorch`, `gpytorch`, `torch`, `numpy`, `matplotlib` (all pre-installed)
- Data files: `updated_inputs - Week 11.npy` and `updated_outputs - Week 11.npy` in each `data/fX/` folder

## How to Verify

1. Activate the environment: `pyenv activate sdd-dev`
2. Open any week 11 notebook, e.g., `functions/f1/f1 - week 11.ipynb`
3. Run All Cells
4. Scroll to the bottom — the last 3 cells should show:
   - A markdown section "ARD Feature Relevance Analysis"
   - A printed table of raw lengthscale values
   - A horizontal bar chart of normalised relevance scores

## Per-Function Details

| Function | Dims | Kernel | Output Transform | Special Notes |
|----------|------|--------|-----------------|---------------|
| F1 | 2 | Matérn-2.5 | log(y + ε) | Log scale for outputs |
| F2 | 2 | Matérn-2.5 | Standardize | — |
| F3 | 3 | Matérn-2.5 | y - y_min (shift) | Negative outputs shifted |
| F4 | 4 | Matérn-2.5 | Standardize | — |
| F5 | 4 | Matérn-1.5 | log(y) + Standardize | Strictly positive outputs |
| F6 | 5 | Matérn-1.5 | Standardize | Bars labelled with ingredient names |
| F7 | 6 | Matérn-2.5 | Standardize | **Diagnostic GP** (production is NN) |
| F8 | 8 | Matérn-2.5 | Standardize | Highest dimensionality |

## Files Modified

- `functions/f1/f1 - week 11.ipynb` — 3 cells appended
- `functions/f2/f2 - week 11.ipynb` — 3 cells appended
- `functions/f3/f3 - week 11.ipynb` — 3 cells appended
- `functions/f4/f4 - week 11.ipynb` — 3 cells appended
- `functions/f5/f5 - week 11.ipynb` — 3 cells appended
- `functions/f6/f6 - week 11.ipynb` — 3 cells appended
- `functions/f7/f7 - week 11.ipynb` — 3 cells appended
- `functions/f8/f8 - week 11.ipynb` — 3 cells appended

No new files created. No existing cells modified.
