# Quickstart: 006-sfgp-mfgp-pe

**Branch**: `006-sfgp-mfgp-pe`  
**Notebook**: `functions/f2/preq-eval-f2.ipynb`

---

## Prerequisites

All dependencies are pre-installed in the existing capstone environment:

| Package | Version | Purpose |
|---------|---------|---------|
| `botorch` | 0.16.1 | `SingleTaskGP`, `MultiTaskGP`, `fit_gpytorch_mll` |
| `gpytorch` | 1.15.1 | `MaternKernel`, `RBFKernel`, `ScaleKernel`, `GaussianLikelihood` |
| `numpy` | any | Array ops, `.npy` file loading |
| `pandas` | any | Results DataFrames |
| `matplotlib` | any | All visualisations |
| `torch` | any | Tensor construction |

**Data files required** (must exist before running):
```
data/f2/updated_inputs - Week 7.npy    # shape (17, 2)
data/f2/updated_outputs - Week 7.npy   # shape (17,)
```

---

## Running the Notebook

1. Open `functions/f2/preq-eval-f2.ipynb` in VS Code or Jupyter Lab  
2. Select the Python kernel that has BoTorch installed  
3. **Kernel → Restart & Run All** (or run cells top to bottom in order)

> The notebook is designed to be run fully from top to bottom on a clean kernel. Running cells out of order will produce `NameError` exceptions.

---

## Expected Runtime

| Section | Approx. Time |
|---------|-------------|
| Imports + data load | < 5 seconds |
| SFGP default run (1 config × 7 steps) | ~10 seconds |
| SFGP 50-config sweep | 3–8 minutes |
| MFGP default run (1 config × 7 steps) | ~20 seconds |
| MFGP 50-config sweep | 10–20 minutes |
| Comparison + visualisation | < 30 seconds |
| **Total** | **~15–30 minutes** |

> If the full sweep is too slow, temporarily reduce `sfgp_configs` and `mfgp_configs` to a subset for development. Both lists can be sliced: `sfgp_configs[:10]`.

---

## Expected Outputs

After running all cells:

1. **SFGP default prediction plot** — 3-panel figure (predictions vs actuals, absolute errors, NLP per step)  
2. **SFGP 50-config results table** — `sfgp_hp_df` displayed as HTML DataFrame  
3. **Best SFGP** — printed: label, MAE, NLP, Coverage  
4. **Best SFGP prediction plot** — same 3-panel style for best config  
5. **MFGP default prediction plot**  
6. **MFGP 50-config results table** — `mfgp_hp_df`  
7. **Best MFGP** — printed  
8. **Best MFGP prediction plot**  
9. **2-way comparison bar chart** — 3-panel (MAE / NLP / Coverage), 2 bars per panel  
10. **100-config sensitivity chart** — horizontal bars, colour-coded by family  
11. **Full ranked results table** — 100 rows sorted by NLP  
12. **Winner detail prediction plot** — best overall surrogate highlighted  
13. **Conclusions cell** — recommendation sentence

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError` on data load | Week 7 `.npy` files missing | Ensure `data/f2/updated_inputs - Week 7.npy` exists |
| `NameError: best_sfgp not defined` | Cells run out of order | Restart kernel and run all |
| >5 NaN rows in `sfgp_hp_df` | Fitting failures on short data | Expected; check that kernel `'matern05'` configs have higher failure rates on 10-pt starts |
| MFGP step 0 always predicts poorly | Step-0 uses LF-only SFGP fallback | Normal; step-0 prediction is less informed than steps 1–6 |
| `RuntimeError: singular matrix` in MFGP | Very low `noise_lb` on HF with 1–2 points | Config records NaN and continues; review configs with `noise_lb=1e-6` and `rank=2` |
| Plot windows don't appear | Inline backend not set | Add `%matplotlib inline` as the first cell, or ensure notebook is running in Jupyter |

---

## Key Variables Reference

| Variable | Type | Contents |
|----------|------|----------|
| `X_all` | `np.ndarray (17,2)` | All 17 input points |
| `y_all` | `np.ndarray (17,)` | All 17 output values |
| `N_INIT` | `int` = 10 | Initial training set size |
| `sfgp_configs` | `list[dict]` length 50 | All SFGP hyperparameter configs |
| `mfgp_configs` | `list[dict]` length 50 | All MFGP hyperparameter configs |
| `sfgp_hp_df` | `pd.DataFrame (50,4)` | SFGP sweep results: label, MAE, NLP, Coverage_95 |
| `mfgp_hp_df` | `pd.DataFrame (50,4)` | MFGP sweep results |
| `best_sfgp` | `pd.Series` | Row from `sfgp_hp_df` with minimum NLP |
| `best_mfgp` | `pd.Series` | Row from `mfgp_hp_df` with minimum NLP |
| `comparison_df` | `pd.DataFrame (2,5)` | Head-to-head: Model, Configuration, MAE, NLP, Coverage_95 |
