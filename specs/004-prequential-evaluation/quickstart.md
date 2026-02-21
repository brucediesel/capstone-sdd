# Quickstart: F6 Prequential Evaluation

**Feature**: 004-prequential-evaluation (F6 extension)

## Prerequisites

- Python 3.14.2 (pyenv virtualenv `sdd-dev`)
- BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch
- Jupyter notebook support (VS Code or JupyterLab)

## Running the Notebook

1. **Activate environment**:
   ```bash
   pyenv activate sdd-dev
   ```

2. **Open the notebook**:
   ```
   functions/f6/preq-eval-f6.ipynb
   ```

3. **Run All Cells** (top to bottom):
   - Cell 1–7: Setup (imports, data loading, helpers)
   - Cell 8–15: NN evaluation (45 configs → nn_hp_df)
   - Cell 16–21: MFGP evaluation (50 configs → mfgp_hp_df)
   - Cell 22–27: SFGP evaluation (40 configs → sfgp_hp_df)
   - Cell 28–29: 3-way comparison (best NN vs SFGP vs MFGP)
   - Cell 30–31: Best model visualisation
   - Cell 32–33: Sensitivity charts (135 configs)
   - Cell 34–35: Full ranked table (135 rows)
   - Cell 36: Conclusions

4. **Expected runtime**: ~5–10 minutes (135 configs × 6 steps each)

## Expected Outputs

| Output | Shape/Size | Description |
|--------|------------|-------------|
| `nn_hp_df` | 45 rows × 4 cols | NN hyperparameter results |
| `sfgp_hp_df` | 40 rows × 4 cols | SFGP hyperparameter results |
| `mfgp_hp_df` | 50 rows × 4 cols | MFGP hyperparameter results |
| `comparison_df` | 3 rows × 5 cols | Best of each family side-by-side |
| `full_ranked` | 135 rows × 5 cols | All configs ranked by NLP |
| Bar chart | 1×3 figure | 3-way comparison (MAE, NLP, Coverage) |
| 3-panel plot | 1 figure | Best model prequential evaluation |
| Sensitivity charts | 1×3 figure | 135 configs by NLP, MAE, Coverage |

## Data Files

```
data/f6/updated_inputs - Week 6.npy    # shape: (26, 5)
data/f6/updated_outputs - Week 6.npy   # shape: (26,)
```
