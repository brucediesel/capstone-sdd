# Quickstart: F3 BART Memory Reduction

**Notebook**: `functions/f3/preq-eval-f3.ipynb`  
**Environment**: `sdd-dev` (Python 3.14.2)  
**Branch**: `005-week7-pe-surrogates`

## What this changes

Three cells in `preq-eval-f3.ipynb` are modified to reduce BART memory usage:

1. **BART default run call** — `m_trees 50→20`, `draws 500→200`, `tune 200→100`
2. **BART HP sweep markdown** — updated parameter range documentation
3. **BART HP sweep configs** — 8 configurations all within `m_trees ≤ 20`, `draws ≤ 200`

All GP and Random Forest cells are **untouched**.

## Running the fix

### 1. Open the notebook

```
functions/f3/preq-eval-f3.ipynb
```

Select kernel: **sdd-dev (Python 3.14.2)**

### 2. Run cells in order

Execute from the **BART section** onward (or restart and run all):

| Step | Cell description | Expected outcome |
|------|-----------------|-----------------|
| BART definition | `bart_prequential_evaluation()` function | Prints `bart_prequential_evaluation() defined.` |
| BART default run | Single call with `m_trees=20, draws=200, tune=100` | 7 step predictions printed, MAE/NLP/Coverage shown — **no crash** |
| BART default viz | `plot_prequential_results(...)` | Prediction vs actual plot rendered |
| BART HP sweep | 8-config loop | 8 rows in results table, no NaN from OOM |
| Best BART | `bart_hp_df['NLP'].idxmin()` | Prints best config label |

### 3. Verify success criteria

```python
# SC-004: Check default parameters
# Look for this line in the BART default run cell:
bart_prequential_evaluation(X_all, y_all, N_INIT, m_trees=20, draws=200, tune=100)

# SC-005: Check all sweep configs are within bounds
assert all(c['m_trees'] <= 20 for c in bart_configs)
assert all(c['draws'] <= 200 for c in bart_configs)

# SC-003: Confirm GP/RF cells unchanged after commit
# git diff HEAD~1 HEAD -- functions/f3/preq-eval-f3.ipynb
# Should show changes only in BART-related cells
```

## Expected BART runtime

With `m_trees=20, draws=200, tune=100, chains=4`, each of the 7 prequential
steps takes approximately **1–3 minutes** on a 4-core CPU. Total runtime for
the BART section (default + 8-sweep) is approximately **1–2 hours**.

## If BART still runs out of memory

If OOM still occurs after this change, reduce further:
- Set `m_trees=10` and `draws=100` in the default run call
- The HP sweep's exception handler will catch remaining failures gracefully and produce `NaN` rows rather than crashing
