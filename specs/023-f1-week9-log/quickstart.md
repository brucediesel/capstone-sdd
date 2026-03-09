# Quickstart: F1 Week 9 — Hurdle Model with log Transform (No Penalties)

**Date**: 2026-03-09  
**Feature**: 023-f1-week9-log

## Prerequisites

1. Week 9 data files must exist:
   - `./data/f1/updated_inputs - Week 9.npy`
   - `./data/f1/updated_outputs - Week 9.npy`
   - If missing, run the results processing notebook at `./functions/results/` first.

2. Python environment with:
   - numpy
   - scikit-learn
   - matplotlib
   - scipy
   - pandas (for tabular display)

## Implementation Steps

### Step 1: Create notebook `./functions/f1/f1 - week 9.ipynb`

Use `f1 - week 8.ipynb` as a structural reference and make the following changes:

### Step 2: Change data loading paths

Update file paths to `Week 9`. Set expected sample count to 19 (10 initial + 9 submissions).

### Step 3: Change log1p to log (THE KEY CHANGE)

In Stage 2 RF training cell, change:
```python
# BEFORE (Week 9):
y_pos_log = np.log1p(y_pos)

# AFTER (Week 9):
y_pos_log = np.log(y_pos)
```

### Step 4: Change back-transformation

In acquisition and visualisation cells, change:
```python
# BEFORE (Week 9):
mu_cand = np.expm1(mu_log_cand)

# AFTER (Week 9):
mu_cand = np.exp(mu_log_cand)
```

**Note**: Per clarification, contour panels display in log-space. The `exp()` back-transform is only needed if displaying original-space values (e.g., in the data table or convergence plot).

### Step 5: Remove local penalization and interior penalty

Remove the local penalization (Gaussian mask) and interior penalty (sinusoidal boundary suppression) from the acquisition function. The acquisition simplifies to:
```python
a_final = a_ucb  # No penalty(x) or interior(x) multiplication
```
Remove the `PENALTY_RADIUS`, `STEEPNESS`, and `FLOOR` hyperparameters. Change KAPPA from 3.0 to 0.5 (exploitation-focused — log transform gives meaningful surrogate signal, only 4 submissions remain).

### Step 6: Update visualisation labels

Update panel titles and axis labels to reference "log" instead of "log1p". Contour panels display log-space values directly (range ~-565 to ~-35).

### Step 7: Update submission count

- Week title: "Week 9"
- Expected samples: 19 (10 initial + 9 submissions)
- Submission points highlighted: indices 10-18

### Step 8: Verify and run

Execute all cells. Confirm:
- [ ] Data loads successfully (19 samples, 2D)
- [ ] RF trains on log-space targets (values ~-565 to ~-35)
- [ ] 3-panel contour shows meaningful variation in log-space
- [ ] No local penalization or interior penalty in acquisition
- [ ] Panel 3 titled "Acquisition (Weighted UCB)"
- [ ] Submission query formatted as `0.xxxxxx-0.xxxxxx`
- [ ] All values clipped to [0.0, 0.999999]

## Key Differences from Week 8

| Aspect | Week 8 (log1p) | Week 9 (log, no penalties) |
|--------|----------------|---------------------------|
| Data files | Week 8 | Week 9 |
| Sample count | 18 | 19 |
| RF training targets | `np.log1p(y_pos)` | `np.log(y_pos)` |
| RF target range | ~[1.9e-245, 7.7e-16] (≈ inputs) | ~[-563.5, -34.8] |
| Back-transform | `np.expm1(mu)` | `np.exp(mu)` |
| Contour display | Back-transformed (near-zero, flat) | Log-space (readable variation) |
| Local penalization | Yes (PENALTY_RADIUS=0.15) | Removed |
| Interior penalty | Yes (STEEPNESS=0.1, FLOOR=0.01) | Removed |
| Acquisition | UCB × penalty × interior | Raw weighted UCB |
| Panel 3 title | "Penalised Acquisition" | "Acquisition (Weighted UCB)" |
| KAPPA | 3.0 | 0.5 (exploitation-focused) |
