# Quickstart: F5 Week 9 — Remove Interior Penalty

**Date**: 2026-03-09  
**Feature**: 025-f5-remove-penalty

## Prerequisites

1. Week 9 data files must exist:
   - `./data/f5/updated_inputs - Week 9.npy`
   - `./data/f5/updated_outputs - Week 9.npy`

2. Python environment with:
   - numpy, torch, copy, warnings
   - botorch (SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf, SobolQMCNormalSampler)
   - gpytorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood, GreaterThan)
   - matplotlib, scipy

## Implementation Steps

### Step 1: Remove penalty constants from hyperparameters cell

In the constants cell (cell 4), **remove** these three lines:
```python
# REMOVE:
STEEPNESS = 1.0
FLOOR = 0.01
EPS_BOUND = 0.005  # tighten [0, 1] → [ε, 1-ε] during penalised optimisation
```

Keep all other constants (`N_INITIAL`, `N_TOTAL`, `N_DIMS`, `N_SUBMISSIONS`, `STALLING_WINDOW`, `STALLING_REL_THRESHOLD`, `N_RESTARTS`, `DIM`).

### Step 2: Remove Step 4 cells entirely

Delete both cells that make up Step 4:
- The **markdown cell** explaining the in-loop interior penalty (contains the mathematical explanation of 4x(1-x), gradient comparison table, etc.)
- The **code cell** containing the `PenalisedAcquisition` class definition, `penalised_nei` wrapping, `BOUNDS_IP` tightened bounds, and the penalised `optimize_acqf` call with diagnostic output

### Step 3: Remove Step 6 penalty visualisation cells

Delete both cells that make up Step 6:
- The **markdown cell** with header "Step 6: Three-Colour Interior Penalty Visualisation (3-Panel)"
- The **code cell** rendering the 3-panel penalty contour plots (GP Mean, 4x(1-x) Penalty, Penalised Mean)

### Step 4: Update Step 5 surrogate visualisation

In the Step 5 visualisation code cell, replace all references to `next_x_ip` with `best_point`:

```python
# BEFORE:
axes[0].scatter(next_x_ip[top2[0]], next_x_ip[top2[1]], ...)
axes[1].scatter(next_x_ip[top2[0]], next_x_ip[top2[1]], ...)

# AFTER:
axes[0].scatter(best_point[top2[0]], best_point[top2[1]], ...)
axes[1].scatter(best_point[top2[0]], best_point[top2[1]], ...)
```

Update the suptitle to remove "IP":
```python
# BEFORE:
plt.suptitle("F5 — GP Matérn-5/2 ARD Surrogate (Week 9, NEI q=4 + IP)", fontsize=14)

# AFTER:
plt.suptitle("F5 — GP Matérn-5/2 ARD Surrogate (Week 9, NEI q=4)", fontsize=14)
```

### Step 5: Update Step 8 submission cell

Simplify the submission cell to show only the base NEI submission. Remove the IP submission block and penalty parameter diagnostics:

```python
# REMOVE the entire IP submission block:
# ── In-Loop Interior Penalty Submission ──
submission_ip = np.clip(next_x_ip, 0.0, 0.999999)
query_ip = "-".join(f"{v:.6f}" for v in submission_ip)
# ... all IP print statements ...

# REMOVE penalty parameter output:
print(f"Interior Penalty: STEEPNESS={STEEPNESS}, FLOOR={FLOOR}, bounds=[{EPS_BOUND}, {1-EPS_BOUND}]")

# KEEP the base NEI submission and format validation
```

### Step 6: Update title and hyperparameter table

Update the title markdown cell:
```markdown
<!-- BEFORE: -->
## Week 9 — GP Matérn-5/2 + qLogNEI + Interior Penalty (4D)

<!-- AFTER: -->
## Week 9 — GP Matérn-5/2 + qLogNEI (4D)
```

Remove "Interior Penalty" from the description paragraph and delete rows 16-17 from the hyperparameter table:
```markdown
<!-- REMOVE these rows: -->
| 16 | IP STEEPNESS | 1.0 | Boundary suppression |
| 17 | IP FLOOR | 0.01 | Minimum penalty at boundary |
```

### Step 7: Update strategy recommendations

Update the strategy markdown cell at the end of the notebook to note that interior penalty was evaluated and removed. Remove any recommendations about adjusting STEEPNESS.

### Step 8: Verify and run

Execute all cells. Confirm:
- [ ] Data loads successfully (29 samples, 4D)
- [ ] GP trains with 15-restart MLL (log1p → z-score, outcome_transform=None)
- [ ] Base NEI acquisition runs with q=4, 50 restarts, 3000 raw samples
- [ ] Distance-based selection produces a candidate
- [ ] Submission query formatted as `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx`
- [ ] All values clipped to [0.0, 0.999999]
- [ ] Surrogate visualisation renders 3-panel plot without penalty reference
- [ ] No penalty visualisation panel exists
- [ ] Convergence plot renders correctly
- [ ] Convergence metrics and exploration spread cells execute unchanged
- [ ] LOO cross-validation completes without errors
- [ ] No references to `PenalisedAcquisition`, `penalised_nei`, `STEEPNESS`, `FLOOR`, `EPS_BOUND`, or `BOUNDS_IP` remain

## Key Differences from Previous Strategy

| Aspect | Before | After |
|--------|--------|-------|
| Acquisition wrapper | PenalisedAcquisition (additive log-space 4x(1-x)) | Plain qLogNEI (no wrapper) |
| Acquisition bounds | Tightened [0.005, 0.995] | Standard [0, 1] |
| Candidate selection | Penalty-weighted distance selection | Base NEI distance selection |
| Penalty constants | STEEPNESS=1.0, FLOOR=0.01, EPS_BOUND=0.005 | Removed |
| Visualisation | 3-panel surrogate + 3-panel penalty | 3-panel surrogate only |
| Submission | Dual (base + IP) | Single (base NEI only) |
