# Quickstart: F1-F8 Week 9 Implementation

**Feature**: 021-f1-f8-week9 | **Date**: 2026-03-02

## Prerequisites

- Python 3.11 with `sdd-dev` environment active
- All Week 9 data files present in `./data/fX/` for F1-F8
- Week 8 notebooks available as templates in `./functions/fX/`

## Implementation Order

### Recommended: F2 first (simplest GP), then F1, F3-F8

1. **F2** (2D, SFGP, no penalty) — simplest GP implementation; validates the full cell structure including performance eval
2. **F1** (2D, Hurdle) — validates sklearn surrogate + LOO with hurdle model edge case
3. **F3** (3D, SFGP with manual z-score) — validates z-score LOO handling
4. **F5** (4D, GP with log1p) — validates double-transform LOO
5. **F4** (4D, MFGP) — validates multi-fidelity LOO
6. **F6** (5D, SFGP) — higher-dim GP
7. **F7** (6D, NN) — validates neural network LOO
8. **F8** (8D, SFGP) — highest dimensionality validates scaling

## Per-Notebook Implementation Steps

For each function, copy `fX - week 8.ipynb` to `fX - week 9.ipynb` and modify:

### Step 1: Update data references (Cell 4)
- Change `Week 8` to `Week 9` in file paths
- Update `N_TOTAL` constant to match Week 9 sample count
- Add `X_initial`, `y_initial`, `X_submissions`, `y_submissions` splits

### Step 2: Add performance eval constants (Cell 3)
```python
STALLING_CONSECUTIVE_THRESHOLD = 3
STALLING_RELATIVE_THRESHOLD = 0.05
```

### Step 3: Add `scipy.spatial.distance` import (Cell 2)
```python
from scipy.spatial.distance import pdist, squareform
```

### Step 4: Update visualisation colour scheme (Cells 7-8)
Replace single-colour training point scatter with:
```python
# Initial samples (blue)
ax.scatter(X_initial[:, d1], X_initial[:, d2], c='tab:blue', s=40,
           edgecolors='white', zorder=5, label='Initial samples')
# Weekly submissions (orange)
ax.scatter(X_submissions[:, d1], X_submissions[:, d2], c='tab:orange', s=60,
           edgecolors='white', zorder=5, label='Weekly submissions')
# Proposed point (green star)
ax.scatter(proposed[d1], proposed[d2], c='tab:green', marker='*', s=200,
           edgecolors='black', zorder=6, label='Proposed next point')
ax.legend()
```

### Step 5: Add Cell 10 — Convergence Metrics
```python
# Best-value trajectory
best_trajectory = np.array([y[:N_INITIAL + k + 1].max() for k in range(N_SUBMISSIONS)])
initial_best = y[:N_INITIAL].max()
per_submission_delta = np.diff(np.concatenate([[initial_best], best_trajectory]))
new_best_flags = per_submission_delta > 0

# Consecutive no-improvement (tail-only: count from most recent backwards)
tail_no_improve = 0
for flag in reversed(new_best_flags):
    if not flag:
        tail_no_improve += 1
    else:
        break
consecutive_no_improvement = tail_no_improve

# Relative improvement
final_best = y.max()
improvement = final_best - initial_best
if abs(initial_best) < 1e-10:
    relative_improvement = 0.0 if improvement < 1e-10 else 1.0
else:
    relative_improvement = improvement / abs(initial_best)

stalling_flag = (consecutive_no_improvement >= STALLING_CONSECUTIVE_THRESHOLD or
                 relative_improvement < STALLING_RELATIVE_THRESHOLD)

print(f"Stalling: {stalling_flag}")
```

### Step 6: Add Cell 11 — Exploration Spread
```python
dists = pdist(X_submissions)
mean_pairwise = dists.mean()
dist_matrix = squareform(dists)
np.fill_diagonal(dist_matrix, np.inf)
nn_dists = dist_matrix.min(axis=1)
max_nn = nn_dists.max()
min_nn = nn_dists.min()

print(f"Mean pairwise distance: {mean_pairwise:.4f}")
print(f"Max nearest-neighbour:  {max_nn:.4f}")
print(f"Min nearest-neighbour:  {min_nn:.4f} (tightest cluster)")
```

### Step 7: Add Cell 12 — LOO Surrogate Error
Implement 9-fold LOO loop specific to the function's surrogate type. See `contracts/notebook-cells.md` and `research.md` for per-surrogate details.

### Step 8: Add Cell 13 — Interpretation Markdown
Write a markdown cell interpreting all metrics and (if stalling) proposing strategy changes from the pre-defined table in `research.md` R4.

## Validation Checklist

For each notebook, verify:
- [ ] Executes end-to-end without errors
- [ ] Proposed point within [0.0, 0.999999]
- [ ] Submission query in correct format
- [ ] Plots show three-colour scheme with legend
- [ ] Performance evaluation section present at end
- [ ] Stalling flag computed correctly
- [ ] LOO metrics displayed
- [ ] Strategy recommendation present (if stalling)

## Key Patterns to Copy from Week 8

- Multi-restart MLL fitting loop (GP functions)
- Interior penalty formula (F1, F5, F6, F7)
- Feature importance for dimension selection (high-dim functions)
- Submission formatting and validation assertions
