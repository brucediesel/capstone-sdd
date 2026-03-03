# Research: F1-F8 Week 9 Performance Evaluation

**Feature**: 021-f1-f8-week9 | **Date**: 2026-03-02

## R1: Leave-One-Out (LOO) Prediction Error by Surrogate Type

### Decision: Retrain-from-scratch for all surrogates (9 refits per function)

**Rationale**: With 19-49 training points and 9 LOO folds, retrain-from-scratch completes in seconds for most surrogates. Analytic GP LOO exists but BoTorch doesn't expose the inverse kernel matrix directly, and the analytic formula assumes fixed hyperparameters. Retraining is simpler and consistent across all surrogate types.

**Alternatives considered**:
- Analytic LOO (GP-only): Rejected because it requires fixed hyperparameters and custom Cholesky decomposition code. Not simpler than retraining.
- K-fold cross-validation: Rejected because 9 submission points naturally map to LOO (9-fold) — it IS the natural fold structure.

### Per-Surrogate LOO Details

| Surrogate | Functions | Key Implementation Note |
|-----------|-----------|------------------------|
| SingleTaskGP | F2, F3, F6, F8 | Construct new `SingleTaskGP(X_loo, Y_loo, ...)` per fold. `Standardize(m=1)` auto-recomputes. |
| GP + manual z-score | F3, F5 | Recompute z-score stats (mean, std) per fold. Un-standardise predictions before error computation. F5 also needs `expm1()` to invert `log1p`. |
| Multi-Fidelity GP | F4 | Retrain `SingleTaskMultiFidelityGP`. Reduce MLL restarts to 5 (from 15) for LOO to limit runtime. |
| Hurdle Model | F1 | Retrain both stages. If removing a point drops `n_positive < 3`, use fallback prediction = 0 for that fold. |
| Neural Network | F7 | Retrain 200 epochs per fold. Use `torch.manual_seed` for reproducibility. Single forward pass (not MC dropout) for LOO prediction. |

### Data Exclusion Rule

For each submission point `i` (indices `N_initial` to `N_total - 1`):
- Training set = all `N_total - 1` points (all initial + all other submissions)
- Predict on the held-out single point
- Compute error in original (un-transformed) output space

---

## R2: Exploration Spread Metrics

### Decision: Raw Euclidean pairwise distance and nearest-neighbour distance, no cross-function normalisation

**Rationale**: All inputs are already in [0, 1]^d. Metrics are interpreted per-function to diagnose clustering, not compared across functions. Normalising by sqrt(d) adds complexity without practical benefit for the capstone context.

**Implementation**:
```python
from scipy.spatial.distance import pdist, squareform

# Mean pairwise Euclidean distance
dists = pdist(X_submissions)
mean_pairwise = dists.mean()

# Maximum nearest-neighbour distance  
dist_matrix = squareform(dists)
np.fill_diagonal(dist_matrix, np.inf)
nn_dists = dist_matrix.min(axis=1)
max_nn_dist = nn_dists.max()
```

**Expected scales** (for uniform random points in [0,1]^d):
- 2D: mean pairwise ~0.52
- 4D: mean pairwise ~0.73
- 8D: mean pairwise ~1.03

Points significantly below these values indicate clustering.

---

## R3: Stalling Detection — Handling Negative/Zero Outputs

### Decision: `(best_final - best_initial) / abs(best_initial)` with zero-guard

**Rationale**: For maximisation, improvement is `best_final - best_initial`. Dividing by `abs(best_initial)` gives correct relative magnitude for both positive and negative starting values.

**Edge cases**:
- `best_initial ≈ 0` (F1): If `abs(best_initial) < 1e-10`, flag stalling if `improvement == 0`, not-stalling if any positive output achieved.
- All negative, no improvement: formula returns 0 → stalling (correct).
- Negative improving: e.g., -0.5 → -0.4 gives 20% relative improvement (correct).

**Implementation**:
```python
best_initial = y[:N_INITIAL].max()
best_final = y.max()
improvement = best_final - best_initial

if abs(best_initial) < 1e-10:
    relative_improvement = 0.0 if improvement < 1e-10 else 1.0
else:
    relative_improvement = improvement / abs(best_initial)

stalling_relative = relative_improvement < 0.05
```

**Consecutive-submission stalling** (tail-only — checks the most recent submissions):
```python
submission_bests = [y[:N_INITIAL + k + 1].max() for k in range(9)]
improvements = [submission_bests[k] - (submission_bests[k-1] if k > 0 else best_initial) 
                for k in range(9)]
# Count consecutive no-improvement from the END of the sequence
tail_no_improve = 0
for imp in reversed(improvements):
    if imp <= 0:
        tail_no_improve += 1
    else:
        break
stalling_consecutive = tail_no_improve >= 3
```

---

## R4: Strategy Recommendations for Stalling Functions

### Decision: 2-3 concrete, function-specific recommendations per function

Each recommendation references a specific hyperparameter or model change. The recommendations are pre-written templates that the markdown interpretation cell will select from based on the stalling diagnosis.

| Function | Recommendation 1 | Recommendation 2 | Recommendation 3 |
|----------|-------------------|-------------------|-------------------|
| F1 (Hurdle+RF) | Switch to SingleTaskGP (if positive outputs now exist) | Raise KAPPA from 3.0 to 5.0 for more exploration | Use LHS candidates instead of random for better space coverage |
| F2 (SFGP 1.5) | Switch kernel to Matérn 2.5 for smoother extrapolation | Increase raw_samples from 512 to 2048 | Add random restart perturbation to escape local optimum |
| F3 (SFGP 2.5) | Use BoTorch Standardize(m=1) instead of manual z-score | Increase acquisition num_restarts from 10 to 20 | Add interior penalty (S=0.5, F=0.01) |
| F4 (MFGP) | Switch to standard SingleTaskGP (no multi-fidelity data) | Reduce q from 4 to 1 for focused acquisition | Add interior penalty (S=0.5, F=0.01) |
| F5 (GP log1p) | Relax interior penalty STEEPNESS from 1.0 to 0.3 | Increase Sobol candidates from 3000 to 5000 | Try Matérn 1.5 kernel for rougher fitting |
| F6 (SFGP 1.5) | Switch kernel to Matérn 2.5 for 5D extrapolation | Reduce noise floor from 1e-2 to 1e-4 | Increase acquisition restarts from 50 to 80 |
| F7 (NN) | Switch to SingleTaskGP Matérn 2.5 (better uncertainty) | Increase MC dropout samples from 50 to 200 | Widen network to 6→10→10→1 |
| F8 (SFGP 2.5) | Switch from qEI to qLogNEI (numeric stability in 8D) | Add interior penalty (S=0.5, F=0.01) | Increase raw_samples from 4096 to 8192 |

---

## Summary of All Research Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| LOO method | Retrain-from-scratch (9 refits) | Simplest, consistent across all surrogates |
| LOO exclusion | Remove from full dataset | Spec requires retraining on all data minus one |
| Exploration metrics | Raw Euclidean, no normalisation | Per-function interpretation, inputs already in [0,1]^d |
| Stalling formula | `(best_final - best_initial) / abs(best_initial)` | Correct for maximisation with positive and negative values |
| Zero initial best | Flag stalling if no improvement; not-stalling if any positive found | Handles F1's zero-inflated case |
| Strategy recommendations | Pre-defined per-function table | Concrete, actionable, specific to each surrogate/acquisition |
