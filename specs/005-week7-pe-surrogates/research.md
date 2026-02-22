# Research: Week 7 — F1 Hurdle Model with Weighted UCB and Local Penalization

**Branch**: `005-week7-pe-surrogates`  
**Phase**: 0 — Resolve all NEEDS CLARIFICATION items from Technical Context  
**Date**: 2026-02-22

---

## Decision 1 — Stage 2 Regressor: Per-tree uncertainty from RandomForestRegressor

**Decision**: Use `rf.estimators_` list comprehension to extract per-tree predictions, then take `.std(axis=0)` across the tree axis.

**Rationale**: `sklearn.ensemble.RandomForestRegressor` exposes `estimators_` as a list of fitted `DecisionTreeRegressor` objects. Calling `.predict()` on each and stacking gives an `(n_trees, n_candidates)` array; the std across axis 0 is the natural ensemble uncertainty estimate. No external library is needed and the pattern is 3 lines of numpy.

**Code pattern**:
```python
tree_preds = np.array([tree.predict(X_cand) for tree in stage2_rf.estimators_])  # (n_trees, n_cand)
mu_log     = tree_preds.mean(axis=0)   # E[log1p(y) | y>0] on log scale
sigma_rf   = tree_preds.std(axis=0)    # σ_RF(x)
```

**Alternatives considered**:
- Bootstrap ensemble of separate regressors — more code, slower, no benefit over estimators_ on an RF already fitted
- Conformal prediction intervals — principled but adds complexity incompatible with the student-notebook simplicity constraint

---

## Decision 2 — Stage 1 Classifier: CalibratedClassifierCV configuration for small dataset

**Decision**: `CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight='balanced'), cv=3, method='sigmoid')`

**Rationale**:
- `cv=3` (not 5): with 17 samples, `cv=5` risks ~3 samples per fold and potential empty-class folds for a sparse positive class. `cv=3` gives ~5–6 per fold which is viable.
- `method='sigmoid'` (Platt scaling): sklearn's documentation explicitly warns that `method='isotonic'` requires ≥1 000 calibration samples; it will overfit severely on 17 samples.
- `class_weight='balanced'`: essential when the positive class may be a minority; prevents the classifier from defaulting to "all-negative" predictions which would suppress the acquisition function to zero everywhere.

**Alternatives considered**:
- `SVC(probability=True)` — internally also uses Platt scaling but adds kernel and gamma hyperparameters, increasing documentation burden for minimal gain
- `RandomForestClassifier` — probability estimates from class proportions per leaf can produce 0/1 outputs with small trees, poor calibration on 17 samples

---

## Decision 3 — Variance approximation in weighted UCB

**Decision**: Simplified form $a(x) = p(x)\cdot\mu(x) + \kappa \cdot p(x) \cdot \sigma_{\text{RF}}(x)$

**Rationale**: The simplified variance $\text{Var}(p\mu) \approx p^2\sigma_\mu^2$ (at constant p) gives $\sqrt{\text{Var}} \approx p\cdot\sigma_\text{RF}$. This is algebraically equivalent to the delta-method leading term, interpretable ("exploration bonus is proportional to probability of positive output"), and one line of code. Full delta method would require estimating $\sigma_p$ from the logistic calibrator, adding complexity without materially changing the exploration pattern at this dataset size.

**Alternatives considered**:
- Full delta method $\sqrt{p^2\sigma_\mu^2 + \mu^2\sigma_p^2 + \sigma_p^2\sigma_\mu^2}$ — more rigorous but requires $\sigma_p$ estimation and is harder to explain in a student notebook
- Ignoring Stage 1 uncertainty entirely ($\kappa\cdot\sigma_\text{RF}$ only) — misses the intent of the weighted UCB formulation

---

## Decision 4 — Local Penalization: formula and radius selection

**Decision**: Multiplicative Gaussian mask applied against **all 17 existing data points**. Penalty radius $r = 0.15$.

**Formula**:
$$\text{penalty}(x) = \prod_{i=1}^{17} \left(1 - \exp\!\left(-\frac{\|x - x_i\|^2}{2r^2}\right)\right)$$

**Vectorized numpy implementation**:
```python
diffs    = X_cand[:, np.newaxis, :] - X_obs[np.newaxis, :, :]   # (C, N, 2)
sq_dists = (diffs ** 2).sum(axis=2)                              # (C, N)
masks    = 1.0 - np.exp(-sq_dists / (2.0 * PENALTY_RADIUS ** 2)) # (C, N)
penalty  = masks.prod(axis=1)                                    # (C,)
acq_penalized = acq_raw * penalty
```

**Radius rationale** ($r = 0.15$): The 2D input space diagonal is $\sqrt{2} \approx 1.414$. Radius 0.15 corresponds to ~10.6% of the diagonal, creating a meaningful exclusion zone around each observation while not blacking out more than ~70% of the space (with 17 points at $r=0.15$, effective coverage is well under 100%). This satisfies SC-003 (proposed point ≥ 0.05 from all existing points) with comfortable margin.

**Alternatives considered**:
- Hard exclusion (set acquisition to 0 within radius) — discontinuous, creates artefacts at boundary; smooth Gaussian is numerically better
- Sliding window (5 most recent points only) — conflicts with the confirmed decision (Q4 clarification) to use all 17 points

---

## Decision 5 — Stage 2 fallback threshold

**Decision**: If fewer than **3** positive-output samples (y > 0) are present, Stage 2 is skipped. The acquisition falls back to pure uncertainty exploration: Gaussian random sampling over the penalized space.

**Rationale**: A Random Forest with fewer than 3 positive samples cannot produce meaningful splits on 2 features without complete overfitting. The threshold of 3 was explicitly confirmed during clarification.

---

## Decision 6 — Visualization panel titles (Week 7 style)

Following the exact pattern from Week 5/6 (`'<Surrogate> <Quantity> (<symbol>)'`):

| Panel | Colormap | Title |
|-------|----------|-------|
| 1 — combined hurdle prediction | `viridis` | `'Hurdle Mean Prediction (ŷ = p·expm1(μ))'` |
| 2 — uncertainty | `YlOrRd` | `'Hurdle Uncertainty (p·σ_RF)'` |
| 3 — penalized acquisition | `plasma` | `f'Penalized UCB Acquisition (κ={KAPPA})'` |

---

## Summary: All NEEDS CLARIFICATION resolved

| Item | Resolved Decision |
|------|------------------|
| Stage 2 regressor algorithm | Random Forest via `estimators_` |
| Variance approximation | Simplified: $p\cdot\sigma_\text{RF}$ |
| Stage 1 classifier | Logistic Regression + CalibratedClassifierCV(cv=3, method='sigmoid') |
| Local penalization scope | All 17 evaluated data points |
| Minimum positive samples for Stage 2 | 3 |
| Panel titles | Decided (see Decision 6) |
