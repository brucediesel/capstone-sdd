# Research: F6 Interior Penalty

**Feature**: 016-f6-interior-penalty  
**Date**: 2025-02-24

---

## Decision 1: Penalty Steepness for 5D Domain

**Context**: The interior penalty `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)` decays faster with more dimensions because the product accumulates. F5 (4D) used `STEEPNESS = 1.0`.

**Analysis**: With rank-based scoring (Decision 2), only the _relative ordering_ of `w(x)` across 4 candidates matters, not absolute magnitudes. The steepness parameter controls the transition width — how far from the boundary the penalty is active. With `STEEPNESS = 1.0` and 5D, a point at `x_i = 0.2` in all dimensions yields `w ≈ 0.014` (very low). This is appropriate because F6's boundary-hugging problem is severe (milk dimension collapses to 0).

**Decision**: `STEEPNESS = 1.0` (same as F5).  
**Rationale**: Rank-based scoring makes the exact w(x) magnitudes irrelevant — only the ordering matters. Strong steepness clearly differentiates boundary from interior candidates, which is the primary goal. The penalty floor (FLOOR = 0.01) prevents division-by-zero or degenerate zero weights.  
**Alternatives considered**:
- `STEEPNESS = 0.5` — softer transition, but weaker differentiation between moderately interior and truly interior points. Unnecessary given rank-based scoring.
- `STEEPNESS = 0.3` — too soft; points at 20% from boundary in all dimensions still get `w ≈ 0.21`, which may not rank-differentiate well with only 4 candidates.

---

## Decision 2: Re-Scoring Strategy for All-Negative Outputs (CRITICAL)

**Context**: F6 outputs are all negative (e.g., `pred_means ≈ [-3.2, -1.5, -2.8, -0.9]`). Maximisation means higher (less negative) is better. The naive F5 approach (`pred_means * w(x)`) fails catastrophically: multiplying a negative value by `w(x) < 1` makes it _less negative_ (closer to 0), which _promotes_ boundary candidates instead of penalising them.

**Strategies evaluated**:

| Strategy | Penalises Boundary | Scale-Invariant | No Tuning | Robust |
|----------|-------------------|-----------------|-----------|--------|
| A: Shift-Penalize-Shift | ✅ | ❌ | ❌ | ❌ (low-spread fragility) |
| B: Additive penalty | ✅ | ❌ | ❌ (magnitude tuning) | ❌ |
| **C: Rank-based** | **✅** | **✅** | **✅** | **✅** |
| D: Tiebreaker only | ❌ (too weak) | ✅ | ✅ | ❌ |
| E: Abs multiplicative | ❌ (weak, caps at 2×) | ❌ | ✅ | ❌ |

**Decision**: Rank-based scoring (Strategy C).

**Formula**:
```python
# Rank 1..4, higher value → higher rank
rank_mean = np.argsort(np.argsort(pred_means)) + 1     # better mean → higher rank
rank_weight = np.argsort(np.argsort(interior_weight)) + 1  # more interior → higher rank
combined_score = rank_mean + rank_weight                 # range [2, 8]
```

**Rationale**:
1. **Scale-invariant** — works identically for `[-0.9, -3.2]` or `[-900, -3200]`.
2. **Sign-invariant** — correct for both positive and negative outputs.
3. **No tuning** — no `penalty_magnitude` or shift anchors to calibrate.
4. **Natural median filter** — `combined_score >= np.median(combined_score)` splits 4 candidates into 2 above-median, matching the existing F6 selection pattern.
5. **No scipy dependency** — uses `np.argsort(np.argsort(...))` instead of `scipy.stats.rankdata`.

**Alternatives considered**:
- Shift-Penalize-Shift (Strategy A): penalty effectiveness collapses when `pred_means` have low spread.
- Additive penalty (Strategy B): requires tuning `penalty_magnitude` relative to output scale.
- Simple multiplicative (F5 approach): mathematically inverted for negative outputs.

---

## Decision 3: Integration with Distance-Based Selection

**Context**: The existing F6 selection flow is:
1. Median filter: `above_median = pred_means >= median(pred_means)` → keeps top 2 of 4.
2. Distance selection: among above-median, pick farthest from training data.

**Decision**: Replace `pred_means` with `combined_score` in the median filter only. Distance selection is unchanged.

**Flow**:
```python
above_median = combined_score >= np.median(combined_score)  # top 2 of 4
above_median_indices = np.where(above_median)[0]
above_median_dists = dists[above_median_indices].cpu().numpy()
best_ip_idx = above_median_indices[np.argmax(above_median_dists)]
best_point = candidates[best_ip_idx].cpu().numpy()
```

**Rationale**: The distance-based tiebreaker already handles exploration. The penalty's job is only to demote boundary candidates in the median filter. Keeping the two mechanisms separate (rank-based filter + distance-based selection) maintains the clarity and composability of the pipeline.  
**Alternatives considered**:
- Incorporating `w(x)` into the distance metric — over-engineers the solution; distance and interiority are separate concerns.
- Replacing distance selection entirely with rank-based selection — loses the exploration benefit of distance-based selection.

---

## Decision 4: Visualisation Panel 3 Replacement

**Context**: The existing Panel 3 shows "Dimension Relevance (1/ℓ, normalised)" — a horizontal bar chart. The interior penalty feature replaces this with a penalty surface heatmap to visualise where the penalty is active.

**Decision**: Replace Panel 3 with a 2D penalty heatmap on the two most relevant dimensions (highest 1/ℓ), with the other 3 dimensions held at 0.5.

**Formula**:
```python
top2 = np.argsort(inv_ls)[-2:][::-1]  # 2 most relevant dimensions
grid = np.linspace(0, 1, 80)
xx, yy = np.meshgrid(grid, grid)
base = np.full((80*80, 5), 0.5)
base[:, top2[0]] = xx.ravel()
base[:, top2[1]] = yy.ravel()
w_grid = FLOOR + (1 - FLOOR) * np.prod(np.sin(np.pi * base) ** (2 * STEEPNESS), axis=1)
```

**Rationale**: A 5D penalty cannot be directly visualised. Projecting onto the 2 most relevant dimensions (by ARD lengthscale) shows the penalty landscape where the GP is most sensitive. Holding other dimensions at 0.5 (centre) shows the best-case penalty — actual penalty for boundary points in held-out dimensions would be even lower.  
**Alternatives considered**:
- Pairs plot (all 10 pairs) — too many panels, overwhelming.
- 1D marginal plots — loses the 2D interaction structure of the product penalty.
- Removing Panel 3 entirely — loses diagnostic value.

---

## Decision 5: Batch Size and Acquisition Parameters

**Context**: F6 uses `q=4` with `num_restarts=50`, `raw_samples=3000`, `SobolQMCNormalSampler(sample_shape=torch.Size([512]))`.

**Decision**: Keep all acquisition parameters unchanged. The interior penalty is post-hoc re-scoring only.

**Rationale**: The penalty is applied _after_ `optimize_acqf` returns candidates. Changing acquisition parameters is orthogonal to the penalty mechanism and would confound the comparison.  
**Alternatives considered**:
- Increasing `q` to get more candidates for ranking — unnecessarily increases computation; 4 candidates give sufficient ranking granularity (ranks 1–4, combined scores 2–8).
- Modifying the acquisition function itself (constrained acquisition) — violates the "post-hoc penalty" design and is significantly more complex.
