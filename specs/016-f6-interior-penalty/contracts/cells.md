# Cell Contracts: F6 Interior Penalty

**Feature**: 016-f6-interior-penalty  
**Date**: 2025-02-24  
**Target**: `functions/f6/f6.ipynb` — append cells 60–65 after existing cell 59

---

## Cell 60 — Markdown: Section Header

**Type**: Markdown  
**Purpose**: Document the interior penalty section and hyperparameters.

**Content**:
```markdown
## Week 7 — Interior Penalty Re-Scoring

Apply a soft interior penalty to discourage boundary-hugging candidates.
The penalty weight `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`
re-scores candidates via rank-based combination (sign-invariant for all-negative outputs).

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| STEEPNESS      | 1.0   | Strong boundary decay, rank-based scoring makes magnitude irrelevant |
| FLOOR          | 0.01  | Prevents zero-weight degeneracy |
```

---

## Cell 61 — Code: Compute Interior Penalty Weights

**Type**: Code  
**Purpose**: Calculate `w(x)` for each of the 4 candidates.

**Inputs** (from existing cells):
| Variable | Type | Shape | Source Cell |
|----------|------|-------|-------------|
| `candidates` | `torch.Tensor` | `(4, 5)` | Cell 54 (NEI acquisition) |

**Outputs**:
| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `STEEPNESS` | `float` | scalar | `1.0` |
| `FLOOR` | `float` | scalar | `0.01` |
| `interior_weight` | `np.ndarray` | `(4,)` | Per-candidate weight ∈ [FLOOR, 1.0] |

**Contract**:
- `STEEPNESS` and `FLOOR` are named constants at cell top.
- `cands_np = candidates.cpu().numpy()` — converts to NumPy for sin/prod.
- `interior_weight = FLOOR + (1 - FLOOR) * np.prod(np.sin(np.pi * cands_np) ** (2 * STEEPNESS), axis=1)`
- Prints `interior_weight` and labels each candidate as "boundary" (`w < 0.5`) or "interior" (`w >= 0.5`).
- Assert `interior_weight.shape == (4,)`.
- Assert `np.all(interior_weight >= FLOOR) and np.all(interior_weight <= 1.0)`.

---

## Cell 62 — Code: Rank-Based Re-Scoring and Selection

**Type**: Code  
**Purpose**: Combine posterior mean rank and interior weight rank; apply median filter; select best point via distance.

**Inputs** (from existing cells):
| Variable | Type | Shape | Source Cell |
|----------|------|-------|-------------|
| `pred_means` | `np.ndarray` | `(4,)` | Cell 54 (NEI acquisition) |
| `interior_weight` | `np.ndarray` | `(4,)` | Cell 61 |
| `candidates` | `torch.Tensor` | `(4, 5)` | Cell 54 |
| `X_train` | `torch.Tensor` | `(27, 5)` | Cell 54 |
| `dists` | `torch.Tensor` | `(4,)` | Cell 54 |

**Outputs**:
| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `rank_mean` | `np.ndarray` | `(4,)` | Rank by posterior mean (1=worst, 4=best) |
| `rank_weight` | `np.ndarray` | `(4,)` | Rank by interior weight (1=boundary, 4=interior) |
| `combined_score` | `np.ndarray` | `(4,)` | `rank_mean + rank_weight`, range [2, 8] |
| `above_median` | `np.ndarray[bool]` | `(4,)` | `combined_score >= median(combined_score)` |
| `best_ip_idx` | `int` | scalar | Index of selected candidate |
| `best_point` | `np.ndarray` | `(5,)` | Final selected point |

**Contract**:
- Ranking via `np.argsort(np.argsort(x)) + 1` (no scipy dependency).
- Median filter: `above_median = combined_score >= np.median(combined_score)`.
- Among above-median indices, select candidate with largest `dists` value.
- Fallback: if no candidates pass filter, select `argmax(combined_score)`.
- Prints comparison table: original `pred_means` vs `rank_mean`, `interior_weight` vs `rank_weight`, `combined_score`, and marks selected candidate with `◄`.
- Assert `best_point.shape == (5,)`.
- Assert all `best_point >= 0.01` and `best_point[4] >= 0.10` (feasibility bounds).

---

## Cell 63 — Code: Updated 3-Panel Visualisation

**Type**: Code  
**Purpose**: Replace the dimension-relevance bar chart (Panel 3) with a penalty surface heatmap, while keeping Panels 1 (posterior mean) and 2 (posterior std) unchanged.

**Inputs** (from existing cells):
| Variable | Type | Shape | Source Cell |
|----------|------|-------|-------------|
| `best_model` | `SingleTaskGP` | — | Cell 53 (GP training) |
| `X_train` | `torch.Tensor` | `(27, 5)` | Cell 54 |
| `candidates` | `torch.Tensor` | `(4, 5)` | Cell 54 |
| `best_point` | `np.ndarray` | `(5,)` | Cell 62 |
| `ls` | `np.ndarray` | `(5,)` | Cell 53 (lengthscales) |
| `STEEPNESS` | `float` | scalar | Cell 61 |
| `FLOOR` | `float` | scalar | Cell 61 |

**Outputs**: Matplotlib figure (3 panels, displayed inline).

**Contract**:
- `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`
- **Panel 1**: Posterior mean heatmap on top-2 ARD dims (80×80 grid, other dims at 0.5). Scatter `X_train` projected, star marker for `best_point`.
- **Panel 2**: Posterior std heatmap (same grid/projection).
- **Panel 3**: Interior penalty `w(x)` heatmap (same grid/projection). Colour scale `[FLOOR, 1.0]`, `cmap='RdYlGn'`. Scatter candidates with marker size proportional to `combined_score`. Star for `best_point`.
- `top2 = np.argsort(1.0 / ls)[-2:][::-1]` (2 most relevant dims by inverse lengthscale).
- Title includes dimension names: `f"x{top2[0]} vs x{top2[1]}"`.
- `plt.tight_layout(); plt.show()`.

---

## Cell 64 — Code: Convergence Plot (Updated)

**Type**: Code  
**Purpose**: Show convergence with the penalty-selected point highlighted.

**Inputs**:
| Variable | Type | Shape | Source Cell |
|----------|------|-------|-------------|
| `y_raw` | `np.ndarray` | `(27,)` | Cell 51 |
| `best_point` | `np.ndarray` | `(5,)` | Cell 62 |
| `pred_means` | `np.ndarray` | `(4,)` | Cell 54 |
| `best_ip_idx` | `int` | scalar | Cell 62 |

**Outputs**: Matplotlib figure (1 panel, displayed inline).

**Contract**:
- Plot `y_raw` as line/scatter for historical points.
- Add horizontal line at `pred_means[best_ip_idx]` labelled "IP-selected predicted mean".
- Add horizontal line at `pred_means.max()` labelled "Best raw candidate mean" (dashed, for comparison).
- Title: "Convergence — F6 with Interior Penalty".
- `plt.show()`.

---

## Cell 65 — Code: Submission Query (Penalty-Adjusted)

**Type**: Code  
**Purpose**: Print the selected point in submission format, with penalty metadata.

**Inputs**:
| Variable | Type | Shape | Source Cell |
|----------|------|-------|-------------|
| `best_point` | `np.ndarray` | `(5,)` | Cell 62 |
| `pred_means` | `np.ndarray` | `(4,)` | Cell 54 |
| `interior_weight` | `np.ndarray` | `(4,)` | Cell 61 |
| `combined_score` | `np.ndarray` | `(4,)` | Cell 62 |
| `best_ip_idx` | `int` | scalar | Cell 62 |
| `STEEPNESS` | `float` | scalar | Cell 61 |
| `FLOOR` | `float` | scalar | Cell 61 |

**Outputs**: Printed text (no new variables).

**Contract**:
- Prints `best_point` formatted to 4 decimal places.
- Validates 5D format: `assert best_point.shape == (5,)`.
- Validates feasibility: `assert np.all(best_point >= 0.01)`, `assert best_point[4] >= 0.10`.
- Prints penalty metadata: `STEEPNESS`, `FLOOR`, `interior_weight[best_ip_idx]`, `combined_score[best_ip_idx]`.
- Prints comparison: raw best (`argmax(pred_means)`) vs penalty-selected (`best_ip_idx`) — flags if they differ.
