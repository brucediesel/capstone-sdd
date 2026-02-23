# Quickstart: F3 Week 7 Implementation

**Feature**: 009-f3-sfgp-nei  
**Branch**: `009-f3-sfgp-nei`  
**Target file**: `functions/f3/f3.ipynb`  
**Estimated effort**: ~1 hour (writing + running cells)  

---

## Prerequisites

1. **Branch**: Confirm you are on `009-f3-sfgp-nei`.
   ```bash
   git branch --show-current   # should print: 009-f3-sfgp-nei
   ```
2. **Data files**: Confirm both Week 7 files exist.
   ```bash
   ls "data/f3/updated_inputs - Week 7.npy"
   ls "data/f3/updated_outputs - Week 7.npy"
   ```
3. **Python environment**: The `sdd-dev` pyenv environment must be active with BoTorch 0.16.1.
   ```bash
   python -c "import botorch; print(botorch.__version__)"   # → 0.16.1
   ```

---

## Implementation Steps

### Step 1 — Open the target notebook

Open `functions/f3/f3.ipynb` in VS Code.  
Scroll to the very last cell (currently the Research markdown cell ending at line 1005).  
All new cells are added **after** this cell.

### Step 2 — Add Cell 1: Section Header (Markdown)

Append a markdown cell with the heading:

```markdown
## Week 7 — SFGP with Matérn-5/2 ARD

This section fits a Single-Fidelity Gaussian Process (SFGP)...
```

See [contracts/week7-cells.md](contracts/week7-cells.md#Cell-1) for the full required text.

### Step 3 — Add Cell 2: Imports and Data Loading (Code)

Append a code cell that:
- Imports BoTorch, GPyTorch, NumPy, Matplotlib, and `copy`
- Loads `updated_inputs - Week 7.npy` and `updated_outputs - Week 7.npy`
- Validates inputs are in [0.0, 1.0]
- Prints sample count, ranges, and best observed value

**Run this cell immediately** to confirm the data files load correctly before proceeding.

### Step 4 — Add Cell 3: Hyperparameter Explanation (Markdown)

Append a markdown cell with a table or list covering all 10 hyperparameters defined in [contracts/week7-cells.md](contracts/week7-cells.md#Cell-3).

### Step 5 — Add Cell 4: Model Training with Restarts (Code)

Append the multi-restart training cell:
- 15 restarts with `torch.manual_seed(seed)`
- `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))`
- Print fitted ℓ_A, ℓ_B, ℓ_C, σ²_f, σ²_n

**Run this cell.** It should take 15–60 seconds. Confirm 5 hyperparameter values are printed.

### Step 6 — Add Cell 5: NEI Acquisition (Code)

Append the acquisition cell:
- BOUNDS at [0.0, 0.0, 0.0] / [0.999999, 0.999999, 0.999999]
- `qLogNoisyExpectedImprovement` with `X_baseline=X_train`
- `optimize_acqf` with `q=1, num_restarts=10, raw_samples=512`

**Run this cell.** `next_x_raw` must be populated as a `(3,)` array.

### Step 7 — Add Cell 6: Surrogate Slice Plots (Code)

Append the visualisation cell with three pairwise 2D panels.  
See [contracts/week7-cells.md](contracts/week7-cells.md#Cell-6) for the exact requirements (labels, title, proposed point marker).

**Run this cell.** Confirm 3 plots render with the yellow star marking the proposed point.

### Step 8 — Add Cell 7: Convergence Plot (Code)

Append the convergence cell.  
**Run this cell.** Confirm axes are labelled "Sample Number" and "Best Observed Output".

### Step 9 — Add Cell 8: Submission Query (Code)

Append the query formatting cell.  
**Run this cell.** Confirm the final output line matches `0.xxxxxx-0.yyyyyy-0.zzzzzz`.

---

## Verification Checklist

Run all 6 new code cells top-to-bottom in a fresh kernel restart. Verify:

- [ ] Cell 2 prints sample count (should be ~22) and best observed value
- [ ] Cell 4 prints exactly 5 hyperparameter values (ℓ_A, ℓ_B, ℓ_C, σ²_f, σ²_n), each labelled
- [ ] Three lengthscale values are distinct (confirms ARD is working)
- [ ] Cell 5 completes without error; `next_x_raw` is defined
- [ ] Cell 6 renders 3 plots; each has labelled axes and the proposed yellow star
- [ ] Cell 7 renders a convergence plot with the correct axis labels
- [ ] Cell 8 prints a string matching pattern `0.XXXXXX-0.YYYYYY-0.ZZZZZZ`
- [ ] Zero existing cells show modifications in `git diff`

---

## Commit

```bash
git add functions/f3/f3.ipynb
git commit -m "feat(f3): add Week 7 SFGP Matern-5/2 ARD + NEI section"
```

---

## Known Constraints

- **Existing cells**: Do not modify any cell before the new Week 7 header. The entire Week 7 section is a pure append.
- **Input clamping**: The `format_query` function must clamp to [0.0, 0.999999] before formatting (not [0.0, 1.0]).
- **dtype**: Use `torch.float64` throughout — mixing float32 will silently break MLL fitting.
- **Student-t likelihood**: Only document it in the markdown cell; implement Gaussian noise. Do not add a second training cell for Student-t.
