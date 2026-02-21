# Implementation Plan: F6 NN, SFGP & MFGP 3-Way Comparison

**Branch**: `004-prequential-evaluation` | **Date**: 2025-07-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-prequential-evaluation/spec.md` (F6 section)

## Summary

Extend the F6 prequential evaluation notebook from a **2-way** (NN + MFGP, 90 configs) comparison to a **3-way** (NN + SFGP + MFGP, 135 configs) comparison. Three changes:

1. **Modify NN grid** — Change from layers {2,3} × nodes {3,4,5,6} × lr(5) = 40 configs → layers {1,2,3} × nodes {4,5,6} × lr(5) = **45 configs**
2. **Add SFGP** — New `sfgp_prequential_with_config()` function evaluating 40 SingleTaskGP configurations (kernel {Matérn 2.5, 1.5, 0.5, RBF} × transform {raw, standardise} × noise_lb(5))
3. **Upgrade comparison** — 2-way (NN vs MFGP) → 3-way (NN vs SFGP vs MFGP), update all downstream cells (comparison table/chart, best model viz, sensitivity charts, ranked table, conclusions)

## Technical Context

**Language/Version**: Python 3.14.2 (pyenv virtualenv `sdd-dev`)
**Primary Dependencies**: BoTorch 0.16.1 (SingleTaskGP, SingleTaskMultiFidelityGP, fit_gpytorch_mll), GPyTorch 1.15.1 (MaternKernel, RBFKernel, ScaleKernel, GaussianLikelihood, GreaterThan, ExactMarginalLogLikelihood), PyTorch (nn.Module, Adam, MSELoss), matplotlib, pandas, numpy
**Storage**: `.npy` files in `data/f6/` (read-only)
**Testing**: None (per constitution — no unit tests)
**Target Platform**: Local Jupyter notebook (macOS)
**Project Type**: Single Jupyter notebook (`functions/f6/preq-eval-f6.ipynb`)
**Performance Goals**: N/A — offline analysis, runtime ~5–10 min for 135 configs
**Constraints**: Small dataset (26 samples, 5D input), negative outputs [−2.571, −0.219]
**Scale/Scope**: 1 notebook, 135 hyperparameter configurations, 810 total predictions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| Code as simple as possible | ✅ PASS | Each step in its own cell with markdown explanation |
| All code in Jupyter notebooks | ✅ PASS | Single .ipynb file |
| No unit tests | ✅ PASS | No tests created |
| BoTorch as default GP library | ✅ PASS | SingleTaskGP (SFGP) and SingleTaskMultiFidelityGP (MFGP) |
| Each problem in own notebook/folder | ✅ PASS | `functions/f6/preq-eval-f6.ipynb` |
| Existing cells NOT replaced (weekly) | ✅ N/A | Not a weekly submission; prequential eval notebook. Cell edits are minor (grid values, comparison logic) |
| Hyperparameters explained | ✅ PASS | Markdown cells document all HP choices and search spaces |
| Visualisations (surrogate + convergence) | ✅ PASS | 3-panel prequential plot, sensitivity charts, comparison bar chart |
| All problems are maximisation | ✅ PASS | F6 maximises (closest to 0) |

**Gate result: ALL PASS** — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/004-prequential-evaluation/
├── plan.md              # This file
├── spec.md              # Feature specification (F6 section: lines 971–1298)
├── research.md          # Phase 0: SFGP kernel config, BoTorch patterns
├── data-model.md        # Phase 1: Entity definitions (configs, results, DataFrames)
├── quickstart.md        # Phase 1: How to run the notebook
├── contracts/           # Phase 1: N/A (no API — notebook-only)
├── tasks.md             # Phase 2: Implementation tasks (created by /speckit.tasks)
└── checklists/
    └── requirements-f6-sfgp-nn-update.md  # Quality checklist (16/16 pass)
```

### Source Code

```text
functions/f6/
└── preq-eval-f6.ipynb   # The single notebook (30 cells → ~36 cells after changes)

data/f6/
├── updated_inputs - Week 6.npy    # 26×5 input array (read-only)
└── updated_outputs - Week 6.npy   # 26×1 output array (read-only)
```

**Structure Decision**: Single Jupyter notebook. No src/, tests/, or multi-project structure. All code lives in notebook cells. This matches the constitution requirement.

## Implementation Plan — Cell-by-Cell

### Phase A: Import & Grid Updates (modify existing cells)

| Step | Cell | Action | Details |
|------|------|--------|---------|
| A1 | Cell 3 (imports) | **Edit** | Add `from botorch.models import SingleTaskGP`; add `from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel` |
| A2 | Cell 12 (markdown) | **Edit** | Update NN search space description: layers {1,2,3} × nodes {4,5,6} = 45 configs |
| A3 | Cell 13 (NN configs) | **Edit** | `layers_grid = [1, 2, 3]`, `nodes_grid = [4, 5, 6]`, update comments from "50" to "45" |

### Phase B: Add SFGP (insert new cells after cell 21)

| Step | Cell | Action | Details |
|------|------|--------|---------|
| B1 | New markdown | **Insert** after cell 21 | "## Single Fidelity GP (SFGP) Evaluation" section header with architecture description |
| B2 | New code | **Insert** | `sfgp_prequential_with_config()` function — builds SingleTaskGP with configurable kernel (MaternKernel/RBFKernel wrapped in ScaleKernel), ARD lengthscales for 5D, output transform (raw/manual z-score), noise constraint via GaussianLikelihood |
| B3 | New markdown | **Insert** | SFGP HP grid description (40 configs = 4 kernels × 2 transforms × 5 noise levels) |
| B4 | New code | **Insert** | Generate 40 SFGP configs + evaluation loop → `sfgp_hp_df` DataFrame (40 rows) |
| B5 | New markdown | **Insert** | "### Best SFGP Configuration" |
| B6 | New code | **Insert** | Select best SFGP by NLP (primary), MAE (secondary); display results |

### Phase C: Upgrade 2-Way → 3-Way (modify existing cells)

| Step | Cell | Action | Details |
|------|------|--------|---------|
| C1 | Cell 22 (markdown) | **Edit** | "2-Way Comparison" → "3-Way Comparison: Best NN vs Best SFGP vs Best MFGP" |
| C2 | Cell 23 (comparison) | **Edit** | Add SFGP entry to `comparison_data`; `colors = ['#FF9800', '#2196F3', '#E91E63']`; update title and chart for 3 models |
| C3 | Cell 25 (best model viz) | **Edit** | Add `elif overall_winner == 'SFGP'` branch to re-run SFGP winner; set `winner_color = '#2196F3'` |
| C4 | Cell 26 (markdown) | **Edit** | Add "SFGP = blue" to sensitivity description |
| C5 | Cell 27 (sensitivity) | **Edit** | Add SFGP rows (blue #2196F3) between NN and MFGP loops; update legend with 3 Patch entries; update title |
| C6 | Cell 28 (markdown) | **Edit** | Update "NN and MFGP" → "NN, SFGP, and MFGP" in ranked table description |
| C7 | Cell 29 (ranked table) | **Edit** | Add `sfgp_summary` block; concat 3 DataFrames (NN + SFGP + MFGP) |
| C8 | Cell 30 (conclusions) | **Edit** | Change "two surrogate families" → "three"; add SFGP bullet; update implications |

### Phase D: Execute & Verify

| Step | Action | Success Criterion |
|------|--------|-------------------|
| D1 | Run all cells top-to-bottom | SC-F6-001: All cells execute without errors |
| D2 | Check `nn_hp_df.shape` | SC-F6-007: 45 rows, layers {1,2,3} × nodes {4,5,6} |
| D3 | Check `sfgp_hp_df.shape` | SC-F6-012: 40 rows |
| D4 | Check `mfgp_hp_df.shape` | SC-F6-008: 50 rows (unchanged) |
| D5 | Check `comparison_df.shape` | SC-F6-009: 3 rows (NN, SFGP, MFGP) side-by-side |
| D6 | Check `full_ranked.shape` | SC-F6-002: 135 rows = 810 predictions ÷ 6 steps |
| D7 | Visual check: bar chart colours | SC-F6-005: Orange (NN), Blue (SFGP), Pink (MFGP) |
| D8 | Visual check: best model 3-panel plot | SC-F6-010: predictions vs actuals, error, NLP per step |

## SFGP Function Design

```python
def sfgp_prequential_with_config(X_all, y_all, n_init, config):
    """
    Run Single Fidelity GP prequential evaluation with a specific configuration.
    
    Uses BoTorch SingleTaskGP with configurable kernel, output transform (manual
    z-score or raw), and noise floor. 5D inputs (no fidelity column).
    
    config dict keys:
        kernel_type      : str ('matern_2.5', 'matern_1.5', 'matern_0.5', 'rbf')
        output_transform : str ('raw' or 'standardise')
        noise_lb         : float — lower bound for noise constraint
        label            : str — descriptive label
    
    Returns:
        dict with 'predictions', 'actuals', 'stds', 'metrics'
    """
    n_total = len(y_all)
    n_steps = n_total - n_init
    
    predictions, actuals_list, pred_stds = [], [], []
    
    for step in range(n_steps):
        n_train = n_init + step
        
        # 1. Apply output transform (manual z-score — matches MFGP pattern)
        if config.get('output_transform', 'raw') == 'standardise':
            train_mean = y_all[:n_train].mean()
            train_std  = y_all[:n_train].std() + 1e-10
            y_work = (y_all - train_mean) / train_std
        else:
            y_work = y_all.copy()
            train_mean, train_std = 0.0, 1.0
        
        X_train = torch.tensor(X_all[:n_train], dtype=torch.float64)
        y_train = torch.tensor(y_work[:n_train], dtype=torch.float64).unsqueeze(-1)
        X_test  = torch.tensor(X_all[n_train:n_train+1], dtype=torch.float64)
        y_actual_orig = y_all[n_train]
        
        # 2. Build kernel: ScaleKernel(base_kernel(ard_num_dims=5))
        kernel_type = config.get('kernel_type', 'matern_2.5')
        if kernel_type == 'rbf':
            base_kernel = RBFKernel(ard_num_dims=5)
        else:
            nu_map = {'matern_2.5': 2.5, 'matern_1.5': 1.5, 'matern_0.5': 0.5}
            base_kernel = MaternKernel(nu=nu_map[kernel_type], ard_num_dims=5)
        covar_module = ScaleKernel(base_kernel)
        
        # 3. Build SingleTaskGP
        noise_lb = config.get('noise_lb', 1e-5)
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))
        
        model = SingleTaskGP(
            X_train, y_train,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # Use current hyperparameters if fitting fails
        
        # 4. Predict
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean_work = posterior.mean.item()
            std_work  = posterior.variance.sqrt().item()
        
        # 5. Convert back to original space
        mean_orig = mean_work * train_std + train_mean
        std_orig  = std_work * train_std
        
        predictions.append(mean_orig)
        actuals_list.append(y_actual_orig)
        pred_stds.append(max(std_orig, 1e-10))
    
    preds_arr = np.array(predictions)
    acts_arr  = np.array(actuals_list)
    stds_arr  = np.array(pred_stds)
    
    metrics = compute_metrics(preds_arr, acts_arr, stds_arr)
    
    return {
        'predictions': predictions,
        'actuals': actuals_list,
        'stds': pred_stds,
        'metrics': metrics
    }
```

**Key design decisions:**

1. **Manual z-score** (not BoTorch `Standardize`) — consistent with existing MFGP code pattern
2. **ScaleKernel wrapping** — standard BoTorch/GPyTorch pattern for scaled kernels with ARD
3. **5D input, no fidelity column** — unlike MFGP, SFGP uses raw 5D inputs
4. **Same try/except pattern** — consistent with NN and MFGP eval loops for error handling
5. **Same function signature** — `(X_all, y_all, n_init, config)` matches both NN and MFGP functions

## Complexity Tracking

No constitution violations. No complexity justifications needed.
