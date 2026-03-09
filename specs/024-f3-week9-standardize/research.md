# Research: F3 Week 9 — BoTorch Standardize with Increased Restarts

**Feature**: 024-f3-week9-standardize  
**Date**: 2026-03-09

## R1: BoTorch Standardize(m=1) with SingleTaskGP

**Task**: Determine how `Standardize(m=1)` integrates with `SingleTaskGP` and whether it is the default behavior.

**Decision**: Use the BoTorch default — `SingleTaskGP` applies `Standardize(m=1)` automatically when no `outcome_transform` is specified. Remove all manual z-score code.

**Rationale**: 
- BoTorch's `SingleTaskGP` applies `Standardize(m=1)` by default when `outcome_transform` is not explicitly set
- The existing F3 week 9 notebook manually computes z-score (`y_mean`, `y_std`, `y_standardised`) and explicitly constructs the GP without specifying `outcome_transform`, which means the default Standardize is also applied — resulting in **double standardisation**
- The fix: remove manual z-score code and pass raw `Y_train` to `SingleTaskGP`. The default Standardize handles everything internally.
- F6 week 8 notebook confirms this pattern works correctly with identical GP construction

**Alternatives considered**:
- Explicit `outcome_transform=Standardize(m=1)` import: Works but unnecessary — the default is identical. Adding it explicitly improves readability at the cost of an extra import. **Decision**: Add the explicit import and parameter for clarity per Constitution Principle I (simplicity = each step clearly explained).
- Keep manual z-score with `outcome_transform=None`: Rejected — spec explicitly requires replacing manual z-score with BoTorch Standardize.

## R2: Standardize and Posterior Predictions — Auto-Untransformation

**Task**: Determine whether `model.posterior(X).mean` returns original-scale or standardised-scale values when using `Standardize(m=1)`.

**Decision**: `posterior.mean` returns values in the **original output scale**. Remove all manual un-standardisation in visualisation and LOO cells.

**Rationale**:
- BoTorch Standardize automatically untransforms the posterior when retrieving predictions
- Confirmed by F6 week 8 notebook comments: "Standardize(m=1) auto-untransforms to original space"
- The existing F3 week 9 visualisation code manually un-standardises: `mean_raw = mean_std * y_std_safe + y_mean`. This must be **removed** because the posterior already returns original-scale values.
- The existing F3 week 9 LOO code manually un-standardises predictions: `pred_raw = pred_std * y_loo_std + y_loo_mean`. This must also be **removed**.

**Alternatives considered**:
- Keep manual un-standardisation on top of auto-untransform: Rejected — would result in double un-standardisation (incorrect values)

## ~~R3: Interior Penalty Integration with qLogNEI~~ [REMOVED per clarification 2026-03-09]

> Interior penalty was evaluated during implementation and removed per user decision. The acquisition function uses plain qLogNEI passed directly to `optimize_acqf` without any wrapper.

## R4: LOO Cross-Validation with Standardize

**Task**: Determine how LOO CV changes when using `Standardize(m=1)` instead of manual z-score.

**Decision**: Each LOO fold simply constructs a `SingleTaskGP` with the held-out sample removed. Standardize handles per-fold normalisation automatically. Remove all per-fold z-score recomputation code.

**Rationale**:
- Currently, each LOO fold manually recomputes `y_loo_mean`, `y_loo_std`, `y_loo_std_z` before building the GP. This is 4 lines of z-score code per fold.
- With `Standardize(m=1)`, each fold's GP automatically standardises its training data. No manual statistics needed.
- The predicted value from `model.posterior(x_held).mean` is already in original scale, so the `pred_raw = pred_std * y_loo_std + y_loo_mean` un-standardisation line must be removed.
- This is a strict simplification — fewer lines of code, fewer opportunities for error.

**Alternatives considered**: None — this is the direct consequence of R1 and R2.

## R5: Hyperparameter Naming Convention

**Task**: Clarify distinction between MLL restarts and acquisition restarts to avoid confusion.

**Decision**: Use `N_RESTARTS = 15` for MLL multi-start optimisation (unchanged) and `NUM_RESTARTS_ACQ = 20` for acquisition optimisation (increased from 10). Both names are defined in the hyperparameters cell.

**Rationale**: The existing notebook uses `N_RESTARTS` for MLL and `ACQ_RESTARTS` for acquisition. Renaming `ACQ_RESTARTS` to `NUM_RESTARTS_ACQ` improves clarity and matches the spec's FR-007 naming convention.

**Alternatives considered**:
- Keep `ACQ_RESTARTS` name: Acceptable but inconsistent with spec. **Decision**: Use spec naming for alignment.

## R6: Contour Visualisation with Standardize

**Task**: Confirm that the contour visualisation approach (3D→2D slicing) works with Standardize outputs.

**Decision**: No changes needed to the slicing approach. Grid prediction code simplifies — remove un-standardisation lines since posterior returns original-scale values. Revert to 1×3 layout (matching Week 8) since interior penalty was removed.

**Rationale**:
- The existing approach creates a 50×50 grid for each 2D pair, fixing the third dimension at the best-observed value
- Posterior predictions via `model.posterior(X_grid)` return original-scale mean and variance when using Standardize
- The existing manual un-standardisation (`mean_raw = mean_std * y_std_safe + y_mean`, `std_raw = std_std * y_std_safe`) must be removed
- With interior penalty removed, the 2×3 layout (Row 2: penalty surface) is no longer needed — revert to 1×3 posterior mean layout with white uncertainty contours
