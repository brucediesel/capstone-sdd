# Feature Specification: F5 Week 9 — Kernel, Standardize & Raw Samples

**Feature Branch**: `026-f5-kernel-standardize` (from `025-f5-remove-penalty`)  
**Created**: 2026-03-09  
**Status**: Active  
**Predecessor**: Penalty removal completed on branch `025-f5-remove-penalty`  
**Input**: Clarification session: boundary-stuck candidates after penalty removal → apply Matérn-1.5, Standardize(m=1), raw_samples=5000

## Clarifications

### Session 2026-03-09

- Q: Should the 3 new changes (kernel, Standardize, raw_samples) extend the current branch or a new feature branch? → A: New feature branch (e.g., `026-f5-kernel-standardize`)
- Q: Should Matérn-1.5 replace Matérn-5/2 outright, or run both and compare? → A: Replace outright (single kernel)
- Q: Should BoTorch Standardize(m=1) replace only the manual z-score (keeping log1p), or replace both transforms? → A: Keep manual log1p, replace only z-score with Standardize(m=1)

## Completed Scope (Branch 025)

> The following user stories were completed on branch `025-f5-remove-penalty` and are preserved here for traceability. All 15 tasks passed validation.
> - US-A: Remove Interior Penalty from Acquisition (P1) ✅
> - US-B: Remove Penalty Visualisation (P1) ✅
> - US-C: Update Documentation and Submission (P2) ✅

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Core Pipeline Changes (Priority: P1)

As a researcher, I want the F5 week 9 notebook to use a Matérn-1.5 kernel with BoTorch Standardize(m=1) outcome transform and 5000 raw samples, so that the GP's rougher assumptions and improved acquisition coverage address the boundary-stuck candidates observed after penalty removal.

**Why this priority**: These are the three functional code changes that directly address the boundary-sticking problem. All downstream cells depend on these changes.

**Independent Test**: Run cells 2→4→8→10 — GP trains with `nu=1.5` and `outcome_transform=Standardize(m=1)`, acquisition uses `raw_samples=5000`, no `y_mean`/`y_std_val`/`y_std` variables exist, inverse transform is `expm1(posterior.mean)`.

**Acceptance Scenarios**:

1. **Given** the GP training cell uses `MaternKernel(nu=2.5, ...)`, **When** the kernel is updated, **Then** it uses `MaternKernel(nu=1.5, ard_num_dims=4)` and prints confirm "Matérn-1.5"
2. **Given** the transform code manually computes `y_mean`, `y_std_val`, `y_std`, **When** Standardize is adopted, **Then** those variables are removed; `Y_train` receives `log1p(y_raw)` directly; GP is constructed with `outcome_transform=Standardize(m=1)`
3. **Given** the acquisition cell uses `raw_samples=3000`, **When** raw_samples is increased, **Then** `optimize_acqf` uses `raw_samples=5000`
4. **Given** the inverse transform is `expm1(pred * y_std_val + y_mean)`, **When** Standardize auto-inverts, **Then** the inverse becomes `expm1(posterior.mean)` with no manual z-score inverse

---

### User Story 2 — Downstream Cell Consistency (Priority: P1)

As a researcher, I want the visualisation, submission, and LOO cells updated to use the simplified inverse transform and reference "Matérn-1.5", so the entire notebook pipeline is internally consistent.

**Why this priority**: Downstream cells that still use old inverse transforms or kernel names will produce incorrect results or misleading labels.

**Independent Test**: Run cells 12, 16, 22 — viz uses `expm1(grid_mu)` inverse and suptitle says "Matérn-1.5", submission prints "Standardize(m=1)", LOO uses `nu=1.5` with `Standardize(m=1)` per fold.

**Acceptance Scenarios**:

1. **Given** the visualisation cell applies `expm1(grid_mu * y_std_val + y_mean)`, **When** Standardize is adopted, **Then** the inverse becomes `expm1(grid_mu)` and suptitle reads "Matérn-1.5"
2. **Given** the LOO cell uses `nu=2.5` and manual z-score per fold, **When** the pipeline is updated, **Then** each fold uses `nu=1.5`, `outcome_transform=Standardize(m=1)`, and the inverse becomes `expm1(pred)`
3. **Given** the submission cell prints "GP Matérn-5/2 ARD (outcome_transform=None)", **When** updated, **Then** it prints "GP Matérn-1.5 ARD (outcome_transform=Standardize(m=1))"

---

### User Story 3 — Documentation Updates (Priority: P2)

As a researcher, I want the notebook title, hyperparameter table, and strategy cell updated to reflect the new kernel, transform, and raw_samples settings, so the notebook accurately documents the current configuration.

**Why this priority**: Documentation consistency is important but secondary to the functional code changes.

**Independent Test**: Title says "Matérn-1.5", hyperparams table shows `nu=1.5`, `raw_samples=5000`, `Standardize(m=1)`, strategy documents the changes.

**Acceptance Scenarios**:

1. **Given** the title references "Matérn-5/2", **When** updated, **Then** it reads "Matérn-1.5"
2. **Given** the hyperparameter table shows `nu=2.5`, `raw_samples=3000`, `outcome_transform=None`, **When** updated, **Then** it shows `nu=1.5`, `raw_samples=5000`, `outcome_transform=Standardize(m=1)`
3. **Given** the strategy section recommends trying Matérn-1.5 and Standardize, **When** updated, **Then** it documents these changes as applied and provides recommendations for future iterations

---

### Edge Cases

- What if Matérn-1.5 still produces boundary-stuck candidates? The notebook still produces valid submissions — LOO MAE and convergence metrics flag issues for evaluation. Strategy cell should recommend further changes (e.g., Matérn-0.5, input warping).
- What if Standardize(m=1) changes LOO MAE significantly? This is expected — the pipeline is mathematically equivalent but numerically more stable. Document the difference.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The `PenalisedAcquisition` class definition MUST be removed from the notebook
- **FR-002**: The `penalised_nei` variable and its usage in `optimize_acqf` MUST be removed; `optimize_acqf` MUST pass `nei` directly as `acq_function`
- **FR-003**: Constants `STEEPNESS`, `FLOOR`, and `EPS_BOUND` MUST be removed from the hyperparameters cell
- **FR-004**: The tightened bounds `BOUNDS_IP` MUST be removed; acquisition MUST use the standard `BOUNDS` (`[0, 1]^4`)
- **FR-005**: The Step 4 interior penalty explanation markdown cell MUST be removed entirely
- **FR-006**: The Step 4 interior penalty code cell MUST be removed entirely (the cell containing `PenalisedAcquisition`, `penalised_nei`, and the penalised optimization)
- **FR-007**: The Step 6 penalty visualisation section (markdown header + code cell) MUST be removed entirely
- **FR-008**: The Step 5 surrogate visualisation MUST reference the base NEI `best_point` instead of penalty-selected `next_x_ip`; the plot title MUST not mention "IP"
- **FR-009**: The Step 8 submission cell MUST show only the base NEI selected point; all penalty diagnostic output and parameter references MUST be removed
- **FR-010**: The notebook title, hyperparameter table, and strategy recommendations MUST be updated to remove all penalty references
- **FR-011**: No changes beyond those specified in FR-012–FR-014 MUST be made. All other settings — acquisition function type (qLogNEI), MC samples (512), q value (4), acquisition restarts (50), distance-based selection logic, MLL restarts (15) — MUST remain unchanged
- **FR-012**: Kernel MUST change from Matérn-5/2 (`nu=2.5`) to Matérn-1.5 (`nu=1.5`), keeping ARD enabled (`ard_num_dims=4`)
- **FR-013**: Manual z-score normalisation (`y_mean`, `y_std_val`, `y_std`) MUST be removed; `BoTorch Standardize(m=1)` MUST be added as `outcome_transform`; manual `log1p` transform MUST be retained; inverse transforms MUST simplify to `expm1(posterior.mean)`
- **FR-014**: Acquisition `raw_samples` MUST increase from 3000 to 5000

### Key Entities

- **F5 Week 9 Notebook**: `functions/f5/f5 - week 9.ipynb` — the single file modified by this feature
- **Interior Penalty Components**: `PenalisedAcquisition` class, `STEEPNESS`/`FLOOR`/`EPS_BOUND` constants, `penalised_nei` variable, `BOUNDS_IP` tightened bounds, Step 4 explanation and code cells, Step 6 visualisation cells

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The notebook executes end-to-end without errors, producing a valid submission query in `0.xxxxxx-0.xxxxxx-0.xxxxxx-0.xxxxxx` format with all values in [0.0, 0.999999]
- **SC-002**: Zero references to `PenalisedAcquisition`, `penalised_nei`, `STEEPNESS`, `FLOOR`, `EPS_BOUND`, or `BOUNDS_IP` exist anywhere in the notebook
- **SC-003**: The surrogate model uses GP Matérn-1.5 (changed from 5/2), ARD, log1p → Standardize(m=1) (replaced manual z-score), 15 MLL restarts
- **SC-004**: The base acquisition configuration uses qLogNEI, q=4, 50 restarts, 5000 raw samples (increased from 3000), distance-based selection
- **SC-005**: All visualisations render correctly — surrogate plots use base NEI point, convergence plot unchanged, no penalty panels present
- **SC-006**: LOO cross-validation executes successfully with updated kernel (`nu=1.5`) and `Standardize(m=1)` configuration, producing valid MAE results

## Assumptions

- The existing base NEI acquisition (Step 3) already produces valid candidates via distance-based selection; removing the penalty simply means using those candidates directly for submission
- The notebook's output transform pipeline changes from manual log1p → manual z-score to manual log1p → BoTorch Standardize(m=1); the kernel changes from Matérn-5/2 to Matérn-1.5; raw_samples increases from 3000 to 5000. All other hyperparameters remain unchanged
- Step 3 (base NEI) remains fully intact — only Step 4 (penalty wrapper), Step 6 (penalty viz), and penalty references in other cells are removed
- Performance evaluation cells (convergence metrics, exploration spread) do not reference interior penalty and need no changes. LOO is updated with the new kernel and Standardize configuration (FR-012, FR-013)
