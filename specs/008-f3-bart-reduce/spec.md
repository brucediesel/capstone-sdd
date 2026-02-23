# Feature Specification: F3 BART Memory Reduction

**Feature Branch**: `005-week7-pe-surrogates`  
**Created**: 2026-02-23  
**Status**: Draft  
**Spec Directory**: `specs/008-f3-bart-reduce`

## Overview

The `preq-eval-f3.ipynb` notebook runs out of memory when executing BART (Bayesian Additive Regression Trees) prequential evaluation. The default BART configuration uses 50 trees, 500 MCMC draws, 200 burn-in samples, and 4 parallel chains — a configuration that exceeds available RAM on the development machine. This feature reduces BART model sizes across the default run and all hyperparameter sweep configurations to allow the notebook to complete successfully, while leaving all GP and Random Forest evaluations unchanged.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - BART Completes Without OOM (Priority: P1)

A student runs `preq-eval-f3.ipynb` end-to-end on a memory-constrained laptop. The BART default run and all 8 BART hyperparameter configurations complete without crashing or running out of memory. Each BART step produces a prediction, error, and uncertainty estimate, and a final results table is displayed.

**Why this priority**: The notebook currently crashes during BART execution, making results unobtainable. This is a blocker — nothing downstream of BART runs.

**Independent Test**: Can be tested by executing only the BART cells (default run + HP sweep). The notebook runs to completion, all 8 configurations produce rows in the results table, and no kernel crash or OOM error occurs.

**Acceptance Scenarios**:

1. **Given** the notebook is opened with `preq-eval-f3.ipynb`, **When** the BART default run cell is executed, **Then** the cell completes without error and prints one prediction per prequential step plus MAE, NLP, and 95% coverage metrics.
2. **Given** the BART default run has completed, **When** the BART HP sweep cell is executed, **Then** all 8 configurations run to completion, errors are printed per step, and a results table is displayed with no NaN rows caused by OOM crashes.
3. **Given** the BART default run uses `m_trees=20, draws=200, tune=100, chains=2`, **When** the notebook is run on a machine with 16 GB RAM, **Then** peak memory usage during BART training does not trigger a kernel restart.

---

### User Story 2 - GP and RF Evaluations Unchanged (Priority: P2)

A student verifies that reducing BART model sizes has not altered any other part of the notebook. All GP hyperparameter configurations and the Random Forest evaluation produce identical results to the version before this change.

**Why this priority**: The user explicitly requires that only BART is changed. Accidental modification of GP or RF cells would invalidate those results.

**Independent Test**: Execute only GP and RF cells before and after the change. Output values (MAE, NLP, coverage) must be identical.

**Acceptance Scenarios**:

1. **Given** the GP prequential evaluation code and its 15 hyperparameter configurations, **When** those cells are executed, **Then** all 15 configurations produce the same metrics as before this change, and no GP cell is modified.
2. **Given** the Random Forest prequential evaluation code, **When** those cells are executed, **Then** results are identical and no RF cell is modified.
3. **Given** the final model comparison cell, **When** it is executed, **Then** the comparison table includes GP, BART (with new smaller config), and RF results side by side.

---

### Edge Cases

- What if a BART configuration still runs out of memory with the reduced settings? The cell must gracefully catch the exception, log "FAILED: OOM", append NaN to the results table, and continue to the next configuration.
- What if reducing draws/tune degrades BART predictive quality so severely that no configuration is meaningful? The results table and comparison chart will still be produced; interpretation is the student's responsibility in the notebook narrative.
- What if a configuration with very few trees (m=5) produces degenerate predictions (infinite NLP)? The NLP metric is already computed with a numerical clip in `compute_metrics`; degenerate rows will show large NLP values rather than crashing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The BART default run in `preq-eval-f3.ipynb` MUST use `m_trees=20, draws=200, tune=100, chains=2` as its baseline parameters.
- **FR-002**: The BART hyperparameter sweep MUST contain exactly 8 configurations, all using `m_trees ≤ 20`, `draws ≤ 200`, `tune ≤ 100`, and `chains = 2`.
- **FR-003**: The BART hyperparameter sweep configurations MUST vary `m_trees` across at least two distinct values (e.g., 5, 10, 20) and vary `draws` across at least two values (e.g., 100, 200).
- **FR-004**: The `bart_prequential_evaluation()` function signature and return structure MUST remain unchanged so that downstream visualisation and comparison cells continue to work without modification.
- **FR-005**: All 15 GP hyperparameter configurations MUST remain exactly as they are; no GP cell may be modified.
- **FR-006**: The Random Forest evaluation cells MUST remain exactly as they are; no RF cell may be modified.
- **FR-007**: The notebook narrative markdown cell that precedes the BART HP sweep MUST be updated to reflect the new, smaller configuration ranges so the documentation is accurate.
- **FR-008**: The final model comparison cell MUST remain functional and include BART results alongside GP and RF results.

### Key Entities

- **BART Configuration**: A parameter set `{m_trees, draws, tune, chains, label}` passed to `bart_prequential_evaluation()`. After this change, all configurations satisfy `m_trees ≤ 20`, `draws ≤ 200`, `tune ≤ 100`, `chains = 2`.
- **Prequential Step**: One iteration of train-on-first-N, predict-on-N+1. F3 uses 15 initial training points and evaluates the remaining samples one at a time.
- **Results Table**: A pandas DataFrame with columns `label`, `MAE`, `NLP`, `Coverage_95` — one row per configuration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The BART default run cell completes successfully (no error, no OOM crash) on a 16 GB RAM machine.
- **SC-002**: All 8 BART HP sweep configurations either complete successfully or fail gracefully with a caught exception, producing a full results table with no unhandled kernel crash.
- **SC-003**: No GP cell and no RF cell in `preq-eval-f3.ipynb` is modified by this change (verified by `git diff` showing zero changes to GP and RF code cells).
- **SC-004**: The BART default configuration uses `m_trees=20, draws=200, tune=100, chains=2` (verified by reading the notebook cell).
- **SC-005**: The BART HP sweep contains only configurations with `m_trees ≤ 20` and `draws ≤ 200` (verified by inspecting `bart_configs` list).
- **SC-006**: The markdown cell describing the BART HP sweep accurately documents the new smaller parameter ranges.

## Assumptions

- The development machine has approximately 16 GB RAM. Reducing to `chains=2, draws=200, m_trees=20` is expected to reduce peak memory by roughly 70% compared to the original `chains=4, draws=500, m_trees=50` setting.
- The 7 prequential evaluation steps for F3 (15 initial + 7 subsequent samples) each require a full BART model re-train. The bottleneck is the MCMC sampling, not the data size.
- PyMC-BART version installed is compatible with `chains=2`; no additional library changes are required.
- Reducing MCMC samples may increase variance in BART posterior estimates; this is an acceptable trade-off for notebook executability and is noted in the spec. The prequential evaluation still produces valid (if noisier) uncertainty estimates.
- The `compute_metrics()` helper already handles degenerate predictions (NaN, inf) gracefully via clipping.
- The `random_seed=42` used in BART calls remains unchanged to preserve reproducibility given the same parameters.

## Dependencies

- Extends specification `005-week7-pe-surrogates` (same branch).
- Notebook: `functions/f3/preq-eval-f3.ipynb` — only BART cells are modified.
- No changes to data files, imports, or helper functions outside BART-specific code.
