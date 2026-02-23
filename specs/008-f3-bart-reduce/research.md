# Research: F3 BART Memory Reduction

**Feature**: `specs/008-f3-bart-reduce`  
**Date**: 2026-02-23  
**Status**: Complete — all unknowns resolved before planning

## Summary

No NEEDS CLARIFICATION items were present in the spec. All decisions are fully pre-resolved from notebook code review and knowledge of PyMC-BART memory behaviour. This document records the rationale for each decision.

---

## Decision 1: Target BART parameters

**Decision**: Default run uses `m_trees=20, draws=200, tune=100, chains=4`. HP sweep uses `m_trees ∈ {5, 10, 20}`, `draws ∈ {100, 200}`, `tune ∈ {50, 100}`.

**Rationale**: The original config (`m_trees=50, draws=500, tune=200, chains=4`) causes OOM. Memory consumption scales roughly linearly with `m_trees × draws`. Reducing `m_trees` from 50 to 20 (60% reduction) and `draws` from 500 to 200 (60% reduction) gives an expected ~64% reduction in per-chain memory. With `chains=4` retained, this brings total peak RAM well within 16 GB for 3-dimensional input data and 22 training points.

**Alternatives considered**:
- Reducing `chains` from 4 to 2: Would halve memory but compromise MCMC chain mixing assessment. User updated spec to keep `chains=4`, so this was rejected.
- Using a fully sequential sampler (`cores=1` already set): Already present in the original code; does not reduce memory.
- Using variational inference instead of MCMC: Would require changing the inference backend — too invasive; spec explicitly requires reducing iterations only.

---

## Decision 2: Scope of change — only BART cells

**Decision**: GP cells (15 configs, BoTorch) and Random Forest cells are not touched.

**Rationale**: The spec explicitly requires (FR-005, FR-006) that GP and RF remain unchanged. The OOM problem is isolated to BART's MCMC sampling, which holds large posterior trace arrays in memory. GP (BoTorch L-BFGS-B MLL) and Random Forest (scikit-learn) do not exhibit this behaviour.

**Alternatives considered**: None — the constraint is explicit in the user request.

---

## Decision 3: Retain `bart_prequential_evaluation()` signature

**Decision**: The function definition cell is not modified. Only the call sites (default run cell and HP sweep cell) are changed.

**Rationale**: The function accepts `m_trees`, `draws`, `tune` as keyword arguments. Passing smaller values achieves the memory reduction without any change to the function body. This satisfies FR-004 (return structure unchanged) and SC-003 (downstream cells unaffected).

**Alternatives considered**: Adding a `max_memory_mb` guard inside the function body — rejected as unnecessary complexity; the constitutional principle is simplicity.

---

## Decision 4: 8-configuration HP sweep design

**Decision**: 8 configs varying `m_trees ∈ {5, 10, 20}` × `draws ∈ {100, 200}` × `tune ∈ {50, 100}`, with some combinations omitted to stay at exactly 8.

**Rationale**: FR-002 requires exactly 8 configurations. FR-003 requires at least 2 distinct `m_trees` values and at least 2 distinct `draws` values. The design covers the low end (`m=5, draws=100`) through the maximum allowed (`m=20, draws=200`) to give informative comparison of the tradeoff between model capacity and memory.

**Alternatives considered**: Using a full 3×2×2 = 12-config grid — rejected because it exceeds the FR-002 constraint of exactly 8.

---

## Decision 5: No new library requirements

**Decision**: No package installation needed.

**Rationale**: `pymc` and `pymc-bart` are already installed in the `sdd-dev` environment (confirmed by existing executed cells in the notebook). The parameter changes use the existing API without any new function calls.
