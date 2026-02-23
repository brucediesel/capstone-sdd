# Implementation Plan: F3 BART Memory Reduction

**Branch**: `005-week7-pe-surrogates` | **Date**: 2026-02-23 | **Spec**: [spec.md](spec.md)

## Summary

`preq-eval-f3.ipynb` crashes with out-of-memory during BART prequential evaluation. The root cause is the original default configuration (`m_trees=50, draws=500, tune=200, chains=4`) and HP sweep containing configs up to `m_trees=200, draws=1000` — all of which require re-training on every one of 7 prequential steps. The fix reduces all BART model sizes (default: `m_trees=20, draws=200, tune=100`; sweep: all `m_trees ≤ 20, draws ≤ 200`) while keeping GP (15 configs) and Random Forest cells entirely unchanged.

## Technical Context

**Language/Version**: Python 3.14.2 (`sdd-dev` pyenv environment)  
**Primary Dependencies**: `pymc`, `pymc-bart` — already installed and used in the notebook  
**Storage**: N/A — changes are confined to notebook cells  
**Testing**: Manual — execute BART cells and confirm no kernel crash, full results table produced  
**Target Platform**: Jupyter notebook running on macOS laptop, ~16 GB RAM  
**Performance Goals**: Each prequential step BART re-train must complete without triggering kernel OOM restart  
**Constraints**: Peak RAM during any single BART call must remain below system available memory (~16 GB)  
**Scale/Scope**: 7 prequential steps × (1 default + 8 sweep) = 63 BART model fits total  

## Constitution Check

*Based on `CONSTITUTION.md` at repo root.*

| Principle | Status | Notes |
|-----------|--------|-------|
| Code as simple as possible, steps clearly explained | ✅ PASS | Reducing model size increases simplicity |
| All code in Jupyter notebooks | ✅ PASS | `preq-eval-f3.ipynb` is the only artefact |
| No unit tests required | ✅ PASS | Manual execution is sufficient |
| Each model must specify hyperparameters with explanations | ✅ PASS | FR-007 requires updating the narrative markdown cell to document new ranges |
| Weekly changes added as new sections, existing cells not replaced | ⚠️ JUSTIFIED | The "no replace" rule applies to the main optimisation notebooks (`f1.ipynb`–`f8.ipynb`) and their weekly submission sections. `preq-eval-f3.ipynb` is a diagnostics/research notebook; editing its BART cells is a bug-fix, not a weekly submission. No SC gate violation. |
| Use BoTorch as default for GP surrogates | ✅ PASS | GP cells (BoTorch SingleTaskGP) remain untouched |
| Visualisations of surrogate + convergence | ✅ PASS | Visualisation cells are not modified |

**Gate result**: PASS — one justified deviation from the "no-replace" rule, scoped only to the diagnostic notebook.

## Project Structure

### Documentation (this feature)

```text
specs/008-f3-bart-reduce/
├── plan.md          ← this file
├── research.md      ← Phase 0 (no unknowns; all decisions pre-resolved)
├── quickstart.md    ← how to run and verify the fix
└── checklists/
    └── requirements.md
```

No `data-model.md` or `contracts/` — this change introduces no new entities or APIs.

### Source Code

```text
functions/f3/
└── preq-eval-f3.ipynb   ← ONLY file modified; three cells changed
```

**Cells to modify** (from notebook review):

| Cell | Type | Change |
|------|------|--------|
| BART default run (line 563) | Code | `m_trees=50, draws=500, tune=200` → `m_trees=20, draws=200, tune=100`. `chains` stays 4. |
| BART HP sweep markdown (lines 572–579) | Markdown | Update documented ranges to match new smaller configs |
| BART HP sweep code (lines 582–623) | Code | Replace 8 configs so all have `m_trees ≤ 20`, `draws ≤ 200`, `tune ≤ 100`, `chains=4` |

**Cells NOT touched**: all GP cells (lines 86–473), Random Forest cell, `compute_metrics`, visualisation cells, comparison cell.

## Phase 0: Research

See [research.md](research.md). All decisions are pre-resolved — no NEEDS CLARIFICATION items remain.

## Phase 1: Design

### Data Model

No new entities. `BART Configuration` dict `{m_trees, draws, tune, label}` already exists; shape and keys are unchanged.

### Contracts

No new API contracts. `bart_prequential_evaluation(X_all, y_all, n_init, m_trees, draws, tune)` signature and return dict are unchanged (FR-004).

### Implementation Steps

**T001 — Update BART default run call** *(US1, FR-001)*  
Change the single invocation line in the "Run BART with Default Hyperparameters" cell:  
```python
# Before
bart_default_results = bart_prequential_evaluation(X_all, y_all, N_INIT, m_trees=50, draws=500, tune=200)
# After
bart_default_results = bart_prequential_evaluation(X_all, y_all, N_INIT, m_trees=20, draws=200, tune=100)
```

**T002 — Replace BART HP sweep configs** *(US1, FR-002, FR-003)*  
Replace the `bart_configs` list with 8 configurations, all with `m_trees ∈ {5, 10, 20}`, `draws ∈ {100, 200}`, `tune ∈ {50, 100}`, `chains=4`.  
Function call stays identical; only the config values change.

**T003 — Update HP sweep markdown narrative** *(US2, FR-007, SC-006)*  
Update the markdown cell preceding the HP sweep to document the new parameter ranges (5–20 trees, 100–200 draws, 50–100 tune) and remove references to `m=100`, `m=200`, `draws=1000`.

**T004 — Verify GP cells unchanged** *(US2, FR-005, SC-003)*  
Run `git diff` after implementation — confirm zero changes to GP code cells.

**T005 — Execute notebook and validate** *(all SCs)*  
Run BART default cell: confirm completion (SC-001).  
Run BART HP sweep: confirm 8-row results table with no unhandled OOM (SC-002).  
Read BART default call in cell: confirm `m_trees=20, draws=200, tune=100` (SC-004).  
Inspect `bart_configs`: confirm all entries have `m_trees ≤ 20, draws ≤ 200` (SC-005).  
Inspect markdown: confirm accurate documentation (SC-006).  
`git diff HEAD~1 HEAD -- functions/f3/preq-eval-f3.ipynb | grep -v BART`: should show no GP/RF changes (SC-003).

**T006 — Commit** *(governance)*  
Commit: `fix(008): reduce BART model sizes in preq-eval-f3.ipynb to prevent OOM`.

## Complexity Tracking

No constitution violations requiring justification — the single "new sections" deviation is a scope clarification, not a violation.
