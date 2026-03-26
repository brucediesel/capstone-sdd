# Implementation Plan: Project README & Documentation Update

**Branch**: `037-project-readme` | **Date**: 26 March 2026 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/037-project-readme/spec.md`

## Summary

Create a comprehensive README.md for the capstone SDD black-box optimisation project, and update the existing modelcards.md and datasheets.md to reflect the full 13-week campaign (Weeks 3–13). The README documents per-function strategies, results with convergence plot references, critical evaluation, and lessons learnt. All three files are Markdown documentation — no code changes required.

## Technical Context

**Language/Version**: Markdown (no code execution)
**Primary Dependencies**: None — documentation-only deliverable
**Storage**: Three Markdown files in project root: README.md (new), modelcards.md (update), datasheets.md (update)
**Testing**: Manual review against acceptance scenarios in spec.md
**Target Platform**: GitHub repository / local file system
**Project Type**: Documentation within single Jupyter notebook project
**Performance Goals**: N/A
**Constraints**: Must be consistent with data in `functions/results/process_results.ipynb` and weekly notebooks
**Scale/Scope**: 3 files, ~8 sections per function × 8 functions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | PASS | Documentation files are inherently simple. No code complexity introduced. |
| II. Per-Function Isolation | PASS | README summarises each function independently. No cross-function code changes. |
| III. Per-Iteration Notebooks | PASS | No notebooks are created or modified. README references existing notebooks. |
| IV. Data Organisation | PASS | No data files created or modified. Documentation references existing data paths. |
| V. BoTorch & PyTorch Stack | N/A | No code changes. |
| VI. Documentation & Visualisation | PASS | This feature directly fulfils the documentation requirement. README references convergence visualisations from process_results.ipynb. |
| VII. Maximisation Objective | PASS | Results reported as best-observed maximum values. |

**Gate result**: PASS — no violations.

## Project Structure

### Documentation (this feature)

```text
specs/037-project-readme/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (document structure model)
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Files (repository root)

```text
README.md                # NEW — project overview, strategies, results, evaluation, lessons
modelcards.md            # UPDATE — extend from Week 9 to full Weeks 3–13 evolution
datasheets.md            # UPDATE — extend from Week 9 to Week 13 final dataset sizes
```

### Referenced Files (read-only)

```text
functions/results/process_results.ipynb    # Convergence plots
functions/f1/ through functions/f8/        # Weekly notebooks (strategy source)
data/f1/ through data/f8/                  # Dataset files (.npy)
CONSTITUTION.md                            # Project governance
```

**Structure Decision**: Documentation-only feature. Three Markdown files at root level — one new (README.md), two existing (modelcards.md, datasheets.md). No source code directories needed.

## Complexity Tracking

No constitution violations to justify.

---

## Phase 0: Research

**Output**: [research.md](research.md) — all unknowns resolved.

### Research Topics Investigated

| ID | Topic | Status | Key Finding |
|----|-------|--------|-------------|
| R1 | F1–F8 strategy evolution (Weeks 3–13) | RESOLVED | Full per-function timelines documented — surrogates, acquisitions, transforms, hyperparameters, pivots |
| R2 | Final dataset sizes (Week 13) | RESOLVED | numpy-verified: F1=23, F2=23, F3=28, F4=43, F5=33, F6=33, F7=43, F8=53 (13 points added per function) |
| R3 | Current state of modelcards.md | RESOLVED | Covers Week 9 only; needs extension to Weeks 3–13 in Details and Performance sections |
| R4 | Current state of datasheets.md | RESOLVED | Covers Week 9 only; needs Composition, Collection Process, and Distribution updates to Week 13 |
| R5 | Convergence plot referencing | RESOLVED | Reference process_results.ipynb by path and description; no image embedding |
| R6 | README structure design | RESOLVED | Per-function grouped sections (strategy + results + evaluation) preferred over separate cross-function sections |

### Best Outputs (Week 13, numpy-verified)

| Function | Dim | Best Output | Interpretation |
|----------|-----|-------------|----------------|
| F1 | 2D | 7.71 × 10⁻¹⁶ | Essentially zero — radiation source never located |
| F2 | 2D | 0.674 | Moderate positive — log-likelihood optimised |
| F3 | 3D | −0.0114 | Near zero (negative) — close to optimum |
| F4 | 4D | 0.532 | Moderate positive |
| F5 | 4D | 8,662.4 | Large positive — chemical yield maximised (3.1× improvement) |
| F6 | 5D | −0.111 | Negative (closer to zero is better) — 85% improvement |
| F7 | 6D | 2.305 | Moderate positive — ML accuracy tuned |
| F8 | 8D | 9.982 | Strong positive — 8D optimisation successful |

---

## Phase 1: Design & Contracts

**Prerequisites**: research.md complete ✓

### Document Structure (data-model.md)

See [data-model.md](data-model.md) for full entity definitions. Summary:

**README.md** (new file):
1. Title & Project Overview
2. Per-function sections ×8 (strategy → results → evaluation)
3. Results summary table (all functions)
4. Convergence plots reference
5. Lessons learnt (≥5 insights)
6. Project structure
7. References to modelcards.md, datasheets.md

**modelcards.md** (update existing):
- Extend introduction from "Weeks 3–9" to "Weeks 3–13"
- Add strategy evolution timeline to each function's Details section
- Update Performance section with Week 13 metrics

**datasheets.md** (update existing):
- Extend introduction from "Weeks 3–9" to "Weeks 3–13"
- Update Composition with Week 13 dataset sizes
- Update Collection Process with extended timeline
- Update Distribution file lists through Week 13

### Contracts

No API contracts required — this is a documentation-only feature. The "contracts" are the document structures defined in data-model.md and the cross-document consistency rules:

| Constraint | Scope | Rule |
|-----------|-------|------|
| Best output values | README ↔ modelcards | Must match numpy-verified Week 13 values |
| Dataset sizes | README ↔ datasheets | Must match numpy-verified Week 13 counts |
| Surrogate descriptions | README ↔ modelcards | Strategy summaries must be consistent |
| Temporal scope | All 3 files | Must state "Weeks 3–13" consistently |
| File paths | README references | Must use correct relative paths |

### Implementation Tasks (Phase 2 preview)

| Task | File | Effort | Dependencies |
|------|------|--------|-------------|
| T1: Create README.md | README.md (new) | Medium | research.md data |
| T2: Update modelcards.md | modelcards.md | Medium | research.md R1, R3 |
| T3: Update datasheets.md | datasheets.md | Low | research.md R2, R4 |
| T4: Cross-document validation | All 3 files | Low | T1, T2, T3 |

### Quickstart

See [quickstart.md](quickstart.md) for implementation order and validation checklist.

---

## Constitution Re-Check (Post-Design)

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Simplicity | PASS | Three Markdown files, no code complexity |
| II. Per-Function Isolation | PASS | Each function documented independently in all three files |
| III. Per-Iteration Notebooks | PASS | No notebooks created or modified |
| IV. Data Organisation | PASS | No data files created or modified |
| V. BoTorch & PyTorch Stack | N/A | No code changes |
| VI. Documentation & Visualisation | PASS | Feature directly fulfils documentation requirements; references existing convergence visualisations |
| VII. Maximisation Objective | PASS | Results reported as best-observed maximum values |

**Post-design gate result**: PASS — no violations. Design is consistent with constitution v1.1.0.
