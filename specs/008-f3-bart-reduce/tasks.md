# Tasks: F3 BART Memory Reduction

**Feature**: `specs/008-f3-bart-reduce`  
**Branch**: `005-week7-pe-surrogates`  
**Plan**: [plan.md](plan.md)

## Phase 1: Setup

- [X] **T001** — Verify notebook state: confirm `preq-eval-f3.ipynb` exists and BART cells are identifiable

## Phase 2: Core Implementation

- [X] **T002** — Update BART default run call: `m_trees=50→20`, `draws=500→200`, `tune=200→100` (FR-001, SC-001, SC-004)
- [X] **T003** — Replace BART HP sweep `bart_configs` list: 8 configs, all `m_trees ≤ 20`, `draws ≤ 200`, `tune ≤ 100`, `chains=4` (FR-002, FR-003, SC-002, SC-005)
- [X] **T004** — Update BART HP sweep markdown narrative: document new parameter ranges (FR-007, SC-006)

## Phase 3: Validation

- [X] **T005** — Verify GP and RF cells are untouched: `git diff` shows no changes to GP or RF code cells (FR-005, FR-006, SC-003)
- [X] **T006** — Execute BART cells in notebook and confirm: default run completes, 8-row HP results table produced, no OOM crash (SC-001, SC-002)

## Phase 4: Commit

- [X] **T007** — Commit: `fix(008): reduce BART model sizes in preq-eval-f3.ipynb to prevent OOM`
- [X] **T008** — Mark all tasks complete in this file
