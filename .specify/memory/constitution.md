<!--
Sync Impact Report
- Version change: 0.0.0 → 1.1.0
- Modified principles:
  - Weekly Updates → Per-Iteration Notebooks (renamed + redefined)
- Added sections: Optimization Methodology, Documentation Requirements
- Removed sections: none
- Templates requiring updates:
  - .specify/templates/plan-template.md — ⚠ pending (generic template, no
    function-specific references to update)
  - .specify/templates/spec-template.md — ⚠ pending (generic template)
  - .specify/templates/tasks-template.md — ⚠ pending (generic template)
- Follow-up TODOs:
  - RATIFICATION_DATE set to project start estimate; refine if known
  - Existing notebooks (f1-f8) are NOT renamed by this amendment
-->

# Capstone SDD Black-Box Optimisation Constitution

## Core Principles

### I. Simplicity
Code MUST be as simple as possible with each step clearly explained.
All code MUST be submitted as Jupyter notebooks. No unit tests are
required. Complexity MUST be justified if introduced.

### II. Per-Function Isolation
Each of the 8 black-box optimisation problems (f1–f8) MUST be solved
in its own folder within `./functions/`. The original notebook for each
function is `fX.ipynb`.

### III. Per-Iteration Notebooks
Each weekly iteration MUST be implemented in a **new notebook** named
`fX - week Y.ipynb` (e.g., `f1 - week 7.ipynb`) stored in the same
folder as the original function notebook (`./functions/fX/`). The
original `fX.ipynb` contains all historical weekly sections up to the
point this convention was adopted and MUST NOT be modified further.
Each iteration notebook MUST be self-contained: imports, data loading,
surrogate fitting, acquisition, visualisation, and submission query.
Existing notebooks from previous iterations MUST NOT be modified —
previous work is preserved as-is. The current iteration's notebook
may be updated until finalised.

### IV. Data Organisation
Data for each problem is stored in `./data/fX/`. Initial data uses
`initial_inputs.npy` / `initial_outputs.npy`. Weekly updates use
`updated_inputs - Week X.npy` / `updated_outputs - Week X.npy`.

### V. BoTorch & PyTorch Stack
BoTorch is the default library for Gaussian Process surrogates.
Additional surrogates (polynomial, tree-based, neural network) MAY
use scikit-learn or PyTorch. Acquisition functions MUST match the
surrogate type.

### VI. Documentation & Visualisation
Every model MUST specify hyperparameters and rationale for their
values. Every model MUST provide visualisations of the surrogate
function and convergence of the objective. Problem context (input/
output dimensions, background) MUST inform visualisation design.

### VII. Maximisation Objective
All 8 problems are maximisation tasks. Acquisition functions and
comparisons MUST treat higher objective values as better.

## Optimization Workflow

1. Train model on current data.
2. Propose next sample point via acquisition function.
3. Submit sample; receive result the following week.
4. Add result to dataset; create new iteration notebook.
5. Repeat from step 1.

## Governance

- This constitution supersedes all other development practices for
  the capstone SDD project.
- Amendments MUST be documented with version bump, rationale, and
  date.
- Versioning follows semantic versioning: MAJOR (breaking governance
  change), MINOR (new principle or material expansion), PATCH
  (clarification or typo fix).
- All specs and plans MUST verify compliance with this constitution
  before implementation begins.

**Version**: 1.1.0 | **Ratified**: 2025-01-15 | **Last Amended**: 2025-02-25
