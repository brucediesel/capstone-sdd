# Implementation Plan: F1-F8 Week 9 -- Bayesian Optimisation with Performance Evaluation

**Branch**: `021-f1-f8-week9` | **Date**: 2026-03-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/021-f1-f8-week9/spec.md`

## Summary

Create 8 self-contained Jupyter notebooks (`f1 - week 9.ipynb` through `f8 - week 9.ipynb`) that:
1. Load Week 9 data for each function, retaining identical surrogate/acquisition strategies from Week 8.
2. Enhance all visualisations with a three-colour scheme: blue (initial samples), orange/red (all 9 weekly submissions), green star (proposed point).
3. Add a new Performance Evaluation section at the end of each notebook with code cells computing convergence metrics, exploration spread, and LOO surrogate error, followed by a markdown interpretation cell that diagnoses stalling and proposes strategy changes where needed.

## Technical Context

**Language/Version**: Python 3.11 (sdd-dev environment)
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch, scikit-learn, NumPy, Matplotlib
**Storage**: `.npy` files in `./data/fX/` directories
**Testing**: Manual notebook execution (no unit tests per constitution)
**Target Platform**: macOS (local Jupyter execution)
**Project Type**: Single project -- 8 independent Jupyter notebooks
**Performance Goals**: Each notebook executes end-to-end in < 5 minutes
**Constraints**: Limited memory (BART excluded historically); notebooks must be self-contained
**Scale/Scope**: 8 notebooks, 286-line spec, ~13 cells per notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity | **PASS** | Same strategies as Week 8; performance eval uses basic metrics (running max, pairwise distance, LOO). No complex new frameworks. |
| II. Per-Function Isolation | **PASS** | Each function has its own notebook in `./functions/fX/`. No shared code files. |
| III. Per-Iteration Notebooks | **PASS** | New `fX - week 9.ipynb` notebooks created; existing notebooks untouched. |
| IV. Data Organisation | **PASS** | Loading `updated_inputs/outputs - Week 9.npy` from `./data/fX/`. |
| V. BoTorch & PyTorch Stack | **PASS** | GP functions use BoTorch. F1 (sklearn hurdle) and F7 (PyTorch NN) use approved alternatives. |
| VI. Documentation & Visualisation | **PASS** | All hyperparameters documented. Surrogate + convergence + performance evaluation plots included. |
| VII. Maximisation Objective | **PASS** | All acquisition functions maximise. Stalling detection uses no-new-maximum criterion. |

**Gate result: ALL PASS -- proceed to Phase 0.**

## Project Structure

### Documentation (this feature)

```text
specs/021-f1-f8-week9/
+-- plan.md              # This file
+-- research.md          # Phase 0: LOO patterns, stalling thresholds
+-- data-model.md        # Phase 1: Data entities and relationships
+-- quickstart.md        # Phase 1: Implementation guide
+-- contracts/           # Phase 1: Notebook cell contracts
+-- tasks.md             # Phase 2: (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
functions/
+-- f1/
|   +-- f1.ipynb                 # Historical (frozen)
|   +-- f1 - week 8.ipynb        # Previous iteration (frozen)
|   +-- f1 - week 9.ipynb        # NEW -- this feature
|   +-- preq-eval-f1.ipynb       # Prequential evaluation (frozen)
+-- f2/ ... f8/                   # Same pattern per function
+-- results/
    +-- process_results.ipynb     # Results processing (frozen)

data/
+-- f1/
|   +-- initial_inputs.npy
|   +-- initial_outputs.npy
|   +-- updated_inputs - Week 9.npy   # Input for this feature
|   +-- updated_outputs - Week 9.npy  # Input for this feature
+-- f2/ ... f8/                        # Same pattern
+-- results/
```

**Structure Decision**: Per-function isolation with new iteration notebooks. No shared utilities -- each notebook is fully self-contained per Constitution III.

## Complexity Tracking

> No violations found. All gates pass.
