# Notebook Cell Contract: Week 12 Optimisation Notebooks

**Feature**: 035-f1-f8-week12-optimisation
**Date**: 2026-03-18

## Overview

Each week 12 notebook follows a fixed cell sequence. This contract defines the cell types, purposes, and expected outputs for each cell position. Two notebook variants exist:

1. **GP variant** (F1–F6, F8): SFGP surrogate with BoTorch acquisition
2. **NN variant** (F7): Neural network surrogate with MC dropout acquisition

## GP Variant Cell Contract (F1–F6, F8)

### Cell 1: Title (Markdown)

```markdown
# FX — Week 12 Bayesian Optimisation

**Objective**: ...
**Function**: FX (Nd input, 1D output, maximisation)
**Strategy**: [strategy summary from week 10]
```

**Required fields**: Objective, Function dimensions, Strategy summary

### Cell 2: Imports & Configuration (Code)

**Imports**: numpy, matplotlib, itertools.combinations, math
**Config constants**: FUNC_NUM, N_DIMS, N_INITIAL, WEEK (=11), USE_LOG_SCALE (F1 only), DATA_DIR

**Expected output**: None (silent)

### Cell 3: Load Data Header (Markdown)

Text: "## Step 1 — Load Data"

### Cell 4: Load Data (Code)

**Loads**: `updated_inputs - Week {WEEK}.npy`, `updated_outputs - Week {WEEK}.npy`
**Prints**: Function summary (dims, counts, shapes, best/worst), full data table

**Expected output**: Printed summary with n_total, n_submissions, shapes, best/worst values, data table

### Cell 5: Convergence Header (Markdown)

Text: "## Step 2 — Convergence Plot"

### Cell 6: Convergence Plot (Code)

**Computes**: `running_best = np.maximum.accumulate(outputs)`
**Produces**: Line plot with blue initial, orange submissions, grey vertical separator
**F1 special**: Log y-axis with non-positive values clamped to 1e-300

**Expected output**: matplotlib figure (10×5)

### Cell 7: Pair Plots Header (Markdown)

Text: "## Step 3 — 2D Pair Plots"

### Cell 8: Pair Plots (Code)

**Produces**: Grid of 2D scatter plots for all $\binom{d}{2}$ dimension pairs
**Markers**: Blue circles (initial, unmarked), orange circles (submissions, numbered by week), green star (best, s=500, zorder=5)
**Legend**: Must include "Initial", "Submissions", "Best" entries

**Expected output**: matplotlib figure with subplots

### Cell 9: Performance Evaluation Header (Markdown)

Text: "## Step 4 — Performance Evaluation" + current strategy description

### Cell 10: Performance Evaluation (Code)

**Computes**: improvements count, max consecutive no-improve, stalling flag (≥3)
**Prints**: Summary metrics + per-submission table (week, output, best-so-far, improved?)

**Expected output**: Printed metrics table

### Cell 11: Evaluation Summary (Markdown)

Strategy evaluation text: performance analysis, stalling status, key observations

### Cell 12: Strategy Proposals (Markdown)

Proposed strategy improvements (informational — no code changes applied)

### Cell 13: Optimisation Strategy Description (Markdown)

Text: "## Step 5 — Optimisation Run" + numbered strategy description

### Cell 14: Optimisation Imports (Code)

**Imports**: torch, botorch (SingleTaskGP, fit_gpytorch_mll, qLogNoisyExpectedImprovement, optimize_acqf, SobolQMCNormalSampler), gpytorch (MaternKernel, ScaleKernel, GaussianLikelihood, constraints, ExactMarginalLogLikelihood)

**Expected output**: "All imports successful."

### Cell 15: Optimisation Configuration (Code)

**Defines**: All hyperparameter constants (KERNEL_NU, LS bounds, NOISE_LB, N_MLL_RESTARTS, MC_SAMPLES, Q_BATCH, etc.)
**Prints**: Configuration summary

### Cell 16: Data Preparation Header (Markdown)

Transform description (function-specific)

### Cell 17: Data Preparation (Code)

**Converts**: numpy arrays to torch tensors
**Applies**: Function-specific output transform (log, shift, or none)

**Expected output**: Tensor shapes and ranges

### Cell 18: GP Fitting Header (Markdown)

Text: "### Step 5.2 — SFGP Fitting" + kernel description

### Cell 19: GP Fitting (Code)

**Loop**: N_MLL_RESTARTS iterations, each with random seed
**Constructs**: ScaleKernel(MaternKernel(...)) + GaussianLikelihood + SingleTaskGP
**Selects**: Model with lowest negative MLL
**Prints**: Per-restart table + best model hyperparameters

### Cell 20: Acquisition Header (Markdown)

Text: "### Step 5.3 — Acquisition & Selection"

### Cell 21: Acquisition Optimisation (Code)

**Creates**: qLogNoisyExpectedImprovement with configured sampler
**Runs**: optimize_acqf with bounds, q, restarts, raw_samples
**Applies**: Quality gate + distance-based selection
**Prints**: Candidate batch, posterior means, selected point

### Cell 22: Interior Penalty Re-scoring (Code) — *if applicable*

**Applies**: `w(x) = FLOOR + (1−FLOOR) · ∏ sin(πxᵢ)^(2·STEEPNESS)`
**Re-selects**: Based on penalised acquisition values
**Prints**: Per-survivor re-scoring table

### Cell 23: Submission Query (Code)

**Outputs**: Formatted submission block with surrogate config, hyperparameters, proposed coordinates, duplicate check

**Expected output**: Clearly labelled "WEEK 12 SUBMISSION QUERY FOR FUNCTION X" block

### Cell 24: Surrogate Visualisation Header (Markdown) — *F1, F2 only*

Text: "### Step 5.4 — Surrogate Visualisation"

### Cell 25: Surrogate Visualisation (Code) — *F1, F2 only*

**Produces**: 3-panel contour (posterior mean, uncertainty, acquisition surface) with data overlay

### Cell 26: Updated Convergence Header (Markdown)

Text: "### Step 5.5 — Updated Convergence Plot"

### Cell 27: Updated Convergence (Code)

**Produces**: Convergence plot with observed trajectory + proposed point (green star) at predicted value

## NN Variant Differences (F7 only)

Cells 14–23 are replaced with:

- **NN Imports** (Code): torch, torch.nn
- **NN Config** (Code): Architecture, LR, epochs, dropout, MC_SAMPLES, N_CANDIDATES, etc.
- **Z-Score Normalisation** (Code): Manual standardisation of X and y
- **NN Training** (Code): Adam optimiser, MSE loss, epoch loop with loss printing
- **MC Dropout Scoring** (Code): N_CANDIDATES random uniform, MC forward passes, mean+EI blend
- **Interior Penalty & Selection** (Code): Penalised acquisition argmax
- **Submission Query** (Code): Same format as GP variant
- **Gradient Importance** (Code): Input gradient magnitude analysis + 2D slice visualisation

## Validation Rules

1. Every code cell must execute without errors
2. Every visualisation cell must produce a matplotlib figure
3. The submission query must contain coordinates in [0, 0.999999] per dimension
4. The duplicate check must report "OK — unique point"
5. The data week (WEEK constant) must be 11 for all notebooks
6. All hyperparameters must exactly match the week 10 values from research.md
