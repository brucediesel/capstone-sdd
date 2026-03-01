# Implementation Plan: F2–F8 Week 8 — Bayesian Optimisation Iteration

**Branch**: `020-f2-f8-week8` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/020-f2-f8-week8/spec.md`

## Summary

Create 7 new self-contained Jupyter notebooks (`f2 - week 8.ipynb` through `f8 - week 8.ipynb`) that load Week 8 data, fit the same surrogate model as Week 7, optimise the same acquisition function, and propose the next sample point for Week 9. Each notebook replicates its function's Week 7 strategy exactly, changing only the data file path and sample count expectations.

## Technical Context

**Language/Version**: Python 3.14.2 (Jupyter kernel `sdd-dev`)  
**Primary Dependencies**: BoTorch, GPyTorch, PyTorch (F2–F6, F8); PyTorch + NumPy (F7 NN)  
**Storage**: `.npy` files in `./data/fX/`  
**Testing**: Manual execution — each notebook runs end-to-end without errors (Constitution I)  
**Target Platform**: macOS (local Jupyter)  
**Project Type**: Jupyter notebooks — no src/ directory  
**Performance Goals**: Each notebook completes in < 5 minutes  
**Constraints**: Each notebook must be fully self-contained (Constitution III)  
**Scale/Scope**: 7 notebooks, ~60–80 code cells total

### Week 8 Data Shapes (verified)

| Function | X shape | y shape | y range | Best value | Best idx |
|----------|---------|---------|---------|------------|----------|
| F2 | (18, 2) | (18,) | [-0.066, 0.674] | 0.674355 | 10 |
| F3 | (23, 3) | (23,) | [-0.399, -0.031] | -0.031427 | 21 |
| F4 | (38, 4) | (38,) | [-32.626, 0.532] | 0.532175 | 32 |
| F5 | (28, 4) | (28,) | [0.113, 3394.680] | 3394.679933 | 26 |
| F6 | (28, 5) | (28,) | [-2.571, -0.206] | -0.205600 | 26 |
| F7 | (38, 6) | (38,) | [0.003, 2.305] | 2.304991 | 33 |
| F8 | (48, 8) | (48,) | [5.592, 9.982] | 9.982473 | 47 |

### Per-Function Week 7 Code Research

#### F2 (2D) — SFGP + qLogNEI

**Cell structure** (8 cells in Week 7 section):

**Imports**:
```python
import numpy as np, matplotlib.pyplot as plt, torch, warnings
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
```

**Surrogate**: `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-3))`, `Normalize(d=2)` input transform, no output standardisation, `.double()`, single `fit_gpytorch_mll` (no restart loop).

**Acquisition**: `qLogNoisyExpectedImprovement`, `X_baseline=X_train_t`, `prune_baseline=True`, `q=1`, `num_restarts=10`, `raw_samples=512`, bounds `[[0,0],[1,1]]`.

**Viz**: 3-panel (18×5): posterior mean (viridis), std (YlOrRd), NEI surface (plasma). 50×50 grid. Red scatter + yellow star. Convergence: running max, boundary axvline at 10.5.

**Constants**: `KERNEL='matern15'`, `NOISE_LB=1e-3`, `ARD=True`, `INPUT_NORM=True`, `N_RESTARTS=10`, `RAW_SAMPLES=512`.

---

#### F3 (3D) — SFGP + qLogNEI

**Cell structure** (8 cells in Week 7 section):

**Imports**:
```python
import copy, warnings, numpy as np, torch, matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
```

**Surrogate**: `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`, no input transform, manual z-score output standardisation. 15 restarts with `torch.manual_seed(seed)`, best model via `copy.deepcopy`. Init: `LENGTHSCALE_INIT=0.25`, `SIGNAL_VAR_INIT=1.0`, `NOISE_VAR_INIT=0.1`.

**Acquisition**: `qLogNoisyExpectedImprovement`, `X_baseline=X_train`, `prune_baseline=True`, `q=1`, `num_restarts=10`, `raw_samples=512`, bounds `[[0,0,0],[0.999999,0.999999,0.999999]]`.

**Viz**: 3-panel pairwise 2D slices (18×5): pairs (0,1), (0,2), (1,2), third dim fixed at best_point. mean (viridis) + 2σ contours (white). 50×50 grid. Labels: Compound A/B/C. Convergence: boundary at 15.5.

**Constants**: `N_RESTARTS=15`, `LENGTHSCALE_INIT=0.25`, `SIGNAL_VAR_INIT=1.0`, `NOISE_VAR_INIT=0.1`, `JITTER=1e-6`.

---

#### F4 (4D) — MFGP + MF-qNEI

**Cell structure** (8 cells in Week 7 section):

**Imports**:
```python
import copy, warnings, numpy as np, torch, matplotlib.pyplot as plt
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
```

**Surrogate**: `SingleTaskMultiFidelityGP` with `nu=2.5`, `linear_truncated=True`, `data_fidelities=[4]`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-4))`, no input transform, manual z-score output. Fidelity column (all 1.0) appended → `X_mf` shape (N, 5). 15 restarts, best via `copy.deepcopy`.

**Acquisition**: `qLogNoisyExpectedImprovement`, `X_baseline=X_mf`, `prune_baseline=True`, `sampler=SobolQMCNormalSampler(64)`, `q=4`, `num_restarts=20`, `raw_samples=512`, `fixed_features={4: 1.0}`, bounds 5D. Best of 4 by posterior mean (de-standardised).

**Viz**: 2-panel (16×6): top-2 dims by shortest ARD lengthscales. mean (viridis), std (plasma). 80×80 grid. Convergence: boundary at 30.5.

**Constants**: `N_RESTARTS=15`, `noise_lb=1e-4`, `q=4`, `MC_samples=64`, `acq_restarts=20`, `raw_samples=512`, `GRID_RES=80`.

---

#### F5 (4D) — GP + qLogNEI + Interior Penalty

**Cell structure** (14 cells including interior penalty section):

**Imports**:
```python
import numpy as np, torch, copy, matplotlib.pyplot as plt, warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
```

**Surrogate**: `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-6))`, `outcome_transform=None` (explicitly disables default Standardize). Manual `log1p` → z-score. Init: `lengthscale=0.5`, `noise=0.1*Y_train.var()`, `outputscale=1.0`. 15 restarts.

**Acquisition**: `qLogNoisyExpectedImprovement`, `sampler=SobolQMCNormalSampler(512)`, `prune_baseline=True`, `q=4`, `num_restarts=50`, `raw_samples=3000`, bounds `[0,1]⁴`. Distance-based selection: filter to `pred_means_orig >= median`, pick farthest (Euclidean via `torch.cdist`). Inverse: `np.expm1(pred_means_std * y_std_val + y_mean)`.

**Interior Penalty**: `STEEPNESS=1.0`, `FLOOR=0.01`. **Multiplicative on posterior mean**: `weighted_means = pred_means_orig * interior_weight`. Median filter on weighted means, then farthest from data.

**Viz**: 3-panel (18×5): mean (viridis), std (magma), dim relevance (1/ℓ, steelblue). 80×80 grid. IP: 3-panel (mean, std, penalised mean plasma). Convergence: boundary at 26.5.

**Constants**: `N_RESTARTS=15`, `STEEPNESS=1.0`, `FLOOR=0.01`.

---

#### F6 (5D) — SFGP + qLogNEI + Interior Penalty (Rank-Based)

**Cell structure** (~9 cells including IP re-scoring):

**Imports**: Same as F5 (SingleTaskGP family), plus `copy`, `warnings`, `re`.

**Surrogate**: `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=5))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-2))`, default `Standardize(m=1)` (NOT disabled). Init: `lengthscale=0.5`, `noise=0.2`, `outputscale=1.0`. 15 restarts. Post-fit assertion: `noise >= 1e-2`.

**Acquisition**: `qLogNoisyExpectedImprovement`, `sampler=SobolQMCNormalSampler(512)`, `prune_baseline=True`, `q=4`, `num_restarts=50`, `raw_samples=3000`. **Feasibility bounds**: `[[0.01,0.01,0.01,0.01,0.10], [1.0,1.0,1.0,1.0,1.0]]`. Distance-based: filter to `pred_means >= median`, farthest. Fallback to highest mean if none above median.

**Interior Penalty**: `STEEPNESS=1.0`, `FLOOR=0.01`. **RANK-BASED** (not multiplicative) because all outputs negative. Ranks: `rank_mean = argsort(argsort(pred_means)) + 1`, `rank_weight = argsort(argsort(interior_weight)) + 1`, `combined_score = rank_mean + rank_weight`. Median filter on combined_score, farthest from data. Assert: `best_point[4] >= 0.10`.

**Viz**: 3-panel (18×5): mean (viridis), std (magma), dim relevance (5 bars). 80×80 grid. IP: 3-panel (mean, std, penalty RdYlGn with vmin=FLOOR). Candidates scattered with size ∝ combined_score. Convergence: boundary markers, IP convergence: observed line, running best dashed, IP-selected hline (red), raw-best hline (orange if different).

**Constants**: `N_RESTARTS=15`, `STEEPNESS=1.0`, `FLOOR=0.01`. Ingredient names: `['flour', 'sugar', 'eggs', 'butter', 'milk']`.

---

#### F7 (6D) — Neural Network + MC Dropout EI + Interior Penalty

**Cell structure** (8 cells):

**Imports**:
```python
import numpy as np, matplotlib.pyplot as plt, torch
import torch.nn as nn, torch.optim as optim
```
**No BoTorch/GPyTorch** — pure PyTorch NN surrogate.

**Surrogate**: Custom `SurrogateNN(nn.Module)` — `6→5→5→1`, `nn.ReLU()`, `nn.Dropout(p=0.1)` after each hidden layer. `optim.Adam(lr=0.005)`, `nn.MSELoss()`, 200 epochs, `torch.manual_seed(42)`. Z-score normalisation on both X (per-dim, `+1e-8` stability) and y. Training loss plotted log-scale. R² on original scale.

**Acquisition**: MC Dropout EI. `MC_SAMPLES=50` stochastic forward passes (`model.train()`). `N_CANDIDATES=20000` uniform random in `[0,1]⁶` (`np.random.seed(42)`). EI: `np.mean(np.maximum(mc_preds_orig - y_best, 0), axis=0)`. No `optimize_acqf`. Selection: `np.argmax(penalised_ei)`, fallback to `np.argmax(interior_weight)` if all EI=0.

**Interior Penalty**: `STEEPNESS=0.1`, `FLOOR=0.01`. **Multiplicative on EI**: `penalised_ei = ei * interior_weight`. Works because F7 outputs are all positive → EI ≥ 0.

**Viz**: Training loss curve (10×4, log scale). 3-panel (18×5): NN mean (viridis), MC uncertainty (YlOrRd), IP heatmap (RdYlGn). 50×50 grid, fixed dims at best observed. Named axes: `['learning_rate', 'reg_strength', 'n_layers', 'dropout', 'batch_size', 'optimizer']`. Convergence: running best, IP-selected mean (green hline), raw-EI best (orange dashed). Weekly boundaries at 30.5, 31.5, 32.5, 33.5, 36.5.

**Constants**: `LEARNING_RATE=0.005`, `EPOCHS=200`, `DROPOUT=0.1`, `MC_SAMPLES=50`, `N_CANDIDATES=20000`, `STEEPNESS=0.1`, `FLOOR=0.01`.

---

#### F8 (8D) — SFGP + qEI

**Cell structure** (8 cells):

**Imports**:
```python
import numpy as np, torch, matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
```
Fallback: `from torch.quasirandom import SobolEngine`

**Surrogate**: `SingleTaskGP` with `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))`, `GaussianLikelihood(noise_constraint=GreaterThan(1e-7))`, default `Standardize(m=1)`. No init overrides, single `fit_gpytorch_mll` (no restart loop).

**Acquisition**: `qExpectedImprovement` (NOT qLogNEI), `best_f=Y_train.max().item()+XI` where `XI=0.01`, `sampler=SobolQMCNormalSampler(256)`, `q=1`, `num_restarts=30`, `raw_samples=4096`, bounds `[0,1]⁸`. **Fallback**: if `acq_val <= 0`, generate 4096 Sobol candidates → pick highest posterior mean.

**Viz**: Feature importance bar chart (8×4): top-2 red, others blue, horizontal, inverted y. 3-panel (18×5): mean (viridis), std (magma), analytic EI (plasma, using `ExpectedImprovement` for speed). 50×50 grid. Best observed = white star + black edge, candidate = red X. Convergence: running best (blue), observations (grey scatter). Weekly boundaries: `{40.5: 'Initial→Wk5', 45.5: 'Wk5→Wk6', 46.5: 'Wk6→Wk7'}`.

**Constants**: `XI=0.01`, `MC_SAMPLES=256`, `NUM_RESTARTS=30`, `RAW_SAMPLES=4096`.

---

### Key Discrepancies: Spec vs. Code (All Resolved)

| Item | Spec-020 says | Code does | Resolution |
|------|---------------|-----------|------------|
| F7 dropout | p=0.1 | p=0.1 | Match (corrected in clarify session 1) |
| F7 epochs | 200 | 200 | Match (corrected in clarify session 1) |
| F7 STEEPNESS | 0.1 | 0.1 | Match (noted in spec) |
| F2 acquisition | qLogNEI | qLogNEI | Match (corrected in clarify session 1) |
| F3 acquisition | qLogNEI | qLogNEI | Match (corrected in clarify session 2) |
| F3 noise floor | ≥ 1e-6 | GreaterThan(1e-6) | Match (corrected in clarify session 2) |
| F3/F5 MLL restarts | 15 | 15 | Match (corrected in clarify session 2) |
| F2/F8 MLL restarts | Not specified | Single fit (no loop) | Code is ground truth |

### Implementation Strategy

**Template Pattern**: Each notebook follows the same high-level structure:
1. Markdown header with week/strategy documentation
2. Hyperparameter constants cell
3. Data loading + validation cell
4. Surrogate training cell
5. Acquisition + candidate selection cell
6. Surrogate visualisation cell(s)
7. Convergence plot cell
8. Submission formatting cell

**Complexity Groups**:
- **Group A** (simple GP, no IP): F2, F3, F8
- **Group B** (GP + interior penalty): F5, F6
- **Group C** (special surrogate): F4 (MFGP), F7 (NN)

**Recommended implementation order**: F2 → F8 → F3 → F4 → F5 → F6 → F7  
Rationale: Start with simplest (F2, 2D), then build up complexity. F8 next as it's a standard GP but 8D. F3 adds multi-restart. F4 adds multi-fidelity. F5/F6 add interior penalty. F7 is unique (NN surrogate).

**Week-to-Week Changes** (what changes from Week 7 → Week 8):
- Data file path: `Week 7` → `Week 8`
- Expected sample count: +1 for each function
- Convergence plot boundary markers: add new `Wk7→Wk8` boundary
- All surrogate hyperparameters, acquisition parameters, and strategies remain **identical**

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity | PASS | Each notebook replicates Week 7 code — no new complexity introduced |
| II. Per-Function Isolation | PASS | Each function gets its own notebook in `./functions/fX/` |
| III. Per-Iteration Notebooks | PASS | New `fX - week 8.ipynb` created; original `fX.ipynb` NOT modified |
| IV. Data Organisation | PASS | Loads from `./data/fX/updated_*- Week 8.npy` |
| V. BoTorch & PyTorch Stack | PASS | F2–F6, F8 use BoTorch; F7 uses PyTorch NN (allowed per principle) |
| VI. Documentation & Visualisation | PASS | Each notebook has hyperparameter tables, surrogate viz, convergence plots |
| VII. Maximisation Objective | PASS | All acquisition functions maximise; higher output = better |

**Gate result: PASS** — No violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/020-f2-f8-week8/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (N/A — notebooks only)
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
functions/
├── f2/
│   └── f2 - week 8.ipynb    # NEW — 2D SFGP + qLogNEI
├── f3/
│   └── f3 - week 8.ipynb    # NEW — 3D SFGP + qLogNEI (15 restarts)
├── f4/
│   └── f4 - week 8.ipynb    # NEW — 4D MFGP + MF-qNEI (q=4)
├── f5/
│   └── f5 - week 8.ipynb    # NEW — 4D GP + qLogNEI + IP (S=1.0)
├── f6/
│   └── f6 - week 8.ipynb    # NEW — 5D SFGP + qLogNEI + IP rank-based
├── f7/
│   └── f7 - week 8.ipynb    # NEW — 6D NN + MC Dropout EI + IP (S=0.1)
└── f8/
    └── f8 - week 8.ipynb    # NEW — 8D SFGP + qEI (q=1)
```

**Structure Decision**: Jupyter notebooks only — no `src/` directory. Each notebook is a self-contained deliverable placed alongside the original `fX.ipynb` per Constitution III.

## Complexity Tracking

> No violations to justify. All principles PASS.

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts are complete.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Simplicity | PASS | Each notebook replicates Week 7 — no new abstractions, surrogates, or acquisition functions introduced |
| II. Per-Function Isolation | PASS | 7 separate notebooks in 7 separate `./functions/fX/` directories |
| III. Per-Iteration Notebooks | PASS | New files `fX - week 8.ipynb`; originals untouched; each notebook is self-contained with imports, data, model, acquisition, viz, submission |
| IV. Data Organisation | PASS | Loads `updated_inputs - Week 8.npy` and `updated_outputs - Week 8.npy` from `./data/fX/` |
| V. BoTorch & PyTorch Stack | PASS | F2–F6, F8 use BoTorch GPs; F7 uses PyTorch NN — both allowed |
| VI. Documentation & Visualisation | PASS | Each notebook includes hyperparameter tables, surrogate contour/slice plots, convergence plots, and submission output |
| VII. Maximisation Objective | PASS | All acquisition functions (qLogNEI, qEI, MC Dropout EI) treat higher output as better; convergence plots show running maximum |

**Post-design gate result: PASS** — No violations. Design artifacts are complete and constitution-compliant.
