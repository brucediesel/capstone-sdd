# Research: ARD Relevance Visualisation

**Feature**: 034-ard-relevance-visualisation
**Date**: 2025-07-18

## R1: Per-Function GP Configuration for ARD Extraction

### Decision
Each function's ARD diagnostic GP replicates the kernel, output transform, and noise settings from its week 10 optimisation notebook.

### Rationale
The spec clarification (Q1) requires matching each function's optimisation notebook output transform so that lengthscale relevance reflects how the actual surrogate "sees" the data.

### Per-Function Configuration Table

> **Note**: MLL Restarts values below are from the source optimisation notebooks for context. `fit_gpytorch_mll` handles restart logic internally — these are not directly configurable parameters for this feature.

| Func | Dims | Kernel | Nu | Output Transform | Noise LB | MLL Restarts |
|------|------|--------|----|-----------------|----------|--------------|
| F1 | 2 | Matérn | 2.5 | Standardize(m=1) | 1e-4 | 15 |
| F2 | 2 | Matérn | 2.5 | Standardize(m=1) | 1e-4 | 50 |
| F3 | 3 | Matérn | 2.5 | Shift: y - y_min | 1e-4 | 40 |
| F4 | 4 | Matérn | 2.5 | Standardize(m=1) | 1e-3 | 30 |
| F5 | 4 | Matérn | 1.5 | log(y) + Standardize(m=1) | 1e-6 | 15 |
| F6 | 5 | Matérn | 1.5 | Standardize(m=1) | 1e-3 | 15 |
| F7 | 6 | Matérn | 2.5 (diagnostic) | Standardize(m=1) | 1e-4 | 20 |
| F8 | 8 | Matérn | 2.5 | Standardize(m=1) | 1e-7 | 30 |

### F7 Diagnostic GP Justification
F7's production surrogate is a neural network (2L×5N, lr=0.005, dropout=0.05). Since NNs have no kernel lengthscales, a separate diagnostic SingleTaskGP with Matérn-2.5 ARD and Standardize(m=1) is fitted purely for relevance analysis. Matérn-2.5 is chosen as the default GP kernel for this project. The diagnostic nature must be clearly annotated in the notebook.

### Alternatives Considered
- **Uniform Standardize across all functions**: Rejected. User chose Option A (match each function's transform) during clarification.
- **Skip F7 entirely**: Rejected. All 8 functions must have ARD visualisation per spec.
- **Use F7's NN feature importance (gradient-based)**: Too complex for diagnostic purposes. GP ARD is simpler and consistent with the other functions.

---

## R2: ARD Lengthscale Extraction Pattern

### Decision
Use the standard BoTorch/GPyTorch pattern to extract per-dimension lengthscales from a fitted SingleTaskGP.

### Code Pattern
```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize

# Fit GP
covar_module = ScaleKernel(MaternKernel(nu=KERNEL_NU, ard_num_dims=N_DIMS))
likelihood = GaussianLikelihood(noise_constraint=GreaterThan(NOISE_LB))
model = SingleTaskGP(X_train, Y_train,
                     covar_module=covar_module,
                     likelihood=likelihood,
                     outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Extract lengthscales
lengthscales = model.covar_module.base_kernel.lengthscale.detach().squeeze().numpy()
```

### Rationale
This is the established pattern used across all week 10 notebooks (F1–F6, F8). It's well-tested in this project and directly produces the ARD lengthscale array.

### Alternatives Considered
- **Manual optimisation loop**: Unnecessary — `fit_gpytorch_mll` handles multi-restart MLL optimisation internally.
- **Different GP library (sklearn)**: Rejected — Constitution Principle V requires BoTorch stack.

---

## R3: Relevance Score Computation

### Decision
Compute normalised relevance as inverse lengthscale, normalised to sum to 1.

### Formula
```
relevance_i = (1 / lengthscale_i) / sum(1 / lengthscale_j for all j)
```

### Rationale
Inverse lengthscale is the standard ARD relevance metric. Smaller lengthscale → model is more sensitive to that dimension → higher relevance score. Normalising to sum to 1 makes scores interpretable as relative importance percentages.

### Alternatives Considered
- **Raw lengthscales without normalisation**: Still shown in the printed table (per Q2 clarification), but not suitable for the bar chart since absolute scale is hard to interpret visually across different functions.
- **Sensitivity-based (gradient) importance**: More principled but significantly more complex. ARD lengthscales are the standard GP-based approach and match the project's simplicity principle.

---

## R4: Week 11 Notebook Integration Pattern

### Decision
Append 3 new cells to the end of each week 11 notebook, after the existing evaluation/strategy markdown cells.

### Cell Structure
1. **Markdown cell**: Section header "## ARD Feature Relevance Analysis" with explanatory text
2. **Code cell**: GP fitting, lengthscale extraction, raw lengthscale table printing
3. **Code cell**: Normalised relevance computation and horizontal bar chart

### Week 11 Common Structure (all 8 notebooks)
All week 11 notebooks share the same 11-cell structure:
- Cells 1-2: Title + imports (numpy, matplotlib, itertools, math)
- Cells 3-4: Config/data loading
- Cells 5-6: Convergence plot
- Cells 7-8: Pair plots with green star marker at best
- Cells 9-10: Evaluation/strategy markdown
- Cell 11: Final notes markdown

New ARD cells appended after Cell 11 (as cells 12, 13, 14).

### Data Loading
Week 11 notebooks load `updated_inputs - Week 11.npy` and `updated_outputs - Week 11.npy`. The ARD GP will reuse the same `inputs` and `outputs` variables already loaded in Cell 4.

### Import Additions
The new code cell needs additional imports not present in the existing week 11 notebooks:
- `torch` (for tensor conversion)
- `botorch.models.SingleTaskGP`
- `botorch.fit.fit_gpytorch_mll`
- `gpytorch.mlls.ExactMarginalLogLikelihood`
- `gpytorch.kernels.ScaleKernel, MaternKernel`
- `gpytorch.likelihoods.GaussianLikelihood`
- `gpytorch.constraints.GreaterThan`
- `botorch.models.transforms.outcome.Standardize` (where applicable)

These imports will be placed at the top of the GP fitting code cell (Cell 13) rather than modifying the existing import cell, to keep the ARD section self-contained.

### Rationale
Self-contained cells make the ARD section easy to understand independently, consistent with the Constitution's simplicity principle. Not modifying existing cells avoids any risk of breaking current functionality.

### Alternatives Considered
- **Add imports to existing Cell 2**: Rejected — modifying existing cells is riskier and breaks self-containment of the ARD section.
- **Single code cell for everything**: Rejected — separating GP fit from visualisation improves readability and matches project convention of one logical step per cell.

---

## R5: Visualisation Design

### Decision
Horizontal bar chart using matplotlib with consistent formatting across all 8 notebooks.

### Chart Specification
- **Type**: `plt.barh()` — horizontal bars
- **Y-axis labels**: Dimension names where available (e.g., F6: 'flour', 'sugar', 'eggs', 'butter', 'milk'), dimension indices elsewhere (e.g., F1: 'x1', 'x2')
- **X-axis**: "Normalised Relevance" (0 to max, with values summing to 1)
- **Title**: "FX: ARD Feature Relevance (Matérn-ν kernel)"
- **Colour**: Single colour for all bars (steelblue) — consistent across functions
- **Value annotations**: Relevance percentage printed on each bar
- **F7 annotation**: Subtitle noting "(Diagnostic GP — production surrogate is Neural Network)"

### Raw Lengthscale Table
- Printed via `print()` in a formatted table showing dimension name/index and raw lengthscale value
- Printed BEFORE the bar chart in the same or preceding code cell

### Rationale
Horizontal bars are easier to read when dimension labels vary in length. Consistent colour and format across all 8 functions satisfies User Story 2 (P2). Value annotations reduce the need to read axis tick marks.

### Alternatives Considered
- **Vertical bar chart**: Rejected — dimension labels would need rotation for longer names (F6 ingredient names).
- **Heatmap**: Over-engineered for per-function analysis with 2–8 dimensions.
- **Radar/spider chart**: Overkill and less intuitive for simple feature importance.

---

## R6: F6 Dimension Names

### Decision
F6 uses ingredient names as dimension labels: ['flour', 'sugar', 'eggs', 'butter', 'milk'].

### Rationale
F6 is a recipe optimisation problem. The week 10 notebook already defines these ingredient names. Using meaningful names instead of x1–x5 makes the ARD chart immediately interpretable in the problem context.

### Other Functions
- F1: x1, x2
- F2: x1, x2
- F3: x1, x2, x3
- F4: x1, x2, x3, x4
- F5: x1, x2, x3, x4
- F7: learning_rate, reg_strength, n_layers, dropout, batch_size, optimizer (hyperparameter names from week 10)
- F8: x1, x2, x3, x4, x5, x6, x7, x8
