# Notebook Cell Contract: fX - week 9.ipynb

**Feature**: 021-f1-f8-week9 | **Date**: 2026-03-02

This contract defines the cell structure for all 8 Week 9 notebooks. Each cell has a type (markdown/code), preconditions, and postconditions.

## Cell Structure

### Cell 1: Title & Strategy Summary (Markdown)

**Content**: Week 9 title, strategy summary (carried from Week 8), rationale for any changes (enhanced visualisation + performance evaluation added).

### Cell 2: Imports (Code)

**Postconditions**:
- All required libraries imported
- Warnings suppressed where appropriate
- `scipy.spatial.distance` imported (new for performance eval)

**GP Functions (F2, F3, F5, F6, F8)**:
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from itertools import groupby
from scipy.spatial.distance import pdist, squareform
import botorch, gpytorch  # + specific submodules per function
```

**F1 (Hurdle)**:
```python
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import groupby
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
```

**F7 (Neural Net)**:
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from itertools import groupby
from scipy.spatial.distance import pdist, squareform
```

### Cell 3: Hyperparameters (Markdown + Code)

**Preconditions**: None
**Postconditions**:
- All hyperparameters defined as ALL_CAPS named constants
- New constants added for performance eval:
  - `STALLING_CONSECUTIVE_THRESHOLD = 3`
  - `STALLING_RELATIVE_THRESHOLD = 0.05`
- All constants printed for auditability

### Cell 4: Data Loading & Validation (Code)

**Preconditions**: Hyperparameters defined
**Postconditions**:
- `X` (inputs) and `y` (outputs) loaded from `../../data/fX/updated_inputs - Week 9.npy` and `updated_outputs - Week 9.npy`
- Shape assertions: `X.shape == (N_TOTAL, N_DIMS)`, `y.shape == (N_TOTAL,)`
- Range validation: all X values in [0.0, 1.0], no NaN in y
- `X_initial`, `y_initial` = first N_INITIAL rows
- `X_submissions`, `y_submissions` = remaining rows (9 per function)
- Best observation identified and printed
- Tabular display of all data points

### Cell 5: Surrogate Model Fitting (Code)

**Preconditions**: Data loaded and validated
**Postconditions**: Surrogate model fitted on full dataset (X, y), ready for predictions

**Varies by function** — see spec FR-F1 through FR-F8 for exact model configuration.

### Cell 6: Acquisition Function Optimisation (Code)

**Preconditions**: Surrogate fitted
**Postconditions**:
- Acquisition function optimised
- `best_candidate` selected (ndarray of shape (n_dims,))
- For q>1 functions (F4, F5, F6): best of q candidates selected by posterior mean
- Interior penalty applied where applicable (F1, F5, F6, F7)

### Cell 7: Visualisation — Surrogate Plots (Code)

**Preconditions**: Surrogate fitted, candidate selected
**Postconditions**:
- 3-panel figure: surrogate mean | uncertainty | acquisition surface
- **NEW colour scheme**:
  - Initial samples: blue markers (`c='tab:blue'`, `s=40`)
  - Weekly submissions: orange/red markers (`c='tab:orange'`, `s=60`, slightly larger)
  - Proposed point: green star (`marker='*'`, `c='tab:green'`, `s=200`)
- Legend with 3 entries: "Initial samples", "Weekly submissions", "Proposed next point"
- For high-dim functions (F4-F8): top-2 dimensions selected by feature importance, remaining fixed at best point

### Cell 8: Visualisation — Convergence Plot (Code)

**Preconditions**: Data loaded
**Postconditions**:
- Running maximum curve plotted
- Individual observations as scatter points with colour scheme:
  - Blue for initial period (indices 0 to N_INITIAL-1)
  - Orange/red for submission period (indices N_INITIAL to N_TOTAL-1)
- Vertical marker at initial/submission boundary
- Grid, labels, legend

### Cell 9: Submission Query (Code)

**Preconditions**: `best_candidate` available
**Postconditions**:
- Candidate clipped to [0.0, 0.999999]
- Formatted as `"0.xxxxxx-0.xxxxxx-...-0.xxxxxx"`
- Validation assertions (dimension count, range check)
- Strategy metadata printed
- Formatted query displayed prominently

### Cell 10: Performance Evaluation — Convergence Metrics (Code) [NEW]

**Preconditions**: Data loaded, `N_INITIAL` defined
**Postconditions**:
- `best_trajectory`: running max after each submission
- `per_submission_delta`: improvement per submission
- `new_best_flags`: boolean array
- `consecutive_no_improvement`: int (trailing streak counted from most recent submission backwards)
- `relative_improvement`: float
- `stalling_flag`: bool (True if trailing consecutive >= 3 OR relative < 0.05)
- All metrics printed in a summary table

### Cell 11: Performance Evaluation — Exploration Spread (Code) [NEW]

**Preconditions**: `X_submissions` available
**Postconditions**:
- `mean_pairwise_distance`: float
- `max_nn_distance`: float
- `min_nn_distance`: float
- Metrics printed
- Optional: scatter plot of submission points (2D/3D functions only)

### Cell 12: Performance Evaluation — LOO Surrogate Error (Code) [NEW]

**Preconditions**: Full dataset available, surrogate configuration known
**Postconditions**:
- 9-fold LOO loop completed
- `loo_predictions`, `loo_actuals`, `loo_errors` computed
- `loo_mae`, `loo_rmse` printed
- Per-point error table displayed
- Note about limited sample size included

### Cell 13: Performance Evaluation — Interpretation & Strategy (Markdown) [NEW]

**Preconditions**: All performance metrics computed in Cells 10-12
**Content**:
- Summary of convergence status (stalling or healthy)
- Interpretation of exploration spread (clustered vs spread)
- Assessment of surrogate prediction accuracy
- If stalling: 1-3 specific, actionable strategy change recommendations
- If not stalling: confirmation of strategy effectiveness + observations

---

## Function-Specific Variations

| Cell | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|------|----|----|----|----|----|----|----|----|
| 5 (Surrogate) | Hurdle two-stage | SFGP Matérn 1.5 | SFGP Matérn 2.5 | MFGP LinTrunc | GP log1p | SFGP Matérn 1.5 | NN 6→5→5→1 | SFGP Matérn 2.5 |
| 6 (Acquisition) | Weighted UCB + local pen | qLogNEI | qLogNEI | MF-qNEI q=4 | qLogNEI q=4 | qLogNEI q=4 | MC Dropout EI | qEI q=1 |
| 7 (Vis type) | 2D contour | 2D contour | 3D slice | 4D slice | 4D slice | 5D slice | 6D slice | 8D slice |
| 12 (LOO) | Retrain hurdle | Retrain SFGP | Retrain SFGP (recompute z-score) | Retrain MFGP (5 restarts) | Retrain GP (undo log1p+zscore) | Retrain SFGP | Retrain NN (seed) | Retrain SFGP |
