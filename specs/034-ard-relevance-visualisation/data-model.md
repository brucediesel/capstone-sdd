# Data Model: ARD Relevance Visualisation

**Feature**: 034-ard-relevance-visualisation
**Date**: 2025-07-18

## Entities

### ARDConfig (per-function configuration)

Defines the GP configuration for ARD extraction per function.

| Field | Type | Description |
|-------|------|-------------|
| func_num | int | Function number (1–8) |
| n_dims | int | Number of input dimensions |
| kernel_nu | float | Matérn kernel smoothness (1.5 or 2.5) |
| noise_lb | float | Noise lower bound for GaussianLikelihood |
| mll_restarts | int | Not directly settable — `fit_gpytorch_mll` handles internally |
| output_transform | str | One of: 'log', 'log+standardize', 'shift', 'standardize', 'none' |
| dim_names | list[str] | Human-readable names for each input dimension |
| is_diagnostic | bool | True for F7 (NN surrogate); False for all others |

### Per-Function Configuration Values

| Func | n_dims | kernel_nu | noise_lb | output_transform | dim_names | is_diagnostic |
|------|--------|-----------|----------|-----------------|-----------|---------------|
| F1 | 2 | 2.5 | 1e-4 | log | [x1, x2] | False |
| F2 | 2 | 2.5 | 1e-4 | standardize | [x1, x2] | False |
| F3 | 3 | 2.5 | 1e-4 | shift | [x1, x2, x3] | False |
| F4 | 4 | 2.5 | 1e-3 | standardize | [x1, x2, x3, x4] | False |
| F5 | 4 | 1.5 | 1e-6 | log+standardize | [x1, x2, x3, x4] | False |
| F6 | 5 | 1.5 | 1e-3 | standardize | [flour, sugar, eggs, butter, milk] | False |
| F7 | 6 | 2.5 | 1e-4 | standardize | [learning_rate, reg_strength, n_layers, dropout, batch_size, optimizer] | True |
| F8 | 8 | 2.5 | 1e-7 | standardize | [x1, x2, x3, x4, x5, x6, x7, x8] | False |

### ARDResult (computed per function)

| Field | Type | Description |
|-------|------|-------------|
| lengthscales | ndarray[float] | Raw ARD lengthscale per dimension (shape: n_dims) |
| relevance | ndarray[float] | Normalised relevance scores (shape: n_dims, sums to 1.0) |

### Relationships

- One ARDConfig → one fitted GP → one ARDResult (1:1:1)
- Each function produces exactly one ARDResult
- No cross-function dependencies (each notebook is self-contained)

### Validation Rules

- `len(lengthscales) == n_dims` for each function
- `sum(relevance) ≈ 1.0` (within floating-point tolerance)
- All lengthscale values must be > 0
- All relevance values must be in [0, 1]

### State Transitions

Not applicable — this is a one-shot computation per notebook (fit → extract → display). No state machine.

## Data Flow

```
.npy files (inputs, outputs)
    ↓  np.load()
numpy arrays
    ↓  torch.tensor()
PyTorch tensors (X_train, Y_train)
    ↓  [apply output transform per function]
Transformed Y_train
    ↓  SingleTaskGP + fit_gpytorch_mll
Fitted GP model
    ↓  model.covar_module.base_kernel.lengthscale.detach().squeeze().numpy()
Raw lengthscales (ndarray)
    ↓  1/ls / sum(1/ls)
Normalised relevance (ndarray)
    ↓  print() table + plt.barh()
Displayed output (table + chart)
```
