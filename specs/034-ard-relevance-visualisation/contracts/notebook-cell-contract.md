# Contracts: ARD Relevance Visualisation

This feature modifies Jupyter notebooks only — no APIs, endpoints, or inter-service contracts.

The "contract" for this feature is the consistent cell interface added to each notebook:

## Notebook Cell Contract

### Cell 12: Markdown — ARD Section Header
- Title: `## ARD Feature Relevance Analysis`
- Brief explanation of what ARD lengthscales mean
- For F7 only: note that this is a diagnostic GP analysis

### Cell 13: Code — GP Fitting & Raw Lengthscale Table
**Inputs**: `inputs` (ndarray), `outputs` (ndarray) from existing Cell 4
**Outputs**: Printed table of raw lengthscales per dimension

**Behaviour**:
1. Import BoTorch/GPyTorch dependencies
2. Convert data to tensors
3. Apply function-specific output transform
4. Construct SingleTaskGP with MaternKernel(nu=NU, ard_num_dims=N_DIMS)
5. Fit via `fit_gpytorch_mll`
6. Extract `model.covar_module.base_kernel.lengthscale`
7. Print formatted table: dimension name → raw lengthscale value

### Cell 14: Code — Normalised Relevance Bar Chart
**Inputs**: `lengthscales` (ndarray from Cell 13)
**Outputs**: Horizontal bar chart displayed inline

**Behaviour**:
1. Compute `relevance = (1/ls) / sum(1/ls)`
2. Create `plt.barh()` with dimension names on y-axis
3. Annotate bars with percentage values
4. Set title, axis labels
5. `plt.tight_layout(); plt.show()`
