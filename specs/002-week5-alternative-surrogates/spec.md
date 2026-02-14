# Feature Specification: Week 5 — Alternative Surrogate Models

**Feature Branch**: `002-week5-alternative-surrogates`  
**Parent Branch**: `master`  
**Created**: 2026-02-14  
**Status**: Draft  
**Input**: Week 5 submission — replace GP surrogates with problem-appropriate alternative models

## Summary

Add a "Week 5" section to each of the 8 function notebooks, replacing the Gaussian Process surrogate with an alternative model tailored to each problem's characteristics. This demonstrates understanding of different surrogate modelling approaches while producing valid Week 6 submission proposals. The GP sections from previous weeks remain untouched (append-only per CONSTITUTION).

**Surrogate Model Assignments:**

| Functions | Current Model | New Surrogate | Rationale |
|-----------|--------------|---------------|-----------|
| f1 (2D) | GP (SingleTaskGP) | Polynomial Response Surface (Quadratic) | Low-dimensional, simple surface — quadratic model is interpretable and fast |
| f2, f3 (2D, 3D) | GP (SingleTaskGP) | Random Forest | Handles noise well, flexible decision boundaries, no distributional assumptions |
| f4, f5 (4D, 4D) | GP (SingleTaskGP) | Gradient Boosted Trees | Strong predictive performance in moderate dimensions, sequential learning captures complex interactions |
| f6, f7, f8 (5D, 6D, 8D) | GP (SingleTaskGP) | Neural Network (PyTorch) | Scales to higher dimensions, flexible function approximation, learns complex nonlinear relationships |

## Clarifications

### Session 2026-02-14

- Q: Branch from current `001-bayesian-optimization-notebooks` or main? → A: **Branch from `master`** (clean branch for Week 5 alternative surrogates)
- Q: For f2–f5, use both Random Forest and Gradient Boost, or split? → A: **Split** — RF for f2/f3, GBM for f4/f5
- Q: Neural network framework for f6/f7/f8? → A: **Plain PyTorch (nn.Module)** — consistent with existing torch dependency
- Q: Update CONSTITUTION for model diversity? → A: **Yes** — update to reflect that surrogate models may now vary per function
- Q: Acquisition strategy for non-GP surrogates? → A: **Confidence-based (UCB)** — use surrogate prediction + uncertainty estimate to compute Upper Confidence Bound
- Q: Polynomial uncertainty method (FR-007)? → A: **Distance-based** — σ(x) = min Euclidean distance from x to nearest observed point. Simpler and more reliable than residual-based with only 15 samples.
- Q: UCB candidate count (n_candidates)? → A: **Fixed 20,000** for all models — good coverage with predictable runtime across all dimensionalities.
- Q: GBM ensemble size for uncertainty? → A: **10 models** with different random seeds — more stable uncertainty estimate for UCB.
- Q: NN feature importance for slice plot dimension selection? → A: **Input gradient magnitude** — average |∂ŷ/∂xᵢ| over training points. PyTorch-native, simple, no extra dependencies.
- Q: Should κ be the same across all models or per-model? → A: **Per-model κ** — tune κ separately for each model type to account for different uncertainty scales. Document rationale per model.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Polynomial Surrogate for f1 (Priority: P1)

As a data scientist working on the radiation detection problem (f1), I need to replace the GP surrogate with a quadratic polynomial response surface to demonstrate an alternative modelling approach, clearly showing all hyperparameters and producing a valid next sample point.

**Why this priority**: f1 is 2D and simplest to implement/visualize. Demonstrates polynomial modelling fundamentals.

**Independent Test**: Execute the new "Week 5" section in f1.ipynb end-to-end using Week 5 data. Verify the polynomial model fits, coefficients are displayed, UCB acquisition proposes a valid next point within [0, 0.999999], and visualizations render.

**Acceptance Scenarios**:

1. **Given** Week 5 cumulative data for f1 (15 samples × 2D), **When** the polynomial response surface cell executes, **Then** a second-degree polynomial (quadratic) is fitted using all 15 data points, the model coefficients and R² score are displayed, and the fitted surface is visualized as a contour plot
2. **Given** fitted polynomial model, **When** the UCB acquisition cell executes, **Then** the acquisition function evaluates the polynomial on a dense grid of candidates, adds an exploration bonus based on distance from observed points, and proposes the next sample point with coordinates within [0, 0.999999]
3. **Given** the proposed next point, **When** the convergence plot cell executes, **Then** a convergence plot shows the running maximum over all 15 observations including BO-submitted points, and the next submission point is formatted as `x1-x2` with 6 decimal places

### User Story 2 — Random Forest Surrogate for f2/f3 (Priority: P1)

As a data scientist working on the log-likelihood (f2, 2D) and drug discovery (f3, 3D) problems, I need to replace the GP surrogate with a Random Forest model to demonstrate ensemble-based surrogate modelling with tree-based uncertainty estimation.

**Why this priority**: RF is a natural alternative to GP — well-understood, handles noise, and provides uncertainty through tree variance.

**Independent Test**: Execute the new "Week 5" section in f2.ipynb and f3.ipynb. Verify RF model trains, feature importance is displayed, UCB acquisition proposes valid next points, convergence plots render.

**Acceptance Scenarios**:

1. **Given** Week 5 cumulative data for f2 (15 samples × 2D) or f3 (20 samples × 3D), **When** the Random Forest cell executes, **Then** a `RandomForestRegressor` trains on all available data with `n_estimators`, `max_depth`, `min_samples_split`, and `random_state` clearly specified and justified, prints feature importance scores per input dimension, and displays OOB score
2. **Given** trained RF model, **When** the UCB acquisition cell executes, **Then** the acquisition evaluates the mean prediction across all trees on a set of random candidates, computes per-candidate standard deviation from individual tree predictions as the uncertainty estimate, applies UCB formula `μ(x) + κ·σ(x)` where κ is a documented exploration parameter, and proposes the next sample point within bounds
3. **Given** the proposed next point, **When** visualization cells execute, **Then** a 2D surrogate contour plot (f2) or 2D slice plot (f3) of the RF mean prediction is displayed along with an uncertainty contour, and a convergence plot shows the running maximum

### User Story 3 — Gradient Boosted Trees Surrogate for f4/f5 (Priority: P1)

As a data scientist working on the warehouse placement (f4, 4D) and chemical yield (f5, 4D) problems, I need to replace the GP surrogate with Gradient Boosted Trees to demonstrate sequential learning-based surrogate modelling.

**Why this priority**: GBM excels at capturing complex interactions in moderate dimensions. Different modelling paradigm from GP and RF.

**Independent Test**: Execute the new "Week 5" section in f4.ipynb and f5.ipynb. Verify GBM model trains, hyperparameters documented, UCB acquisition proposes valid next points.

**Acceptance Scenarios**:

1. **Given** Week 5 cumulative data for f4 (35 samples × 4D) or f5 (25 samples × 4D), **When** the Gradient Boost cell executes, **Then** a `GradientBoostingRegressor` trains on all available data with `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, and `subsample` clearly specified and justified, and prints feature importance per input dimension
2. **Given** trained GBM model, **When** the UCB acquisition cell executes, **Then** 10 GBM models with different random seeds are trained (ensemble for uncertainty), mean and std of predictions across the ensemble are computed, UCB formula `μ(x) + κ·σ(x)` is applied, and the next sample point is proposed within bounds
3. **Given** the proposed next point, **When** visualization cells execute, **Then** a pairwise slice visualization of the GBM predictions is displayed showing the learned landscape, feature importance is visualized as a bar chart, and a convergence plot shows the running maximum

### User Story 4 — Neural Network Surrogate for f6/f7/f8 (Priority: P1)

As a data scientist working on the cake recipe (f6, 5D), ML hyperparameter tuning (f7, 6D), and high-dimensional (f8, 8D) problems, I need to replace the GP surrogate with a feedforward neural network to demonstrate deep learning-based surrogate modelling with explicit architecture documentation.

**Why this priority**: NNs scale better to higher dimensions and demonstrate a fundamentally different modelling approach. Architecture choices (layers, nodes) must be clearly specified.

**Independent Test**: Execute the new "Week 5" section in f6.ipynb, f7.ipynb, and f8.ipynb. Verify NN trains, architecture is documented (hidden layers, nodes per layer, activation, optimizer), UCB acquisition proposes valid next points.

**Acceptance Scenarios**:

1. **Given** Week 5 cumulative data for f6 (25×5D), f7 (35×6D), or f8 (45×8D), **When** the Neural Network cell executes, **Then** a PyTorch feedforward NN is trained with hyperparameters clearly documented:
   - Number of hidden layers (document each layer's node count)
   - Activation function per layer (e.g., ReLU, Tanh)
   - Learning rate, optimizer (e.g., Adam), number of epochs
   - Input/output normalization approach
   - Loss function (MSE)
   - Training loss curve is plotted showing convergence

2. **Given** trained NN model, **When** the UCB acquisition cell executes, **Then** uncertainty is estimated via MC Dropout (training-mode forward passes) or an ensemble of NNs with different initializations, mean and std are computed from multiple forward passes, UCB formula `μ(x) + κ·σ(x)` is applied on random candidates, and the best point is proposed within [0, 0.999999]

3. **Given** the proposed next point for each function, **When** convergence cells execute, **Then** convergence plot shows running maximum over all observations, input gradient magnitude (avg |∂ŷ/∂xᵢ|) is computed to identify the top 2 most important dimensions, surrogate predictions are visualized via 2D slice plots for those dimensions, and the next submission point is formatted correctly

### Edge Cases

- **Polynomial overfitting (f1)**: 15 data points fitting a full quadratic in 2D (6 coefficients) — small margin. Monitor R² and consider regularization if needed.
- **RF/GBM with small datasets**: f2 has only 15 samples. Tree ensembles may overfit. Cap `max_depth` and `min_samples_split` appropriately.
- **NN with small datasets (f6)**: 25 samples for a 5D NN is very small. Use regularization (weight decay, dropout) and keep architecture small (1–2 hidden layers).
- **Uncertainty calibration**: UCB requires uncertainty estimates. RF uses tree variance, GBM uses ensemble variance, NN uses MC Dropout. Document that these are approximations.
- **Existing cells unchanged**: Per CONSTITUTION, the new Week 5 sections MUST be appended after all existing cells. No existing cells modified or deleted.
- **Model comparison**: Each notebook should briefly compare the new surrogate's prediction vs the previous GP prediction (e.g., agree/disagree on best region) in a markdown comment.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each notebook (f1–f8) MUST add a new "## Week 5 — [Model Name] Surrogate" section after all existing cells, without modifying any prior cells
- **FR-002**: f1.ipynb MUST implement a quadratic polynomial response surface using `sklearn.preprocessing.PolynomialFeatures(degree=2)` and `sklearn.linear_model.LinearRegression` (or Ridge), displaying all coefficients, interaction terms, R² score, and intercept
- **FR-003**: f2.ipynb and f3.ipynb MUST implement `sklearn.ensemble.RandomForestRegressor` with documented hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `random_state`, and OOB score (`oob_score=True`)
- **FR-004**: f4.ipynb and f5.ipynb MUST implement `sklearn.ensemble.GradientBoostingRegressor` with documented hyperparameters: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample`
- **FR-005**: f6.ipynb, f7.ipynb, and f8.ipynb MUST implement a PyTorch feedforward neural network (`nn.Module`) with documented architecture: number of hidden layers, nodes per layer, activation function, optimizer, learning rate, epochs, and loss function
- **FR-006**: All notebooks MUST implement a UCB-based acquisition function: `UCB(x) = μ(x) + κ·σ(x)` where μ is the surrogate's mean prediction, σ is the estimated uncertainty, and κ is a per-model documented exploration parameter tuned to each model type's uncertainty scale:
  - Polynomial (f1): κ to be justified based on distance-based σ magnitude
  - Random Forest (f2, f3): κ to be justified based on tree variance σ magnitude
  - Gradient Boost (f4, f5): κ to be justified based on ensemble σ magnitude
  - Neural Network (f6, f7, f8): κ to be justified based on MC Dropout σ magnitude
  - Each κ value MUST include a markdown explanation of why it was chosen for that model type
- **FR-007**: Uncertainty estimation MUST be implemented per model type:
  - Polynomial: distance-based — σ(x) = min Euclidean distance from x to nearest observed point
  - RF: standard deviation across individual tree predictions
  - GBM: ensemble of models with different random seeds (10 models), std across ensemble
  - NN: MC Dropout (multiple forward passes with dropout enabled) or NN ensemble
- **FR-008**: Each notebook MUST generate a surrogate function visualization appropriate to its dimensionality:
  - 2D (f1, f2): full contour plot of surrogate predictions over [0,1]²
  - 3D (f3): 2D slice plot fixing least-important dimension at best observed value
  - 4D+ (f4–f8): pairwise 2D slice plots for top 2 most important input dimensions (RF/GBM: `feature_importances_`; NN: input gradient magnitude avg |∂ŷ/∂xᵢ|)
- **FR-009**: Each notebook MUST generate a convergence plot showing the running maximum (best found) over all observations (initial + weekly submissions)
- **FR-010**: Each notebook MUST display the proposed next sample point in submission format `x1-x2-...-xn` with 6 decimal places, clamped to [0, 0.999999]
- **FR-011**: Each notebook MUST load Week 5 cumulative data from `../../data/f{N}/updated_inputs - Week 5.npy` and `../../data/f{N}/updated_outputs - Week 5.npy`
- **FR-012**: All hyperparameters MUST include text justifications explaining why values were chosen, including dimension-specific rationale where applicable
- **FR-013**: The CONSTITUTION.md MUST be updated to reflect that surrogate models may differ per function and are chosen based on problem characteristics

### Non-Functional Requirements

- **NFR-001**: Each notebook's Week 5 section MUST execute in <2 minutes on a standard laptop
- **NFR-002**: Code MUST be as simple as possible with each step clearly explained via markdown cells (per CONSTITUTION)
- **NFR-003**: Visualizations MUST render inline within Jupyter using matplotlib
- **NFR-004**: New dependencies (scikit-learn) MUST be verified as installed before use
- **NFR-005**: NN training MUST use float64 precision for consistency with existing notebook code
- **NFR-006**: All notebooks MUST follow consistent Week 5 section structure: data loading → model hyperparameters → training → acquisition → visualization → convergence → submission format

### Key Entities *(include if feature involves data)*

- **PolynomialSurrogate** (f1):
  - Attributes: degree (2), n_features (2), n_coefficients (6 for quadratic 2D), R² score, coefficients array, intercept
  - Hyperparameters: degree, regularization alpha (if Ridge)

- **RandomForestSurrogate** (f2, f3):
  - Attributes: n_estimators (100–200), max_depth (5–10), min_samples_split (2–5), min_samples_leaf (1–2), random_state (42), oob_score, feature_importances_
  - Uncertainty: std across tree predictions

- **GradientBoostSurrogate** (f4, f5):
  - Attributes: n_estimators (100–200), learning_rate (0.05–0.1), max_depth (3–5), min_samples_split (2–5), subsample (0.8), feature_importances_
  - Uncertainty: ensemble of 10 models with different seeds, std across ensemble

- **NeuralNetworkSurrogate** (f6, f7, f8):
  - Attributes: architecture varies by problem dimensionality:
    - f6 (5D): 2 hidden layers [32, 16] nodes
    - f7 (6D): 2 hidden layers [64, 32] nodes
    - f8 (8D): 3 hidden layers [128, 64, 32] nodes
  - Hyperparameters: activation (ReLU), optimizer (Adam), learning_rate (1e-3), epochs (500–1000), weight_decay (1e-4), dropout_rate (0.1 for uncertainty)
  - Uncertainty: MC Dropout (50 forward passes)
  - Feature importance: input gradient magnitude — avg |∂ŷ/∂xᵢ| over training points (for slice plot dimension selection)

- **UCBAcquisition** (all notebooks):
  - Attributes: κ (per-model exploration parameter), n_candidates (20,000 random points, fixed for all models), bounds [0, 0.999999]^D
  - Formula: `UCB(x) = μ(x) + κ·σ(x)` — maximize to find next sample point
  - κ is tuned per model type to account for different uncertainty scales (distance-based vs tree variance vs ensemble std vs MC Dropout std). Each notebook documents its κ choice with justification. Higher κ → more exploratory, lower κ → more exploitative.

## Technical Context

### New Dependencies

- **scikit-learn** (existing or to be installed): `PolynomialFeatures`, `LinearRegression`/`Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `StandardScaler`
- **PyTorch** (already installed): `torch.nn`, `torch.optim` for NN surrogates

### Data Shapes (Week 5 Cumulative)

| Function | Inputs Shape | Outputs Shape | Total Samples |
|----------|-------------|---------------|--------------|
| f1 | (15, 2) | (15,) | 10 initial + 5 weekly |
| f2 | (15, 2) | (15,) | 10 initial + 5 weekly |
| f3 | (20, 3) | (20,) | 15 initial + 5 weekly |
| f4 | (35, 4) | (35,) | 30 initial + 5 weekly |
| f5 | (25, 4) | (25,) | 20 initial + 5 weekly |
| f6 | (25, 5) | (25,) | 20 initial + 5 weekly |
| f7 | (35, 6) | (35,) | 30 initial + 5 weekly |
| f8 | (45, 8) | (45,) | 40 initial + 5 weekly |

### UCB Acquisition Strategy

Since non-GP models don't integrate with BoTorch's `optimize_acqf`, the acquisition is implemented manually:

```python
# Generate random candidates within bounds (fixed 20,000 for all models)
candidates = np.random.uniform(0, 0.999999, size=(20000, D))

# Get mean prediction and uncertainty from surrogate
mean = model.predict(candidates)       # RF/GBM: .predict()
std = compute_uncertainty(candidates)   # model-specific

# UCB acquisition
ucb = mean + kappa * std
next_point = candidates[np.argmax(ucb)]
```

### Uncertainty Estimation by Model Type

| Model | Method | Implementation |
|-------|--------|---------------|
| Polynomial | Distance-based | σ(x) = min Euclidean distance from x to nearest observed point (normalized) |
| Random Forest | Tree variance | σ(x) = std of predictions across individual trees |
| Gradient Boost | Ensemble | Train 10 GBMs with different random seeds, σ(x) = std across models |
| Neural Network | MC Dropout | 50 forward passes with dropout enabled, σ(x) = std of outputs |

### Known Constraints

- Small dataset sizes (15–45 samples) require careful regularization for all model types
- Polynomial model limited to 2D (f1 only) — coefficient count grows combinatorially with dimensions
- RF/GBM feature importance provides interpretability similar to GP lengthscales
- NN requires input/output normalization for stable training
- All models are maximization — UCB naturally maximizes by choosing highest upper bound

## Out of Scope

- Modifying any existing notebook cells (per CONSTITUTION)
- Bayesian hyperparameter optimization of the surrogates themselves
- Cross-validation or hold-out evaluation (insufficient data for meaningful splits)
- GPU training for neural networks (CPU is sufficient for these dataset sizes)
- Deploying models or creating APIs
- Changing the results processing notebook (already completed in feature 001)

## Success Criteria

- All 8 notebooks execute their Week 5 sections without errors
- Each notebook uses the specified alternative surrogate model (polynomial/RF/GBM/NN)
- Every hyperparameter has a documented justification in markdown
- UCB acquisition proposes valid next points within [0, 0.999999] for all dimensions
- Surrogate visualizations render appropriately for each dimensionality
- Convergence plots show running maximum over all observations
- Submission format `x1-x2-...-xn` with 6 decimal places is produced for each function
- CONSTITUTION.md updated to reflect model diversity
- No existing notebook cells are modified — Week 5 sections are purely appended

## Resolved Questions

- **κ tuning**: ~~Should κ=2.0 be the same across all models?~~ → **Per-model κ**, tuned separately for each model type to account for different uncertainty scales. Each κ value documented with justification.
- **NN architecture search**: Commit to the specified architecture (f6: [32,16], f7: [64,32], f8: [128,64,32]). Document reasoning for each choice. No multi-architecture comparison needed.
