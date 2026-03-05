# Model Cards

This document provides model cards for each of the eight black-box optimisation functions (F1–F8) in the capstone project. Model cards are structured transparency documents, inspired by Mitchell et al. (2019), that describe what a model does, how it was built, how it performs, and what its limitations are. They support reproducibility, responsible interpretation, and informed decision-making by readers who may not have access to the underlying notebooks.

This project tackled eight black-box optimisation problems as part of a certificate course in AI and ML. For each function, a surrogate model was trained on available data and used to guide the selection of new query points via an acquisition function. Over the course of seven weekly submission rounds (Weeks 3–9), the surrogate and acquisition strategy were refined based on observed results and systematic evaluation of alternative approaches. The model cards below document the final (Week 9) configuration for each function.

---

## F1 — Radiation Source Detection

### Overview

**Approach**: Hurdle Model (Logistic Regression classifier + Random Forest regressor)  
**Function**: F1 — Radiation Source Detection  
**Dimensionality**: 2D input  
**Type**: Two-stage hurdle model combining a probabilistic classifier with a tree-based regressor

This model addresses the challenge of a zero-inflated output distribution where the vast majority of observations return zero or near-zero values. A hurdle model was chosen because standard Gaussian Process surrogates cannot effectively model the sharp discontinuity between zero and non-zero regions of the output space.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F1 black-box function, guiding candidate selection toward regions likely to contain non-zero (radiation source) signals
- Uncertainty-guided exploration of the 2D input space to locate the radiation source

**Not suitable for**:
- Production deployment for actual radiation source detection without validation against ground-truth measurements
- Extrapolation beyond the [0, 0.999999] input bounds
- Use as a standalone radiation detection system — the surrogate approximates a synthetic challenge function, not a physical detector

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: Hurdle Model — Stage 1: `CalibratedClassifierCV(LogisticRegression(C=1.0))` predicting P(y > 0); Stage 2: `RandomForestRegressor(n_estimators=100, max_depth=3)` trained on `log1p(y)` for positive outputs
- **Architecture**: Two-stage pipeline — classifier gates the regressor; if fewer than 3 truly positive samples exist, FALLBACK_MODE activates and the model switches to pure exploration
- **Key Hyperparameters**: C=1.0 (logistic regularisation), n_estimators=100, max_depth=3 (RF), FALLBACK_MODE threshold=3 positive samples

**Acquisition Function**:
- **Type**: Weighted UCB — a(x) = p(x)·μ(x) + κ·p(x)·σ_RF(x)
- **Configuration**: κ=3.0 (strong exploration bias), 20,000 random candidate evaluations

**Special Techniques**:
- Interior penalty: S=0.1 (gentle steepness), F=0.01 (sinusoidal boundary suppression ensuring candidates remain interior)
- Local penalisation: Gaussian mask with radius=0.15, applied to previously sampled regions to encourage spatial diversity
- Fallback exploration: When FALLBACK_MODE is active, the model reverts to random exploration weighted by interior penalty

*This model was selected after systematic evaluation of alternative surrogates including standard GP, polynomial response surface, and Random Forest/Gradient Boosted Tree surrogates during prequential evaluation in Weeks 5–7.*

### Performance

- **Dataset size**: 19 observations (10 initial + 9 acquired)
- **Output range**: [−3.606 × 10⁻³, 0.000]
- **Output characteristics**: Zero-inflated — 14 near-zero values, 5 slightly negative values; the radiation source has not been located
- **LOO Surrogate Error**: MAE and RMSE computed on 9 submission-round folds
- **Best observed value**: 0.000 (no positive signal detected)

### Assumptions and Limitations

1. **Zero-inflated output distribution**: The vast majority of the output space returns zero, making standard regression surrogates ineffective. The hurdle model mitigates this but relies on having at least 3 positive samples to train Stage 2 — a threshold not yet met.
2. **Small dataset regime**: With only 19 observations in a 2D space, the model has extremely limited coverage. The fallback to pure exploration reflects insufficient data to build a reliable surrogate.
3. **Source not located**: After 9 rounds of optimisation, the radiation source has not been found. The optimisation is stalling, indicating that the search strategy may need a fundamentally different exploration approach (e.g., space-filling design) rather than model-guided search.

### Ethical Considerations

- **Domain context**: Radiation source detection has real-world safety implications. While this is a synthetic challenge function, the domain underscores the importance of transparent reporting — missed detections in a real scenario could have serious consequences.
- **Transparency**: Documenting the hurdle model's failure mode (FALLBACK_MODE activation) and the stalling optimisation ensures that readers understand the model's current limitations rather than assuming successful optimisation.
- **Limitations awareness**: The surrogate is an approximation of a synthetic challenge function and should not be interpreted as a validated radiation detection model. The zero-inflated output structure means the model's predictions are dominated by the classifier stage, whose accuracy is limited by the small dataset.

---

## F2 — Noisy Log-Likelihood

### Overview

**Approach**: Single-task Gaussian Process with Matérn-1.5 kernel  
**Function**: F2 — Noisy Log-Likelihood  
**Dimensionality**: 2D input  
**Type**: Gaussian Process (exact inference)

This model uses a standard single-task GP with a Matérn-1.5 kernel and automatic relevance determination (ARD) to model a noisy log-likelihood surface. The Matérn-1.5 kernel was chosen for its ability to model rough, non-smooth functions, which suits the noisy nature of this function.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F2 black-box function, identifying input configurations that maximise the noisy log-likelihood
- Uncertainty-guided exploration of the 2D input space, leveraging GP posterior variance for principled exploration–exploitation trade-off

**Not suitable for**:
- Direct use as a log-likelihood estimator for real statistical models — the surrogate approximates a synthetic function
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications requiring deterministic or noise-free predictions

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-1.5 with ARD (2 lengthscale parameters), lengthscale bounds [0.01, 2.0]
- **Key Hyperparameters**: noise_lb=1e-3 (noise lower bound), Normalize input transform, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI (q-Log Noisy Expected Improvement)
- **Configuration**: q=4 (batch of 4 candidates), MC=512 Monte Carlo samples, 20 restarts, 1024 raw Sobol samples → distance-based selection (candidates with mean ≥ median and farthest from existing data)

**Special Techniques**:
- Lengthscale bounds [0.01, 2.0] to prevent GP degeneracy (overly smooth or overly rough fits)
- Distance-based candidate selection: from q=4 candidates, those with predicted mean below the median are filtered out, and the remaining are ranked by distance from existing data to promote spatial diversity

*This model was selected after systematic evaluation of alternative surrogates including SFGP and MFGP during prequential evaluation.*

### Performance

- **Dataset size**: 19 observations (10 initial + 9 acquired)
- **Output range**: [−0.066, 0.674]
- **Output characteristics**: Mixed sign — 2 negative values and 17 positive values
- **LOO Surrogate Error**: MAE and RMSE computed on 9 submission-round folds
- **Best observed value**: 0.674

### Assumptions and Limitations

1. **Small dataset regime**: With only 19 observations in 2D, the GP posterior is dominated by the prior in regions far from data. The lengthscale bounds mitigate pathological fits but limit the model's ability to capture fine-scale features.
2. **Noise floor assumption**: The noise lower bound of 1e-3 assumes a minimum level of observation noise. If the true function is nearly deterministic in some regions, this may over-smooth the surrogate.
3. **Local optimum risk**: The distance-based candidate selection strategy was introduced because the optimisation showed signs of clustering around a local optimum. This heuristic does not guarantee escape from local optima.

### Ethical Considerations

- **Domain context**: Log-likelihood estimation is fundamental to statistical inference. While this is a synthetic function, transparent reporting of the surrogate's noise assumptions supports responsible use in contexts where likelihood values inform decisions.
- **Transparency**: Documenting the lengthscale bounds and distance-based selection strategy enables readers to assess whether these design choices were appropriate for the function's characteristics.
- **Limitations awareness**: The GP surrogate provides uncertainty estimates that are only as reliable as the kernel assumptions. Readers should not treat posterior confidence intervals as calibrated probability statements without further validation.

---

## F3 — Drug Discovery

### Overview

**Approach**: Single-task Gaussian Process with Matérn-2.5 kernel  
**Function**: F3 — Drug Discovery  
**Dimensionality**: 3D input  
**Type**: Gaussian Process (exact inference)

This model uses a standard single-task GP with a Matérn-2.5 kernel and ARD to model a drug discovery objective function. The Matérn-2.5 kernel provides a good balance between smoothness and flexibility for this moderately complex 3D landscape.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F3 black-box function, guiding candidate selection to identify molecular configurations or process parameters that maximise the drug discovery objective
- Uncertainty-guided exploration of the 3D input space

**Not suitable for**:
- Direct use as a drug efficacy predictor — the surrogate models a synthetic challenge function, not a real pharmacological response
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications where all-negative output values may be misinterpreted as errors rather than a genuine characteristic of the function

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (3 lengthscale parameters), lengthscale initialisation=0.25
- **Key Hyperparameters**: noise_lb=1e-6 (very low noise assumption), 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI (q-Log Noisy Expected Improvement)
- **Configuration**: 10 restarts, 512 raw Sobol samples

**Special Techniques**:
- Manual z-score standardisation: outputs are transformed as (y − mean) / std before GP fitting, recomputed per LOO fold to prevent data leakage

*This model was selected after systematic evaluation of alternative surrogates including GP, BART, and Random Forest during prequential evaluation.*

### Performance

- **Dataset size**: 24 observations (15 initial + 9 acquired)
- **Output range**: [−0.399, −0.031]
- **Output characteristics**: All negative (24 out of 24 observations)
- **LOO Surrogate Error**: MAE and RMSE computed with z-score recomputation per fold
- **Best observed value**: −0.031

### Assumptions and Limitations

1. **All-negative output domain**: Every observed output is negative. The GP must model relative differences within a narrow negative range, which can be challenging for acquisition functions that use zero as a reference point. Manual z-score standardisation shifts the outputs to have zero mean, mitigating this issue.
2. **Small dataset regime**: With 24 observations in 3D, the GP has limited ability to resolve the full landscape. The low noise assumption (1e-6) may cause overfitting to observed points.
3. **Z-score recomputation**: The manual standardisation requires recomputing mean and standard deviation for each LOO fold, adding complexity and potential for implementation errors compared to BoTorch's built-in Standardize transform.

### Ethical Considerations

- **Domain context**: Drug discovery has direct implications for human health. While this is a synthetic function, the domain highlights the importance of documenting model limitations — an under-explored region of the surrogate could correspond to a promising drug candidate being overlooked.
- **Transparency**: Documenting the manual z-score approach and the all-negative output characteristics ensures readers understand the preprocessing choices and can assess their appropriateness.
- **Limitations awareness**: The surrogate approximates a synthetic challenge function and does not capture real pharmacological interactions. Any insights should not be transferred to actual drug discovery without extensive validation.

---

## F4 — Warehouse Product Placement

### Overview

**Approach**: Multi-Fidelity Gaussian Process with Matérn-5/2 + Linear Truncated Fidelity Kernel  
**Function**: F4 — Warehouse Product Placement  
**Dimensionality**: 4D input (+ 1 fidelity dimension)  
**Type**: Multi-Fidelity Gaussian Process

This model uses a multi-fidelity GP architecture as a deliberate regularisation strategy. Although all data is collected at a single fidelity level (fidelity=1.0), the MFGP's LinearTruncatedFidelityKernel acts as an additional regulariser on the kernel structure, which was found to outperform standard single-fidelity GPs during prequential evaluation.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F4 black-box function, identifying warehouse product placement configurations that maximise the objective
- Leveraging the MFGP's regularisation effect to produce smoother, more robust surrogate fits in a moderate-dimensional space

**Not suitable for**:
- Direct use as a warehouse layout optimiser — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- True multi-fidelity applications — despite using MFGP architecture, all data is at fidelity=1.0

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskMultiFidelityGP (BoTorch)
- **Kernel**: Matérn-5/2 + LinearTruncatedFidelityKernel, ARD (4 spatial + 1 fidelity lengthscale)
- **Key Hyperparameters**: noise_lb=1e-4, fidelity=1.0 for all data points, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI (MF-qNEI variant)
- **Configuration**: q=4, MC=64 Monte Carlo samples, 20 restarts, 512 raw Sobol samples, fixed_features={4: 1.0} (fidelity dimension fixed), prune_baseline=True

**Special Techniques**:
- Multi-fidelity GP used as a regulariser: the LinearTruncatedFidelityKernel adds structure to the kernel that acts as implicit regularisation, improving generalisation
- Prequential evaluation winner with best NLP score (−1.35) among all surrogates evaluated for F4
- Manual z-score standardisation on outputs

*This model was selected after systematic evaluation of alternative surrogates including SFGP, MFGP, and GBT during prequential evaluation, where it achieved the best negative log-predictive density.*

### Performance

- **Dataset size**: 39 observations (30 initial + 9 acquired)
- **Output range**: [−32.626, 0.532]
- **Output characteristics**: Mostly negative — 34 negative values and 5 positive values
- **LOO Surrogate Error**: MAE, RMSE, Median, and Max error computed on submission points
- **Best observed value**: 0.532

### Assumptions and Limitations

1. **Synthetic multi-fidelity usage**: The MFGP is used with constant fidelity (1.0) for all points, which is a non-standard application. The regularisation benefit was empirically validated through PE but may not generalise to other functions or datasets.
2. **Wide output range**: The output spans over 33 units (−32.6 to 0.5), making standardisation critical. Outlier outputs could disproportionately influence the GP fit.
3. **Fixed fidelity dimension**: The fidelity dimension is always fixed at 1.0 during acquisition, adding a constant column to the input data. This increases the effective dimensionality without adding information, potentially affecting lengthscale estimation.

### Ethical Considerations

- **Domain context**: Warehouse product placement optimisation affects operational efficiency and potentially worker ergonomics. While this is a synthetic function, the domain suggests that misoptimisation could have real-world impacts on supply chain operations.
- **Transparency**: Documenting the unconventional use of MFGP as a regulariser (rather than for true multi-fidelity modelling) is important for readers to correctly interpret the modelling choice.
- **Limitations awareness**: The surrogate's predictions are most reliable near observed data points. Regions of the 4D space with sparse coverage may have poorly calibrated uncertainty estimates.

---

## F5 — Chemical Process Yield

### Overview

**Approach**: Single-task Gaussian Process with Matérn-5/2 kernel  
**Function**: F5 — Chemical Process Yield  
**Dimensionality**: 4D input  
**Type**: Gaussian Process (exact inference)

This model uses a standard single-task GP with a Matérn-5/2 kernel and ARD to model a chemical process yield function characterised by an extremely heavy-tailed output distribution spanning several orders of magnitude (0.11 to 3,395). A log1p→z-score transform chain addresses the heavy tail, and an in-loop interior penalty wrapper prevents boundary-pinning.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F5 black-box function, identifying chemical process parameters that maximise yield
- Uncertainty-guided exploration with interior penalty to avoid degenerate boundary solutions

**Not suitable for**:
- Direct use as a chemical process control model — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Prediction of actual chemical yields without calibration against real experimental data

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-5/2 with ARD (4 lengthscale parameters), lengthscale initialisation=0.5
- **Key Hyperparameters**: noise_lb=1e-6, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI with in-loop PenalisedAcquisition wrapper
- **Configuration**: q=4, MC=512 Monte Carlo samples, 50 restarts, 3000 raw Sobol samples → distance-based candidate selection

**Special Techniques**:
- Interior penalty: S=1.0, F=0.01, penalty function 4x(1−x) applied additively in log-space, input bounds tightened to [0.005, 0.995]
- In-loop penalty wrapper: The penalty is applied within the acquisition function evaluation (additive in log-space) rather than as a post-hoc multiplier, which fixes a failure mode where multiplicative penalties in log-space inverted the acquisition landscape
- Output transform chain: log1p(y) → z-score standardisation; inverse: expm1(z·std + mean)

*This model was selected after systematic evaluation of alternative surrogates including GP with various kernels, GBT, and MFGP during prequential evaluation.*

### Performance

- **Dataset size**: 29 observations (20 initial + 9 acquired)
- **Output range**: [0.113, 3,394.680]
- **Output characteristics**: All positive, extremely heavy-tailed — values span nearly 5 orders of magnitude
- **LOO Surrogate Error**: MAE, RMSE, Median, and Max error computed on submission points
- **Best observed value**: 3,394.680

### Assumptions and Limitations

1. **Heavy-tailed distribution**: The output range spans 0.11 to 3,395, requiring a log1p transform to stabilise the GP fit. The transform assumes a log-normal-like distribution of yields, which may not hold across the entire input space.
2. **Boundary-pinning risk**: Without the interior penalty, the acquisition function tends to propose candidates at the input boundaries (0 or 0.999999), which is a known failure mode for GP-based optimisation in bounded domains. The in-loop penalty mitigates this but introduces additional hyperparameters (S, F) that must be tuned.
3. **Transform chain complexity**: The log1p→z-score→expm1 inverse transform chain adds complexity and potential for numerical instability, particularly for values near zero where log1p approximation effects are strongest.

### Ethical Considerations

- **Domain context**: Chemical process optimisation affects safety, environmental impact, and product quality. While this is a synthetic function, the domain highlights the importance of understanding model limitations — an incorrectly optimised process could lead to waste, unsafe conditions, or poor yields in a real setting.
- **Transparency**: Documenting the in-loop penalty mechanism and the transform chain enables readers to understand exactly how candidates are selected and why boundary solutions are suppressed.
- **Limitations awareness**: The surrogate is trained on 29 observations in 4D, providing sparse coverage. High-yield regions identified by the surrogate may not correspond to robust operating conditions in a real chemical process.

---

## F6 — Cake Recipe Optimisation

### Overview

**Approach**: Single-task Gaussian Process with Matérn-1.5 kernel  
**Function**: F6 — Cake Recipe Optimisation  
**Dimensionality**: 5D input  
**Type**: Gaussian Process (exact inference)

This model uses a standard single-task GP with a Matérn-1.5 kernel and ARD to model a cake recipe scoring function where all outputs are negative (representing deductions from an ideal recipe). The Matérn-1.5 kernel was selected to handle the relatively rough response surface and a rank-based interior penalty was developed specifically for this function's all-negative output domain.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F6 black-box function, identifying cake recipe configurations that minimise quality deductions (i.e., maximise the negative score toward zero)
- Uncertainty-guided exploration of the 5D recipe parameter space with feasibility constraints

**Not suitable for**:
- Direct use as a recipe optimiser for actual baking — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Interpretation of outputs as absolute quality scores — values represent deductions within the challenge scoring system

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-1.5 with ARD (5 lengthscale parameters), lengthscale initialisation=0.5
- **Key Hyperparameters**: noise_lb=1e-2 (higher noise floor reflecting recipe variability), Standardize(m=1) output transform, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI
- **Configuration**: q=4, MC=512 Monte Carlo samples, 50 restarts, 3000 raw Sobol samples → distance-based candidate selection, feasibility bounds (x₄ ≥ 0.10, other dimensions ≥ 0.01)

**Special Techniques**:
- Rank-based interior penalty: S=1.0, F=0.01, sign-invariant design — uses rank-based scoring instead of multiplicative penalty to avoid inverting rankings for all-negative output values
- Feasibility constraints: dimension 4 requires values ≥ 0.10; all other dimensions require values ≥ 0.01, reflecting physical recipe constraints

*This model was selected after systematic evaluation of alternative surrogates including SFGP, MFGP, and Neural Network during prequential evaluation.*

### Performance

- **Dataset size**: 29 observations (20 initial + 9 acquired)
- **Output range**: [−2.571, −0.111]
- **Output characteristics**: All negative (29 out of 29 observations) — scores represent deductions from an ideal recipe
- **LOO Surrogate Error**: MAE, RMSE, Median, and Max error computed on submission points
- **Best observed value**: −0.111

### Assumptions and Limitations

1. **All-negative output domain**: Standard multiplicative interior penalties would invert the acquisition rankings because multiplying a negative acquisition value by a positive penalty reverses the ordering. The rank-based penalty was designed specifically to handle this, but adds implementation complexity.
2. **Feasibility constraints**: The feasibility bounds (x₄ ≥ 0.10) represent domain knowledge about valid recipe proportions. These hard constraints reduce the effective search space and could exclude potentially optimal regions if the bounds are too conservative.
3. **Higher noise assumption**: The noise_lb=1e-2 is significantly higher than for other functions, reflecting the assumption that recipe scoring has inherent variability. If the function is actually deterministic, this over-estimates noise and reduces surrogate precision.

### Ethical Considerations

- **Domain context**: Recipe optimisation is a consumer-facing domain. While this is a synthetic function, transparent reporting ensures that recipe recommendations derived from such models are understood to be surrogate-guided suggestions, not guaranteed outcomes.
- **Transparency**: Documenting the rank-based penalty mechanism and feasibility constraints enables readers to understand the design choices specific to handling all-negative outputs and domain-specific constraints.
- **Limitations awareness**: The 5D recipe space is sparsely sampled with 29 observations. The surrogate's recommendation may not generalise to actual baking conditions where additional factors (oven calibration, ingredient freshness) affect outcomes.

---

## F7 — ML Hyperparameter Tuning

### Overview

**Approach**: Neural Network Surrogate with MC Dropout  
**Function**: F7 — ML Hyperparameter Tuning  
**Dimensionality**: 6D input  
**Type**: Neural Network (feedforward with dropout-based uncertainty)

This model uses a compact neural network as the surrogate function instead of a Gaussian Process. The NN was chosen because it scales better to higher-dimensional input spaces and can model complex, non-stationary response surfaces. MC Dropout provides approximate uncertainty estimates for acquisition function evaluation.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F7 black-box function, identifying ML hyperparameter configurations that maximise model performance
- Approximate uncertainty-guided exploration via MC Dropout, providing a computationally efficient alternative to GP posterior inference in 6D

**Not suitable for**:
- Direct use as a hyperparameter optimiser for production ML systems — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications requiring well-calibrated uncertainty estimates — MC Dropout uncertainty is known to be approximate and may be poorly calibrated

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SurrogateNN — custom feedforward neural network
- **Architecture**: 6 → 5 → 5 → 1 (input → hidden → hidden → output), 71 trainable parameters, ReLU activation, Dropout(0.1) between layers
- **Key Hyperparameters**: Adam optimiser with lr=0.005, MSE loss, 200 training epochs, seed=42 for reproducibility

**Acquisition Function**:
- **Type**: MC Dropout Expected Improvement
- **Configuration**: 50 forward passes with dropout enabled to estimate μ(x) and σ(x), standard EI formula applied, 20,000 random candidates evaluated

**Special Techniques**:
- Interior penalty: S=0.1 (gentle), F=0.01 (multiplicative, safe because all outputs are positive)
- MC Dropout for uncertainty estimation: dropout remains active during inference, and multiple forward passes produce a distribution of predictions from which mean and standard deviation are computed
- Manual z-score standardisation on both inputs (X) and outputs (Y)

*This model was selected after systematic evaluation of alternative surrogates including SFGP, MFGP, and Neural Network during prequential evaluation across Weeks 5–7.*

### Performance

- **Dataset size**: 39 observations (30 initial + 9 acquired)
- **Output range**: [0.003, 2.305]
- **Output characteristics**: All positive (39 out of 39 observations)
- **LOO Surrogate Error**: MAE and RMSE computed (NN retrained for 200 epochs per fold), Train R² reported
- **Best observed value**: 2.305

### Assumptions and Limitations

1. **Poorly calibrated uncertainty**: MC Dropout provides approximate uncertainty estimates that are not guaranteed to be well-calibrated. The dropout rate (0.1) and number of forward passes (50) were chosen heuristically, and the resulting uncertainty may underestimate or overestimate true model uncertainty in different regions of the input space.
2. **Non-GP surrogate**: Unlike GP-based approaches used for other functions, the NN does not provide analytical posterior distributions. This means theoretical guarantees of Bayesian optimisation (e.g., convergence bounds) do not apply.
3. **Retraining cost**: The NN must be retrained from scratch (200 epochs) for each LOO fold, making cross-validation more computationally expensive than for GP surrogates which can leverage kernel matrix updates.
4. **Small network capacity**: With only 71 parameters and 2 hidden layers of 5 nodes each, the network has limited capacity to model complex 6D surfaces. This is a deliberate trade-off to prevent overfitting on 39 data points.

### Ethical Considerations

- **Domain context**: ML hyperparameter tuning directly affects the performance and fairness of downstream ML models. While this is a synthetic function, the domain illustrates how surrogate-guided hyperparameter selection can embed biases if the surrogate's uncertainty is poorly calibrated — a model confident in a suboptimal region could lead to underperforming ML systems.
- **Transparency**: Documenting the NN architecture, dropout rate, and MC inference procedure ensures reproducibility. The choice to use a non-GP surrogate is a significant design decision that readers should understand.
- **Limitations awareness**: The MC Dropout uncertainty estimates should be interpreted cautiously. In practice, NN surrogates for hyperparameter optimisation should be validated against established tools (e.g., Optuna, BOHB) before deployment.

---

## F8 — 8D ML Hyperparameters

### Overview

**Approach**: Single-task Gaussian Process with Matérn-2.5 kernel  
**Function**: F8 — 8D ML Hyperparameters  
**Dimensionality**: 8D input  
**Type**: Gaussian Process (exact inference)

This model uses a standard single-task GP with a Matérn-2.5 kernel and ARD to model an 8-dimensional ML hyperparameter optimisation function. Despite the high dimensionality, a GP was retained because the relatively large initial dataset (40 points) provides sufficient coverage for kernel fitting, and the Standardize transform simplifies output handling.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F8 black-box function, identifying 8-dimensional hyperparameter configurations that maximise model performance
- Uncertainty-guided exploration of the 8D input space, with a Sobol fallback strategy for regions where the acquisition function becomes flat

**Not suitable for**:
- Direct use as a hyperparameter optimiser for production ML systems — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- High-throughput batch optimisation — this function uses q=1 (single candidate per round), unlike the q=4 batch approach used for most other functions

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (8 lengthscale parameters)
- **Key Hyperparameters**: noise_lb=1e-7 (extremely low noise assumption), Standardize(m=1) output transform

**Acquisition Function**:
- **Type**: qEI (Expected Improvement)
- **Configuration**: q=1 (single candidate), MC=256 Monte Carlo samples, best_f=y_max + 0.01 (XI=0.01 exploration bonus), 30 restarts, 4096 raw Sobol samples

**Special Techniques**:
- Sobol fallback: When all qEI values are zero (flat acquisition surface), the model selects the candidate with the highest posterior mean from 4096 Sobol samples, ensuring progress even when EI provides no signal
- XI=0.01 exploration bonus: a small offset added to best_f to encourage exploration beyond the current best observation

*This model was selected after systematic evaluation of alternative surrogates including SFGP, MFGP, and Neural Network during prequential evaluation.*

### Performance

- **Dataset size**: 49 observations (40 initial + 9 acquired)
- **Output range**: [5.592, 9.982]
- **Output characteristics**: All positive (49 out of 49 observations)
- **LOO Surrogate Error**: MAE and RMSE computed (GP retrained per fold with auto Standardize)
- **Best observed value**: 9.982

### Assumptions and Limitations

1. **High dimensionality with limited data**: 49 observations in 8D provides extremely sparse coverage (approximately 2 points per input dimension above the initial set). The GP posterior will be dominated by the prior in most of the input space, and lengthscale estimation for 8 ARD parameters requires substantial data.
2. **Single candidate per round (q=1)**: Unlike other functions that use q=4 batch acquisition, F8 proposes only one candidate per round. This conservative approach limits the rate of exploration but avoids the computational challenges of batch acquisition in 8D.
3. **qEI flatness**: The acquisition function frequently returns zero for all candidates, triggering the Sobol fallback. This indicates that the GP's posterior improvement expectations are negligible across the evaluated candidate set, potentially due to high prior uncertainty or a well-optimised current best.
4. **Very low noise assumption**: The noise_lb=1e-7 essentially assumes a noise-free function. If the true function has any stochasticity, this may cause the GP to interpolate rather than smooth, leading to overconfident predictions.

### Ethical Considerations

- **Domain context**: As with F7, ML hyperparameter tuning affects downstream model performance and fairness. The 8D nature of this function illustrates the curse of dimensionality in hyperparameter optimisation — sparse data coverage means the surrogate's recommendations may be unreliable in unexplored regions.
- **Transparency**: Documenting the Sobol fallback mechanism and the q=1 constraint ensures readers understand the practical limitations of applying GP-based optimisation in higher-dimensional spaces.
- **Limitations awareness**: The surrogate's predictions should be treated with particular caution in 8D given the sparse data coverage. The qEI flatness and resulting fallback behaviour suggest that the GP's uncertainty estimates may not provide meaningful guidance in large portions of the input space.

---

## Summary

| Function | Domain | Dims | Final Surrogate | Final Acquisition | Interior Penalty |
|----------|--------|------|-----------------|-------------------|------------------|
| F1 | Radiation Source Detection | 2 | Hurdle (LR + RF) | Weighted UCB κ=3 | Yes (S=0.1, F=0.01) |
| F2 | Noisy Log-Likelihood | 2 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | No |
| F3 | Drug Discovery | 3 | SFGP Matérn-2.5 ARD | qLogNEI | No |
| F4 | Warehouse Product Placement | 4 | MFGP Matérn-5/2 | qLogNEI q=4 | No |
| F5 | Chemical Process Yield | 4 | SFGP Matérn-5/2 ARD | qLogNEI q=4 + IP | Yes (S=1.0, F=0.01) |
| F6 | Cake Recipe Optimisation | 5 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Yes (S=1.0, rank-based) |
| F7 | ML Hyperparameter Tuning | 6 | NN 6→5→5→1 | MC Dropout EI | Yes (S=0.1, F=0.01) |
| F8 | 8D ML Hyperparameters | 8 | SFGP Matérn-2.5 ARD | qEI q=1 | No |
