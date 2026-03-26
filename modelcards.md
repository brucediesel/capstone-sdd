# Model Cards

This document provides model cards for each of the eight black-box optimisation functions (F1–F8) in the capstone project. Model cards are structured transparency documents, inspired by Mitchell et al. (2019), that describe what a model does, how it was built, how it performs, and what its limitations are. They support reproducibility, responsible interpretation, and informed decision-making by readers who may not have access to the underlying notebooks.

This project tackled eight black-box optimisation problems as part of a certificate course in AI and ML. For each function, a surrogate model was trained on available data and used to guide the selection of new query points via an acquisition function. Over the course of twelve weekly submission rounds (Weeks 3–13), the surrogate and acquisition strategy were refined based on observed results and systematic evaluation of alternative approaches. The model cards below document the full strategy evolution and final (Week 13) configuration for each function.

---

## F1 — Radiation Source Detection

### Overview

**Approach**: SFGP Matérn-2.5 (final); Hurdle Model (Weeks 3–9)  
**Function**: F1 — Radiation Source Detection  
**Dimensionality**: 2D input  
**Type**: Single-task Gaussian Process (final strategy); previously a two-stage hurdle model

This function's zero-inflated output distribution posed a persistent challenge. A hurdle model was used for Weeks 3–9 but was replaced by a standard GP from Week 10 onward after the hurdle failed to provide structured uncertainty estimates suitable for BO acquisition functions.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F1 black-box function, guiding candidate selection toward regions likely to contain non-zero (radiation source) signals
- Uncertainty-guided exploration of the 2D input space to locate the radiation source

**Not suitable for**:
- Production deployment for actual radiation source detection without validation against ground-truth measurements
- Extrapolation beyond the [0, 0.999999] input bounds
- Use as a standalone radiation detection system — the surrogate approximates a synthetic challenge function, not a physical detector

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | Hurdle Model (Logistic + RF) | Weighted UCB (κ=3.0) | log1p transform, interior penalty S=0.1, local penalisation r=0.15, FALLBACK_MODE |
| Week 10 | SFGP Matérn-2.5 + qLogNEI | qLogNEI q=4 | Switched to GP; Hurdle lacked structured uncertainty |
| Weeks 11–13 | SFGP Matérn-2.5 + qLogNEI | qLogNEI q=1 (Week 13) | Exploitation focus; interior penalty removed Week 13 |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch) — Matérn-2.5 with ARD
- **Acquisition**: qLogNEI, q=1 (single greedy candidate for exploitation)
- **Key change**: Interior penalty removed in Week 13 to focus on pure exploitation

**Previous Surrogate (Weeks 3–9)**:
- **Type**: Hurdle Model — Stage 1: `CalibratedClassifierCV(LogisticRegression(C=1.0))` predicting P(y > 0); Stage 2: `RandomForestRegressor(n_estimators=100, max_depth=3)` trained on `log1p(y)` for positive outputs
- **Architecture**: Two-stage pipeline — classifier gates the regressor; if fewer than 3 truly positive samples exist, FALLBACK_MODE activates and the model switches to pure exploration
- **Key Hyperparameters**: C=1.0 (logistic regularisation), n_estimators=100, max_depth=3 (RF), FALLBACK_MODE threshold=3 positive samples
- **Acquisition**: Weighted UCB — a(x) = p(x)·μ(x) + κ·p(x)·σ_RF(x), κ=3.0

*The Hurdle Model was selected after systematic evaluation of alternative surrogates (standard GP, polynomial response surface, RF/GBT) during prequential evaluation in Weeks 5–7. It was replaced in Week 10 when SFGP proved more effective with the growing dataset.*

### Performance

- **Dataset size**: 23 observations (10 initial + 13 acquired over Weeks 3–13)
- **Output range**: [−3.606 × 10⁻³, 7.71 × 10⁻¹⁶]
- **Output characteristics**: Zero-inflated — the vast majority of observations return zero or near-zero; the radiation source has not been located
- **Best observed value**: 7.71 × 10⁻¹⁶ (essentially zero — source never located)

### Assumptions and Limitations

1. **Zero-inflated output distribution**: The vast majority of the output space returns zero, making standard regression surrogates ineffective. The hurdle model (Weeks 3–9) mitigated this but relied on having at least 3 positive samples to train Stage 2 — a threshold never met. Switching to SFGP in Week 10 provided structured uncertainty but did not resolve the fundamental challenge.
2. **Small dataset regime**: With 23 observations in a 2D space, the model has limited coverage. The sustained failure to locate the source suggests the search strategy may need a fundamentally different exploration approach (e.g., space-filling design).
3. **Source not located**: After 12 rounds of optimisation across two distinct surrogate strategies, the radiation source has not been found. This is the least successful function in the campaign.

### Ethical Considerations

- **Domain context**: Radiation source detection has real-world safety implications. While this is a synthetic challenge function, the domain underscores the importance of transparent reporting — missed detections in a real scenario could have serious consequences.
- **Transparency**: Documenting the hurdle model's failure mode (FALLBACK_MODE activation) and the stalling optimisation ensures that readers understand the model's current limitations rather than assuming successful optimisation.
- **Limitations awareness**: The surrogate is an approximation of a synthetic challenge function and should not be interpreted as a validated radiation detection model. The zero-inflated output structure means the model's predictions are dominated by the classifier stage, whose accuracy is limited by the small dataset.

---

## F2 — Noisy Log-Likelihood

### Overview

**Approach**: Single-task Gaussian Process with Matérn-2.5 kernel (final); Matérn-1.5 (Weeks 3–9)  
**Function**: F2 — Noisy Log-Likelihood  
**Dimensionality**: 2D input  
**Type**: Gaussian Process (exact inference)

This model uses a single-task GP with ARD to model a noisy log-likelihood surface. The kernel was upgraded from Matérn-1.5 to Matérn-2.5 in Week 10 as the growing dataset supported a smoother fit.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F2 black-box function, identifying input configurations that maximise the noisy log-likelihood
- Uncertainty-guided exploration of the 2D input space, leveraging GP posterior variance for principled exploration–exploitation trade-off

**Not suitable for**:
- Direct use as a log-likelihood estimator for real statistical models — the surrogate approximates a synthetic function
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications requiring deterministic or noise-free predictions

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Standardize transform, distance-based selection, noise_lb=1e-3 |
| Week 10 | SFGP Matérn-2.5 ARD | qLogNEI q=4 | Smoother kernel for well-behaved surface |
| Weeks 11–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Exploitation focus |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (2 lengthscale parameters)
- **Key Hyperparameters**: noise_lb=1e-3, Normalize input transform, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI (q-Log Noisy Expected Improvement)
- **Configuration**: q=1 (exploitation-focused in final weeks)

**Special Techniques**:
- Lengthscale bounds [0.01, 2.0] to prevent GP degeneracy
- Distance-based candidate selection during batch acquisition phases (q=4)

*This model was selected after systematic evaluation of alternative surrogates including SFGP and MFGP during prequential evaluation.*

### Performance

- **Dataset size**: 23 observations (10 initial + 13 acquired over Weeks 3–13)
- **Output range**: [−0.066, 0.674]
- **Output characteristics**: Mixed sign — predominantly positive values
- **Best observed value**: 0.674

### Assumptions and Limitations

1. **Small dataset regime**: With only 23 observations in 2D, the GP posterior is dominated by the prior in regions far from data. The lengthscale bounds mitigate pathological fits but limit the model's ability to capture fine-scale features.
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

This model uses a single-task GP with a Matérn-2.5 kernel and ARD to model a drug discovery objective function with an entirely negative output domain. The strategy remained consistent throughout the campaign, with refinements to batch size and standardisation approach.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F3 black-box function, guiding candidate selection to identify molecular configurations or process parameters that maximise the drug discovery objective
- Uncertainty-guided exploration of the 3D input space

**Not suitable for**:
- Direct use as a drug efficacy predictor — the surrogate models a synthetic challenge function, not a real pharmacological response
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications where all-negative output values may be misinterpreted as errors rather than a genuine characteristic of the function

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–7 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | Standardize transform |
| Week 8 | SFGP Matérn-2.5 ARD | qLogNEI q=3 | Increased batch; output shifting to handle negative values |
| Week 9 | SFGP Matérn-2.5 ARD | qLogNEI q=3 | Standardize(m=1), 3-colour diagnostics |
| Weeks 10–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Tuning refinements; exploitation focus |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (3 lengthscale parameters), lengthscale initialisation=0.25
- **Key Hyperparameters**: noise_lb=1e-6, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI
- **Configuration**: q=1 (exploitation-focused in final weeks)

**Special Techniques**:
- Manual z-score standardisation: outputs transformed as (y − mean) / std before GP fitting, recomputed per LOO fold to prevent data leakage
- Output shifting to handle negative values during batch acquisition phases

*This model was selected after systematic evaluation of alternative surrogates including GP, BART, and Random Forest during prequential evaluation.*

### Performance

- **Dataset size**: 28 observations (15 initial + 13 acquired over Weeks 3–13)
- **Output range**: [−0.399, −0.0114]
- **Output characteristics**: All negative (28 out of 28 observations)
- **Best observed value**: −0.0114

### Assumptions and Limitations

1. **All-negative output domain**: Every observed output is negative. The GP must model relative differences within a narrow negative range, which can be challenging for acquisition functions that use zero as a reference point. Manual z-score standardisation shifts the outputs to have zero mean, mitigating this issue.
2. **Small dataset regime**: With 28 observations in 3D, the GP has limited ability to resolve the full landscape. The low noise assumption (1e-6) may cause overfitting to observed points.
3. **Z-score recomputation**: The manual standardisation requires recomputing mean and standard deviation for each LOO fold, adding complexity and potential for implementation errors compared to BoTorch's built-in Standardize transform.

### Ethical Considerations

- **Domain context**: Drug discovery has direct implications for human health. While this is a synthetic function, the domain highlights the importance of documenting model limitations — an under-explored region of the surrogate could correspond to a promising drug candidate being overlooked.
- **Transparency**: Documenting the manual z-score approach and the all-negative output characteristics ensures readers understand the preprocessing choices and can assess their appropriateness.
- **Limitations awareness**: The surrogate approximates a synthetic challenge function and does not capture real pharmacological interactions. Any insights should not be transferred to actual drug discovery without extensive validation.

---

## F4 — Warehouse Product Placement

### Overview

**Approach**: SFGP Matérn-2.5 (final); Multi-Fidelity GP (Weeks 3–9)  
**Function**: F4 — Warehouse Product Placement  
**Dimensionality**: 4D input  
**Type**: Single-task Gaussian Process (final); previously Multi-Fidelity GP

This model transitioned from a multi-fidelity GP architecture to a standard single-task GP in Week 10. Although the MFGP's LinearTruncatedFidelityKernel provided useful regularisation early on, the simpler SFGP proved competitive as more data accumulated.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F4 black-box function, identifying warehouse product placement configurations that maximise the objective
- Standard GP-based uncertainty-guided exploration in a 4D space

**Not suitable for**:
- Direct use as a warehouse layout optimiser — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- True multi-fidelity applications — despite using MFGP architecture in early weeks, all data is at fidelity=1.0

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | MFGP (Multi-Fidelity GP) | qLogNEI q=4 | LinearTruncatedFidelityKernel as regulariser; fidelity=1.0 |
| Week 10 | SFGP Matérn-2.5 ARD | qLogNEI q=4 | Switched from MFGP; 4D single-task GP simpler and competitive |
| Weeks 11–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Exploitation focus |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (4 lengthscale parameters)
- **Key Hyperparameters**: 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI
- **Configuration**: q=1 (exploitation-focused in final weeks)

**Previous Surrogate (Weeks 3–9)**:
- **Type**: SingleTaskMultiFidelityGP (BoTorch)
- **Kernel**: Matérn-5/2 + LinearTruncatedFidelityKernel, ARD (4 spatial + 1 fidelity lengthscale)
- **Key Hyperparameters**: noise_lb=1e-4, fidelity=1.0 for all data points
- Prequential evaluation winner with best NLP score (−1.35) among surrogates evaluated for F4
- Manual z-score standardisation on outputs

*The MFGP was selected via prequential evaluation in early weeks. It was replaced in Week 10 when accumulating data made the simpler SFGP equally competitive without the fidelity dimension overhead.*

### Performance

- **Dataset size**: 43 observations (30 initial + 13 acquired over Weeks 3–13)
- **Output range**: [−32.626, 0.532]
- **Output characteristics**: Mostly negative — extreme outliers present
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

**Approach**: Single-task Gaussian Process with Matérn-1.5 kernel (final); Matérn-5/2 and GBT explored earlier  
**Function**: F5 — Chemical Process Yield  
**Dimensionality**: 4D input  
**Type**: Gaussian Process (exact inference)

This model uses a single-task GP with a Matérn-1.5 kernel and ARD. The rougher Matérn-1.5 kernel was adopted from Week 9 onward after a series of surrogate experiments (GBT in Week 5, Matérn-5/2 in Week 8). A log1p→z-score transform chain addresses the extremely heavy-tailed output distribution, and an in-loop interior penalty prevents boundary-pinning.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F5 black-box function, identifying chemical process parameters that maximise yield
- Uncertainty-guided exploration with interior penalty to avoid degenerate boundary solutions

**Not suitable for**:
- Direct use as a chemical process control model — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Prediction of actual chemical yields without calibration against real experimental data

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | Standard BO; best ≈1,089 |
| Week 5 | GBT Ensemble (10 models) | UCB κ=2.5 | 20,000 candidates; departure from GP |
| Week 8 | SFGP Matérn-5/2 ARD | qLogNEI q=4 | log1p + z-score, additive interior penalty (S=1.0, F=0.01); major jump to ~3,000+ |
| Week 9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Rougher kernel for small data; 5,000 raw samples |
| Weeks 10–12 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | MLL restarts 50→60; raw samples 5,000→8,000; distance gate relaxed |
| Week 13 | SFGP Matérn-1.5 ARD | qLogNEI q=1 | Interior penalty removed; single greedy point |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-1.5 with ARD (4 lengthscale parameters)
- **Key Hyperparameters**: noise_lb=1e-6, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI
- **Configuration**: q=1 (exploitation-focused), interior penalty removed in Week 13

**Special Techniques**:
- Interior penalty (Weeks 8–12): S=1.0, F=0.01, in-loop additive penalty in log-space
- Output transform chain: log1p(y) → z-score standardisation; inverse: expm1(z·std + mean)
- Distance-based candidate selection during batch phases

*Strategy evolved through systematic evaluation — GBT was eliminated by prequential evaluation, and Matérn-1.5 proved superior to Matérn-5/2 for the heavy-tailed surface.*

### Performance

- **Dataset size**: 33 observations (20 initial + 13 acquired over Weeks 3–13)
- **Output range**: [0.113, 8,662.4]
- **Output characteristics**: All positive, extremely heavy-tailed — values span nearly 5 orders of magnitude
- **Best observed value**: 8,662.4 (approximately 3.1× improvement factor from initial best)

### Assumptions and Limitations

1. **Heavy-tailed distribution**: The output range spans 0.11 to 3,395, requiring a log1p transform to stabilise the GP fit. The transform assumes a log-normal-like distribution of yields, which may not hold across the entire input space.
2. **Boundary-pinning risk**: Without the interior penalty, the acquisition function tends to propose candidates at the input boundaries (0 or 0.999999), which is a known failure mode for GP-based optimisation in bounded domains. The in-loop penalty mitigates this but introduces additional hyperparameters (S, F) that must be tuned.
3. **Transform chain complexity**: The log1p→z-score→expm1 inverse transform chain adds complexity and potential for numerical instability, particularly for values near zero where log1p approximation effects are strongest.

### Ethical Considerations

- **Domain context**: Chemical process optimisation affects safety, environmental impact, and product quality. While this is a synthetic function, the domain highlights the importance of understanding model limitations — an incorrectly optimised process could lead to waste, unsafe conditions, or poor yields in a real setting.
- **Transparency**: Documenting the in-loop penalty mechanism and the transform chain enables readers to understand exactly how candidates are selected and why boundary solutions are suppressed.
- **Limitations awareness**: The surrogate is trained on 33 observations in 4D, providing sparse coverage. High-yield regions identified by the surrogate may not correspond to robust operating conditions in a real chemical process.

---

## F6 — Cake Recipe Optimisation

### Overview

**Approach**: Single-task Gaussian Process with Matérn-1.5 kernel (final); NN explored in Week 5  
**Function**: F6 — Cake Recipe Optimisation  
**Dimensionality**: 5D input  
**Type**: Gaussian Process (exact inference)

This model uses a single-task GP with a Matérn-1.5 kernel and ARD to model a cake recipe scoring function where all outputs are negative. A rank-based interior penalty was specifically developed for this function's all-negative output domain, where standard multiplicative penalties would invert acquisition rankings.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F6 black-box function, identifying cake recipe configurations that minimise quality deductions (i.e., maximise the negative score toward zero)
- Uncertainty-guided exploration of the 5D recipe parameter space with feasibility constraints

**Not suitable for**:
- Direct use as a recipe optimiser for actual baking — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Interpretation of outputs as absolute quality scores — values represent deductions within the challenge scoring system

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | best ≈−0.714 |
| Week 5 | Neural Network (5→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates |
| Week 8 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Rank-based interior penalty; milk constraint ≥0.10 |
| Week 9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Enhanced diagnostics; rank-based IP maintained |
| Weeks 10–13 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Noise floor 1e-2→1e-3; milk 0.10→0.12; raw samples 3,000→5,000 |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-1.5 with ARD (5 lengthscale parameters)
- **Key Hyperparameters**: noise_lb=1e-3, Standardize(m=1) output transform, 15-restart marginal log-likelihood optimisation

**Acquisition Function**:
- **Type**: qLogNEI
- **Configuration**: q=4, MC=512 Monte Carlo samples, 50 restarts, 5,000 raw Sobol samples, feasibility bounds (x₄ ≥ 0.12, other dimensions ≥ 0.01)

**Special Techniques**:
- Rank-based interior penalty: S=1.0, F=0.01, sign-invariant design — uses rank-based scoring instead of multiplicative penalty to avoid inverting rankings for all-negative output values
- Feasibility constraints: dimension 4 (milk) requires values ≥ 0.12; all other dimensions require values ≥ 0.01

*This model was selected after systematic evaluation of alternative surrogates including SFGP, MFGP, and Neural Network during prequential evaluation.*

### Performance

- **Dataset size**: 33 observations (20 initial + 13 acquired over Weeks 3–13)
- **Output range**: [−2.571, −0.111]
- **Output characteristics**: All negative — scores represent deductions from an ideal recipe
- **Best observed value**: −0.111 (approximately 85% improvement from initial best)

### Assumptions and Limitations

1. **All-negative output domain**: Standard multiplicative interior penalties would invert the acquisition rankings because multiplying a negative acquisition value by a positive penalty reverses the ordering. The rank-based penalty was designed specifically to handle this, but adds implementation complexity.
2. **Feasibility constraints**: The feasibility bounds (x₄ ≥ 0.10) represent domain knowledge about valid recipe proportions. These hard constraints reduce the effective search space and could exclude potentially optimal regions if the bounds are too conservative.
3. **Higher noise assumption**: The noise_lb=1e-2 is significantly higher than for other functions, reflecting the assumption that recipe scoring has inherent variability. If the function is actually deterministic, this over-estimates noise and reduces surrogate precision.

### Ethical Considerations

- **Domain context**: Recipe optimisation is a consumer-facing domain. While this is a synthetic function, transparent reporting ensures that recipe recommendations derived from such models are understood to be surrogate-guided suggestions, not guaranteed outcomes.
- **Transparency**: Documenting the rank-based penalty mechanism and feasibility constraints enables readers to understand the design choices specific to handling all-negative outputs and domain-specific constraints.
- **Limitations awareness**: The 5D recipe space is sparsely sampled with 33 observations. The surrogate's recommendation may not generalise to actual baking conditions where additional factors (oven calibration, ingredient freshness) affect outcomes.

---

## F7 — ML Hyperparameter Tuning

### Overview

**Approach**: Neural Network Surrogate with MC Dropout  
**Function**: F7 — ML Hyperparameter Tuning  
**Dimensionality**: 6D input  
**Type**: Neural Network (feedforward with dropout-based uncertainty)

This model uses a compact neural network as the surrogate function instead of a Gaussian Process. The NN was chosen because it scales better to higher-dimensional input spaces and can model complex, non-stationary response surfaces. The architecture was progressively compacted from a larger initial network (Week 5) to a minimal design, and the acquisition function evolved from pure EI to a blended exploitation-focused strategy.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F7 black-box function, identifying ML hyperparameter configurations that maximise model performance
- Approximate uncertainty-guided exploration via MC Dropout, providing a computationally efficient alternative to GP posterior inference in 6D

**Not suitable for**:
- Direct use as a hyperparameter optimiser for production ML systems — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- Applications requiring well-calibrated uncertainty estimates — MC Dropout uncertainty is known to be approximate and may be poorly calibrated

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | Standard BO |
| Week 5 | NN (6→128→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates; NN for 6D scalability |
| Week 8 | Compact NN (6→5→5→1) | EI via MC Dropout | Interior penalty (S=0.1, F=0.01); 200 epochs, lr=0.005 |
| Weeks 9–13 | Compact NN (6→5→5→1) | Blended: 0.7×mean + 0.3×EI | Dropout 0.1→0.05; relaxed IP (S=0.05, F=0.02); exploitation-focused |

**Final Surrogate Model (Week 13)**:
- **Type**: SurrogateNN — custom feedforward neural network
- **Architecture**: 6 → 5 → 5 → 1 (input → hidden → hidden → output), 71 trainable parameters, ReLU activation, Dropout(0.05) between layers
- **Key Hyperparameters**: Adam optimiser with lr=0.005, MSE loss, 200 training epochs, seed=42 for reproducibility

**Acquisition Function**:
- **Type**: Blended acquisition — 0.7×posterior_mean + 0.3×EI via MC Dropout
- **Configuration**: 50 forward passes with dropout enabled to estimate μ(x) and σ(x), 20,000 random candidates evaluated

**Special Techniques**:
- Interior penalty: relaxed from S=0.1→0.05, F=0.01→0.02 over campaign
- MC Dropout for uncertainty estimation: dropout remains active during inference
- Manual z-score standardisation on both inputs (X) and outputs (Y)

*This model was selected after systematic evaluation of alternative surrogates (SFGP, MFGP) during prequential evaluation in Weeks 5–7. The NN was retained as the only non-GP surrogate in the campaign.*

### Performance

- **Dataset size**: 43 observations (30 initial + 13 acquired over Weeks 3–13)
- **Output range**: [0.003, 2.305]
- **Output characteristics**: All positive
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

This model uses a single-task GP with a Matérn-2.5 kernel and ARD to model an 8-dimensional ML hyperparameter optimisation function. Despite the high dimensionality, a GP was retained throughout because the relatively large initial dataset (40 points) provides sufficient coverage for kernel fitting. A brief NN experiment in Week 5 was abandoned in favour of returning to GP with qEI.

### Intended Use

**Suitable for**:
- Surrogate-based optimisation of the F8 black-box function, identifying 8-dimensional hyperparameter configurations that maximise model performance
- Uncertainty-guided exploration of the 8D input space, with a Sobol fallback strategy for regions where the acquisition function becomes flat

**Not suitable for**:
- Direct use as a hyperparameter optimiser for production ML systems — the surrogate approximates a synthetic challenge function
- Extrapolation beyond the [0, 0.999999] input bounds
- High-throughput batch optimisation — this function uses q=1 (single candidate per round)

### Details

**Strategy Evolution (Weeks 3–13)**:

| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | 8D; 40 initial samples |
| Week 5 | NN (8→128→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates |
| Week 8 | SFGP Matérn-2.5 ARD | qEI q=1, XI=0.01 | Returned to GP; 256 MC samples, 30 restarts, 4,096 raw; fallback to posterior mean |
| Weeks 9–13 | SFGP Matérn-2.5 ARD | qEI q=1, XI=0.01 | Stable strategy; 3-colour diagnostics added Week 9 |

**Final Surrogate Model (Week 13)**:
- **Type**: SingleTaskGP (BoTorch/GPyTorch)
- **Kernel**: Matérn-2.5 with ARD (8 lengthscale parameters)
- **Key Hyperparameters**: noise_lb=1e-7, Standardize(m=1) output transform

**Acquisition Function**:
- **Type**: qEI (Expected Improvement)
- **Configuration**: q=1 (single candidate), MC=256, best_f=y_max + 0.01 (XI=0.01 exploration bonus), 30 restarts, 4,096 raw Sobol samples

**Special Techniques**:
- Sobol fallback: When all qEI values are zero, the model selects the candidate with the highest posterior mean from 4,096 Sobol samples
- XI=0.01 exploration bonus to encourage exploration beyond the current best

*The GP approach was maintained across the campaign. An NN experiment in Week 5 was abandoned as the GP proved more reliable in 8D with the large initial dataset.*

### Performance

- **Dataset size**: 53 observations (40 initial + 13 acquired over Weeks 3–13)
- **Output range**: [5.592, 9.982]
- **Output characteristics**: All positive — relatively narrow range with high baseline
- **Best observed value**: 9.982

### Assumptions and Limitations

1. **High dimensionality with limited data**: 53 observations in 8D provides extremely sparse coverage (approximately 2 points per input dimension above the initial set). The GP posterior will be dominated by the prior in most of the input space, and lengthscale estimation for 8 ARD parameters requires substantial data.
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
| F1 | Radiation Source Detection | 2 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | Removed (Week 13) |
| F2 | Noisy Log-Likelihood | 2 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | No |
| F3 | Drug Discovery | 3 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | No |
| F4 | Warehouse Product Placement | 4 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | No |
| F5 | Chemical Process Yield | 4 | SFGP Matérn-1.5 ARD | qLogNEI q=1 | Removed (Week 13) |
| F6 | Cake Recipe Optimisation | 5 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Yes (S=1.0, rank-based) |
| F7 | ML Hyperparameter Tuning | 6 | NN 6→5→5→1 | Blended 0.7×mean + 0.3×EI | Yes (S=0.05, F=0.02) |
| F8 | 8D ML Hyperparameters | 8 | SFGP Matérn-2.5 ARD | qEI q=1 | No |
