# Research: Datasheets & Model Cards

**Feature**: 022-datasheets-modelcards  
**Date**: 2026-03-05  
**Purpose**: Resolve all unknowns from Technical Context and gather per-function data for document generation

## Research Tasks & Findings

### R1: Document Format & Structure Best Practices

**Decision**: Standard markdown with consistent heading hierarchy and summary tables  
**Rationale**: Model cards (Mitchell et al., 2019) and datasheets (Gebru et al., 2021) are industry-standard transparency documents. Markdown is the natural format for a GitHub-hosted project and renders natively in all code viewers.  
**Alternatives considered**: PDF (rejected — not version-controllable), LaTeX (rejected — overkill for course deliverable), HTML (rejected — extra tooling)

### R2: Model Card Scope — Historical vs. Final

**Decision**: Document only the Week 9 final model configuration, with a one-sentence note that the model was selected after systematic evaluation of alternatives  
**Rationale**: Clarification from user — "only document the model cards from the last week." Full evolution history is available in weekly notebooks for readers who want it.  
**Alternatives considered**: Full 10-round evolution table (rejected per clarification), strictly final-only with zero history (rejected — brief context aids understanding)

### R3: Data Verification — Shapes & Ranges

All data verified from actual `.npy` files:

| Fn | Init In Shape | Init Out Shape | W9 In Shape | W9 Out Shape | Output Min | Output Max |
|----|---------------|----------------|-------------|--------------|------------|------------|
| F1 | (10, 2) | (10, 1) | (19, 2) | (19, 1) | −3.606e-03 | 0.000e+00 |
| F2 | (10, 2) | (10, 1) | (19, 2) | (19, 1) | −0.065624 | 0.674355 |
| F3 | (15, 3) | (15, 1) | (24, 3) | (24, 1) | −0.398926 | −0.031427 |
| F4 | (30, 4) | (30, 1) | (39, 4) | (39, 1) | −32.625660 | 0.532175 |
| F5 | (20, 4) | (20, 1) | (29, 4) | (29, 1) | 0.112940 | 3394.679933 |
| F6 | (20, 5) | (20, 1) | (29, 5) | (29, 1) | −2.571170 | −0.111490 |
| F7 | (30, 6) | (30, 1) | (39, 6) | (39, 1) | 0.002701 | 2.304991 |
| F8 | (40, 8) | (40, 1) | (49, 8) | (49, 1) | 5.592193 | 9.982473 |

### R4: Per-Function Final Model Details (Week 9)

#### F1 — Radiation Source Detection (2D)
- **Surrogate**: Hurdle Model — Stage 1: `CalibratedClassifierCV(LogisticRegression(C=1.0))` for P(y>0); Stage 2: `RandomForestRegressor(n_estimators=100, max_depth=3)` on `log1p(y)` for positive outputs; FALLBACK_MODE=True (< 3 truly positive samples → pure exploration)
- **Acquisition**: Weighted UCB — a(x) = p(x)·μ(x) + κ·p(x)·σ_RF(x), κ=3.0, 20,000 random candidates
- **Interior penalty**: Yes, S=0.1 (gentle), F=0.01 (sinusoidal boundary suppression)
- **Special**: Local penalisation (Gaussian mask, radius=0.15), fallback exploration
- **Output characteristics**: Zero-inflated — 14 near-zero, 5 negative, radiation source not yet located
- **Preprocessing**: log1p(y) on positive outputs for Stage 2
- **LOO metrics**: MAE/RMSE on 9 submission folds
- **Stalling**: Yes — zero improvement; source not located

#### F2 — Noisy Log-Likelihood (2D)
- **Surrogate**: SingleTaskGP, Matérn-1.5, ARD (2 LS), Normalize, noise_lb=1e-3, LS bounds [0.01, 2.0], 15-restart MLL
- **Acquisition**: qLogNEI, q=4, MC=512, 20 restarts, 1024 raw → distance-based selection (mean ≥ median, farthest from data)
- **Interior penalty**: No
- **Special**: LS bounds to prevent GP degeneracy, distance-based candidate selection
- **Output characteristics**: Mixed (2 negative, 17 positive)
- **Preprocessing**: BoTorch Normalize
- **LOO metrics**: MAE/RMSE on 9 submission folds
- **Stalling**: Monitor — distance-based selection is recovery strategy

#### F3 — Drug Discovery (3D)
- **Surrogate**: SingleTaskGP, Matérn-2.5, ARD (3 LS), noise_lb=1e-6, LS init=0.25, 15-restart MLL
- **Acquisition**: qLogNEI, 10 restarts, 512 raw samples
- **Interior penalty**: No
- **Special**: Manual z-score standardisation (recomputed per LOO fold)
- **Output characteristics**: All negative (24/24)
- **Preprocessing**: Manual z-score: (y − mean) / std
- **LOO metrics**: MAE/RMSE with z-score recomputation per fold
- **Stalling**: Monitor

#### F4 — Warehouse Product Placement (4D)
- **Surrogate**: SingleTaskMultiFidelityGP, Matérn-5/2 + LinearTruncatedFidelityKernel, ARD (4 spatial + 1 fidelity dim), noise_lb=1e-4, fidelity=1.0 for all points, 15-restart MLL
- **Acquisition**: qLogNEI (MF-qNEI), q=4, MC=64, 20 restarts, 512 raw, fixed_features={4: 1.0}, prune_baseline=True
- **Interior penalty**: No
- **Special**: Multi-fidelity GP as regulariser (synthetic single-fidelity), PE winner (NLP=−1.35)
- **Output characteristics**: Mostly negative (34 neg, 5 pos)
- **Preprocessing**: Manual z-score: (y − mean) / std
- **LOO metrics**: MAE/RMSE/Median/Max on submission points
- **Stalling**: Monitor

#### F5 — Chemical Process Yield (4D)
- **Surrogate**: SingleTaskGP, Matérn-5/2, ARD (4 LS), noise_lb=1e-6, LS init=0.5, 15-restart MLL
- **Acquisition**: qLogNEI, q=4, MC=512, 50 restarts, 3000 raw → distance-based selection + in-loop PenalisedAcquisition wrapper (additive in log-space)
- **Interior penalty**: Yes, S=1.0, F=0.01, penalty=4x(1−x), additive log-space, bounds tightened to [0.005, 0.995]
- **Special**: In-loop penalty wrapper (fixes multiplicative failure in log-space), expm1 inverse
- **Output characteristics**: All positive, heavy-tailed (0.11 to 3395)
- **Preprocessing**: log1p → manual z-score; inverse: expm1(z·std + mean)
- **LOO metrics**: MAE/RMSE/Median/Max on submissions
- **Stalling**: Boundary-pinning resolved by in-loop penalty

#### F6 — Cake Recipe Optimisation (5D)
- **Surrogate**: SingleTaskGP, Matérn-1.5, ARD (5 LS), noise_lb=1e-2, Standardize(m=1), LS init=0.5, 15-restart MLL
- **Acquisition**: qLogNEI, q=4, MC=512, 50 restarts, 3000 raw → distance-based selection, feasibility bounds (x4 ≥ 0.10, others ≥ 0.01)
- **Interior penalty**: Yes, S=1.0, F=0.01, rank-based (sign-invariant for all-negative outputs)
- **Special**: Rank-based penalty (multiplicative would invert rankings for negative outputs), feasibility constraints
- **Output characteristics**: All negative (29/29, score of deductions)
- **Preprocessing**: Standardize(m=1) (BoTorch auto z-score)
- **LOO metrics**: MAE/RMSE/Median/Max on submissions
- **Stalling**: Monitor

#### F7 — ML Hyperparameter Tuning (6D)
- **Surrogate**: Neural Network SurrogateNN(6→5→5→1), 71 params, ReLU + Dropout(0.1), Adam(lr=0.005), MSE loss, 200 epochs, seed=42
- **Acquisition**: MC Dropout EI — 50 forward passes for μ(x), σ(x), standard EI formula, 20,000 random candidates
- **Interior penalty**: Yes, S=0.1, F=0.01 (multiplicative, safe for positive outputs)
- **Special**: Non-GP surrogate, MC Dropout uncertainty estimation
- **Output characteristics**: All positive (39/39)
- **Preprocessing**: Manual z-score on both X and Y
- **LOO metrics**: MAE/RMSE (NN retrained 200 epochs/fold), Train R²
- **Stalling**: Monitor — NN dropout uncertainty poorly calibrated

#### F8 — 8D ML Hyperparameters (8D)
- **Surrogate**: SingleTaskGP, Matérn-2.5, ARD (8 LS), noise_lb=1e-7, Standardize(m=1)
- **Acquisition**: qEI, q=1, MC=256, best_f=y_max+0.01 (XI=0.01), 30 restarts, 4096 raw
- **Interior penalty**: No
- **Special**: Sobol fallback (if all qEI=0 → select highest posterior mean from 4096 candidates), exploration bonus XI=0.01
- **Output characteristics**: All positive (49/49)
- **Preprocessing**: Standardize(m=1) (BoTorch auto z-score)
- **LOO metrics**: MAE/RMSE (retrained per fold with auto Standardize)
- **Stalling**: Monitor — qEI flatness triggers fallback

### R5: Prequential Evaluation Methodology

**Decision**: All 8 functions underwent prequential evaluation during Weeks 5–7  
**Rationale**: PE was used to systematically compare surrogate alternatives (GP, RF, GBT, BART, NN, SFGP, MFGP) before selecting final models  
**Metrics recorded**: MAE, NLP (Negative Log Predictive density), 95% Prediction Interval Coverage  
**LOO evaluation**: All Week 9 notebooks include LOO (Leave-One-Out) surrogate error analysis on the 9 submission-round data points

### R6: Ethical Considerations Scope

**Decision**: Domain-appropriate ethical considerations at capstone course level  
**Rationale**: These are surrogate models for a coursework challenge, not production systems. Ethical considerations focus on transparency, reproducibility, and responsible interpretation of surrogate approximations.  
**Alternatives considered**: Full ethics review (rejected — disproportionate for course deliverable)

## Summary Comparison Table

| Property | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| **Domain** | Radiation source | Log-likelihood | Drug discovery | Warehouse | Chemical yield | Cake recipe | ML hyperparams | 8D ML hyperparams |
| **Dims** | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |
| **Init→W9 pts** | 10→19 | 10→19 | 15→24 | 30→39 | 20→29 | 20→29 | 30→39 | 40→49 |
| **Output sign** | Zero-inflated | Mixed | All negative | Mostly neg | All positive | All negative | All positive | All positive |
| **Surrogate** | Hurdle (LR+RF) | SFGP Matérn-1.5 | SFGP Matérn-2.5 | MFGP Matérn-5/2 | SFGP Matérn-5/2 | SFGP Matérn-1.5 | NN 6→5→5→1 | SFGP Matérn-2.5 |
| **Acquisition** | Weighted UCB κ=3 | qLogNEI q=4 | qLogNEI | qLogNEI q=4 | qLogNEI q=4 | qLogNEI q=4 | MC Dropout EI | qEI q=1 |
| **Interior penalty** | Yes (S=0.1) | No | No | No | Yes (S=1.0) | Yes (S=1.0, rank) | Yes (S=0.1) | No |
| **Transform** | log1p | Normalize | z-score | z-score | log1p→z-score | Standardize | z-score (X+Y) | Standardize |
