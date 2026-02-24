# Research: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Feature**: 017-f7-nn-interior-penalty  
**Date**: 2025-02-24

---

## Decision 1: Epoch Count for 71-Parameter Network

**Decision**: 200 epochs

**Rationale**: The 6→5→5→1 architecture has ~71 parameters (30+5 + 25+5 + 5+1). With 37 training samples, the parameter-to-sample ratio is ~1.9:1 (overparameterised). At lr=0.005 with Adam, convergence to low training loss happens in 100–200 epochs. Beyond that, the network memorises noise. Previous weeks used 500–800 epochs for 12,769-parameter networks — 200 is proportionally appropriate for 71 parameters. Dropout regularisation slows convergence slightly, justifying 200 over 100.

**Alternatives considered**:
- 500 (Week 5 default): Overkill; loss curve flat for 300+ epochs. Wastes time and risks noise memorisation.
- 100: Slightly risky — dropout may require a few more epochs for stable convergence.
- Early stopping (patience=20): Sound but adds code complexity. At 200 epochs the overhead is minimal.

---

## Decision 2: Dropout Rate for 5-Node Layers

**Decision**: 0.1 (reduced from 0.2)

**Rationale**: With 5 nodes per layer, dropout=0.2 drops 1 node on average — removing 20% of layer capacity per forward pass. The effective sub-network is only 4 nodes wide, potentially too narrow. At dropout=0.1, expected active nodes = 4.5. The probability of dropping 2+ nodes per layer drops from ~26% (p=0.2) to ~7% (p=0.1), preserving representational capacity while still enabling stochastic MC variance estimation. This is a deliberate change from Weeks 5–6 where layers had 128/64/32 nodes.

**Alternatives considered**:
- 0.2 (Weeks 5–6): Designed for 128/64/32-node layers. Inappropriate for 5-node layers.
- 0.05: Too low — MC variance negligible, EI collapses to deterministic.
- 0.15: Viable but 0.1 is cleaner and safer for 5-node layers.

---

## Decision 3: MC Dropout EI Computation Method

**Decision**: Sample-then-average (non-parametric MC estimator)

Formula: `EI(x) = (1/N) · Σᵢ max(fᵢ(x) − y_best, 0)` where `fᵢ(x)` is the i-th stochastic forward pass prediction (un-normalised).

**Rationale**: This is the exact MC estimator of Expected Improvement. EI is defined as E[max(f(x) − y_best, 0)] where the expectation is over the predictive posterior. MC Dropout approximates this posterior by sampling network weights via dropout masks. The sample-then-average approach is non-parametric — no Gaussianity assumption on the MC predictive distribution. For a 5-node network, the predictive distribution may be multimodal or skewed, so assuming Gaussian (as the analytic EI formula does) could be inaccurate. With 50 MC samples, the estimator has ~14% relative standard error — sufficient for ranking candidates.

**Implementation detail**: Un-normalise MC predictions before computing EI:
```python
mc_preds_orig = mc_predictions * y_std + y_mean  # back-transform
y_best = y_outputs.max()  # best observed in original scale
ei = np.mean(np.maximum(mc_preds_orig - y_best, 0), axis=0)  # (N_CANDIDATES,)
```

**Alternatives considered**:
- Analytic EI from (μ_MC, σ_MC): Adds Gaussianity assumption; questionable for 5-node network.
- UCB (Weeks 5–6 method): Valid but spec requires NEI. Also, UCB doesn't naturally compose with interior penalty as cleanly as EI × w.

---

## Decision 4: Candidate Count

**Decision**: 20,000 uniform random candidates in [0, 1]⁶

**Rationale**: For N=20,000 in D=6, expected nearest-neighbour distance ≈ N^(−1/D) ≈ 0.192. Every point in the unit hypercube is within ~0.19 of some candidate — adequate for a smooth acquisition surface. The interior penalty concentrates "interesting" candidates toward the interior, improving effective coverage. Total compute: 50 × 20,000 = 1M forward passes through a 71-parameter network — trivial.

**Alternatives considered**:
- 50,000: Only ~12% improvement in coverage (0.165 vs 0.192). 2.5M MC passes — still fast but unnecessary.
- Sobol/LHS sampling: Better space-filling but adds scipy dependency. Uniform is sufficient at 20k.

---

## Decision 5: Interior Penalty Approach — Multiplicative

**Decision**: Multiplicative penalty `penalised_EI(x) = EI(x) · w(x)` with STEEPNESS=1.0, FLOOR=0.01

**Rationale**: All F7 outputs are positive (range [0.003, 2.305]), so EI values are non-negative. Multiplying by w(x) ∈ [0.01, 1.0] correctly suppresses boundary candidates (lower penalised EI) and promotes interior candidates (higher penalised EI). This is simpler than the rank-based approach used for F6 (which was needed because F6 had all-negative outputs making multiplicative incorrect). STEEPNESS=1.0 and FLOOR=0.01 are consistent with F5/F6 specs.

**Edge case — all EI = 0**: If no candidate improves over y_best, penalised_EI = 0 everywhere. Fallback: select the candidate with the highest interior weight (most interior point) to encourage exploration.

**Alternatives considered**:
- Rank-based scoring (F6 approach): Unnecessary for positive EI. Adds complexity without benefit.
- Additive penalty: `EI(x) + λ·w(x)` — requires tuning λ to balance EI and penalty scales. Multiplicative is scale-invariant.

---

## Summary Table

| Hyperparameter | Value | Previous (Wk 5–6) | Rationale |
|---|---|---|---|
| Architecture | 6→5→5→1 | 6→128→64→32→1 / 6→64→32→1 | User-specified (2L×5N) |
| Parameters | ~71 | ~12,769 / ~2,209 | 180× / 31× smaller |
| Learning rate | 0.005 | 0.005 / 0.01 | User-specified |
| Epochs | 200 | 800 / 500 | 71 params converge fast; longer = memorisation |
| Dropout | 0.1 | 0.2 | 5-node layers can't afford 20% capacity loss |
| MC samples | 50 | 50 | Consistent; sufficient for EI ranking |
| Acquisition | MC Dropout EI × penalty | UCB | Non-parametric EI; multiplicative penalty (positive outputs) |
| Candidates | 20,000 | 20,000 | NN distance ~0.19 in 6D; adequate |
| STEEPNESS | 1.0 | 1.0 (F5/F6) | 6D product already strong at S=1.0 |
| FLOOR | 0.01 | 0.01 (F5/F6) | Consistent with prior specs |
