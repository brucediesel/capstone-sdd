# Research: F2–F8 Week 8 Bayesian Optimisation

**Feature**: `020-f2-f8-week8` | **Date**: 2026-03-01

## Research Questions

### RQ-1: What are the exact Week 7 code-level implementations for each function?

**Decision**: Extract and replicate the Week 7 code cells from each `fX.ipynb` notebook as the ground truth for Week 8. The spec-level descriptions are secondary to the actual notebook code.

**Rationale**: Multiple discrepancies exist between prior specs and the actual notebook implementations (e.g., F7 DROPOUT=0.1 vs spec's 0.2, F7 EPOCHS=200 vs spec's 500). The running notebooks are the deliverables being graded, so code is ground truth.

**Alternatives considered**:
- Use spec values: Rejected — would change the strategy, which contradicts "keeping the same strategy as week 7"
- Re-derive from first principles: Rejected — unnecessary complexity for a week-to-week continuation

### RQ-2: What data shape changes occur from Week 7 to Week 8?

**Decision**: Each function gains exactly 1 sample (the Week 7 submission result). All dimensions remain unchanged.

| Function | Week 7 Samples | Week 8 Samples | Dimensions |
|----------|---------------|----------------|------------|
| F2 | 17 | 18 | 2 |
| F3 | 22 | 23 | 3 |
| F4 | 37 | 38 | 4 |
| F5 | 27 | 28 | 4 |
| F6 | 27 | 28 | 5 |
| F7 | 37 | 38 | 6 |
| F8 | 47 | 48 | 8 |

**Rationale**: Each week's submission adds one observation per function. This was confirmed by loading the actual `.npy` files.

**Alternatives considered**: None — this is factual data.

### RQ-3: What are the key spec-vs-code discrepancies to resolve?

**Decision**: Use code values in all cases. Three corrections were applied to spec.md during the clarify session. All discrepancies now resolved.

| Item | Spec Says | Code Does | Resolution | Status |
|------|-----------|-----------|------------|--------|
| F7 DROPOUT | 0.2 → **0.1** | 0.1 | Spec corrected | ✅ |
| F7 EPOCHS | 500 → **200** | 200 | Spec corrected | ✅ |
| F7 STEEPNESS | 1.0 (spec-017) | 0.1 | Noted in spec-020 | ✅ |
| F2 acquisition | NEI → **qLogNEI** | qLogNEI | Spec corrected | ✅ |
| F3 acquisition | NEI → **qLogNEI** | qLogNEI | Spec corrected (clarify session 2) | ✅ |
| F3 noise floor | not specified → **≥ 1e-6** | GreaterThan(1e-6) | Spec corrected (clarify session 2) | ✅ |
| F3/F5 restarts | 10–20 → **15** | 15 | Spec corrected (clarify session 2) | ✅ |
| F2 multi-restart | Implied | Single fit | Code is ground truth | ✅ |
| F8 multi-restart | Implied | Single fit | Code is ground truth | ✅ |

**Rationale**: "Keep the same strategy as week 7" means replicating the actual Week 7 code, not the prior spec descriptions.

**Alternatives considered**: Correcting to spec values — rejected as it would change the strategy.

### RQ-4: What are the output characteristics that affect surrogate behaviour?

**Decision**: Document the output ranges to anticipate surrogate-specific handling:

| Function | Output Range | All Positive? | All Negative? | Special Handling |
|----------|-------------|---------------|---------------|-----------------|
| F2 | [-0.066, 0.674] | No | No | None |
| F3 | [-0.399, -0.031] | No | **Yes** | Maximisation of negative values |
| F4 | [-32.63, 0.532] | No | No | Large dynamic range |
| F5 | [0.113, 3394.68] | **Yes** | No | log1p transform essential |
| F6 | [-2.571, -0.206] | No | **Yes** | Rank-based scoring for penalty |
| F7 | [0.003, 2.305] | **Yes** | No | z-score normalisation |
| F8 | [5.592, 9.982] | **Yes** | No | None special |

**Rationale**: Functions with all-negative outputs (F3, F6) require careful handling — standard EI works but rank-based penalty selection is needed for F6. Functions with large dynamic range (F5) require log transforms.

**Alternatives considered**: None — data characteristics are fixed.

### RQ-5: What is the recommended implementation order?

**Decision**: Implement in order of increasing complexity:

1. **F2** (Group A) — Simplest: 2D SFGP, single fit, no penalty, qLogNEI
2. **F8** (Group A) — 8D SFGP, single fit, qEI with fallback
3. **F3** (Group A) — 3D SFGP, 15-restart, manual z-score, pairwise slices
4. **F4** (Group C) — 4D MFGP, fidelity column, MF-qNEI q=4
5. **F5** (Group B) — 4D GP, log1p + z-score, interior penalty post-hoc
6. **F6** (Group B) — 5D SFGP, feasibility bounds, rank-based penalty
7. **F7** (Group C) — 6D NN, MC dropout, no BoTorch

**Rationale**: Start simple to validate the template pattern, then tackle progressively more complex surrogates and acquisition functions. F7 is last because it uses an entirely different stack (PyTorch NN instead of BoTorch GP).

**Alternatives considered**:
- Alphabetical order: Rejected — doesn't account for complexity dependencies
- All in parallel: Rejected — easier to debug when building incrementally
