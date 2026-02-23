# Research: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Date**: 2026-02-23 | **Branch**: `010-f4-mfgp-nei`

## RES-001: MFGP Configuration Selection

**Decision**: Use `SingleTaskMultiFidelityGP` with `nu=2.5`, `linear_truncated=True`, z-score standardisation, and `noise_lb=1e-4`.

**Rationale**: This exact configuration was the **PE winner** for F4 across 45 tested configurations (15 SFGP + 15 MFGP + 15 GBT). It achieved the best NLP of -1.35 (vs SFGP -1.18 and GBT 2.55), indicating the best-calibrated uncertainty. The LinearTruncatedFidelityKernel provides beneficial regularisation even with single-fidelity data.

**Alternatives considered**:
- SFGP (Matérn-5/2 ARD): Second-best NLP (-1.18), but less well-calibrated uncertainty
- GBT ensemble: Lowest MAE (1.55) but terrible uncertainty calibration (NLP 2.55)
- MFGP with `linear_truncated=False` (ExponentialDecayFidelityKernel): Tested in PE sweep, inferior NLP

## RES-002: Fidelity Column Strategy

**Decision**: Append a constant fidelity column of 1.0 to all training inputs, creating a 5D augmented input tensor.

**Rationale**: All F4 observations are at a single (high) fidelity. The constant fidelity column is a synthetic construct that enables the `SingleTaskMultiFidelityGP` to use its `LinearTruncatedFidelityKernel`, which introduces a structured prior on how the spatial kernel interacts with the fidelity dimension. Even with single-fidelity data, this provides a different inductive bias/regularisation than a plain `SingleTaskGP`. This approach was validated in the F4 PE sweep (preq-eval-f4.ipynb).

**Alternatives considered**:
- Plain `SingleTaskGP` (no fidelity column): Simpler, but empirically worse NLP in PE
- `MultiTaskGP` with ICM: Used for F2/F3 but not appropriate for F4's problem structure

## RES-003: Acquisition Function — qLogNoisyExpectedImprovement with Fixed Fidelity

**Decision**: Use `qLogNoisyExpectedImprovement` from `botorch.acquisition.logei` with `q=4`, `fixed_features={4: 1.0}`, and a `SobolQMCNormalSampler(sample_shape=torch.Size([64]))`.

**Rationale**:
- `qLogNoisyExpectedImprovement` works directly with `SingleTaskMultiFidelityGP` — no special MF acquisition wrapper needed.
- The `fixed_features` parameter in `optimize_acqf` pins the fidelity column (index 4) to 1.0 during optimisation, so only spatial dimensions are searched.
- 64 MC samples (via sampler) matches the user's "fantasies=64" specification. (Note: "fantasies" is technically `num_fantasies` in qKG, but the equivalent concept in qNEI is MC samples for the posterior expectation.)
- `q=4` returns a joint batch of 4 candidates; the best individual candidate is selected for submission.

**Alternatives considered**:
- `qKnowledgeGradient` with `num_fantasies=64`: More principled for noise handling, but slower and more complex
- `optimize_acqf_mixed`: For enumerating over discrete fidelity values — unnecessary since we only have fidelity=1.0
- `qExpectedImprovement` (non-noisy): Doesn't account for observation noise

## RES-004: Candidate Selection from q=4 Batch

**Decision**: Evaluate each of the 4 candidates individually through the model posterior and select the one with the highest posterior mean at fidelity=1.0 as the primary submission.

**Rationale**: The q=4 batch is jointly optimised (candidates are correlated). For serial evaluation (submit one point per week), we need to pick the single best. Posterior mean ranking is appropriate for maximisation.

**Alternatives considered**:
- Individual acquisition value ranking: More principled but more complex; posterior mean is simpler and aligns with constitution's simplicity principle
- Always use q=1: Loses diversity information; q=4 explores more of the space before selecting
- Submit all 4 (if supported): Challenge only accepts one query per week

## RES-005: MLL Multi-Restart Training

**Decision**: Use 15 random restarts of `fit_gpytorch_mll` with manual seed loop, deepcopy of best model, select by lowest negative MLL.

**Rationale**: Consistent with the F3 Week 7 approach (proven pattern in this project). The MFGP has more hyperparameters than SFGP (4 ARD lengthscales + signal variance + noise variance + fidelity kernel power), so multiple restarts are critical for avoiding local optima in the MLL landscape. 15 restarts balances thoroughness vs compute time.

**Alternatives considered**:
- BoTorch's default single `fit_gpytorch_mll`: Risk of local optima with 7+ hyperparameters
- 20 restarts: Marginally better but slower; 15 has been validated in F3
- Bayesian hyperparameter optimisation: Overkill for this context

## RES-006: Surrogate Visualisation Strategy for 4D

**Decision**: Use 2D contour slices through the two dimensions with the shortest ARD lengthscales (most important features), fixing the other two dimensions at the proposed point's values. Two panels: posterior mean and posterior standard deviation.

**Rationale**: 4D space cannot be directly visualised. ARD lengthscales provide a natural importance ranking — shorter lengthscales mean the function varies more rapidly along that dimension, making it more important to visualise. This approach is consistent with the Week 6 GBT visualisation (which used feature importance for the same purpose).

**Alternatives considered**:
- All 6 pairwise 2D slices: Too many panels, harder to interpret
- PCA projection: Loses interpretability of individual dimensions
- 1D marginal plots: Less informative than 2D contours

## RES-007: Bounds Configuration

**Decision**: Spatial bounds [0, 0.999999]⁴ for input dimensions; fidelity bounds [1.0, 1.0] (locked). Total bounds shape (2, 5).

**Rationale**: Submission format requires values starting with "0." and 6 decimal places, so 0.999999 is the maximum valid value. Fidelity is locked at 1.0 via both bounds and `fixed_features` to ensure consistency.

**Alternatives considered**:
- Bounds [0, 1]⁴: Could produce values like 1.000000 which violate submission format
- Separate optimisation ignoring fidelity dim: Possible but `fixed_features` is cleaner
