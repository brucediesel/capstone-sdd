# Research: F1, F2 & F5 Interior Penalty

**Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)

## R1: Interior Penalty Formula — Established Pattern

**Decision**: Use the sin-based interior penalty formula already established in the project.

**Formula**: `w(x) = FLOOR + (1 - FLOOR) · ∏ᵢ sin(πxᵢ)^(2·STEEPNESS)`

**Rationale**: This formula is used in F6 (STEEPNESS=1.0, FLOOR=0.01) and F7 (STEEPNESS=0.02, FLOOR=0.02) week 10 notebooks, and was documented in prior specs 014/015/016. Reusing it ensures consistency across the project and requires no new validation.

**Alternatives considered**:
- Polynomial boundary penalty (`∏ xᵢ(1-xᵢ)`) — Simpler but lacks the steepness parameter for tuning.
- Log-barrier penalty — More aggressive near boundaries; risk of numerical issues.
- Hard clipping (e.g., reject candidates within ε of boundary) — Too coarse; not differentiable.

## R2: Hyperparameter Values — STEEPNESS=0.02, FLOOR=0.01

**Decision**: STEEPNESS=0.02, FLOOR=0.01 for all three functions.

**Rationale**:
- STEEPNESS=0.02 produces exponent 2·0.02=0.04, so sin(πx)^0.04 ≈ 1.0 for all x except within ~0.01 of the boundary. This is the "very shallow" value used in F7 week 10 and confirmed during clarification.
- FLOOR=0.01 matches F6's value and provides maximum dynamic range while preventing zero-valued penalty weights.
- Uniform values across F1, F2, F5 — no per-function customisation needed since the penalty is intentionally near-no-op in the interior.

**Alternatives considered**:
- STEEPNESS=0.05 — Upper bound of "very shallow" range; slightly more aggressive; rejected as user wants "very shallow".
- FLOOR=0.02 — Matches F7; rejected because lower FLOOR gives more dynamic range at the boundary.
- Per-function tuning — Unnecessary complexity; all three functions share the same [0,1]^d domain.

## R3: Integration Point — Post-Distance-Filter Re-Ranking

**Decision**: Apply interior penalty after the existing distance-based selection filter, then select by highest penalised acquisition value among survivors.

**Rationale**: Confirmed during clarification (Q1). The distance filter serves a different purpose (preventing near-duplicate evaluations) and must be preserved. The penalty re-ranks survivors to prefer interior candidates.

**Pipeline order**:
1. `optimize_acqf` returns q=4 candidates
2. Distance-based filter discards near-duplicate candidates (median gate for F1, 25th percentile gate for F2/F5)
3. Interior penalty computes `w(x)` for each survivor
4. Each survivor's acquisition value is multiplied by its penalty weight
5. Survivor with highest penalised acquisition value is selected

**Alternatives considered**:
- Apply penalty before distance filter — Risk of selecting a penalty-boosted candidate that is too close to existing observations.
- Replace distance filter entirely — Loses the duplicate-prevention mechanism.
- Embed penalty in BoTorch acquisition function — Violates FR-004 (must not modify the GP or acquisition construction); much more complex for no benefit.

## R4: F1/F2 vs F5 Selection Pattern Differences

**Decision**: Adapt the penalty integration to each function's existing selection pattern.

**Findings**:
- **F1**: Uses torch-based distance selection with median gate on posterior means. Variable names: `candidates`, `quality_mask`, `qualified_idx`, `qualified_candidates`, `min_dists`, `selected_idx`, `x_new`.
- **F2**: Identical pattern to F1 (same variable names, same median gate approach with 25th percentile).
- **F5**: Uses numpy-based distance selection with 25th percentile gate and `np.exp()` inverse log transform. Variable names: `candidates`, `qualified_mask`, `qualified_indices`, `distances`, `min_distances`, `selected_idx`, `x_new`.

**Integration approach**: After each function's existing distance selection picks the "best" survivor, add a new cell that:
1. Computes `interior_weight` for all survivors
2. Retrieves the acquisition value for each survivor
3. Computes `penalised_acq = acq_value * interior_weight`
4. Selects `argmax(penalised_acq)` instead of the distance-selected one
5. Prints comparison (original selection vs penalised selection)

**Note**: The existing notebooks use `optimize_acqf` with q=4 and return a single `acq_value` scalar (not per-candidate values). The acquisition values for individual candidates need to be evaluated separately by calling `acqf(candidate.unsqueeze(0))` for each candidate.

## R5: Acquisition Value Per-Candidate Evaluation

**Decision**: Evaluate the acquisition function on each individual candidate to get per-candidate acquisition values for re-scoring.

**Rationale**: `optimize_acqf` returns a single joint acquisition value for the batch, not per-candidate values. To apply the multiplicative penalty, we need individual acquisition values. This can be done by calling `acqf(candidates[i:i+1].unsqueeze(0))` for each candidate `i`.

**Implementation pattern** (from F6):
```python
# Evaluate acquisition on each candidate individually
with torch.no_grad():
    acq_values = torch.tensor([
        acqf(candidates[i:i+1].unsqueeze(0)).item()
        for i in range(len(candidates))
    ])
```

**Alternative considered**:
- Use posterior mean as proxy for acquisition value — Simpler but doesn't capture the exploration component of qLogNEI. Rejected because the penalty should re-score actual acquisition values.
