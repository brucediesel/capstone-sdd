# Research: F5 Week 9 — Remove Interior Penalty

**Feature**: 025-f5-remove-penalty  
**Date**: 2026-03-09

## Research Summary

This is a narrowly scoped deletion feature with no technology choices or unknowns to resolve. All research items below confirm decisions already made in the specification.

## R1: Interior Penalty Removal Impact

**Decision**: Remove all interior penalty code from F5 week 9 notebook  
**Rationale**: The interior penalty was added to suppress boundary-hugging candidates (spec 015-f5-interior-penalty). After evaluation across multiple weeks, the penalty has shown mixed results — in F3 it was found to "swamp acq results" and was removed (spec 024). The user has now requested the same removal for F5. The base NEI with distance-based selection already produces valid candidates.  
**Alternatives considered**:
- Reduce STEEPNESS (e.g., 1.0 → 0.3) — rejected because user explicitly requested full removal
- Keep penalty but reduce FLOOR — rejected for same reason

## R2: Base NEI Candidate Selection Behaviour

**Decision**: Use existing Step 3 distance-based selection logic unchanged  
**Rationale**: Step 3 already implements a complete acquisition pipeline: qLogNEI with q=4, 50 restarts, 3000 raw samples, followed by distance-based selection (above-median posterior mean, farthest from training data). This produces valid candidates without the penalty wrapper. The penalty was added as an enhancement on top of this working pipeline, so removing it simply reverts to the base behaviour.  
**Alternatives considered**: None — the user explicitly said "don't make any other changes"

## R3: Cells to Remove vs Edit

**Decision**: Remove 4 cells entirely (Step 4 markdown, Step 4 code, Step 6 markdown, Step 6 code); edit 6 cells (title, hyperparameters table, constants code, Step 5 viz, Step 8 submission, strategy); leave all other cells unchanged  
**Rationale**: 
- Step 4 (penalty explanation + penalty code): These cells define and execute the `PenalisedAcquisition` wrapper. Without the penalty, they serve no purpose and would error on missing variables.
- Step 6 (penalty visualisation): Renders penalty weight surface and penalised mean — meaningless without penalty code.
- Title cell: References "Interior Penalty" — must be updated for accuracy.
- Hyperparameters table cell: Contains IP STEEPNESS and IP FLOOR rows — must be removed.
- Constants code cell: Contains `STEEPNESS`, `FLOOR`, `EPS_BOUND` — must be removed.
- Step 5 viz: References `next_x_ip` (penalty-selected point) — must change to `best_point` (base NEI selected).
- Step 8 submission: Shows both base and IP submissions — must show only base.
- Strategy cell: Contains penalty recommendations — must note penalty was removed.
- Steps 1-3, 7, 9-13: No penalty references, remain unchanged.  
**Alternatives considered**: None required — this is a mechanical mapping from spec requirements to notebook cells

## R4: Notebook Structure After Removal

**Decision**: Notebook will have ~23 cells (down from ~27) after removing 4 penalty cells  
**Rationale**: Current structure:
1. Title (md) — EDIT
2. Imports (code) — unchanged
3. Hyperparameters (md) — EDIT
4. Constants (code) — EDIT (remove STEEPNESS, FLOOR, EPS_BOUND)
5. Step 1 header (md) — unchanged
6. Step 1 data (code) — unchanged
7. Step 2 header (md) — unchanged
8. Step 2 GP training (code) — unchanged
9. Step 3 header (md) — unchanged
10. Step 3 base NEI (code) — unchanged
11. Step 4 explanation (md) — REMOVE
12. Step 4 penalty code (code) — REMOVE
13. Step 5 header (md) — unchanged
14. Step 5 viz (code) — EDIT (next_x_ip → best_point, remove "IP" from title)
15. Step 6 header (md) — REMOVE
16. Step 6 penalty viz (code) — REMOVE
17. Step 7 header (md) — unchanged
18. Step 7 convergence (code) — unchanged
19. Step 8 header (md) — unchanged
20. Step 8 submission (code) — EDIT (remove IP submission, penalty params)
21. Performance header (md) — unchanged
22. Convergence metrics (code) — unchanged
23. Exploration spread header (md) — unchanged
24. Exploration spread (code) — unchanged
25. LOO header (md) — unchanged
26. LOO (code) — unchanged
27. Strategy (md) — EDIT (remove penalty recommendations)
