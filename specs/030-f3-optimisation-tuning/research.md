# Research: F3 Week 10 — Optimisation Tuning

**Feature**: 030-f3-optimisation-tuning  
**Date**: 2026-03-12

## Research Questions

### R1: Shift transform (y - y_min) — behaviour with GP fitting

**Decision**: Use manual shift transform `Y_shifted = Y - Y.min()` applied once to the tensor before GP construction. Do NOT use Standardize(m=1) simultaneously.

- F3 outputs are all negative: approximately [-0.399, -0.031]
- After shift: Y_shifted ∈ [0, 0.368] — non-negative, suitable for GP fitting
- `y_min` must be stored to reverse-transform predictions: `y_original = y_shifted + y_min`
- No existing function in the project uses this pattern — F3 is the first
- F6 also has all-negative outputs but uses Standardize(m=1) + rank-based interior penalty instead
- The shift is simpler than Standardize(m=1) and preserves absolute differences between outputs
- GP posterior predictions will be in shifted space; reverse-shift needed for convergence plots and display

**Alternatives considered**:
- Keep Standardize(m=1) — rejected because it centres on mean, and for all-negative outputs the zero-crossing creates an artificial boundary in GP space
- log1p or log — impossible since outputs are negative
- Shift + Standardize together — rejected for simplicity; shift alone maps to a workable range [0, 0.368]

**Rationale**: The shift transform is the simplest approach (constitution Principle I) that addresses the negative-output problem. The GP operates on non-negative targets, which improves numerical stability and avoids sign-related issues in the posterior.

### R2: Noise floor 1e-4 — impact on GP fitting with shifted outputs

**Decision**: Use NOISE_LB=1e-4, matching F1 and F2 patterns.

- Week 9 used noise_lb=1e-6, which is extremely tight for 3D data with only 25 samples
- After shift transform, outputs are in [0, 0.368] — similar scale to F2 outputs [0.25, 0.67]
- F2 uses NOISE_LB=1e-4 with Standardize(m=1) on similar output ranges and works well
- F6 (also all-negative) uses noise_lb=1e-2 — higher floor due to recipe variability
- For F3's shifted outputs (scale ~0.37), 1e-4 represents ~0.03% of the range — sufficient to prevent overfitting while preserving signal
- The noise constraint operates directly on the shifted outputs (no Standardize to rescale)

**Alternatives considered**:
- Keep 1e-6 — rejected; causes overfitting on 25 samples in 3D
- Use 1e-3 (F2 week 8 value) — acceptable but 1e-4 gives MLL more freedom to find the right noise level
- Use 1e-2 (F6 value) — too aggressive for F3's well-behaved output surface

**Rationale**: 1e-4 is the standard floor used across F1, F2, and F4. It prevents numerical singularity while allowing the GP to model smooth trends in the shifted output space.

### R3: q=3 candidate selection in 3D space

**Decision**: Reuse the proven two-stage distance-based selection pattern from F1/F2.

**Pattern** (established in F1/F2 week 10):
1. Get posterior means for all q=3 candidates (in shifted space)
2. Quality gate: keep candidates with mean ≥ median of the batch
3. Exploration bonus: from qualified set, select the one with maximum minimum-distance to all training points
4. Uses `torch.cdist` for efficient Euclidean distance in 3D

- With q=3 instead of q=4 (F1/F2), the median filter is still meaningful: keeps top 2 of 3 candidates (or all if only 2 pass median)
- 3D distance computation with 25 training points is trivial (25×3 distance matrix per candidate)
- This pattern balances quality (promising predictions) with spatial diversity (exploration)

**Alternatives considered**:
- Pure exploitation (pick highest shifted mean) — rejected; F3 has 25 samples in 3D and needs spatial diversity
- Pick all 3 as submissions — rejected; challenge requires single submission per round
- Random selection from candidates — rejected; discards model predictions

**Rationale**: Two-stage filter is proven across F1 and F2 implementations. Adapts naturally to q=3 in 3D.

### R4: 2D contour slice visualisation for 3D inputs

**Decision**: Follow the established F3 week 9 pattern exactly — 3 panels for 3 input pairs.

**Pattern** (from F3 week 9):
- Pairs: (0,1), (0,2), (1,2) — 3 unique pairs for 3 dimensions
- For each pair (d1, d2): fix remaining dimension d3 at the proposed point's coordinate
- Grid: 50×50 over [0,1]² for each pair
- Render: contourf for posterior mean (viridis), contour lines for uncertainty (white), overlaid data points
- Layout: 2 rows × 3 columns (row 1: posterior mean, row 2: acquisition surface) or 1×3 per surface

Key adaptations for shift transform:
- Posterior mean values are in shifted space [0, 0.368] — display in shifted space for consistency
- Acquisition surface computed from shifted-space model — no reverse-transform needed for acqf values
- Point overlays use original input coordinates (shift only affects outputs)

**Alternatives considered**: 3D scatter plots — rejected for poor readability and constitution simplicity; Plotly interactive — rejected per constitution.

**Rationale**: Matches existing F3 visualisation convention. Students and evaluators expect consistent presentation across weeks.

### R5: 40 MLL restarts — performance and implementation

**Decision**: Use 40 restarts with independent random seeds, keeping the model with lowest negative MLL.

- Week 9 used 15–20 restarts; F2 week 10 uses 50 restarts
- 3D Matérn-2.5 ARD with shift transform creates a 5-parameter likelihood surface (3 lengthscales + noise + outputscale) — more complex than F1/F2's 4 parameters
- Each restart: construct model, set random hyperparameters, fit via L-BFGS-B, record loss
- Expected time: ~2–3s per restart × 40 = ~80–120s — within the 120s budget
- Implementation uses `torch.manual_seed(seed)` for reproducibility
- Copy best model via `copy.deepcopy(model)` when loss improves

**Alternatives considered**:
- 20 restarts (week 9 value) — rejected per spec; multi-modal likelihood in 3D needs more coverage
- 50 restarts (F2 value) — acceptable but 40 is sufficient for 3D and keeps time manageable
- Single fit + hyperparameter priors — rejected; unreliable for multi-modal surfaces

**Rationale**: 40 is a good balance between reliability and computation time for 5 hyperparameters in 3D.

### R6: Existing F3 week 10 notebook variables in scope

**Decision**: New cells reuse variables from existing cells 1–12. No re-loading of data needed.

Variables available after running existing cells:
- `inputs` (ndarray, shape [25, 3]) — all input data
- `outputs` (ndarray, shape [25,]) — all output values, range [-0.399, -0.031]
- `N_INITIAL` = 15
- `FUNC_NUM` = 3
- `DATA_DIR` = '../../data/f3/'
- `USE_LOG_SCALE` = False
- `WEEK` = 10
- `N_DIMS` = 3
- `n_total` = 25
- `n_submissions` = 10
- `running_best` (ndarray from `np.maximum.accumulate(outputs)`)
- `stalling` = True
- `improvements` = 3

New cells will import BoTorch/GPyTorch, convert data to tensors, apply shift transform, and proceed with the GP pipeline.

### R7: 2048 raw samples — coverage adequacy for 3D

**Decision**: 2048 raw samples is adequate for [0,1]³ acquisition search.

- Week 9 used 512 raw samples — equivalent to ~8 points per dimension (8³ = 512)
- 2048 ≈ 12.7 points per dimension (12.7³ ≈ 2048) — 60% improvement in per-dimension coverage
- F2 (2D) uses 4096; F5 (4D) uses 5000; F1 (2D) uses 10000
- Per-dimension density: F1 = 100, F2 = 64, F5 ≈ 8.4, F3 proposed ≈ 12.7
- For 3D, 2048 Sobol points provide good quasi-random coverage of the unit cube
- The L-BFGS restarts (20) then refine from the best Sobol seeds

**Alternatives considered**:
- 512 (week 9) — rejected per spec; too sparse for 3D
- 4096 — acceptable but doubles computation time for modest gain
- 10000 — overkill for 3D; appropriate for 2D where per-dim density matters more

**Rationale**: 2048 matches the user's explicit request and provides materially better 3D coverage than 512.
