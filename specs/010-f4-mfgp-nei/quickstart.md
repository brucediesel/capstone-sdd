# Quickstart: F4 Week 7 — MFGP + Cost-Aware MF-qNEI

**Date**: 2026-02-23 | **Branch**: `010-f4-mfgp-nei`

## Prerequisites

- Python 3.14 via pyenv (`sdd-dev` environment)
- BoTorch 0.16.1 / GPyTorch 1.15.1 / PyTorch installed
- Data files present:
  - `data/f4/updated_inputs - Week 7.npy` (37×4)
  - `data/f4/updated_outputs - Week 7.npy` (37,)
- VS Code with Jupyter extension or equivalent notebook runner

## Steps

1. **Checkout branch**: `git checkout 010-f4-mfgp-nei`
2. **Activate environment**: `pyenv shell sdd-dev`
3. **Open notebook**: `functions/f4/f4.ipynb`
4. **Scroll to Week 7 section**: After the existing Week 6 section and Research markdown (cell 52)
5. **Run Cell 54** (imports + data loading): Verify 37 samples, inputs in [0, 1]
6. **Read Cell 55** (hyperparameter docs): Review the 10+ parameter table
7. **Run Cell 56** (MFGP training): Wait for 15 restarts; confirm noise ≥ 1e-4
8. **Run Cell 57** (MF-qNEI acquisition): Verify 4 candidates with 4 coords each
9. **Run Cell 58** (surrogate plot): Confirm 2-panel contour renders
10. **Run Cell 59** (convergence plot): Confirm running-best line with boundary at 30.5
11. **Run Cell 60** (submission query): Copy the `SUBMISSION QUERY` output

## Verification Checklist

- [ ] 37 samples loaded with all inputs in [0, 1]
- [ ] MFGP trains with noise variance ≥ 1e-4
- [ ] All fitted HPs printed (ℓ₁–ℓ₄, σ²_f, σ²_n, fidelity power)
- [ ] 4 candidates returned from MF-qNEI
- [ ] All 4 candidates have coordinates in [0, 0.999999]
- [ ] Best candidate selected by posterior mean
- [ ] Surrogate plot renders (2 panels: mean + std)
- [ ] Convergence plot renders with boundary at 30.5
- [ ] Submission string matches format `0.XXXXXX-0.YYYYYY-0.ZZZZZZ-0.WWWWWW`
- [ ] No existing cells modified (52 original cells unchanged)

## Common Issues

| Issue | Resolution |
|-------|------------|
| `SingleTaskMultiFidelityGP` import error | Verify BoTorch ≥ 0.12.0: `pip show botorch` |
| NaN gradients during MLL | Training loop skips failed restarts; check at least 3/15 succeed |
| `optimize_acqf` slow with q=4 | Expected: 1-3 minutes. Reduce `raw_samples` to 256 if needed |
| Fidelity column mismatch | Ensure `data_fidelities=[4]` matches the 5th column (0-indexed) |
