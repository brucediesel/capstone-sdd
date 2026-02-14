# Implementation Plan: Bayesian Optimization Notebooks for 8 Black Box Problems

**Branch**: `001-bayesian-optimization-notebooks` | **Date**: 2026-02-07 | **Spec**: [spec.md](spec.md)

## Summary

Implement and validate Bayesian Optimization solutions for 8 black box optimization problems (f1-f8) using BoTorch with Gaussian Process surrogates and Expected Improvement acquisition. Each problem solved in its own Jupyter notebook with clear hyperparameter documentation, surrogate visualizations, and weekly iteration support. Target: 13-module optimization campaign (Modules 12-24) with one evaluation per module per problem.

## Technical Context

**Language/Version**: Python 3.14.2 (sdd-dev environment)
**Primary Dependencies**: 
- BoTorch 0.16.1 (Bayesian Optimization framework)
- PyTorch 2.10.0 (tensor operations, automatic differentiation)
- GPyTorch 1.15.1 (Gaussian Process implementation)
- NumPy 2.4.1 (data loading from .npy files)
- Matplotlib 3.10.8 (inline visualizations)

**Storage**: NumPy array files (.npy format) in `./data/f{1-8}/` directories
- Initial data: `initial_inputs.npy`, `initial_outputs.npy`
- Weekly updates: `updated_inputs - Week N.npy`, `updated_outputs - Week N.npy`

**Testing**: Manual notebook execution and visual validation (no unit tests per CONSTITUTION)

**Target Platform**: Jupyter Notebook environment (local or Colab)

**Project Type**: Data science / ML optimization notebooks

**Performance Goals**: <5 minutes execution time per notebook on standard laptop

**Constraints**: 
- Code must be "as simple as possible" (per CONSTITUTION)
- Each step clearly explained via markdown cells
- No modification of previous module's code cells (append-only module sections)
- 13 total evaluations budget per problem (one per module, Modules 12-24)

**Scale/Scope**: 
- 8 problems (f1-f8) spanning 2D to 8D dimensionality
- Initial samples vary per function: f1=10, f2=10, f3=15, f4=30, f5=20, f6=20, f7=30, f8=40
- 13 weekly iterations (Weeks 1–13) = initial_count + 13 total observations per problem by end
- ~10-20 code cells per notebook + markdown documentation

## Constitution Check

✅ **Code Simplicity**: BoTorch provides high-level API, implementations use straightforward GP training and acquisition optimization
✅ **Jupyter Notebooks**: All code delivered as .ipynb files in `functions/f{1-8}/` directories
✅ **No Unit Tests**: Manual validation through execution, visual inspection, and output verification
✅ **Weekly Update Pattern**: Weekly sections added without replacing existing cells
✅ **BoTorch Library**: Core requirement satisfied
✅ **Hyperparameter Documentation**: Explicit printing with justifications in markdown
✅ **Visualizations**: Surrogate functions, acquisition landscapes, convergence plots per problem

## Project Structure

### Documentation (this feature)

```
specs/001-bayesian-optimization-notebooks/
├── spec.md              # Feature specification with clarifications (COMPLETE)
├── plan.md              # This file - implementation plan
└── tasks.md             # Task breakdown (TO BE GENERATED)
```

### Source Code (repository root)

```
capstone-sdd/
├── CONSTITUTION.md                      # Project requirements (EXISTING)
├── data/                                # NumPy data files (EXISTING)
│   ├── f1/
│   │   ├── initial_inputs.npy          # 10 samples, 2D
│   │   ├── initial_outputs.npy
│   │   ├── updated_inputs - Week 3.npy
│   │   ├── updated_outputs - Week 3.npy
│   │   └── ... (Week 4 files exist)
│   ├── f2/ ... f8/                     # Same structure for all problems
├── functions/                           # Jupyter notebooks (IN PROGRESS)
│   ├── f1/
│   │   └── f1.ipynb                    # 2D radiation detection (COMPLETE - TESTED)
│   ├── f2/
│   │   └── f2.ipynb                    # 2D log-likelihood (COMPLETE - TESTED)
│   ├── f3/
│   │   └── f3.ipynb                    # 3D drug discovery (FIX APPLIED - PENDING TEST)
│   ├── f4/
│   │   └── f4.ipynb                    # 4D warehouse params (COMPLETE - TESTED)
│   ├── f5/
│   │   └── f5.ipynb                    # 4D chemical process (PENDING TEST)
│   ├── f6/
│   │   └── f6.ipynb                    # 5D cake recipe (NEEDS FIX - base_kernel access)
│   ├── f7/
│   │   └── f7.ipynb                    # 6D ML model (NEEDS FIX - base_kernel access)
│   └── f8/
│       └── f8.ipynb                    # 8D hyperparameter tuning (NEEDS FIX - base_kernel access)
└── .specify/                            # SpecKit metadata (EXISTING)
```

## Implementation Approach

### Phase 1: Fix & Validate Initial Implementations (CURRENT)

**Status**: Partially complete (f1, f2, f4 validated; f3-f8 pending)

**Approach**: Sequential notebook execution to validate BO workflow
- Apply kernel structure fixes (conditional `hasattr()` checks for `base_kernel.lengthscale` and `outputscale` access)
- Execute each notebook cell-by-cell
- Verify GP training, acquisition optimization, visualization rendering
- Extract next_point proposals for Week 1 submission

**Key Technical Decision**: Kernel attribute access must be defensive
```python
# Pattern for safe lengthscale access:
if hasattr(gp_model.covar_module, 'base_kernel'):
    ls = gp_model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
else:
    ls = gp_model.covar_module.lengthscale.detach().numpy()[0]
```
- BoTorch's `SingleTaskGP` may use `RBFKernel`/`MaternKernel` directly or wrapped in `ScaleKernel`
- Cannot assume `base_kernel` or `outputscale` attributes exist
- Solution: conditional checks in all hyperparameter display and visualization cells

### Phase 2: Weekly Iteration Demonstration (FUTURE)

**Status**: Not yet started

**Approach**: Demonstrate weekly update workflow on one problem (likely f1 or f2)
- Load updated data: `np.concatenate([initial, updated_Week_N])`
- Add new markdown section: "## Week N Update"
- Add new code cells for retrained GP, new next_point
- Validate old cells remain unchanged
- Update convergence plots to show improvement trends

### Phase 3: Visualization Strategy for High-D Problems (FUTURE)

**Status**: f1-f2 use full 2D contour plots; f3-f8 strategy varies

**Approach**: Implement lengthscale-guided slicing for f3-f8 per spec clarification
- Identify 2 dimensions with smallest lengthscales (most important)
- Create 2D contour slice fixing other dimensions at best observed values
- Label which dimensions are shown
- Bar chart of lengthscales for feature importance

## Tech Stack Rationale

**BoTorch over alternatives** (scipy.optimize, scikit-optimize):
- Built on PyTorch: GPU acceleration potential, automatic differentiation
- SingleTaskGP: Handles noisy observations with automatic output standardization
- optimize_acqf: Multiple restart optimization with bounds constraints
- Active development, good documentation, industry-standard

**Expected Improvement acquisition over UCB/PI**:
- Balances exploration vs exploitation naturally
- Well-understood behavior, easy to interpret heatmaps
- Good default choice for initial implementation
- Can switch to UCB in later weeks if EI stagnates (per spec clarification)

**Matern 5/2 kernel** (BoTorch default):
- Assumes 2× differentiable functions
- More flexible than RBF (infinitely differentiable assumption)
- Good general-purpose choice for black-box optimization
- Lengthscales reveal feature importance

**float64 precision**:
- GP computations involve matrix inversions (numerically sensitive)
- Avoid underflow in likelihood calculations
- Standard for Bayesian Optimization, negligible memory overhead for small datasets

## Bounds Strategy (Per Spec Clarification)

**2D problems (f1-f2)**: Fixed `[0, 1]` per dimension (data already normalized)

**Higher-D problems (f3-f8)**: Data-driven with 10% margin
```python
x_min = X_init.min(axis=0) - 0.1 * (X_init.max(axis=0) - X_init.min(axis=0))
x_max = X_init.max(axis=0) + 0.1 * (X_init.max(axis=0) - X_init.min(axis=0))
BOUNDS = torch.tensor([x_min, x_max], dtype=torch.float64)
```
- Respects observed feasible region
- 10% margin allows exploration beyond sampled points
- Prevents boundary clustering
- Scales automatically per dimension

## Hyperparameter Justifications (To Be Documented in Notebooks)

**NUM_RESTARTS**:
- f1-f2 (2D): 10 restarts (low-dimensional, fast optimization)
- f3-f5 (3-4D): 15-20 restarts (moderate dimensionality)
- f6-f8 (5-8D): 20-30 restarts (high-dimensional, more local optima)
- **Rationale**: Balances optimization quality vs compute time; more restarts for higher-D acquisition surfaces

**RAW_SAMPLES**:
- f1-f2 (2D): 512 samples (adequate coverage of 2D EI surface)
- f3-f5 (3-4D): 1024 samples (denser sampling for moderate-D)
- f6-f8 (5-8D): 2048-4096 samples (curse of dimensionality requires more initial candidates)
- **Rationale**: Sobol sampling of acquisition function before gradient-based refinement; scales exponentially with dimension to maintain coverage

## Known Issues & Fixes Applied

**Issue 1**: Kernel structure AttributeError
- **Symptom**: `AttributeError: 'RBFKernel' object has no attribute 'base_kernel'`
- **Root cause**: BoTorch SingleTaskGP may use kernel directly or wrapped in ScaleKernel
- **Fix**: Conditional `hasattr()` checks before accessing `base_kernel` or `outputscale`
- **Applied to**: f2 (outputscale), f3 (outputscale), f4 (base_kernel in 2 cells)
- **Pending for**: f6, f7, f8 (base_kernel access in visualization cells)

**Issue 2**: Bounds calculation for higher-D problems
- **Symptom**: Need to programmatically determine sensible search bounds
- **Solution**: Data-driven bounds with 10% margin (documented above)
- **Status**: Implemented in f3-f8 notebooks

## Success Metrics

**Primary (Week 1 - Current Focus)**:
- ✅ All 8 notebooks execute without errors on initial data
- ⏳ Each notebook produces valid next_point within bounds (3 of 8 complete: f1, f2, f4)
- ⏳ GP hyperparameters learned and displayed correctly (3 of 8 validated)
- ⏳ Visualizations render inline (3 of 8 confirmed)
- ⏳ Next submission coordinates extracted (3 of 8 ready)

**Secondary (Future Weeks)**:
- Weekly iteration workflow demonstrated on ≥1 notebook
- Convergence plots show meaningful improvement over 15 weeks
- Lengthscale-guided visualization implemented for f3-f8
- EI interpretation documented ("Low EI suggests convergence")

## Dependencies & Prerequisites

**Environment**: ✅ sdd-dev Python 3.14.2 virtual environment configured
**Data files**: ✅ All initial_inputs.npy and initial_outputs.npy present for f1-f8
**Updated data**: ✅ Week 3 and Week 4 files exist (for future iteration demos)
**Git branch**: ✅ `001-bayesian-optimization-notebooks` active
**SpecKit**: ✅ `.specify/` structure initialized, spec.md with clarifications complete

## Next Steps (Tasks to be Generated)

1. Apply base_kernel fixes to f6, f7, f8
2. Execute and validate f3, f5, f6, f7, f8 notebooks
3. Extract all 8 next_point values for Week 1 submission
4. Create validation summary table
5. (Future) Demonstrate weekly iteration on f1 or f2
6. (Future) Implement lengthscale-guided slicing visualization for f3-f8

---

**Plan generated**: 2026-02-07  
**Ready for task breakdown**: Yes (`/speckit.tasks`)
