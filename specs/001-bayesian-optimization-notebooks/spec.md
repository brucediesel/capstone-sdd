# Feature Specification: Bayesian Optimization Notebooks for 8 Black Box Problems

**Feature Branch**: `001-bayesian-optimization-notebooks`  
**Created**: 2026-02-07  
**Status**: Draft  
**Input**: AI/ML capstone project - 8 black box optimization problems solved using BoTorch

## Clarifications

### Session 2026-02-07

- Q: Higher-dimensional visualization strategy for problems f5-f8? → A: Slice top 2 dimensions identified by smallest lengthscales (most important per GP), fix other dimensions at best observed values
- Q: Bounds strategy for higher-dimensional problems (f3-f8)? → A: Data range with 10% margin [min(X) - 0.1*range, max(X) + 0.1*range] for each dimension
- Q: Convergence stagnation response when EI drops to near-zero? → A: Document EI value with interpretation (e.g., "Low EI suggests convergence"), defer alternative acquisition functions to future weeks if stagnation actually occurs
- Q: Evaluation budget constraint per problem? → A: 13 total evaluations (one point submitted per module from Module 12-24) beyond initial 10 samples (Module 12), resulting in 23 total observations by Module 24. Data grows: 10 points (Module 12), 11 points (Module 13), up to 22 points (Module 24)
- Q: Convergence success threshold - global maximum or just improvement? → A: Meaningful improvement required over initial best value. **Quantitative guideline: ≥5% improvement over initial best value OR EI > 1e-6 (indicating non-trivial exploration potential)** qualifies as successful optimization. Success ultimately judged holistically based on convergence trends, problem difficulty, and evaluation budget constraints.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Initial BO Implementation & Validation (Priority: P1)

As a data scientist working on the capstone project, I need to implement and validate Bayesian Optimization solutions for all 8 black box problems to generate initial sample point submissions using BoTorch with GP surrogates and Expected Improvement acquisition.

**Why this priority**: This is the foundational MVP - without working BO implementations, no submissions can be made and no iterative optimization can occur. Delivers immediate value by enabling Module 12 submissions for the 13-module optimization campaign (Modules 12-24).

**Independent Test**: Can be fully tested by executing each notebook (f1-f8) end-to-end with initial data, verifying GP training completes, acquisition optimization produces valid next points within bounds, and visualizations render correctly.

**Acceptance Scenarios**:

1. **Given** initial_inputs.npy and initial_outputs.npy for problem f1-f8, **When** notebook is executed cell-by-cell, **Then** GP model trains successfully, hyperparameters are learned and displayed, next sample point is proposed with coordinates and acquisition value, and no errors occur
2. **Given** trained GP model, **When** surrogate visualization cell executes, **Then** two plots are generated showing GP mean prediction and uncertainty (std dev) across the input space with observed points and next point marked
3. **Given** optimized acquisition function, **When** acquisition visualization cell executes, **Then** Expected Improvement heatmap is displayed showing why the next point was selected
4. **Given** initial data with best observed value, **When** progress tracking cell executes, **Then** convergence plot displays best value over iterations and prints next submission coordinates

### User Story 2 - Module-by-Module Iterative Updates (Priority: P2)

As a data scientist receiving new evaluation results each module, I need to add updated data to each problem, re-run optimization in new notebook sections, and generate the next submission without losing historical code.

**Why this priority**: Enables the core capstone workflow of iterative optimization across modules. Critical for project success but depends on P1 working first.

**Independent Test**: Can be tested by adding updated_inputs/outputs files to data/f1/, creating a new "Module N" section in the notebook, re-executing BO workflow with expanded dataset (growing from 10 to 22 points), and verifying new next_point differs from previous module.

**Acceptance Scenarios**:

1. **Given** existing notebook with Module 12 code, **When** Module 13 section is added below with updated data loading, **Then** old cells remain unchanged, new cells load combined initial+updated data, GP retrains on expanded dataset, and new next_point is proposed
2. **Given** updated data with more samples, **When** progress plot is regenerated in new module section, **Then** convergence curve shows improvement across iterations including new observations

### User Story 3 - Hyperparameter Documentation & Justification (Priority: P1)

As a project reviewer (instructor), I need to see explicit hyperparameter values and justifications for why they were chosen to assess understanding of Bayesian Optimization methodology.

**Why this priority**: Mandatory project requirement per CONSTITUTION. Must be included from Week 1 submission.

**Independent Test**: Read each notebook markdown and code cells to verify hyperparameters (NUM_RESTARTS, RAW_SAMPLES, kernel choice, acquisition function) are explicitly printed with text explanations of selection rationale.

**Acceptance Scenarios**:

1. **Given** notebook hyperparameters section, **When** cell executes, **Then** prints NUM_RESTARTS=10 (justification: balances optimization quality vs runtime), RAW_SAMPLES=512 (justification: sufficient for Expected Improvement surface exploration), bounds definition, and standardization approach
2. **Given** GP model training section, **When** hyperparameters display cell runs, **Then** learned noise variance, outputscale (if applicable), and lengthscales are printed with interpretation of what they reveal about the problem
3. **Given** acquisition optimization result, **When** next point is displayed, **Then** prints EI value with interpretation (e.g., "High EI indicates potential improvement" or "Low EI suggests convergence")

### User Story 4 - Problem-Specific Visualizations (Priority: P2)

As a data scientist analyzing optimization progress, I need visualizations tailored to each problem's dimensionality and context (e.g., 2D radiation detection vs 8D parameter tuning) to understand surrogate model behavior and guide strategy adjustments.

**Why this priority**: Required by CONSTITUTION for all problems. Enhances interpretability but secondary to working optimization.

**Independent Test**: Execute visualization cells for 2D problems (f1-f2) to verify contour plots with observed points, and for higher-D problems (f3-f8) to verify appropriate dimension-reduction or slice visualizations are generated.

**Acceptance Scenarios**:

1. **Given** 2D problem (f1, f2), **When** surrogate visualization runs, **Then** displays contour plots of GP mean and uncertainty over full [0,1]² space with observed points marked
2. **Given** higher-dimensional problem (f3-f8), **When** surrogate visualization runs, **Then** identifies the 2 dimensions with smallest GP lengthscales, creates 2D contour plot slicing along those dimensions with other dimensions fixed at best observed values, and labels which dimensions are shown

### Edge Cases

- **Sparse/noisy outputs**: What happens when all initial outputs are near-zero (e.g., f1 radiation detection with outputs in [-0.004, 0.0])? GP should still learn lengthscales and propose exploratory next points.
- **High dimensionality**: How does BO scale to 8D problems (f8)? May require more initial samples or different acquisition optimization strategies (NUM_RESTARTS, RAW_SAMPLES tuning).
- **Kernel structure variations**: How to handle BoTorch SingleTaskGP using RBFKernel without explicit ScaleKernel wrapper? Hyperparameter display code must check `hasattr(gp_model.covar_module, 'outputscale')` before accessing.
- **Boundary constraints**: What if acquisition optimization proposes points outside [0,1] bounds? BoTorch's `optimize_acqf` respects bounds parameter but should validate next_point before submission.
- **Convergence stagnation**: What if Expected Improvement drops to near-zero after several weeks? Document EI value with interpretation (e.g., "EI=1e-7 suggests strong convergence, exploration exhausted") in output, defer switching to alternative acquisition functions (UCB, PI) to future weekly sections if needed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load initial_inputs.npy and initial_outputs.npy from `../../data/f{N}/` directory for each problem f1-f8
- **FR-002**: System MUST train a BoTorch SingleTaskGP Gaussian Process model on loaded data with automatic output standardization
- **FR-003**: System MUST optimize Expected Improvement acquisition function using `NUM_RESTARTS=10-30` (scaled by problem dimensionality) and `RAW_SAMPLES=512-4096` (scaled by problem dimensionality) to propose next sample point. See plan.md Hyperparameter Justifications section for dimension-specific rationale.
- **FR-004**: System MUST validate next_point coordinates fall within problem bounds: [0,1] for 2D problems (f1-f2), [min(X)-0.1*range, max(X)+0.1*range] per dimension for higher-D problems (f3-f8) where range=max(X)-min(X)
- **FR-005**: System MUST display learned GP hyperparameters including noise variance and lengthscales with conditional check for outputscale
- **FR-006**: System MUST generate surrogate function visualizations showing GP mean prediction and uncertainty across input space
- **FR-007**: System MUST generate acquisition function visualization showing Expected Improvement heatmap with observed points and proposed next point marked
- **FR-008**: System MUST generate convergence plot tracking best observed value over iterations
- **FR-009**: System MUST print next submission point coordinates in format `[x1, x2, ..., xD]` for easy copy-paste
- **FR-010**: Each module iteration MUST add a new markdown section titled "Module N" with cells for loading updated data and re-executing BO workflow
- **FR-011**: Cells from previous modules MUST NOT be modified or deleted when adding new module sections
- **FR-012**: System MUST handle problems across dimensionalities: 2D (f1-f2), 3D (f3), 4D (f4), 5D (f5), 6D (f6), 7D (f7), 8D (f8)

### Non-Functional Requirements

- **NFR-001**: Notebook execution time MUST be reasonable for iterative development (<5 minutes per notebook on standard laptop)
- **NFR-002**: Code MUST be as simple as possible with each step clearly explained via markdown cells (per CONSTITUTION)
- **NFR-003**: Visualizations MUST render inline within Jupyter notebook using matplotlib
- **NFR-004**: Hyperparameter choices MUST include text justifications explaining why values were selected
- **NFR-005**: All notebooks MUST follow consistent structure: data loading → hyperparameters → GP training → acquisition optimization → visualization → progress tracking
- **NFR-006**: Code MUST use PyTorch float64 precision for numerical stability in GP computations

### Key Entities *(include if feature involves data)*

- **Problem**: Represents one of 8 black box optimization challenges (f1-f8)
  - Attributes: problem ID (f1-f8), input dimensionality (2-8D), initial sample count (10), evaluation budget (13 additional evaluations), timeline (Modules 12-24, 13 modules total), background context (e.g., "radiation source detection")
  - Relationships: has associated initial and updated datasets, has one notebook, belongs to maximization task category

- **Dataset**: Collection of input-output samples for a problem
  - Attributes: X (inputs, shape [N, D]), y (outputs, shape [N, 1]), source (initial vs updated_Module_N), best observed value, best input location
  - Relationships: belongs to one Problem, loaded by notebook cells, split into training tensors for GP

- **Gaussian Process Model**: BoTorch SingleTaskGP surrogate
  - Attributes: likelihood (noise variance), covar_module (kernel with lengthscales, optional outputscale), training data (X_train, y_train)
  - Relationships: trained on Dataset, used by AcquisitionFunction, generates Posterior predictions

- **AcquisitionFunction**: Expected Improvement function for selecting next point
  - Attributes: best_f (current best value), gp_model (trained GP), optimization result (next_point, acq_value)
  - Relationships: uses GP model, outputs next sample point proposal

- **Hyperparameters**: Configuration for BO workflow
  - Attributes: NUM_RESTARTS (10), RAW_SAMPLES (512), BOUNDS (Tensor shape [2, D]), kernel type (Matern 5/2 default)
  - Justifications: NUM_RESTARTS balances optimization quality vs compute time, RAW_SAMPLES ensures good coverage of acquisition surface, Matern kernel assumes smooth but not infinitely differentiable functions

- **ModuleIteration**: Represents one optimization cycle
  - Attributes: module number (12-24), dataset snapshot, proposed next_point, GP hyperparameters learned, convergence metrics
  - Relationships: builds on previous module's data, generates new notebook section, produces one submission

## Technical Context

### Libraries & Environment

- **BoTorch 0.16.1**: Bayesian Optimization framework built on PyTorch
- **PyTorch 2.10.0**: Tensor operations, automatic differentiation backend
- **GPyTorch 1.15.1**: Gaussian Process implementation used by BoTorch
- **NumPy 2.4.1**: Data file loading (.npy format)
- **Matplotlib 3.10.8**: Visualization generation
- **Python 3.14.2**: Runtime environment (sdd-dev kernel)

### Data Format

- Inputs: NumPy arrays shape `(N_samples, D_dimensions)`, float64
- Outputs: NumPy arrays shape `(N_samples,)` or `(N_samples, 1)`, float64
- File naming: `{initial|updated}_inputs.npy`, `{initial|updated}_outputs.npy` (note: existing data uses "Week N" naming, new data will follow module naming)
- Directory structure: `./data/f{1-8}/` with subdirectories for each problem

### Optimization Configuration

- **Kernel**: Matern 5/2 (BoTorch default for SingleTaskGP) - assumes 2-times differentiable functions
- **Acquisition**: Expected Improvement (EI) - balances exploitation of best known region vs exploration of uncertain areas
- **Bounds**: [0, 1] for standardized 2D problems (f1-f2), [min(X)-0.1*range, max(X)+0.1*range] per dimension for higher-D problems (f3-f8) to allow slight exploration beyond observed data
- **Optimization**: L-BFGS-B via BoTorch's `optimize_acqf` with multiple random restarts

### Known Issues & Fixes Applied

- **Issue**: AttributeError when accessing `gp_model.covar_module.outputscale` - BoTorch's SingleTaskGP may use RBFKernel (Matern) without ScaleKernel wrapper
- **Fix**: Conditional check `if hasattr(gp_model.covar_module, 'outputscale')` before accessing, fallback to base_kernel.lengthscale
- **Status**: Fixed in f1/f1.ipynb, needs to be applied to f2-f8

## Out of Scope

- Unit testing (per CONSTITUTION - not required for capstone)
- Alternative acquisition functions (UCB, PI, qEI) - may be added in later modules if EI underperforms
- Hyperparameter tuning via cross-validation - manual selection with justification is sufficient
- Multi-fidelity or constrained optimization - all problems are unconstrained single-fidelity
- Production deployment or API endpoints - notebooks are final deliverable format
- Automated module submission pipeline - manual extraction of next_point and submission expected

## Success Criteria

- All 8 notebooks (f1-f8) execute without errors on initial data
- Each notebook produces a valid next_point proposal within bounds
- GP hyperparameters (noise, lengthscales) are learned and interpretable
- Visualizations correctly render surrogate, acquisition, and convergence for all problems
- Hyperparameter justifications are documented in markdown cells
- Module update workflow is demonstrated for at least one problem (adding Module 13 section)
- Code follows simplicity principle with clear step-by-step explanations
- Notebooks scale to handle 23 total observations (10 initial in Module 12 + 13 module-by-module additions through Module 24) by project end
- Convergence plots show meaningful improvement over initial best values (qualitative assessment based on problem context, not fixed percentage threshold)

## Open Questions / Needs Clarification

- Should notebooks include comparison of multiple acquisition functions, or commit to Expected Improvement throughout?
