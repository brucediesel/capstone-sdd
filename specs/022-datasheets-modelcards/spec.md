# Feature Specification: Datasheets & Model Cards

**Feature Branch**: `022-datasheets-modelcards`  
**Created**: 2025-07-06  
**Status**: Draft  
**Input**: User description: "Create two new markdown documents in the root folder: datasheets.md and modelcards.md. Model cards for each of 8 functions with: Overview, Intended use, Details, Performance, Assumptions/limitations, Ethical considerations. Datasheets for each of 8 functions with: Motivation, Composition, Collection process, Preprocessing/uses, Distribution/maintenance."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Model Cards Document (Priority: P1)

As a capstone assessor or peer reviewer, I want to read a single `modelcards.md` document in the project root that contains a model card for each of the eight optimisation functions (F1–F8), so that I can understand what surrogate model was used for each function, why it was chosen, how it performed, and what its limitations are.

Each model card covers:
- **Overview**: What the model does and which function it targets
- **Intended Use**: The optimisation context and how the model supports decision-making
- **Details**: The surrogate modelling strategy across all ten rounds (initial + Weeks 3–9), including model evolution, key hyperparameters, acquisition function, and any special techniques (interior penalty, hurdle model, multi-fidelity, etc.)
- **Performance**: Observed outputs, best-found values, and any evaluation metrics (prequential evaluation, LOO error) recorded in the notebooks
- **Assumptions & Limitations**: Known constraints, data limitations, failure modes, and boundary conditions
- **Ethical Considerations**: Responsible use context, potential for misuse, and fairness considerations relevant to the function's domain

**Why this priority**: Model cards are the primary deliverable demonstrating mastery of the modelling decisions. Assessors need to understand the reasoning behind model selection and evolution for each function.

**Independent Test**: Open `modelcards.md` in a markdown viewer and verify that all eight function sections are present, each containing the six required subsections with substantive content drawn from the actual notebook results.

**Acceptance Scenarios**:

1. **Given** the project repository, **When** a reader opens `modelcards.md`, **Then** they see eight clearly separated model card sections (F1–F8), each with all six subsections populated
2. **Given** the F1 model card, **When** a reader examines the Details section, **Then** they find a description of the hurdle model approach (logistic classifier + random forest), weighted UCB acquisition, local penalisation, and interior penalty — with a summary of model evolution from GP through polynomial/RF/GBT to the final hurdle model
3. **Given** any model card, **When** a reader checks the Performance section, **Then** they find concrete output statistics (data size, output range, best-found value) and any recorded evaluation metrics
4. **Given** any model card, **When** a reader reviews Assumptions & Limitations, **Then** they find at least two specific limitations tied to the function's characteristics (e.g., zero-inflated outputs for F1, all-negative outputs for F3/F6)

---

### User Story 2 - Datasheets Document (Priority: P1)

As a capstone assessor or peer reviewer, I want to read a single `datasheets.md` document in the project root that contains a datasheet for each of the eight optimisation functions (F1–F8), so that I can understand the data used to train and evaluate the surrogate models.

Each datasheet covers:
- **Motivation**: Why the data was collected (black-box optimisation challenge), the function's real-world domain, and what purpose the data serves
- **Composition**: Number of instances, feature dimensions, input bounds, output characteristics (sign, range, special properties like zero-inflation), and how data grew across rounds
- **Collection Process**: How initial data was provided and how subsequent data points were generated (model-guided candidate selection, external evaluation oracle)
- **Preprocessing & Uses**: Any transformations applied to inputs or outputs (standardisation, log transforms, z-scores), and the intended downstream use (surrogate model training)
- **Distribution & Maintenance**: How the data is stored in the repository, file naming conventions, and the static nature of the dataset (no ongoing collection)

**Why this priority**: Datasheets are equally important as model cards for responsible AI documentation. They provide the data provenance and transparency required for the capstone assessment.

**Independent Test**: Open `datasheets.md` in a markdown viewer and verify that all eight function sections are present, each containing the five required subsections with substantive content drawn from the actual data files and notebook records.

**Acceptance Scenarios**:

1. **Given** the project repository, **When** a reader opens `datasheets.md`, **Then** they see eight clearly separated datasheet sections (F1–F8), each with all five subsections populated
2. **Given** the F3 datasheet, **When** a reader examines the Composition section, **Then** they find that the dataset is 3-dimensional, grew from 15 initial points to 24 by Week 9, and all outputs are negative
3. **Given** any datasheet, **When** a reader checks the Collection Process section, **Then** they find a description of the initial data provision and the iterative model-guided acquisition process across ten rounds
4. **Given** any datasheet, **When** a reader reviews Distribution & Maintenance, **Then** they find the file storage location (`data/fN/`), file naming convention (initial_inputs/outputs.npy, updated_inputs/outputs - Week N.npy), and confirmation that the dataset is static

---

### Edge Cases

- **Functions with zero-inflated outputs (F1)**: The datasheet and model card must explicitly describe the zero-inflation characteristic and how it influenced model choice (hurdle model)
- **Functions with all-negative outputs (F3, F6)**: The documents must note the all-negative output domain and any special handling (manual z-score for F3, rank-based scoring for F6)
- **Multi-fidelity function (F4)**: The datasheet must explain the constant fidelity column appended to inputs; the model card must describe the multi-fidelity GP approach
- **Non-GP surrogate (F7)**: The model card must explain why a neural network was chosen over a GP for the final model and describe MC Dropout for uncertainty estimation
- **Single-query function (F8)**: The model card must note that q=1 was used (unlike q=4 for other late-stage functions) and the fallback to posterior mean when qEI values are zero
- **Interior penalty usage**: Model cards for F1, F5, F6, and F7 must describe interior penalty; model cards for F2, F3, F4, and F8 must note its absence and explain why

## Requirements *(mandatory)*

### Functional Requirements

#### Model Cards (`modelcards.md`)

- **FR-001**: The document MUST be a single markdown file named `modelcards.md` located in the project root directory
- **FR-002**: The document MUST contain exactly eight model card sections, one for each function (F1–F8), clearly separated with headings
- **FR-003**: Each model card MUST include an **Overview** subsection describing what the surrogate model does and which black-box function it targets, including the function's real-world domain (e.g., Radiation Source Detection for F1, Drug Discovery for F3)
- **FR-004**: Each model card MUST include an **Intended Use** subsection describing the optimisation context — that the model serves as a cheap-to-evaluate surrogate for an expensive black-box function, guiding candidate selection to find optimal input configurations
- **FR-005**: Each model card MUST include a **Details** subsection containing:
  - The final surrogate model type and its key hyperparameters
  - The final acquisition function and its configuration
  - A summary of model evolution across rounds (which models were tried in which weeks)
  - Any special techniques used (interior penalty, hurdle model, multi-fidelity, custom transforms)
- **FR-006**: Each model card MUST include a **Performance** subsection containing:
  - The total number of data points available at final round
  - The output range observed (minimum and maximum values)
  - The output sign characteristics (all positive, all negative, zero-inflated, mixed)
  - Any evaluation metrics recorded in the notebooks (prequential evaluation scores, LOO surrogate error)
- **FR-007**: Each model card MUST include an **Assumptions & Limitations** subsection identifying at least two specific limitations, covering:
  - Data size constraints (small dataset regime)
  - Any known failure modes or boundary conditions
  - Domain-specific challenges affecting model performance
- **FR-008**: Each model card MUST include an **Ethical Considerations** subsection addressing:
  - The function's domain and potential real-world impact
  - Responsible use context (model is a surrogate approximation, not ground truth)
  - Any fairness or bias concerns relevant to the domain

#### Datasheets (`datasheets.md`)

- **FR-009**: The document MUST be a single markdown file named `datasheets.md` located in the project root directory
- **FR-010**: The document MUST contain exactly eight datasheet sections, one for each function (F1–F8), clearly separated with headings
- **FR-011**: Each datasheet MUST include a **Motivation** subsection describing:
  - The purpose of data collection (black-box optimisation challenge for a capstone AI/ML course)
  - The function's real-world domain and why it matters
  - Who created the dataset (challenge organisers provided initial data; subsequent points were generated by the student's optimisation pipeline and evaluated by the challenge oracle)
- **FR-012**: Each datasheet MUST include a **Composition** subsection describing:
  - Input dimensionality (number of features)
  - Input bounds (all normalised to [0, 1] per dimension)
  - Total number of instances at initial round and at final round (Week 9)
  - Output characteristics: sign (positive/negative/zero/mixed), range (min to max), and any special properties (zero-inflation for F1)
  - Data growth trajectory across rounds
- **FR-013**: Each datasheet MUST include a **Collection Process** subsection describing:
  - How initial data was provided (pre-generated by challenge organisers)
  - How subsequent data points were acquired (surrogate model trained → acquisition function optimised → candidates submitted → oracle returned true function values)
  - The iterative nature: 10 rounds total (initial + Weeks 3–9)
  - Number of new points per round (varies by function: some submit 1, others up to 4)
- **FR-014**: Each datasheet MUST include a **Preprocessing & Uses** subsection describing:
  - Input preprocessing (if any — e.g., constant fidelity column for F4)
  - Output preprocessing (if any — e.g., log1p transform for F5, manual z-score for F3, rank-based scoring for F6)
  - Standardisation approach (e.g., Standardize transform, manual normalisation)
  - Intended downstream use (training surrogate models for Bayesian optimisation)
- **FR-015**: Each datasheet MUST include a **Distribution & Maintenance** subsection describing:
  - Storage location within the repository (`data/fN/` directories)
  - File format (NumPy `.npy` files)
  - File naming convention (`initial_inputs.npy`, `initial_outputs.npy`, `updated_inputs - Week N.npy`, `updated_outputs - Week N.npy`)
  - Static nature of the dataset (collection is complete, no further updates planned)

#### Cross-Cutting Requirements

- **FR-016**: Both documents MUST use consistent formatting and heading structure across all eight function sections
- **FR-017**: Both documents MUST use plain language accessible to non-technical stakeholders while retaining necessary technical precision for model and data descriptions
- **FR-018**: Both documents MUST reference concrete values from the actual project data (real data sizes, real output ranges, real model configurations) rather than generic placeholders
- **FR-019**: The model cards document MUST include a summary comparison table at the top or bottom showing all eight functions side-by-side with key attributes (domain, dimensions, final surrogate, final acquisition, interior penalty usage)
- **FR-020**: The datasheets document MUST include a summary comparison table at the top or bottom showing all eight functions side-by-side with key attributes (domain, dimensions, initial size, final size, output sign)

### Key Entities

- **Function (F1–F8)**: A black-box optimisation target. Key attributes: real-world domain name, input dimensionality, input bounds, output sign characteristics, total data points
- **Model Card**: A structured transparency document for one function's surrogate model. Contains six sections: Overview, Intended Use, Details, Performance, Assumptions & Limitations, Ethical Considerations
- **Datasheet**: A structured transparency document for one function's dataset. Contains five sections: Motivation, Composition, Collection Process, Preprocessing & Uses, Distribution & Maintenance
- **Surrogate Model**: The machine learning model trained to approximate the black-box function. Key attributes: model type, kernel, acquisition function, hyperparameters, special techniques
- **Round**: One iteration of the optimisation loop (initial data provision or a weekly submission). Ten rounds total: initial + Weeks 3–9

## Assumptions

- The content draws exclusively from data files in `data/f1/` through `data/f8/` and the existing notebooks in `functions/f1/` through `functions/f8/`. No external data sources are needed.
- Performance metrics are limited to what was recorded in the notebooks (output ranges, prequential evaluation, LOO error). No new experiments or evaluations are required.
- The ethical considerations sections will address domain-appropriate concerns at a level suitable for a capstone course deliverable, not a full ethics review.
- "Ten rounds" refers to the initial data provision plus nine weekly submissions (Weeks 3–9).
- The function domain names (Radiation Source Detection, Noisy Log-Likelihood, Drug Discovery, Warehouse Product Placement, Chemical Process Yield, Cake Recipe, ML Hyperparameter Tuning, 8D ML Hyperparameters) are taken from the notebook descriptions and may be approximate labels assigned by the challenge organisers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Both `modelcards.md` and `datasheets.md` exist in the project root and render correctly in a standard markdown viewer
- **SC-002**: Each document contains exactly 8 function sections (F1–F8) with all required subsections populated (6 subsections per model card, 5 subsections per datasheet) — 100% section coverage
- **SC-003**: Every model card's Details section references the actual final surrogate model, acquisition function, and at least one special technique or design decision specific to that function
- **SC-004**: Every datasheet's Composition section contains the correct input dimensionality, initial data size, final data size, and output range matching the actual `.npy` data files
- **SC-005**: Both documents include a summary comparison table covering all 8 functions
- **SC-006**: A reader unfamiliar with the project can understand each function's purpose, data characteristics, and modelling approach by reading only these two documents, without consulting the notebooks
- **SC-007**: No section contains generic placeholder text — every statement is grounded in the actual project data and notebook content
