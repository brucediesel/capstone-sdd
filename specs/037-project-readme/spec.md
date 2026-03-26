# Feature Specification: Project README

**Feature Branch**: `037-project-readme`  
**Created**: 26 March 2026  
**Status**: Draft  
**Input**: User description: "Generate a readme.md file for the project, stored in the root folder. Provide a summary of the strategy adopted for each function. Provide the results of the optimisation process for each function using the convergence plots in the process_results.ipynb notebook. Provide a critical evaluation of the success of each strategy. Provide a section on lessons learnt."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read Project Overview (Priority: P1)

A reader (e.g., assessor, peer, or future contributor) opens the project repository and wants to quickly understand what the project is about, what approach was taken, and what the outcomes were — without needing to open any notebooks or code files.

**Why this priority**: The README is the single entry point for understanding the project. Without a clear overview, no other section has context.

**Independent Test**: Open the README in isolation and verify it conveys the project purpose, the eight optimisation functions, the overall approach (Bayesian Optimisation with tailored surrogates), and the 13-week timeline.

**Acceptance Scenarios**:

1. **Given** a reader with no prior knowledge of the project, **When** they read the README introduction, **Then** they understand the project is a black-box optimisation challenge with 8 functions solved over 13 weekly iterations using Bayesian Optimisation.
2. **Given** a reader scanning the README, **When** they look at the table of contents or section headings, **Then** they can navigate to any function's strategy, results, evaluation, or lessons learnt within seconds.

---

### User Story 2 - Understand Per-Function Strategy (Priority: P1)

A reader wants to understand the specific optimisation strategy adopted for each of the eight functions (F1–F8), including the surrogate model, acquisition function, and key adaptations made over the 13-week campaign.

**Why this priority**: The per-function strategy summary is the core intellectual content of the README — it captures the engineering decisions and rationale.

**Independent Test**: For each function, verify the README states the surrogate model type, kernel/architecture, acquisition function, output transform, and notable strategy changes across weeks.

**Acceptance Scenarios**:

1. **Given** a reader looking at the F1 section, **When** they read the strategy summary, **Then** they learn that F1 used a Hurdle Model (Logistic Classifier + RF Regressor) with Weighted UCB, log-transformed outputs, and shifted to SFGP with qLogNEI in later weeks.
2. **Given** a reader comparing strategies, **When** they read F7 and F8 sections, **Then** they see distinct approaches (Neural Network with MC Dropout for F7 vs GP with qEI for F8) with justification for each choice.

---

### User Story 3 - Review Optimisation Results (Priority: P1)

A reader wants to see the outcomes of the optimisation process for each function, including convergence behaviour and best-observed values, with references to the convergence plots generated in the process_results.ipynb notebook.

**Why this priority**: Results are the evidence of whether the strategies succeeded. Without them, the README is aspirational rather than evidential.

**Independent Test**: Verify the README includes a results summary table with best-observed output per function, a description of convergence behaviour, and a reference to the convergence plots in process_results.ipynb.

**Acceptance Scenarios**:

1. **Given** a reader reviewing results, **When** they look at the results section, **Then** they see a summary table listing each function's best output, number of improving iterations, and convergence trend.
2. **Given** a reader wanting visual evidence, **When** they read the results section, **Then** they find a reference to the process_results.ipynb convergence plot grid with guidance on how to interpret it.

---

### User Story 4 - Read Critical Evaluation (Priority: P2)

A reader wants an honest, reflective assessment of what worked, what did not work, and why — for each function's strategy.

**Why this priority**: Critical evaluation demonstrates analytical depth and self-awareness, which is essential for academic work but depends on strategies and results being documented first.

**Independent Test**: Verify the README includes per-function evaluation noting successes, failures, and possible causes (e.g., local optima trapping, dimensionality challenges, transform artefacts).

**Acceptance Scenarios**:

1. **Given** a reader assessing the F1 evaluation, **When** they read the critical evaluation, **Then** they see an acknowledgement that F1's zero-inflated output made progress extremely difficult, and the strategy pivots (Hurdle → SFGP) were justified but ultimately limited by the function's sparse reward landscape.
2. **Given** a reader looking for candour, **When** they read the evaluation section, **Then** they find specific examples of both successful strategies (e.g., F5 achieving large improvements) and stalled strategies (e.g., F1's radiation source never located despite pivoting from Hurdle Model to SFGP).

---

### User Story 5 - Learn from Lessons Learnt (Priority: P2)

A reader wants to understand the generalisable insights and methodological lessons from the 13-week optimisation campaign.

**Why this priority**: Lessons learnt synthesise the experience into transferable knowledge, but require the per-function detail to have been established first.

**Independent Test**: Verify the README includes a dedicated lessons learnt section covering at least: surrogate selection, acquisition function tuning, output transforms, dimensionality challenges, and adaptive strategy changes.

**Acceptance Scenarios**:

1. **Given** a reader interested in methodology, **When** they read the lessons learnt section, **Then** they find actionable insights such as "Prequential Evaluation proved valuable for surrogate selection before committing to a strategy" and "Interior penalties must be carefully calibrated — too aggressive and they swamp the acquisition signal."
2. **Given** a future practitioner, **When** they read the lessons learnt section, **Then** they can identify at least 5 concrete recommendations for running a multi-week Bayesian Optimisation campaign.

---

### Edge Cases

- What if convergence plots are not available (notebook not executed)? The README should describe the expected content of the plots and reference the notebook file path so the reader can run it themselves.
- What if a function showed no improvement at all? The README should still document the strategy and results honestly, noting zero improvement as a result rather than omitting the function.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The README MUST be a single Markdown file named `README.md` located in the project root folder.
- **FR-002**: The README MUST include a project overview section explaining the black-box optimisation challenge, the 8 functions, input/output constraints, and the 13-week iterative structure.
- **FR-003**: The README MUST include a strategy summary for each function (F1–F8), covering surrogate model, acquisition function, output transformation, key hyperparameters, and notable strategy changes across weeks.
- **FR-004**: The README MUST include a results section for each function summarising the best-observed output, convergence behaviour, and improvement trajectory (e.g., number of improving iterations or improvement factor where available).
- **FR-005**: The README MUST reference the convergence plots from the `functions/results/process_results.ipynb` notebook, describing what they show and how to generate them.
- **FR-006**: The README MUST include a critical evaluation section assessing the success or failure of each function's strategy, with specific evidence and reasoning.
- **FR-007**: The README MUST include a lessons learnt section with generalisable insights from the optimisation campaign.
- **FR-008**: The README MUST include a project structure section describing the repository layout (data/, functions/, specs/, research/ folders).
- **FR-009**: The README MUST be written for a general academic audience — clear, professional prose that does not assume familiarity with the specific challenge.
- **FR-010**: The README MUST reference existing project documentation (modelcards.md, datasheets.md) for readers wanting deeper detail.
- **FR-011**: The `modelcards.md` file MUST be updated to document the full strategy evolution (Weeks 3–13) for each function, including surrogate model changes, hyperparameter adjustments, and final Week 13 performance results.
- **FR-012**: The `datasheets.md` file MUST be updated to reflect the final (Week 13) dataset sizes, collection timeline, and any preprocessing changes made during the campaign.

### Key Entities

- **Function (F1–F8)**: A black-box optimisation target with a defined input dimensionality (2D–8D), output characteristics, and domain context.
- **Strategy**: The combination of surrogate model, acquisition function, output transform, and hyperparameters used for a given function in a given week.
- **Convergence Plot**: A visual showing the running maximum output over iterations, distinguishing initial samples from BO-acquired samples.
- **Results Summary**: Per-function quantitative outcomes including best output, improvement trajectory, and stalling indicators.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reader with no prior knowledge of the project can understand the purpose, approach, and outcomes within 5 minutes of reading the README.
- **SC-002**: All 8 functions are individually documented with strategy, results, and evaluation — no function is omitted.
- **SC-003**: The results section includes quantitative data (best output values, number of improving samples) for every function.
- **SC-004**: The critical evaluation section identifies at least one strength and one weakness for each function's strategy.
- **SC-005**: The lessons learnt section contains at least 5 distinct, actionable insights drawn from the campaign experience.
- **SC-006**: The README correctly references existing documentation files (modelcards.md, datasheets.md, process_results.ipynb) with accurate relative paths.
- **SC-007**: The modelcards.md file documents the full strategy evolution (Weeks 3–13) for all 8 functions, with Week 13 performance metrics matching numpy-verified values.
- **SC-008**: The datasheets.md file reflects Week 13 final dataset sizes for all 8 functions, with file lists extending through Week 13.

## Clarifications

### Session 2026-03-26

- Q: Should updating datasheets.md and modelcards.md be part of this spec or a separate feature? → A: Add to this spec as additional functional requirements (same branch, same implementation).
- Q: How should model cards be updated — full evolution, final-only, or addendum? → A: Show the full strategy evolution (Weeks 3–13) in each model card's Details section.

## Assumptions

- The process_results.ipynb notebook has been executed and the convergence plots are available. The README will describe these plots and reference the notebook but will not embed images directly.
- Results data up to Week 13 (the final submission) is available in the data folders.
- The modelcards.md and datasheets.md files exist but need updating to reflect Week 13 data and final models (covered by FR-011 and FR-012).
- The README is a documentation deliverable — it does not execute code or modify any existing notebooks.
- The strategy descriptions in the README will be high-level summaries referencing the weekly notebooks for implementation detail.
