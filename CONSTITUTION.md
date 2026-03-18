# Project Constitution

## Project Overview
This is a capstone project as part of a certificate course in AI and ML, focused on solving 8 separate black box optimization problems.

## Core Principles

### Code Requirements
- Code should be **as simple as possible** with each step clearly explained
- All code must be submitted in the form of **Jupyter notebooks**
- No unit tests are required

### Project Structure
- 8 separate black box optimization problems (f1-f8)
- Each problem solved in its own Jupyter notebook
- Each notebook stored in its own folder within `./functions` project folder
- Folders and .ipynb files already exist

### Data Organization
- Data for each problem stored in `./data` folder structure
- Individual folder for each problem (f1-f8)
- Initial data files:
  - `initial_inputs.npy`
  - `initial_outputs.npy`
- Updated data files (added weekly):
  - `updated_inputs - Week X.npy`
  - `updated_outputs - Week X.npy`

## Optimization Methodology

### Library and Framework
- Use the **BoTorch library** as the default for Gaussian Process surrogates
- Surrogate models are chosen per function based on problem characteristics (e.g., GP, polynomial response surface, tree-based ensembles, neural networks)
- Acquisition functions are chosen per surrogate type (e.g., Expected Improvement for GP, Upper Confidence Bound for non-GP surrogates)
- Additional libraries: **scikit-learn** (polynomial, Random Forest, Gradient Boost), **PyTorch** (neural network surrogates)

### Workflow Process
1. Train models on initial data
2. Propose next sample point
3. Submit samples weekly
4. Receive results and add to dataset
5. Re-execute optimization with updated data
6. Propose new sample point
7. Repeat process

### Weekly Updates
- Each weekly iteration MUST be implemented in a **new notebook** named `fX - week Y.ipynb` where X is the function number and Y is the week number (e.g., `f1 - week 7.ipynb`, `f3 - week 8.ipynb`)
- New iteration notebooks are stored in the same folder as the original function notebook (`./functions/fX/`)
- The original `fX.ipynb` notebook contains all historical weekly sections up to the point this convention was adopted and MUST NOT be modified further
- Each iteration notebook MUST be self-contained: imports, data loading, surrogate fitting, acquisition, visualisation, and submission query
- **Existing notebooks from previous iterations MUST NOT be modified** — previous work is preserved as-is. The current iteration's notebook may be updated until finalised.

## Documentation Requirements

### Model Specifications
Each model must clearly specify:
- **Hyperparameters used**
- **Explanation of why hyperparameter values were chosen**

### Visualizations
Each model must provide clear visualizations of:
- **Surrogate function** in the problem space
- **Convergence of the objective** function

### Problem Information
- Input and output dimensions specified in each notebook
- Background information for each black box provided
- Use problem context to create appropriate visualizations

## Optimization Goals
All problems are **maximization** tasks.
