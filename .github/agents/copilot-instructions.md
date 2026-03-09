# capstone-sdd Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-17

## Active Technologies
- Python 3.14.2 (pyenv virtualenv `sdd-dev`) + BoTorch 0.16.1 (SingleTaskGP, SingleTaskMultiFidelityGP, fit_gpytorch_mll), GPyTorch 1.15.1 (MaternKernel, RBFKernel, ScaleKernel, GaussianLikelihood, GreaterThan, ExactMarginalLogLikelihood), PyTorch (nn.Module, Adam, MSELoss), matplotlib, pandas, numpy (004-prequential-evaluation)
- `.npy` files in `data/f6/` (read-only) (004-prequential-evaluation)
- Python 3 (Jupyter notebook, kernel already active in workspace) + `numpy`, `matplotlib`, `scikit-learn` (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor) — all already installed in the project environmen (005-week7-pe-surrogates)
- Read-only `.npy` files from `data/f1/`; no writes except notebook state (005-week7-pe-surrogates)
- Python 3.x (existing notebook kernel) + `botorch==0.16.1`, `gpytorch==1.15.1`, `numpy`, `pandas`, `matplotlib`, `torch` (all pre-installed) (006-sfgp-mfgp-pe)
- `.npy` files in `data/f2/`; notebook output cells (no database) (006-sfgp-mfgp-pe)
- Python 3.11 (005-week7-pe-surrogates)
- NumPy `.npy` files under `data/f1/`, `data/f2/`, `data/f3/` — read-only. (005-week7-pe-surrogates)
- Python 3.11 (pyenv `sdd-dev` environment) + BoTorch (GP, NEI), GPyTorch (kernel, likelihood, MLL), PyTorch (tensor operations), NumPy (data loading), Matplotlib (visualisations) (009-f3-sfgp-nei)
- File-based — `.npy` arrays in `data/f3/`; outputs are new notebook cells only, no other files written (009-f3-sfgp-nei)
- Python 3.14 (pyenv `sdd-dev`) + BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch, NumPy, Matplotlib (010-f4-mfgp-nei)
- `.npy` files in `data/f4/` (cumulative weekly snapshots) (010-f4-mfgp-nei)
- `.npy` files in `data/f5/` (read-only); notebook cells in `functions/f5/f5.ipynb` (append-only) (011-f5-gp-nei)
- `.npy` files in `data/f5/` (27 samples × 4 dims) (011-f5-gp-nei)
- `.npy` files in `data/f6/` (27 samples × 5 dims) (012-f6-sfgp-nei)
- Python 3.14.2 (pyenv `sdd-dev` on macOS) + BoTorch 0.16.1, GPyTorch 1.15.1, PyTorch (double precision), NumPy, Matplotlib (012-f6-sfgp-nei)
- `.npy` files in `data/f6/` (27×5 inputs, 27 outputs); notebook cells in `functions/f6/f6.ipynb` (012-f6-sfgp-nei)
- Python 3.11 + BoTorch (SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf), GPyTorch (ExactMarginalLogLikelihood, Matérn-5/2, GaussianLikelihood), PyTorch, NumPy, Matplotlib (015-f5-interior-penalty)
- `.npy` files in `data/f5/` (read-only); notebook state in `functions/f5/f5.ipynb` (015-f5-interior-penalty)
- Python 3.11 + BoTorch (SingleTaskGP, qLogNoisyExpectedImprovement, optimize_acqf), GPyTorch (ExactMarginalLogLikelihood, Matérn-1.5, GaussianLikelihood), PyTorch, NumPy, Matplotlib (016-f6-interior-penalty)
- `.npy` files in `data/f6/` (read-only); notebook state in `functions/f6/f6.ipynb` (016-f6-interior-penalty)
- Python 3.11+ (pyenv, macOS) + PyTorch (nn, optim), NumPy, Matplotlib — no BoTorch needed for NN surrogate (017-f7-nn-interior-penalty)
- `.npy` files in `data/f7/` (read-only); notebook state in `functions/f7/f7.ipynb` (017-f7-nn-interior-penalty)
- Python 3.x (Jupyter Notebook) + numpy, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor) (019-f1-week8-hurdle)
- `.npy` files in `./data/f1/` (019-f1-week8-hurdle)
- Python 3.x (sdd-dev conda env) + BoTorch, GPyTorch, PyTorch, NumPy, Matplotlib (020-f2-f8-week8)
- Python 3.14.2 (Jupyter kernel `sdd-dev`) + BoTorch, GPyTorch, PyTorch (F2–F6, F8); PyTorch + NumPy (F7 NN) (020-f2-f8-week8)
- `.npy` files in `./data/fX/` (020-f2-f8-week8)
- Python 3.11 (sdd-dev environment) + BoTorch, GPyTorch, PyTorch, scikit-learn, NumPy, Matplotlib (021-f1-f8-week9)
- `.npy` files in `./data/fX/` directories (021-f1-f8-week9)
- Markdown (no code execution required) + None — output is static markdown documents (022-datasheets-modelcards)
- Two `.md` files in project root (`modelcards.md`, `datasheets.md`) (022-datasheets-modelcards)
- Python 3.x (Jupyter notebook) + NumPy, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), matplotlib, scipy (pdist/squareform) (023-f1-week9-log)
- Python 3.x (Jupyter Notebook) + numpy, pandas, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), scipy (pdist, squareform) (023-f1-week9-log)
- NumPy `.npy` files in `./data/f1/` (023-f1-week9-log)
- Python 3.x (Jupyter Notebook) + numpy, pandas, matplotlib, scikit-learn (CalibratedClassifierCV, LogisticRegression, RandomForestRegressor), scipy (cdist) (023-f1-week9-log)
- NumPy .npy files in `./data/f1/` (023-f1-week9-log)
- Python 3.x (Jupyter Notebook) + numpy, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), scipy (pdist, squareform) (023-f1-week9-log)
- Python 3.x (Jupyter Notebook) + BoTorch (SingleTaskGP, Standardize, qLogNEI, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood), PyTorch, NumPy, Matplotlib, SciPy (024-f3-week9-standardize)
- `.npy` files in `./data/f3/` (24 samples: 15 initial + 9 submissions) (024-f3-week9-standardize)
- Python 3.11 + BoTorch (SingleTaskGP, Standardize, qLogNEI, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood), NumPy, Matplotlib, SciPy (024-f3-week9-standardize)

- Python 3.14.2 (pyenv sdd-dev) + numpy, matplotlib, scikit-learn (Ridge, RF, GBT), PyTorch (NN), scipy (f1 only) (003-week6-focus-on-exploitation)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.14.2 (pyenv sdd-dev): Follow standard conventions

## Recent Changes
- 024-f3-week9-standardize: Added Python 3.11 + BoTorch (SingleTaskGP, Standardize, qLogNEI, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood, ExactMarginalLogLikelihood), NumPy, Matplotlib, SciPy
- 024-f3-week9-standardize: Added Python 3.x (Jupyter Notebook) + BoTorch (SingleTaskGP, Standardize, qLogNEI, optimize_acqf), GPyTorch (MaternKernel, ScaleKernel, GaussianLikelihood), PyTorch, NumPy, Matplotlib, SciPy
- 023-f1-week9-log: Added Python 3.x (Jupyter Notebook) + numpy, matplotlib, scikit-learn (LogisticRegression, CalibratedClassifierCV, RandomForestRegressor), scipy (pdist, squareform)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
