# Implementation Plan: Week 7 — SFGP and MFGP Prequential Evaluation on Function 2

**Branch**: `006-sfgp-mfgp-pe` | **Date**: 2026-02-22 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/006-sfgp-mfgp-pe/spec.md`

## Summary

Rewrite `functions/f2/preq-eval-f2.ipynb` to replace the three existing surrogate families (GP, BART, RF) with a **Single-Fidelity GP (SFGP)** using BoTorch `SingleTaskGP` and a **Multi-Fidelity GP (MFGP)** using BoTorch `MultiTaskGP`. Data is extended to Week 7 (17 total samples: 10 initial + 7 one-step-ahead prequential steps). Each family is swept over 50 hyperparameter configurations. The best configuration from each family is compared head-to-head and the winner is visualised in detail, reusing the existing `compute_metrics()` and `plot_prequential_results()` utility functions.

## Technical Context

**Language/Version**: Python 3.x (existing notebook kernel)  
**Primary Dependencies**: `botorch==0.16.1`, `gpytorch==1.15.1`, `numpy`, `pandas`, `matplotlib`, `torch` (all pre-installed)  
**Storage**: `.npy` files in `data/f2/`; notebook output cells (no database)  
**Testing**: No unit tests (constitution); manual end-to-end execution of all cells in order  
**Target Platform**: Jupyter notebook, local laptop  
**Project Type**: Single Jupyter notebook  
**Performance Goals**: Full 100-configuration sweep completes in under 30 minutes on 8 GB RAM  
**Constraints**: No new pip dependencies; no modification to existing utility functions; all hyperparameter choices documented inline  
**Scale/Scope**: 17 data points, 2D input, 100 prequential evaluations (7 steps each)

## Constitution Check

*GATE: Must pass before implementation. Re-checked after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Code as simple as possible | ✅ PASS | Two surrogate families, each following the same prequential loop pattern as the existing notebook; no new abstraction layers |
| Submitted as Jupyter notebook | ✅ PASS | Feature modifies `functions/f2/preq-eval-f2.ipynb` in place |
| No unit tests required | ✅ PASS | End-to-end notebook execution is the verification method |
| Uses BoTorch as default for GP surrogates | ✅ PASS | Both SFGP (`SingleTaskGP`) and MFGP (`MultiTaskGP`) are native BoTorch models |
| No new pip dependencies | ✅ PASS | `MultiTaskGP` is in BoTorch 0.16 which is already installed |
| Weekly sections added, existing cells not replaced | ✅ PASS | This is a rewrite of a prequential evaluation notebook (not a weekly BO notebook); the constitution's "add new section" rule applies to the per-function BO notebooks; the prequential notebook is an analysis notebook and is rewritten in full |
| Hyperparameter values explained | ✅ PASS | Required by FR-003/FR-004; each config list entry includes a `label` with key values; a summary markdown cell explains each axis |
| Visualisations of surrogate + convergence | ✅ PASS | `plot_prequential_results()` covers per-step predictions + errors + NLP; bar charts cover comparative metrics |

**No violations.** Complexity Tracking section not required.

## Project Structure

### Documentation (this feature)

```text
specs/006-sfgp-mfgp-pe/
├── plan.md              ← this file
├── research.md          ← Phase 0 complete
├── data-model.md        ← Phase 1 complete
├── quickstart.md        ← Phase 1 complete
├── contracts/
│   └── notebook-cells.md   ← Phase 1 complete
└── tasks.md             ← Phase 2 (created by /speckit.tasks, not this command)
```

### Source Code

```text
functions/f2/
└── preq-eval-f2.ipynb   ← the single file being modified
```

**Structure Decision**: Single-notebook project. All changes are confined to one existing `.ipynb` file. No new source files, no new directories, no new dependencies.

---

## Implementation Phases

### Phase 1 — Notebook Cell Plan

The notebook is rewritten section by section. The cell order below maps directly to the implementation sequence.

#### Section 0 — Title & Overview (Markdown)

Update the notebook title and overview table:
- Title: `Prequential Evaluation of Surrogate Models — Function 2 (Week 7: SFGP vs MFGP)`  
- Update the surrogate models table: replace GP/BART/RF entries with SFGP and MFGP  
- Update `Total samples` row to 17, `Evaluation steps` to 7

#### Section 1 — Imports (Code)

Keep all existing imports. Add:
```python
from botorch.models import MultiTaskGP
```
Remove BART-specific imports (`pymc`, `pymc_bart`). Keep `RandomForestRegressor` import removal is optional (it is unused after this rewrite — remove for cleanliness).

#### Section 2 — Load Data and Fidelity Split (Code)

Set `WEEK = 7`, `N_INIT = 10`. Load Week 7 `.npy` files with `os.path.exists` guard.  
Print summary: shape, output range, fidelity split counts (10 LF / 7 HF).

#### Section 3 — Define Evaluation Metrics (reuse existing `compute_metrics()`)

Move the existing `compute_metrics()` function definition here unchanged as a utility.

#### Section 4 — Define Visualisation Utility (reuse existing `plot_prequential_results()`)

Move the existing `plot_prequential_results()` function here unchanged.

#### Section 5 — SFGP: Default Run (Markdown + Code)

- Markdown: explain SFGP = `SingleTaskGP` (standard GP, single fidelity), Matérn 5/2 default
- Code: define `sfgp_prequential_evaluation(X_all, y_all, n_init)` — identical to existing `gp_prequential_evaluation` but renamed; run default config; call `plot_prequential_results()`

#### Section 6 — SFGP: Define 50-Config Sweep Function (Code)

Define `sfgp_prequential_with_config(X_all, y_all, n_init, config)`.  
Config dict keys: `kernel_type`, `noise_lb`, `ard`, `log_transform`, `input_normalize`, `label`.  
Kernel type mapping:
- `'matern05'` → `MaternKernel(nu=0.5, ard_num_dims=d)`
- `'matern15'` → `MaternKernel(nu=1.5, ard_num_dims=d)` (ard=True) or `ard_num_dims=1` (ard=False)
- `'matern25'` → `MaternKernel(nu=2.5, ard_num_dims=d)`
- `'rbf'` → `RBFKernel(ard_num_dims=d)`

Input normalize: prepend `Normalize(d=d)` transform on `X_train` via `botorch.models.transforms.input.Normalize`.  
Output log-transform: apply `y_work = np.log(np.abs(y_all) + EPS)` as in existing GP code.

#### Section 7 — SFGP: 50 Configuration Definitions (Code)

Define `sfgp_configs` list with 50 entries.

**Block A — 32 configs** (4 kernels × 4 noise bounds × 2 ARD settings, no log-transform, input_normalize=True):
```python
for kernel in ['matern05', 'matern15', 'matern25', 'rbf']:
    for noise_lb in [1e-6, 1e-5, 1e-4, 1e-3]:
        for ard in [True, False]:
            label = f'{kernel}, noise>={noise_lb:.0e}, ard={ard}'
```

**Block B — 18 configs** (3 kernels × 3 noise bounds × log-transform=True × ard=True = 9; + kernel=matern25 × noise_lb={1e-5,1e-4,1e-3} × input_normalize=False × ard={True,False} = 6; + rbf × noise_lb={1e-6,1e-5,1e-4} × log_transform=True × ard=False = 3):

```python
# 9: log-transform configs
for kernel in ['matern15', 'matern25', 'rbf']:
    for noise_lb in [1e-6, 1e-5, 1e-4]:
        {'kernel_type': kernel, 'noise_lb': noise_lb, 'ard': True,
         'log_transform': True, 'input_normalize': True, ...}
# 9: no-normalise configs (ard=True only; 3 kernels × 3 noise_lb values = 9)
for noise_lb in [1e-5, 1e-4, 1e-3]:
    for kernel in ['matern25', 'rbf', 'matern15']:
        {'kernel_type': kernel, 'noise_lb': noise_lb, 'ard': True,
         'log_transform': False, 'input_normalize': False, ...}
```

Total: exactly 50 configs. All defined as a literal list `sfgp_configs`.

#### Section 8 — SFGP: Run Sweep and Print Results Table (Code)

Run the 50-config sweep, accumulate into `sfgp_hp_df`. Print running progress. Display DataFrame.

#### Section 9 — SFGP: Best Configuration (Code)

Select `best_sfgp = sfgp_hp_df.loc[sfgp_hp_df['NLP'].idxmin()]`. Print label, MAE, NLP, Coverage.

#### Section 10 — SFGP: Best Config Prediction Plot (Code)

Re-run `sfgp_prequential_with_config(...)` for the best config with `return_details=True`; call `plot_prequential_results()`.

---

#### Section 11 — MFGP: Overview (Markdown)

Explain:
- MFGP uses BoTorch `MultiTaskGP`; two tasks: task 0 = LF (initial 10), task 1 = HF (weekly submissions)
- Inter-task correlation is the `rank`-dimensional ICM (Intrinsic Coregionalization Model)
- Step 0 fallback: when no HF training points are available, use `SingleTaskGP` on LF data only; this is applied uniformly across all MFGP configs
- Hyperparameter axes: kernel, rank, noise_lb, output_standardize

#### Section 12 — MFGP: Define Prequential Evaluation Function (Code)

Define `mfgp_prequential_with_config(X_all, y_all, n_init, config)`.

```python
for step in range(n_steps):
    t = step          # number of HF points available for training
    # t=0: fallback to SFGP on LF data
    # t>=1: build augmented X_train with task feature, fit MultiTaskGP
```

On successful fit: posterior at `(X_test_aug, task=1)`; extract mean and std.  
On exception: append NaN and continue.

#### Section 13 — MFGP: 50 Configuration Definitions (Code)

Define `mfgp_configs` list with 50 entries.

**Core 48**: `{kernel: [matern15, matern25, rbf]} × {rank: [1, 2]} × {noise_lb: [1e-6, 1e-5, 1e-4, 1e-3]} × {output_standardize: [True, False]}`  
= 3 × 2 × 4 × 2 = 48

**Extra 2**: `{'kernel_type': 'matern25', 'rank': 1, 'noise_lb': 5e-6, 'output_standardize': True}` and `{'kernel_type': 'matern25', 'rank': 1, 'noise_lb': 5e-5, 'output_standardize': True}`

Total: 50 configs. Defined as `mfgp_configs`.

#### Section 14 — MFGP: Default Run (Code)

Run default MFGP config (matern25, rank=1, noise_lb=1e-5, output_standardize=True). Call `plot_prequential_results()`.

#### Section 15 — MFGP: Run Sweep and Print Results Table (Code)

Run 50-config sweep, accumulate into `mfgp_hp_df`. Display DataFrame.

#### Section 16 — MFGP: Best Configuration (Code)

Select `best_mfgp = mfgp_hp_df.loc[mfgp_hp_df['NLP'].idxmin()]`. Print label, MAE, NLP, Coverage.

#### Section 17 — MFGP: Best Config Prediction Plot (Code)

Re-run best MFGP config with full detail; call `plot_prequential_results()`.

---

#### Section 18 — 2-Way Comparison: SFGP vs MFGP (Markdown + Code)

- Build 2-row `comparison_df` with Model, Configuration, MAE, NLP, Coverage_95
- Print metric winners (Best NLP: ..., Best MAE: ..., Best Coverage: ...)
- Print overall winner sentence

#### Section 19 — Visual Comparison Bar Chart (Code)

3-panel figure (MAE / NLP / Coverage), 2 bars per panel, same colour scheme (SFGP=#2196F3, MFGP=#FF9800), value labels. Title: `F2: Best SFGP vs Best MFGP — 2-Way Comparison`.

#### Section 20 — Hyperparameter Sensitivity: All 100 Configs (Code)

Horizontal bar chart of all 100 configurations, colour-coded by family, ranked by NLP. Title: `F2: All 100 Configurations — Hyperparameter Sensitivity`.

#### Section 21 — Full Ranked Results Table (Code)

Concatenate `sfgp_hp_df` and `mfgp_hp_df` with a `Model` column; sort by NLP; display.

#### Section 22 — Winner Detail Visualisation (Code)

Determine overall winner (`best_overall`). If tie: break by MAE, then coverage proximity. Print winner name prominently. Call `plot_prequential_results()` for the winner.

#### Section 23 — Conclusions (Markdown)

- State winning surrogate family and best configuration
- Summarise the 7-step NLP and MAE values
- Note MFGP step-0 fallback limitation
- State recommendation for BO pipeline

---

## Phase 1 Re-check: Constitution Gates

| Gate | Re-check Result |
|------|----------------|
| Simple code | ✅ Each section follows a clear, documented pattern |
| BoTorch default | ✅ Both models are native BoTorch |
| Hyperparameters explained | ✅ Each config list entry has a `label`; markdown cells explain each sweep axis |
| Visualisations | ✅ Sections 10, 17, 19, 20, 22 all produce figures |
| No new dependencies | ✅ `MultiTaskGP` is in BoTorch 0.16 (installed) |
