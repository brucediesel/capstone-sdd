# Tasks: Week 7 — SFGP and MFGP Prequential Evaluation on Function 2

**Branch**: `006-sfgp-mfgp-pe`  
**Input**: Design documents from `/specs/006-sfgp-mfgp-pe/`  
**Notebook**: `functions/f2/preq-eval-f2.ipynb`  
**No tests** — constitution prohibits unit tests; verification is end-to-end notebook execution

---

## Phase 1: Setup

**Purpose**: Clear the existing notebook content and establish the new title/imports baseline

- [X] T001 Clear all existing cells from `functions/f2/preq-eval-f2.ipynb` (GP, BART, RF sections and old overview)
- [X] T002 Write title and overview markdown cell: `Prequential Evaluation of Surrogate Models — Function 2 (Week 7: SFGP vs MFGP)`; update surrogate table (SFGP, MFGP), Total samples = 17, Evaluation steps = 7
- [X] T003 Write imports cell in `functions/f2/preq-eval-f2.ipynb`: keep existing botorch/gpytorch imports, add `from botorch.models import MultiTaskGP`, add `import os`, remove pymc/pymc_bart/sklearn imports; set `np.random.seed(42)` and `torch.manual_seed(42)`; print `'All imports successful.'`

**Checkpoint**: Notebook opens cleanly, all imports resolve, no BART/RF symbols present

---

## Phase 2: Foundational (Utilities — blocks all user stories)

**Purpose**: Define `compute_metrics()` and `plot_prequential_results()` which are called by every surrogate section; plus the data-loading cell. All US phases depend on this phase being complete.

**⚠️ CRITICAL**: No user story implementation can begin until this phase is complete.

- [X] T004 Write metrics markdown cell and define `compute_metrics(predictions, actuals, pred_means, pred_stds) → dict` in `functions/f2/preq-eval-f2.ipynb` — body identical to existing function; print `'compute_metrics() defined.'`
- [X] T005 [P] Write visualisation markdown cell and define `plot_prequential_results(results, model_name) → None` in `functions/f2/preq-eval-f2.ipynb` — body identical to existing function (3-panel: predictions vs actuals, absolute errors, NLP per step); print `'plot_prequential_results() defined.'`

**Checkpoint**: Both utility functions defined. Can be tested standalone with synthetic data before proceeding.

---

## Phase 3: User Story 1 — Load Week 7 Data and Confirm Prequential Setup (Priority: P1) 🎯 MVP

**Goal**: Load 17-sample f2 dataset, validate dimensions and output range, confirm prequential constants and fidelity split — gates all subsequent modelling.

**Independent Test**: Run cells T006 only. Verify printed output shows `X shape: (17, 2)`, `y shape: (17,)`, `Initial training points: 10`, `Evaluation steps: 7`, `LF observations: 10`, `HF observations: 7`, output range within [-0.1, 0.8], no NaN/inf.

- [X] T006 [US1] Write data-loading and validation cell in `functions/f2/preq-eval-f2.ipynb`: set `WEEK = 7`, `N_INIT = 10`, `LF_TASK = 0`, `HF_TASK = 1`; add `assert os.path.exists(...)` guards for both `.npy` paths; load and flatten `X_all` (17,2) and `y_all` (17,); compute `n_total`, `n_steps`; print full summary including output range, NaN/inf check, and fidelity split counts (per plan.md Section 2 / contract C-02)

**Checkpoint**: US1 passes its independent test. Data variables `X_all`, `y_all`, `N_INIT`, `n_steps`, `LF_TASK`, `HF_TASK` are all in scope.

---

## Phase 4: User Story 2 — Evaluate SFGP Across 50 Configurations (Priority: P1)

**Goal**: Implement SFGP prequential loop, define 50 hyperparameter configurations, run the full sweep, identify the best config by NLP, and visualise its predictions.

**Independent Test**: Run cells T007–T012. Verify SFGP default plot renders; 50-row `sfgp_hp_df` displays with ≤5 NaN rows; `best_sfgp` is printed with label, MAE, NLP, Coverage; best-config plot renders.

### Implementation for User Story 2

- [X] T007 [US2] Write SFGP overview markdown cell in `functions/f2/preq-eval-f2.ipynb`: explain SFGP = `SingleTaskGP` (BoTorch, single fidelity, analytic posterior), Matérn 5/2 ARD default; note hyperparameter axes (kernel, noise_lb, ARD, log-transform, input_normalize)
- [X] T008 [US2] Define `sfgp_prequential_evaluation(X_all, y_all, n_init)` in `functions/f2/preq-eval-f2.ipynb`: Matérn 5/2 ARD default, per-step print, call `compute_metrics()`; run it immediately with `sfgp_default_results = sfgp_prequential_evaluation(X_all, y_all, N_INIT)`; call `plot_prequential_results(sfgp_default_results, 'SFGP Default (Matern 5/2 ARD)')` (per plan.md Section 5 / contract C-05)
- [X] T009 [P] [US2] Define `sfgp_prequential_with_config(X_all, y_all, n_init, config)` in `functions/f2/preq-eval-f2.ipynb`: handle `kernel_type` in `{matern05, matern15, matern25, rbf}`, `ard` bool, `noise_lb` float, `log_transform` bool, `input_normalize` bool; wrap fit in try/except; return dict with `pred_means`, `pred_stds`, `actuals`, `metrics`; print `'sfgp_prequential_with_config() defined.'` (per plan.md Section 6 / contract C-06)
- [X] T010 [P] [US2] Define `sfgp_configs` list of exactly 50 dicts in `functions/f2/preq-eval-f2.ipynb`: Block A — 32 configs (4 kernels × 4 noise_lb × 2 ARD, log_transform=False, input_normalize=True); Block B — 18 configs (9 log-transform, 9 no-normalise as specified in plan.md Section 7); each dict has `label`, `kernel_type`, `noise_lb`, `ard`, `log_transform`, `input_normalize`; print `'50 SFGP configurations defined.'` (per contract C-07)
- [X] T011 [US2] Write and run SFGP 50-config sweep cell in `functions/f2/preq-eval-f2.ipynb`: iterate `sfgp_configs`, call `sfgp_prequential_with_config()`, catch failures as NaN, accumulate `sfgp_hp_results`; build `sfgp_hp_df = pd.DataFrame(sfgp_hp_results)` with columns `label`, `MAE`, `NLP`, `Coverage_95`; print running progress per config; display DataFrame (per plan.md Section 8 / contract C-08). Note: T009 and T010 must both be complete before this cell runs.
- [X] T012 [US2] Write best SFGP selection and visualisation cells in `functions/f2/preq-eval-f2.ipynb`: (a) selection cell — `best_sfgp_idx = sfgp_hp_df['NLP'].idxmin()`, `best_sfgp = sfgp_hp_df.loc[best_sfgp_idx]`, print label/MAE/NLP/Coverage; (b) re-run best config with full detail into `best_sfgp_results`; call `plot_prequential_results(best_sfgp_results, f'Best SFGP: {best_sfgp["label"]}')` (per plan.md Sections 9–10 / contracts C-09, C-10)

**Checkpoint**: US2 passes its independent test. Variables `sfgp_hp_df`, `best_sfgp`, `best_sfgp_results` are in scope.

---

## Phase 5: User Story 3 — Evaluate MFGP Across 50 Configurations (Priority: P1)

**Goal**: Implement MFGP prequential loop (with step-0 fallback), define 50 configurations, run the full sweep, identify the best MFGP config by NLP, and visualise its predictions.

**Independent Test**: Run cells T013–T018. Verify MFGP default plot renders; 50-row `mfgp_hp_df` displays with ≤10 NaN rows; `best_mfgp` is printed; best-config plot renders.

### Implementation for User Story 3

- [X] T013 [US3] Write MFGP overview markdown cell in `functions/f2/preq-eval-f2.ipynb`: explain `MultiTaskGP` (BoTorch ICM), two tasks (task 0 = LF indices 0–9, task 1 = HF indices 10–16), task-feature augmentation tensor format, step-0 fallback to `SingleTaskGP` on LF data, hyperparameter axes (kernel, rank, noise_lb, output_standardize)
- [X] T014 [US3] Define `mfgp_prequential_with_config(X_all, y_all, n_init, config)` in `functions/f2/preq-eval-f2.ipynb`: at `step=0` use `SingleTaskGP` fallback on LF data; at `step>=1` build task-augmented `X_train` (concatenate LF rows with task=0 and HF rows with task=1 per the tensor format in research.md Decision 3), fit `MultiTaskGP(X_train, y_train, task_feature=-1, rank=config['rank'])`, extract posterior at `X_test_aug` with `task=HF_TASK`; wrap each step in try/except appending NaN on failure; return full detail dict; print `'mfgp_prequential_with_config() defined.'` (per plan.md Section 12 / contract C-11)
- [X] T015 [P] [US3] Define `mfgp_configs` list of exactly 50 dicts in `functions/f2/preq-eval-f2.ipynb`: Core 48 = `{kernel: [matern15, matern25, rbf]} × {rank: [1,2]} × {noise_lb: [1e-6,1e-5,1e-4,1e-3]} × {output_standardize: [True,False]}`; Extra 2 = matern25/rank=1/noise_lb=5e-6/True and matern25/rank=1/noise_lb=5e-5/True; each dict has `label`, `kernel_type`, `rank`, `noise_lb`, `output_standardize`, `step0_fallback='lf_sfgp'`; print `'50 MFGP configurations defined.'` (per plan.md Section 13 / contract C-12)
- [X] T016 [US3] Write and run MFGP default-config cell in `functions/f2/preq-eval-f2.ipynb`: run `mfgp_default_results = mfgp_prequential_with_config(X_all, y_all, N_INIT, mfgp_configs[0])`; call `plot_prequential_results(mfgp_default_results, 'MFGP Default (Matern 5/2, rank=1)')` (per plan.md Section 14 / contract C-13). Note: T014 and T015 must both complete before this cell runs.
- [X] T017 [US3] Write and run MFGP 50-config sweep cell in `functions/f2/preq-eval-f2.ipynb`: iterate `mfgp_configs`, call `mfgp_prequential_with_config()`, catch total-failure configs as all-NaN row, accumulate `mfgp_hp_results`; build `mfgp_hp_df = pd.DataFrame(mfgp_hp_results)` with columns `label`, `MAE`, `NLP`, `Coverage_95`; print running progress; display DataFrame (per plan.md Section 15 / contract C-14)
- [X] T018 [US3] Write best MFGP selection and visualisation cells in `functions/f2/preq-eval-f2.ipynb`: (a) guard for all-NaN MFGP edge case — print warning if `mfgp_hp_df['NLP'].isna().all()`; else `best_mfgp_idx = mfgp_hp_df['NLP'].idxmin()`, `best_mfgp = mfgp_hp_df.loc[best_mfgp_idx]`, print label/MAE/NLP/Coverage; (b) re-run best config into `best_mfgp_results`; call `plot_prequential_results()` (per plan.md Sections 16–17 / contracts C-15, C-16)

**Checkpoint**: US3 passes its independent test. Variables `mfgp_hp_df`, `best_mfgp`, `best_mfgp_results` are in scope.

---

## Phase 6: User Story 4 — Head-to-Head Comparison and Winner Visualisation (Priority: P2)

**Goal**: Build the 2-row comparison table, render all three comparison charts, declare the overall winner, and show the winner's detailed prediction plot.

**Independent Test**: Run cells T019–T023 (requires Phase 4 + Phase 5 complete). Verify 2-row table displays, bar chart renders, sensitivity chart renders, winner is declared and plotted.

### Implementation for User Story 4

- [X] T019 [US4] Write comparison markdown cell and build `comparison_df` in `functions/f2/preq-eval-f2.ipynb`: 2-row `pd.DataFrame` with columns `Model`, `Configuration`, `MAE`, `NLP`, `Coverage_95`; determine metric-by-metric winners; determine `best_overall` (primary: NLP; tiebreak 1: MAE; tiebreak 2: coverage proximity to 0.95); print winner announcement sentence (per plan.md Section 18 / contract C-17)
- [X] T020 [P] [US4] Write 3-panel comparison bar chart cell in `functions/f2/preq-eval-f2.ipynb`: MAE / NLP / Coverage panels; 2 bars per panel (SFGP=#2196F3, MFGP=#FF9800); value labels on each bar; 0.95 reference line on Coverage panel; title `'F2: Best SFGP vs Best MFGP — 2-Way Comparison'` (per plan.md Section 19 / contract C-18)
- [X] T021 [P] [US4] Write 100-config horizontal sensitivity bar chart cell in `functions/f2/preq-eval-f2.ipynb`: concatenate `sfgp_hp_df` (blue) and `mfgp_hp_df` (orange) into one list of 100 rows; sort by NLP ascending; plot horizontal bars colour-coded by family; title `'F2: All 100 Configurations — Hyperparameter Sensitivity'` (per plan.md Section 20 / contract C-19)
- [X] T022 [P] [US4] Write full ranked results table cell in `functions/f2/preq-eval-f2.ipynb`: build `full_summary` by concatenating `sfgp_hp_df`/`mfgp_hp_df` with `Model` column added; sort by NLP; set 1-based rank as index; display as HTML DataFrame (per plan.md Section 21 / contract C-20)
- [X] T023 [US4] Write winner-detail visualisation cell in `functions/f2/preq-eval-f2.ipynb`: print prominent winner heading; dispatch to the correct evaluation function by checking whether the `best_overall` label is found in `sfgp_hp_df` (call `sfgp_prequential_with_config`) or `mfgp_hp_df` (call `mfgp_prequential_with_config`); re-run the winning config to get detailed results; call `plot_prequential_results()` with title including winner model name and config label (per plan.md Section 22 / contract C-21)

**Checkpoint**: US4 passes its independent test. All comparison figures render. `best_overall` identified and plotted.

---

## Phase 7: Polish & Conclusions

**Purpose**: Wrap up with written conclusions and validate end-to-end execution.

- [X] T024 Write conclusions markdown cell in `functions/f2/preq-eval-f2.ipynb`: state winning surrogate and best configuration; summarise 7-step MAE and NLP; note MFGP step-0 fallback limitation; note that 7 steps is a small sample for reliable coverage estimation; state recommendation for BO pipeline (per plan.md Section 23)
- [ ] T025 Execute all cells top-to-bottom on a clean kernel (`Kernel → Restart & Run All`) and confirm: all 7 phases run without unhandled exceptions; `sfgp_hp_df` has 50 rows (≤5 NaN); `mfgp_hp_df` has 50 rows (≤10 NaN); all plots render; winner is declared (per quickstart.md and SC-001–SC-008)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **blocks all user stories**
- **Phase 3 (US1 — Data)**: Depends on Phase 2
- **Phase 4 (US2 — SFGP)**: Depends on Phase 3 (needs `X_all`, `y_all`, `N_INIT`)
- **Phase 5 (US3 — MFGP)**: Depends on Phase 3 (needs same variables); **independent of Phase 4**
- **Phase 6 (US4 — Comparison)**: Depends on **both** Phase 4 and Phase 5
- **Phase 7 (Polish)**: Depends on Phase 6

### User Story Dependencies

- **US1 (P1)**: Depends only on Foundational — can start as soon as Phase 2 is complete
- **US2 (P1)**: Depends on US1 — start after T006 completes
- **US3 (P1)**: Depends on US1 — can run **in parallel with US2** (different cells)
- **US4 (P2)**: Depends on US2 **and** US3 — requires both sweeps complete

### Within Each User Story

- Markdown explanation cells (T007, T013) can be written simultaneously with function definitions (T009, T014) since they touch different cells
- Config list definitions (T010, T015) can be written simultaneously with their respective function definitions (T009, T014)
- All comparison visualisations (T020, T021, T022) depend only on `comparison_df` (T019) existing, and can be written concurrently

---

## Parallel Execution: US2 + US3 Simultaneously

Since US2 (SFGP) and US3 (MFGP) are independent after the foundational phase, their *code authoring* can proceed concurrently (though cells must be inserted into the notebook in the correct sequential order):

```
# After T006 (US1) completes:

Worker A (SFGP):                   Worker B (MFGP):
T007 SFGP markdown                 T013 MFGP markdown
T009 sfgp_prequential_with_config  T014 mfgp_prequential_with_config
T010 sfgp_configs (50 entries)     T015 mfgp_configs (50 entries)
T008 default run                   T016 default run
T011 50-config sweep               T017 50-config sweep
T012 best + plot                   T018 best + plot
          ↓                                  ↓
                 T019 Comparison (US4)
```

---

## Summary

| Phase | Tasks | User Story | Priority |
|-------|-------|-----------|----------|
| 1: Setup | T001–T003 | — | Blocker |
| 2: Foundational | T004–T005 | — | Blocker |
| 3: Data | T006 | US1 | P1 |
| 4: SFGP | T007–T012 | US2 | P1 |
| 5: MFGP | T013–T018 | US3 | P1 |
| 6: Comparison | T019–T023 | US4 | P2 |
| 7: Polish | T024–T025 | — | Final |
| **Total** | **25 tasks** | **4 user stories** | |

**Suggested MVP scope**: Complete Phases 1–4 (US1 + US2 only) to have a working SFGP prequential evaluation with all 50 configurations. The MFGP sweep (Phase 5) and comparison (Phase 6) can then be layered on top.
