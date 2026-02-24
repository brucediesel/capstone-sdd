# Tasks: F7 Week 7 — NN Surrogate with NEI & Interior Penalty

**Input**: Design documents from `/specs/017-f7-nn-interior-penalty/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cells.md, quickstart.md

**Tests**: No tests (per constitution — no unit tests required; manual execution only).

**Organization**: Tasks grouped by user story. All tasks target a single file: `functions/f7/f7.ipynb` (append cells 50–57 after existing cell 49).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different cells, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1, US2, US3)
- All paths relative to repository root

---

## Phase 1: Setup

**Purpose**: Verify environment and existing notebook state before adding new cells.

- [ ] T001 Verify Python 3.11+ environment is active with torch, numpy, matplotlib available
- [ ] T002 Verify `functions/f7/f7.ipynb` has 49 cells and existing Week 5–6 cells execute without errors

**Checkpoint**: Notebook runs end-to-end; all Week 6 variables are populated. Data files `data/f7/updated_inputs - Week 7.npy` and `data/f7/updated_outputs - Week 7.npy` exist.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data loading and normalisation that ALL user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Append code cell 51 — load Week 7 data from `data/f7/updated_inputs - Week 7.npy` and `data/f7/updated_outputs - Week 7.npy`, z-score normalise inputs and outputs, create torch tensors, print summary (37 samples × 6 dims, output range, best observed), assert shapes and all-positive outputs in `functions/f7/f7.ipynb`

**Checkpoint**: `X_raw` (37, 6), `y_raw` (37,), `X_norm`, `y_norm`, `X_tensor`, `y_tensor`, normalisation stats all available in kernel.

---

## Phase 3: User Story 1 — Train NN Surrogate & Propose Sample via NEI + Interior Penalty (Priority: P1) 🎯 MVP

**Goal**: Train a compact 6→5→5→1 neural network, compute MC Dropout EI with multiplicative interior penalty, and produce a submission-ready query point.

**Independent Test**: Run all cells top-to-bottom. Submission query printed in `x1-x2-...-x6` format (6 decimal places, all in [0, 0.999999]). Interior penalty effect reported (whether selection changed vs raw EI).

### Implementation for User Story 1

- [ ] T004 [US1] Append code cell 52 — define `SurrogateNN(nn.Module)` class (6→5→5→1, ReLU, Dropout(0.1)), train with Adam (lr=0.005), MSE loss, 200 epochs, print progress every 40 epochs, plot training loss curve (log scale), compute and print training R² on original scale in `functions/f7/f7.ipynb`
- [ ] T005 [US1] Append code cell 53 — generate 20,000 random candidates in [0,1]⁶, run 50 MC Dropout forward passes, un-normalise predictions, compute EI via sample-then-average (`mean(max(f_i - y_best, 0))`), compute interior penalty `w(x) = 0.01 + 0.99·∏sin(πxᵢ)²`, multiply `penalised_ei = ei * w`, select best candidate (fallback to argmax interior_weight if all EI=0), print comparison table and penalty effect in `functions/f7/f7.ipynb`
- [ ] T006 [US1] Append code cell 57 — clip `best_point` to [0, 0.999999], format as `x1-x2-...-x6` with 6 decimal places, validate format (6 parts, each in [0, 0.999999]), print submission query with hyperparameter metadata and per-dimension coordinates in `functions/f7/f7.ipynb`

**Checkpoint**: Cells 52, 53, 57 execute without errors. `best_point` shape is (6,), all coordinates in [0, 0.999999], penalty effect reported, submission query printed.

---

## Phase 4: User Story 2 — Document Hyperparameters with Rationale (Priority: P2)

**Goal**: Provide a markdown cell listing all hyperparameters (architecture, lr, dropout, epochs, MC samples, STEEPNESS, FLOOR) with rationale for each choice, satisfying capstone examiner requirements.

**Independent Test**: Read cell 50 markdown — hyperparameter table present with 8 rows (Parameter, Value, Rationale).

### Implementation for User Story 2

- [ ] T007 [P] [US2] Append markdown cell 50 — section header "## Week 7 — Neural Network + NEI with Interior Penalty" with approach explanation and hyperparameter table (architecture 6→5→5→1, lr=0.005, dropout=0.1, epochs=200, MC samples=50, candidates=20k, STEEPNESS=1.0, FLOOR=0.01) in `functions/f7/f7.ipynb`

> **Note**: Cell 50 is the first new cell in notebook order (markdown header before code cells). T007 is marked [P] because it has no code dependencies, but must be appended first in notebook cell order.

**Checkpoint**: Markdown cell with complete hyperparameter documentation visible above all code cells.

---

## Phase 5: User Story 3 — Visualise Surrogate, Penalty, and Convergence (Priority: P2)

**Goal**: Produce the same visualisation style as Weeks 5–6: 3-panel surrogate plot (NN mean, MC uncertainty, interior penalty heatmap) and convergence plot, projected onto the two most important input dimensions.

**Independent Test**: 3-panel figure renders with distinct heatmaps. Convergence plot shows running best across 37 observations with IP-selected predicted mean indicated.

### Implementation for User Story 3

- [ ] T008 [US3] Append code cell 54 — compute gradient-based feature importance (mean absolute gradient per input dim), normalise to sum to 1, identify top-2 dimensions, print importance with bar indicators in `functions/f7/f7.ipynb`
- [ ] T009 [US3] Append code cell 55 — build 50×50 grid on top-2 dimensions (other dims fixed at best-observed values), run MC Dropout on grid, compute interior penalty on grid, plot 3-panel figure: Panel 1 (NN mean, viridis), Panel 2 (MC uncertainty, YlOrRd), Panel 3 (interior penalty, RdYlGn) with observed data scatter and best_point star in `functions/f7/f7.ipynb`
- [ ] T010 [US3] Append code cell 56 — convergence plot with running best across 37 observations, IP-selected candidate predicted mean horizontal line, raw-EI best candidate dashed line, weekly data boundary markers in `functions/f7/f7.ipynb`

**Checkpoint**: Cells 54–56 execute without errors. 3-panel figure and convergence plot render correctly. Feature importance identifies top-2 dimensions for slice projection.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and commit.

- [ ] T011 Run all notebook cells top-to-bottom (cells 0–57) and verify no errors, all assertions pass, all plots render
- [ ] T012 Verify cells 0–49 are unmodified (diff check against master — only new cells appended)
- [ ] T013 Run quickstart.md verification checklist (11 items) against notebook output

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify environment first
- **Foundational (Phase 2)**: Depends on Setup — data loading blocks everything
- **US1 (Phase 3)**: Depends on Foundational — core NN + acquisition
- **US2 (Phase 4)**: No code dependencies — markdown only, but appended first in notebook cell order
- **US3 (Phase 5)**: Depends on US1 (needs `model`, `best_point`, `penalised_ei`, `interior_weight`, `mu`, `sigma` from cells 52–53)
- **Polish (Phase 6)**: Depends on all user stories complete

### Task Dependencies

```text
T001 ──► T002 ──► T007 ──► T003 ──► T004 ──► T005 ──► T008 ──► T009 ──► T010 ──► T006
                                                                                    │
                                                                                    ▼
                                                                          T011 ──► T012 ──► T013
```

### Cell Ordering Constraints

All cells are appended sequentially to one notebook — no parallelism within the notebook file. Notebook cell order:

| Order | Cell | Task | Content |
|-------|------|------|---------|
| 1 | 50 | T007 | Markdown header + hyperparameter table |
| 2 | 51 | T003 | Load data & normalise |
| 3 | 52 | T004 | Define & train NN |
| 4 | 53 | T005 | MC Dropout EI + interior penalty |
| 5 | 54 | T008 | Feature importance via gradients |
| 6 | 55 | T009 | 3-panel visualisation |
| 7 | 56 | T010 | Convergence plot |
| 8 | 57 | T006 | Submission query |

---

## Parallel Example: User Story 1

```text
# No parallel tasks — all cells append to one file sequentially.
# Execute in notebook cell order: T007 → T003 → T004 → T005 → T008 → T009 → T010 → T006
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup verification (T001–T002)
2. Complete Phase 4: US2 markdown header (T007) — appended first in notebook
3. Complete Phase 2: Foundational data loading (T003)
4. Complete Phase 3: US1 core NN + penalty (T004–T005, T006) — cells 52, 53, 57
5. **STOP and VALIDATE**: Submission query produced, penalty effect reported, feasibility passes
6. This is a functional submission-ready notebook

### Incremental Delivery

1. Setup → verify environment and notebook state
2. US2 (cell 50) + Foundational (cell 51) → documentation and data ready
3. US1 (cells 52–53, 57) → core computation, submission-ready
4. US3 (cells 54–56) → visualisation and convergence, examiner-ready
5. Polish → final validation against quickstart checklist (11 items)

---

## Notes

- All tasks modify a single file: `functions/f7/f7.ipynb`
- No new files created, no new dependencies added (PyTorch, NumPy, Matplotlib already available)
- Cell ordering in notebook: 50 (md), 51–56 (code), 57 (code)
- US2 has one dedicated task (T007 — markdown header) which is appended first in notebook order
- Multiplicative penalty `EI × w(x)` is correct because all F7 outputs are positive → EI ≥ 0
- Fallback logic in T005: if all penalised_EI = 0, select argmax(interior_weight) for exploration
- Total: 13 tasks across 6 phases
- Key hyperparameters: 6→5→5→1 (71 params), lr=0.005, dropout=0.1, 200 epochs, 50 MC samples, 20k candidates, STEEPNESS=1.0, FLOOR=0.01
