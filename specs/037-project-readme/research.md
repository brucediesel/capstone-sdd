# Research: Project README & Documentation Update

**Phase 0 output for feature 037-project-readme**

---

## R1: Strategy Evolution per Function (Weeks 3–13)

### Decision
Document each function's full strategy timeline from initial submission through Week 13, capturing surrogate model, acquisition function, output transforms, and key hyperparameter changes.

### Findings

#### F1 — Radiation Source Detection (2D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | Hurdle Model (Logistic + RF) | Weighted UCB (κ=3.0) | log1p transform, interior penalty S=0.1, local penalisation r=0.15, FALLBACK_MODE |
| Week 10 | SFGP Matérn-2.5 + qLogNEI | qLogNEI q=4 | Switched to GP; Hurdle lacked structured uncertainty |
| Weeks 11–13 | SFGP Matérn-2.5 + qLogNEI | qLogNEI q=1 (Week 13) | Exploitation focus; interior penalty removed Week 13 |
| **Best output** | **7.71 × 10⁻¹⁶** (essentially zero — source never located) | | |

#### F2 — Noisy Log-Likelihood (2D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Standardize transform, distance-based selection, noise_lb=1e-3 |
| Week 10 | SFGP Matérn-2.5 ARD | qLogNEI q=4 | Smoother kernel for well-behaved surface |
| Weeks 11–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Exploitation focus |
| **Best output** | **0.674** | | |

#### F3 — Negative-Valued 3D Function (3D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–7 | SFGP Matérn-2.5 ARD | qLogNEI q=1 | Standardize transform |
| Week 8 | SFGP Matérn-2.5 ARD | qLogNEI q=3 | Increased batch; output shifting to handle negative values |
| Week 9 | SFGP Matérn-2.5 ARD | qLogNEI q=3 | Standardize(m=1), 3-colour diagnostics |
| Weeks 10–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Tuning refinements; exploitation focus |
| **Best output** | **−0.0114** | | |

#### F4 — 4D Multi-Fidelity Function (4D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–9 | MFGP (Multi-Fidelity GP) | qLogNEI q=4 | Multiple fidelity levels; high-fidelity-only final selection |
| Week 10 | SFGP Matérn-2.5 ARD | qLogNEI q=4 | Switched from MFGP; 4D single-task GP simpler and competitive |
| Weeks 11–13 | SFGP Matérn-2.5 ARD | qLogNEI q=1 (Week 13) | Exploitation focus |
| **Best output** | **0.532** | | |

#### F5 — Chemical Process Yield (4D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | Standard BO; best ≈1,089 |
| Week 5 | GBT Ensemble (10 models) | UCB κ=2.5 | 20,000 candidates; departure from GP |
| Week 8 | SFGP Matérn-5/2 ARD | qLogNEI q=4 | log1p + z-score, additive interior penalty (S=1.0, F=0.01), distance-based selection; major jump to ~3,000+ |
| Week 9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Rougher kernel for small data; 5,000 raw samples |
| Weeks 10–12 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | MLL restarts 50→60; raw samples 5,000→8,000; distance gate relaxed to 25th percentile |
| Week 13 | SFGP Matérn-1.5 ARD | qLogNEI q=1 | Interior penalty removed; single greedy point |
| **Best output** | **8,662.4** | | |

#### F6 — Cake Recipe Optimisation (5D, All-Negative Outputs)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | best ≈−0.714 |
| Week 5 | Neural Network (5→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates |
| Week 8 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Rank-based interior penalty (sign-invariant for negative outputs); milk constraint ≥0.10 |
| Week 9 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Enhanced diagnostics; rank-based IP maintained |
| Weeks 10–13 | SFGP Matérn-1.5 ARD | qLogNEI q=4 | Noise floor 1e-2→1e-3; milk 0.10→0.12; raw samples 3,000→5,000 |
| **Best output** | **−0.111** (closer to zero is better) | | |

#### F7 — ML Hyperparameter Tuning (6D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | Standard BO |
| Week 5 | NN (6→128→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates; NN for 6D scalability |
| Week 8 | Compact NN (6→5→5→1) | EI via MC Dropout | Interior penalty (S=0.1, F=0.01); 200 epochs, lr=0.005 |
| Weeks 9–13 | Compact NN (6→5→5→1) | Blended: 0.7×mean + 0.3×EI | Dropout 0.1→0.05; relaxed IP (S=0.05, F=0.02); exploitation-focused |
| **Best output** | **2.305** | | |

#### F8 — 8D High-Dimensional Optimisation (8D)
| Period | Surrogate | Acquisition | Key Features |
|--------|-----------|-------------|--------------|
| Weeks 3–4 | SFGP Matérn-5/2 ARD | EI | 8D; 40 initial samples |
| Week 5 | NN (8→128→64→32→1) | UCB via MC Dropout κ=2.5 | 20,000 candidates |
| Week 8 | SFGP Matérn-2.5 ARD | qEI q=1, XI=0.01 | Returned to GP; 256 MC samples, 30 restarts, 4,096 raw; fallback to posterior mean |
| Weeks 9–13 | SFGP Matérn-2.5 ARD | qEI q=1, XI=0.01 | Stable strategy; 3-colour diagnostics added Week 9 |
| **Best output** | **9.982** | | |

### Alternatives Considered
- **Alternative surrogates explored in Weeks 5–7**: Gradient-boosted trees, polynomial response surfaces, neural networks, multi-fidelity GP — evaluated via prequential evaluation before settling on function-specific strategies.
- **Prequential evaluation** (research/Prequential Evaluation.ipynb): Systematic leave-future-out validation to compare surrogates before committing.

---

## R2: Final Dataset Sizes (Week 13)

### Decision
Use numpy-verified Week 13 dataset shapes as authoritative final sizes.

### Findings

| Function | Dimensionality | Initial Samples | Final Samples (Week 13) | Points Added |
|----------|---------------|----------------|------------------------|-------------|
| F1 | 2D | 10 | 23 | 13 |
| F2 | 2D | 10 | 23 | 13 |
| F3 | 3D | 15 | 28 | 13 |
| F4 | 4D | 30 | 43 | 13 |
| F5 | 4D | 20 | 33 | 13 |
| F6 | 5D | 20 | 33 | 13 |
| F7 | 6D | 30 | 43 | 13 |
| F8 | 8D | 40 | 53 | 13 |

**Submission pattern**: Week 3 added 3 points (first submission), then 1 point per week for Weeks 4–13
**Total per function**: 13 points added (3 + 10×1)

### Rationale
numpy `.shape[0]` on `updated_outputs - Week 13.npy` provides exact counts. All 8 functions received exactly 13 additional points over the campaign.

---

## R3: Current State of modelcards.md

### Decision
modelcards.md currently documents Week 9 configurations. Must be extended to cover full Weeks 3–13 evolution per FR-011.

### Findings
- **Current scope**: Week 9 final configuration per function
- **Structure**: Overview, Intended Use, Details, Performance, Assumptions and Limitations, Ethical Considerations
- **Key gap**: No strategy evolution history — only the Week 9 snapshot
- **Update approach**: Extend the "Details" section of each function's model card to include a strategy evolution timeline (Weeks 3–13), and update the "Performance" section to reflect Week 13 final results

### Alternatives Considered
- Option A: Add evolution as appendix to each card — rejected (breaks self-contained card structure)
- Option B: Extend "Details" section inline — **chosen** (maintains card readability)
- Option C: Create separate evolution document — rejected (FR-011 requires updates within modelcards.md)

---

## R4: Current State of datasheets.md

### Decision
datasheets.md currently documents Week 9 dataset sizes. Must be updated to Week 13 final sizes per FR-012.

### Findings
- **Current scope**: Week 9 final dataset sizes and collection process
- **Structure**: Motivation, Composition, Collection Process, Preprocessing and Uses, Distribution and Maintenance
- **Key gaps**: Dataset sizes reference 19 observations (F1/F2), collection timeline says "Weeks 3–9", file lists stop at Week 9
- **Update approach**: Update Composition (final sizes), Collection Process (Weeks 3–13), and Distribution (file lists through Week 13)

### Rationale
Minimal targeted updates to existing sections rather than restructuring. Preserves existing documentation quality while extending temporal coverage.

---

## R5: Convergence Plot References

### Decision
README references convergence plots in `functions/results/process_results.ipynb` by description and file path, without embedding images.

### Findings
- **Plot type**: 2×4 grid of convergence plots (one per function), showing running best-observed output over iterations
- **Notebook**: `functions/results/process_results.ipynb` — must be executed to generate plots
- **Referencing approach**: Describe plot content in README and provide notebook path for readers to render

### Rationale
Embedding images would require image export and path management. Referencing the notebook is more maintainable and aligns with the Jupyter-notebook-centric constitution.

---

## R6: README Structure Design

### Decision
README follows a linear structure: overview → per-function sections (strategy + results + evaluation) → lessons learnt → project structure → references.

### Findings
- Grouping strategy + results + evaluation per function (vs. separate sections) improves readability and reduces cross-referencing
- Summary table at the start of results provides quick overview before per-function detail
- Lessons learnt as a standalone section after per-function detail enables synthesis

### Alternatives Considered
- **Option A**: Separate sections for strategy, results, evaluation — rejected (forces readers to jump between sections for same function)
- **Option B**: Per-function grouped sections — **chosen** (self-contained per function)
- **Option C**: Tabular-only presentation — rejected (insufficient depth for critical evaluation)
