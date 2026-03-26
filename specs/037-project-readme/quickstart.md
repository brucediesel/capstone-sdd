# Quickstart: Project README & Documentation Update

**Phase 1 output for feature 037-project-readme**

## Prerequisites

- Git branch `037-project-readme` checked out
- Access to all weekly notebooks in `functions/f1/` through `functions/f8/`
- Access to `functions/results/process_results.ipynb` (for convergence plot reference)
- Access to existing `modelcards.md` and `datasheets.md` at project root

## Implementation Order

### Step 1: Create README.md

Create `README.md` in the project root. Structure:

1. **Title & Overview** — project purpose, 8 functions, 13-week BO campaign
2. **Per-function sections (F1–F8)** — each containing:
   - Strategy summary (surrogate, acquisition, transforms, key changes)
   - Results (best output, convergence, improving iterations)
   - Critical evaluation (strengths, weaknesses, evidence)
3. **Results summary table** — all 8 functions at a glance
4. **Convergence plots reference** — link to `functions/results/process_results.ipynb`
5. **Lessons learnt** — ≥5 cross-function insights
6. **Project structure** — repository layout
7. **References** — links to modelcards.md, datasheets.md

**Source data**: `specs/037-project-readme/research.md` (strategy timelines and dataset sizes)

### Step 2: Update modelcards.md

For each of the 8 functions:

1. Update the introduction paragraph: "Weeks 3–9" → "Weeks 3–13"
2. In the **Details** section, add a "Strategy Evolution" subsection with a timeline table
3. Update the **final model** description to reflect Week 13 configuration
4. Update the **Performance** section with Week 13 dataset size and best output

### Step 3: Update datasheets.md

For each of the 8 functions:

1. Update the introduction paragraph: "Weeks 3–9" → "Weeks 3–13"
2. Update **Composition** with final (Week 13) dataset sizes
3. Update **Collection Process** with extended timeline and final strategy
4. Update **Distribution** file list to include Week 10–13 files

## Key Data Sources

| Data Point | Source |
|-----------|--------|
| Strategy evolution (F1–F8) | [research.md](research.md) section R1 |
| Final dataset sizes | [research.md](research.md) section R2 |
| Document structure | [data-model.md](data-model.md) |
| Convergence plots | `functions/results/process_results.ipynb` |
| Current modelcards | `modelcards.md` (root) |
| Current datasheets | `datasheets.md` (root) |

## Validation

After implementation, verify:

- [ ] README.md exists at project root
- [ ] All 8 functions documented in README (SC-002)
- [ ] Quantitative results for every function (SC-003)
- [ ] ≥1 strength + ≥1 weakness per function evaluation (SC-004)
- [ ] ≥5 lessons learnt (SC-005)
- [ ] Correct file references (SC-006)
- [ ] modelcards.md reflects Weeks 3–13 evolution (FR-011)
- [ ] datasheets.md reflects Week 13 sizes (FR-012)
- [ ] Cross-document consistency (best outputs, sizes, strategies match)
