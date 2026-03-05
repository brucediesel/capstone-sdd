# Quickstart: Datasheets & Model Cards

**Feature**: 022-datasheets-modelcards  
**Goal**: Create `modelcards.md` and `datasheets.md` in the project root

## Prerequisites

- Branch `022-datasheets-modelcards` checked out
- All Week 9 notebooks (f1–f8) merged to main (feature 021 complete)
- All data files present in `data/f1/` through `data/f8/`

## Implementation Steps

### Step 1: Create `modelcards.md`

1. Create file at project root: `modelcards.md`
2. Add document title and introduction (1–2 paragraphs explaining what model cards are and the project context)
3. For each function F1–F8, add a section following the template in [contracts/modelcard-template.md](contracts/modelcard-template.md)
4. Populate each section using data from [research.md](research.md) — the R4 section has all per-function details
5. Add the summary comparison table (FR-019)

**Data sources per subsection**:
- Overview: Function domain names from research.md R4
- Intended Use: Standard BO context (same for all, with function-specific nuances)
- Details: Week 9 model config from research.md R4 (NOT historical evolution)
- Performance: Output ranges from research.md R3, LOO metrics from R4
- Assumptions & Limitations: Function-specific challenges from R4
- Ethical Considerations: Domain-appropriate at capstone level (R6)

### Step 2: Create `datasheets.md`

1. Create file at project root: `datasheets.md`
2. Add document title and introduction
3. For each function F1–F8, add a section following the template in [contracts/datasheet-template.md](contracts/datasheet-template.md)
4. Populate each section using data from [research.md](research.md)
5. Add the summary comparison table (FR-020)

**Data sources per subsection**:
- Motivation: Domain names from R4, standard BO challenge context
- Composition: Verified shapes/ranges from R3
- Collection Process: Standard BO loop (same structure, function-specific point counts)
- Preprocessing & Uses: Output transforms from R4
- Distribution & Maintenance: Repository structure (consistent across all functions)

### Step 3: Validate

1. Verify both files render correctly in markdown viewer
2. Count sections: 8 functions × 6 subsections = 48 model card subsections
3. Count sections: 8 functions × 5 subsections = 40 datasheet subsections
4. Spot-check data accuracy against `.npy` files (dimensions, sizes, ranges)
5. Verify no placeholder text remains
6. Verify summary tables are complete (8 rows each)

## Key Reference Data

| Fn | Domain | Dims | Init | W9 | Output Range | Surrogate | Acquisition |
|----|--------|------|------|----|-------------|-----------|-------------|
| F1 | Radiation Source | 2 | 10 | 19 | [−3.6e-3, 0] | Hurdle (LR+RF) | Weighted UCB κ=3 |
| F2 | Log-Likelihood | 2 | 10 | 19 | [−0.07, 0.67] | SFGP Matérn-1.5 | qLogNEI q=4 |
| F3 | Drug Discovery | 3 | 15 | 24 | [−0.40, −0.03] | SFGP Matérn-2.5 | qLogNEI |
| F4 | Warehouse | 4 | 30 | 39 | [−32.6, 0.53] | MFGP Matérn-5/2 | qLogNEI q=4 |
| F5 | Chemical Yield | 4 | 20 | 29 | [0.11, 3395] | SFGP Matérn-5/2 | qLogNEI q=4 + IP |
| F6 | Cake Recipe | 5 | 20 | 29 | [−2.57, −0.11] | SFGP Matérn-1.5 | qLogNEI q=4 |
| F7 | ML Hyperparams | 6 | 30 | 39 | [0.003, 2.30] | NN 6→5→5→1 | MC Dropout EI |
| F8 | 8D ML Hyperparams | 8 | 40 | 49 | [5.59, 9.98] | SFGP Matérn-2.5 | qEI q=1 |

## Estimated Effort

- `modelcards.md`: ~400–500 lines of markdown
- `datasheets.md`: ~350–450 lines of markdown
- Total: 2 files, ~800–950 lines, pure documentation (no code)
