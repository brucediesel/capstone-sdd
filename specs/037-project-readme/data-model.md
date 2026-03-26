# Data Model: Project README & Documentation Update

**Phase 1 output for feature 037-project-readme**

This feature produces Markdown documents, not code entities. The "data model" describes the document structure — sections, content types, and relationships between the three deliverable files.

---

## Entity: README.md (New)

### Sections

| Section | Content Type | Source | Notes |
|---------|-------------|--------|-------|
| Title & Badges | Static text | N/A | Project name, brief tagline |
| Project Overview | Prose | Spec FR-002 | Challenge description, 8 functions, 13-week timeline, BO approach |
| Strategy & Results (×8) | Per-function subsection | Research R1, R2 | Surrogate, acquisition, transforms, key changes, best output, convergence |
| Critical Evaluation (×8) | Per-function subsection | Research R1 | Strengths, weaknesses, evidence-based reasoning |
| Results Summary Table | Table | Research R1, R2 | All 8 functions: dimensionality, best output, improvement, convergence trend |
| Convergence Plots | Reference | FR-005 | Path to process_results.ipynb, description of 2×4 grid |
| Lessons Learnt | Prose (≥5 insights) | FR-007, SC-005 | Cross-function synthesis: surrogates, acquisition, transforms, dimensionality, adaptation |
| Project Structure | Tree diagram + descriptions | FR-008 | data/, functions/, specs/, research/ |
| References | Links | FR-010 | modelcards.md, datasheets.md, process_results.ipynb |

### Relationships
- References `modelcards.md` for detailed model documentation
- References `datasheets.md` for data provenance
- References `functions/results/process_results.ipynb` for convergence plots
- Content derived from weekly notebooks in `functions/f1/` through `functions/f8/`

### Validation Rules
- All 8 functions must appear (SC-002)
- Each function must have strategy + results + evaluation (SC-002, SC-004)
- Results must include quantitative data (SC-003)
- At least 5 lessons learnt (SC-005)
- All file references must use correct relative paths (SC-006)

---

## Entity: modelcards.md (Update)

### Current Structure (Week 9)
Each function has: Overview, Intended Use, Details, Performance, Assumptions and Limitations, Ethical Considerations

### Changes Required (FR-011)

| Section | Change | Detail |
|---------|--------|--------|
| Introduction paragraph | Update temporal scope | "Weeks 3–9" → "Weeks 3–13" |
| Details (×8) | Add strategy evolution | Insert timeline table showing surrogate/acquisition changes per period |
| Details (×8) | Update final model | Reflect Week 13 configuration (not Week 9) |
| Performance (×8) | Update metrics | Week 13 dataset size, output range, best observed value |

### Validation Rules
- All 8 functions updated (SC-002 by analogy)
- Evolution covers Weeks 3–13, not just final week
- Final Week 13 performance numbers match numpy-verified values from R2

---

## Entity: datasheets.md (Update)

### Current Structure (Week 9)
Each function has: Motivation, Composition, Collection Process, Preprocessing and Uses, Distribution and Maintenance

### Changes Required (FR-012)

| Section | Change | Detail |
|---------|--------|--------|
| Introduction paragraph | Update temporal scope | "Weeks 3–9" → "Weeks 3–13" |
| Composition (×8) | Update final sizes | Week 9 → Week 13 counts (from R2) |
| Composition (×8) | Update output range | Extend to Week 13 observed range |
| Collection Process (×8) | Update rounds | "Weeks 3–9" → "Weeks 3–13"; 8 rounds → 12 rounds |
| Collection Process (×8) | Update acquisition strategy | Reflect final Week 13 strategy if changed |
| Distribution (×8) | Extend file list | Add `updated_*- Week 10.npy` through `updated_*- Week 13.npy` |
| Distribution (×8) | Update status | Remove "no further updates planned" if already present |

### Validation Rules
- All 8 functions updated
- Final sizes match numpy-verified values: F1=23, F2=23, F3=28, F4=43, F5=33, F6=33, F7=43, F8=53
- File lists include all Week 3–13 entries

---

## Cross-Document Consistency

| Data Point | README.md | modelcards.md | datasheets.md |
|-----------|-----------|---------------|---------------|
| Best output per function | Results table | Performance section | N/A |
| Dataset sizes | Project overview / results | Performance section | Composition section |
| Surrogate model | Strategy summary | Details section | Collection Process (brief) |
| Temporal scope | "Weeks 3–13" | "Weeks 3–13" | "Weeks 3–13" |
| File paths | Project structure | N/A | Distribution section |
