# Datasheet Template

This template defines the exact structure for each function's datasheet in `datasheets.md`.
Repeat this structure for F1 through F8.

---

```markdown
## F{N} — {Domain Name}

### Motivation

- **Purpose**: {Why the dataset was created — black-box optimisation challenge context}
- **Domain**: {The function's real-world domain and why it matters}
- **Creators**: {Who produced the data — challenge organisers (initial) + student optimisation pipeline (acquired) + oracle (evaluation)}
- **Funding/Context**: {Capstone project for certificate course in AI and ML}

### Composition

- **Input dimensionality**: {N} features
- **Input bounds**: [0, 0.999999] per dimension (normalised)
- **Instances**:
  - Initial: {N} observations
  - Final (Week 9): {N} observations
  - Growth: {N} points added across 9 weekly submissions
- **Output characteristics**:
  - Sign: {all positive / all negative / mixed / zero-inflated}
  - Range: [{min}, {max}]
  - Special properties: {e.g., zero-inflated, heavy-tailed, narrow range}
- **Missing data**: None — all queries received valid responses
- **Confidentiality**: No personally identifiable information

### Collection Process

- **Initial data**: {N} pre-generated observations provided by challenge organisers
- **Acquisition strategy**: Surrogate model trained on available data → acquisition function optimised → candidate input(s) submitted → oracle returned true function value(s)
- **Rounds**: 10 total (initial provision + Weeks 3–9)
- **Points per round**: {typical number — e.g., 1, or up to 4 with q-batch}
- **Time frame**: {Duration — approximately 7 weeks of weekly submissions}
- **Oracle**: Black-box function evaluator provided by course organisers (no access to function internals)

### Preprocessing and Uses

- **Input preprocessing**: {Description — e.g., none, constant fidelity column appended}
- **Output preprocessing**: {Description — e.g., log1p transform, manual z-score, Standardize(m=1)}
- **Intended use**: Training surrogate models for Bayesian optimisation
- **Inappropriate uses**:
  - Direct use as ground truth for the real-world domain (data represents a synthetic challenge function)
  - Extrapolation beyond the [0, 0.999999] input domain

### Distribution and Maintenance

- **Storage**: `data/f{N}/` directory in the project repository
- **Format**: NumPy `.npy` files (binary array format)
- **Files**:
  - `initial_inputs.npy` — initial input observations ({shape})
  - `initial_outputs.npy` — initial output observations ({shape})
  - `updated_inputs - Week {W}.npy` — cumulative inputs through Week {W}
  - `updated_outputs - Week {W}.npy` — cumulative outputs through Week {W}
- **Status**: Static — data collection is complete, no further updates planned
- **Licence/Terms**: Academic use within capstone course
- **Maintainer**: Student (project author)
```

---

## Summary Table Contract

The document must end (or begin) with a comparison table:

```markdown
## Summary

| Function | Domain | Dims | Initial Size | Final Size | Output Sign |
|----------|--------|------|--------------|------------|-------------|
| F1 | {domain} | {d} | {init} | {final} | {sign} |
| ... | ... | ... | ... | ... | ... |
| F8 | {domain} | {d} | {init} | {final} | {sign} |
```
