# Model Card Template

This template defines the exact structure for each function's model card in `modelcards.md`.
Repeat this structure for F1 through F8.

---

```markdown
## F{N} — {Domain Name}

### Overview

**Approach**: {Model name/type}  
**Function**: F{N} — {Domain Name}  
**Dimensionality**: {N}D input  
**Type**: {Surrogate model category — e.g., Gaussian Process, Neural Network, Hurdle Model}

{1–2 sentence description of what the model does and why this approach was chosen for this function.}

### Intended Use

**Suitable for**:
- {Primary use case — surrogate-based optimisation of the black-box function}
- {Secondary use case — uncertainty-guided exploration of the input space}

**Not suitable for**:
- {Limitation — e.g., production deployment without validation against ground truth}
- {Limitation — e.g., extrapolation beyond [0, 0.999999] input bounds}

### Details

**Final Surrogate Model (Week 9)**:
- **Type**: {Exact model type with key hyperparameters}
- **Kernel/Architecture**: {Kernel type and parameters OR network architecture}
- **Key Hyperparameters**: {List of critical hyperparameters with values}

**Acquisition Function**:
- **Type**: {Exact acquisition function name}
- **Configuration**: {q value, MC samples, restarts, other parameters}

**Special Techniques**:
- {Technique 1 — e.g., interior penalty with S and F values}
- {Technique 2 — e.g., local penalisation, fallback strategy}

*This model was selected after systematic evaluation of alternative surrogates including {list of alternatives evaluated}.*

### Performance

- **Dataset size**: {N} observations ({initial} initial + {added} acquired)
- **Output range**: [{min}, {max}]
- **Output characteristics**: {sign description — e.g., all positive, zero-inflated}
- **LOO Surrogate Error**: MAE = {value}, RMSE = {value}
- **Best observed value**: {value}

### Assumptions and Limitations

1. **{Limitation 1 title}**: {Description}
2. **{Limitation 2 title}**: {Description}
{3+. Additional limitations as appropriate}

### Ethical Considerations

- **Domain context**: {How the function's real-world domain relates to responsible use}
- **Transparency**: {How documenting the model supports reproducibility}
- **Limitations awareness**: {What users should understand about surrogate approximation fidelity}
```

---

## Summary Table Contract

The document must end (or begin) with a comparison table:

```markdown
## Summary

| Function | Domain | Dims | Final Surrogate | Final Acquisition | Interior Penalty |
|----------|--------|------|-----------------|-------------------|------------------|
| F1 | {domain} | {d} | {surrogate} | {acquisition} | {Yes/No} |
| ... | ... | ... | ... | ... | ... |
| F8 | {domain} | {d} | {surrogate} | {acquisition} | {Yes/No} |
```
