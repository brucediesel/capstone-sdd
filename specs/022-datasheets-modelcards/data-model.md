# Data Model: Datasheets & Model Cards

**Feature**: 022-datasheets-modelcards  
**Date**: 2026-03-05

## Overview

This feature produces two static markdown documents. There is no runtime data model, database, or API. The "data model" describes the document structure — the entities, their fields, and relationships that define the content schema for each document.

## Entities

### Function

The central entity. All model cards and datasheets are organised per function.

| Field | Type | Source | Values |
|-------|------|--------|--------|
| id | string | Fixed | F1–F8 |
| domain_name | string | Notebooks | Radiation Source Detection, Noisy Log-Likelihood, Drug Discovery, Warehouse Product Placement, Chemical Process Yield, Cake Recipe, ML Hyperparameter Tuning, 8D ML Hyperparameters |
| dimensions | integer | Data files | 2, 2, 3, 4, 4, 5, 6, 8 |
| initial_points | integer | Data files | 10, 10, 15, 30, 20, 20, 30, 40 |
| week9_points | integer | Data files | 19, 19, 24, 39, 29, 29, 39, 49 |
| output_sign | enum | Data files | zero-inflated, mixed, all-negative, mostly-negative, all-positive |
| output_min | float | Data files | Per function |
| output_max | float | Data files | Per function |
| input_bounds | range | Constitution | [0, 0.999999] per dimension |

### Model Card

One per function. Contains 6 required subsections.

| Subsection | Required Fields | Source |
|------------|----------------|--------|
| Overview | model_name, function_id, domain_name, model_type | Notebooks |
| Intended Use | optimisation_context, suitable_tasks, unsuitable_tasks | Spec |
| Details | final_surrogate (type, kernel, hyperparams), final_acquisition (type, q, MC_samples, params), selection_note (1 sentence), special_techniques | Week 9 notebooks |
| Performance | total_points, output_range, output_sign, evaluation_metrics (LOO MAE/RMSE, PE scores) | Data files + notebooks |
| Assumptions & Limitations | data_size_constraint, failure_modes (≥2), domain_challenges | Notebooks + research |
| Ethical Considerations | domain_impact, responsible_use_note, fairness_concerns | Domain knowledge |

### Datasheet

One per function. Contains 5 required subsections.

| Subsection | Required Fields | Source |
|------------|----------------|--------|
| Motivation | purpose, domain_importance, dataset_creators | Spec + notebooks |
| Composition | dimensions, input_bounds, initial_size, final_size, output_sign, output_range, growth_trajectory | Data files |
| Collection Process | initial_provision, acquisition_loop, total_rounds (10), points_per_round | Notebooks |
| Preprocessing & Uses | input_transforms, output_transforms, standardisation, downstream_use | Week 9 notebooks |
| Distribution & Maintenance | storage_path, file_format, naming_convention, static_status | Repository structure |

### Summary Table (×2)

Each document includes one summary comparison table.

| Table | Columns |
|-------|---------|
| Model Cards Summary | Function, Domain, Dims, Final Surrogate, Final Acquisition, Interior Penalty |
| Datasheets Summary | Function, Domain, Dims, Initial Size, Final Size, Output Sign |

## Relationships

```
Function (F1–F8)
  ├── has exactly 1 Model Card
  │     ├── Overview
  │     ├── Intended Use
  │     ├── Details
  │     ├── Performance
  │     ├── Assumptions & Limitations
  │     └── Ethical Considerations
  └── has exactly 1 Datasheet
        ├── Motivation
        ├── Composition
        ├── Collection Process
        ├── Preprocessing & Uses
        └── Distribution & Maintenance
```

## Validation Rules

- Every function (F1–F8) must have exactly one model card and one datasheet
- Composition.dimensions must match actual `.npy` input shape[1]
- Composition.initial_size must match `initial_inputs.npy` shape[0]
- Composition.final_size must match `updated_inputs - Week 9.npy` shape[0]
- Composition.output_range must match actual min/max from `updated_outputs - Week 9.npy`
- No subsection may contain placeholder text
- Details section must NOT contain weekly evolution table (per clarification)

## State Transitions

N/A — both documents are static, write-once artefacts. No state machine.
