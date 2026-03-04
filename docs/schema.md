# DPP/BOM Schema (Day 1)

## Dataset Summary
- `row_definition`: one row = one BOM component/item
- `real_seed_file`: `data/seed_dpp_real.csv` (internal/raw style)
- `modeling_seed_file`: `data/seed_dpp_modeling.csv` (clean flat table for modeling/SynthCity)
- `rows`: 200
- `columns`: 8
- `target_task`: predict `repairability_bin` (`Low` / `Medium` / `High`)
- `fairness_grouping`: `supplier_region`

## Removed/Generalized Sensitive or Non-Modeling Fields
- `supplier_name`: removed from modeling table (direct partner identifier).
- `product_id`, `component_id`: removed from SynthCity/modeling-ready table (identifier-like and high-cardinality).

## Column Definitions and Constraints
| column | type | nullable | allowed/range | sensitive | keep/drop reason |
|---|---|---|---|---|---|
| `component_type` | categorical | no | `{cell, connector, housing, pcb}` | no | kept as core structural BOM feature |
| `material_main` | categorical | no | `{aluminum, copper, glass, plastic, steel}` | no | kept as key material driver |
| `mass_g` | numeric | no | min `3.18`, max `96.08`, expected `>0` | no | kept as physical quantitative feature |
| `recycled_content_pct` | numeric | no | min `0.0`, max `97.4`, expected `[0, 100]` | no | kept as circularity indicator |
| `supplier_region` | categorical | no | `{EU, APAC, NA, Other}` | low | kept for risk/fairness grouping |
| `compliance_rohs` | binary | no | `{0,1}` | no | kept as compliance signal |
| `contains_hazardous_substance` | binary | no | `{0,1}` | medium | kept as risk/circularity signal |
| `repairability_bin` | ordinal target | no | `{Low, Medium, High}` | no | downstream prediction label |

## Data Quality Notes (Day 1)
- Missingness in modeling table: `0%` across all columns.
- Duplicate rows removed during cleaning: `0`.
- Unit consistency: `mass_g` in grams, percentages in `[0,100]`.
- Category normalization applied for string fields and yes/no coercion to 0/1 where relevant.

## Validation Rules for Day 2/Day 3
- `mass_g > 0`
- `0 <= recycled_content_pct <= 100`
- `compliance_rohs ∈ {0,1}`
- `contains_hazardous_substance ∈ {0,1}`
- `supplier_region ∈ {EU, APAC, NA, Other}`
- `repairability_bin ∈ {Low, Medium, High}`

