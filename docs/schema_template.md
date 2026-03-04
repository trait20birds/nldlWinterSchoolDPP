# DPP/BOM Schema (Day 1)

## Dataset overview
- **Project:** CE-RISE DPP synthetic data feasibility with SynthCity
- **Row definition:** (e.g., one row = one BOM component/item)
- **Source:** Internal partner BOM-derived DPP subset (restricted)
- **Date prepared:** 
- **Prepared by:** 

## Column inventory

| Column | Type (numeric/categorical/binary/ordinal) | Nullable | Example | Allowed values / range | Sensitive? | Keep/Drop | Notes |
|---|---|---:|---|---|---:|---|---|
| product_id | categorical/id | no | a1b2... | hashed | yes | keep(hashed) | pseudonymized |
| component_type | categorical | no | battery_cell | set/list | no | keep | |
| material_main | categorical | yes | aluminum | set/list | no | keep | |
| mass_g | numeric | yes | 24.5 | >=0 | no | keep | standardized to grams |
| recycled_content_pct | numeric | yes | 35 | 0-100 | no | keep | |
| supplier_region | categorical | yes | EU | {EU,APAC,NA,Other} | yes-ish | keep (grouped) | fairness grouping |
| supplier_name | text | yes | PartnerX | n/a | yes | drop | direct identifier |
| repairability_bin | ordinal/label | yes | High | {Low,Medium,High} | no | keep (target) | downstream task |

## Hard constraints to validate later (Day 2/3)
- `*_pct` columns should be in [0, 100]
- `mass_g >= 0`
- categories normalized (no duplicates like "EU" vs "Europe")
- no raw partner names / direct identifiers in modeling table
- optional domain-specific rules: (add here)

## Fairness grouping choice
- **Primary group column:** (e.g., `supplier_region` or `supplier_tier`)
- Motivation: 

## Downstream task choice
- **Target column:** (e.g., `repairability_bin` / `recyclability_bin`)
- Why this target is useful for utility evaluation: 
