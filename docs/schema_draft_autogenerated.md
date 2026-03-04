# Auto-generated Day 1 Schema Draft
## Dataset overview
- Original shape: {'rows': 200, 'cols': 11}
- Cleaned shape: {'rows': 200, 'cols': 10}
- Duplicates removed: 0
- Target column: repairability_bin
- Group column: supplier_region

## Columns
| Column | Dtype | Inferred role | Missing % | Unique (non-null) | Target? | Group? |
|---|---|---|---:|---:|---:|---:|
| product_id | object | text_or_id | 0.00 | 50 | False | False |
| component_id | object | text_or_id | 0.00 | 200 | False | False |
| component_type | string | categorical | 0.00 | 4 | False | False |
| material_main | string | categorical | 0.00 | 5 | False | False |
| mass_g | float64 | numeric | 0.00 | 196 | False | False |
| recycled_content_pct | float64 | numeric | 0.00 | 154 | False | False |
| supplier_region | string | categorical | 0.00 | 4 | False | True |
| compliance_rohs | int64 | binary | 0.00 | 2 | False | False |
| contains_hazardous_substance | int64 | binary | 0.00 | 2 | False | False |
| repairability_bin | string | categorical | 0.00 | 3 | True | False |
