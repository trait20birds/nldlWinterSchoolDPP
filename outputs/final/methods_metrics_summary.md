# Methods + Metrics Summary (Day 3 One-Pager)

## Objective
Generate and evaluate synthetic DPP/BOM component data for safe benchmarking when real data is constrained.

## Data and setup
- Real cleaned table: `outputs/day1/cleaned_synthcity_input.csv`
- Rows/cols: `200 x 8`
- Target: `repairability_bin`
- Fairness group: `supplier_region`
- Split: 80/20 stratified (`seed=42`) -> train `160`, test `40`

## Generators compared
- `ctgan`
- `tvae`
- `privbayes`

## Evaluation blocks
1. Validation:
   - domain constraints, invalid categories, missingness, duplicates, exact row overlap
2. Utility:
   - TRTR baseline on real
   - TSTR synthetic-to-real performance
   - distribution/correlation similarity
3. Privacy:
   - exact overlap rate
   - synthetic-to-real nearest-neighbor distance
4. Fairness:
   - group proportion distortion
   - group-wise TSTR macro-F1 gap

## Core results
- TRTR macro-F1 (real baseline): `0.9623`
- TSTR macro-F1:
  - TVAE: `0.5319`
  - CTGAN: `0.4757`
  - PrivBayes: `0.3736`
- Optional TVAE 5x TSTR macro-F1: `0.5365`
- Validity violations: `0.0000` for all compared models
- Exact overlap with real-train rows: `0.0000` for all compared models

## Decision
- Selected release model: **TVAE**
- Release dataset: `outputs/final/synthetic_dpp.csv`
- Backup model: **CTGAN**

## Limitation to highlight
Synthetic utility is meaningfully below real-data baseline (TVAE delta `-0.4304` macro-F1 vs TRTR).  
This release is appropriate for prototyping/benchmarking, not production-grade model replacement.
