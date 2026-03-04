# Synthetic DPP Release Card

## 1. Dataset purpose
This synthetic dataset is generated from a cleaned BOM/DPP component table to support DPP research prototyping, model benchmarking, and pipeline testing when real DPP data cannot be shared directly.

## 2. Source data summary
- Origin: prepared CE-RISE-style workshop subset (`data/seed_dpp_modeling.csv`) derived from Day 1 cleaned data.
- Row definition: one row = one BOM component/item.
- Real rows used for generator training: 160 (`data/real_train.csv`).
- Real columns used: 8.
- Sensitive/non-release fields removed before synthesis: `supplier_name`, `product_id`, `component_id`.

## 3. Schema summary
- Numeric fields: `mass_g`, `recycled_content_pct`.
- Categorical fields: `component_type`, `material_main`, `supplier_region`.
- Binary fields: `compliance_rohs`, `contains_hazardous_substance`.
- Target label used during evaluation: `repairability_bin` (`Low`, `Medium`, `High`).
- Fairness grouping: `supplier_region`.
- Constraint checks applied:
  - `mass_g > 0`
  - `0 <= recycled_content_pct <= 100`
  - binary flags in `{0,1}`
  - category values constrained to observed real categories

## 4. Synthetic generation method(s)
- Tool: SynthCity.
- Candidate generators evaluated: `ctgan`, `tvae`, `privbayes`.
- Selected release generator: `tvae`.
- Training/evaluation split: 80/20 stratified split from Day 1 cleaned table (`seed=42`).
- Release file: `outputs/final/synthetic_dpp.csv` (TVAE 1x, 160 rows).
- Additional non-canonical augmentation file: `outputs/final/synthetic_dpp_5x.csv`.
- Optional Day 3 check: TVAE 5x TSTR macro-F1 `0.5365` (slight gain over 1x `0.5319`).

## 5. Utility evaluation
- Real baseline (TRTR): Accuracy `0.9750`, Macro-F1 `0.9623`.
- Release model (TVAE) TSTR: Accuracy `0.5250`, Macro-F1 `0.5319`.
- TVAE 5x optional TSTR: Accuracy `0.6500`, Macro-F1 `0.5365`.
- Explicit utility gap: TVAE macro-F1 is `-0.4304` below TRTR.
- Distribution/correlation checks (TVAE):
  - Numeric correlation MAE: `0.0848`
  - Supplier region TVD: `0.0813`
  - See `outputs/final/utility_stats_tvae.json` and `outputs/final/tvae_numeric_overlays.png`.

## 6. Privacy evaluation
- Exact synthetic-to-real row overlap: `0.0000`.
- Duplicate synthetic row rate: `0.0000`.
- Suspiciously close NN rate (vs real-real 1st percentile threshold): `0.0750`.
- Rare-combination leakage indicator: `0.6875` (watchlist metric; no exact row copying observed).
- See `outputs/final/privacy_summary_tvae.json` and `outputs/final/tvae_nn_distance.png`.

## 7. Fairness evaluation
- Group variable: `supplier_region`.
- Group proportion distortion (TVD): `0.0813`.
- Group-wise TSTR macro-F1 gap: `0.4286` (max - min across groups).
- See `outputs/final/fairness_summary_tvae.json` and `outputs/final/group_metrics_tvae.csv`.

## 8. Intended use
- Research benchmarking of tabular synthetic-data workflows for DPP.
- Pipeline and tooling validation without exposing original records.
- Demonstration dataset for workshops, presentations, and method comparisons.

## 9. Out-of-scope / prohibited use
- Real supplier or product decision-making.
- Regulatory/compliance claims without validation on real governed data.
- Re-identification attempts or any reverse-engineering effort against source data.

## 10. Limitations
- Utility is materially below real-data baseline for downstream label prediction.
- Flattened component-level table does not represent full hierarchical BOM relations.
- Fairness findings are limited by small pilot data volume and group coverage.
- This release should be treated as a pilot synthetic benchmark, not a production-quality replacement for real data.

## 11. Reproducibility artifacts
- Run context: `outputs/final/run_context.json`
- Day 2 run log: `outputs/final/synthcity_run_log.json`
- Comparison table: `outputs/final/model_comparison_table.csv`
- Day 2 report: `outputs/final/day2_report.md`
- Final project report: `outputs/final/final_project_report.md`
- Methods summary: `outputs/final/methods_metrics_summary.md`
