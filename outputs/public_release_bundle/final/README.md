# Final Bundle README

This folder contains the Day 3 packaged deliverables for the synthetic DPP project.

## Canonical release artifacts
- `synthetic_dpp.csv`: official release dataset (TVAE 1x, 160 rows).
- `synthetic_dpp_release_card.md`: completed release card with utility/privacy/fairness and limitations.
- `final_project_report.md`: integrated Day 1 + Day 2 + Day 3 report.

## Supporting artifacts
- `synthetic_dpp_5x.csv`: optional augmentation/demo version.
- `model_comparison_table.csv`: cross-model utility/privacy/fairness table.
- `day2_report.md`: detailed Day 2 execution report.
- `day2_decision_notes.md`: winner selection notes.
- `methods_metrics_summary.md`: one-page presentation summary.
- `schema.md`: Day 1 schema and constraints.
- `run_context.json`: split context (seed, rows, target/group).
- `synthcity_run_log.json`: runtime/model logs.
- `plugin_availability_summary.json`: available plugin audit.
- `trtr_metrics_real_baseline.json`: real baseline metrics.
- `tstr_metrics_tvae.json`: selected-model TSTR metrics.
- `tstr_metrics_tvae_5x.json`: optional Day 3 TSTR check for TVAE 5x.
- `utility_stats_tvae.json`: selected-model distribution/correlation stats.
- `privacy_summary_tvae.json`: selected-model privacy summary.
- `fairness_summary_tvae.json`: selected-model fairness summary.
- `group_metrics_tvae.csv`: selected-model group-wise metrics.
- `tvae_numeric_overlays.png`: utility plot.
- `tvae_nn_distance.png`: privacy distance plot.

## Reproducibility (commands)
Run from project root:

```bash
# Day 1
python src/day1_profile_clean.py \
  --input-csv data/my_bom_dpp.csv \
  --output-dir outputs/day1 \
  --drop-cols supplier_name,partner_internal_id \
  --hash-cols product_id,component_id \
  --target-col repairability_bin \
  --group-col supplier_region

python src/day1_baseline_classification.py \
  --input-csv outputs/day1/cleaned_modeling.csv \
  --target-col repairability_bin \
  --group-col supplier_region \
  --output-dir outputs/day1

# Day 2 full pipeline
python src/day2_run_pipeline.py \
  --input-csv outputs/day1/cleaned_synthcity_input.csv \
  --target-col repairability_bin \
  --group-col supplier_region \
  --output-dir outputs/day2 \
  --seed 42 \
  --test-size 0.2 \
  --ctgan-iters 120 \
  --tvae-iters 120
```

## Notes
- Optional plugin warnings (`GReaT`, `dgl`) are non-fatal for the selected generators.
- Utility gap is documented and should be acknowledged in any external presentation or release.
- CSV parsing caveat: `supplier_region` uses `NA` (North America) as a valid category value.
  Load the release CSV with:

```python
import pandas as pd
df = pd.read_csv("synthetic_dpp.csv", keep_default_na=False)
```
