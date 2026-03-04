# DPP Synthetic Data Project — Day 1 (Do-it-now starter)

This starter pack helps you complete **Day 1** of your SynthCity project on a BOM/DPP table.

## What this does for you today
- Profiles your CSV (missingness, duplicates, inferred schema)
- Cleans it into a modeling table (basic normalization + optional hashing/dropping)
- Creates a schema/summary file for your report
- Trains a baseline classifier (LogReg + RandomForest) on real data
- Saves metrics + plots for your later synthetic comparison

## Folder outputs (after running)
- `outputs/day1/profile_summary.json`
- `outputs/day1/missingness.csv`
- `outputs/day1/schema_inferred.csv`
- `outputs/day1/cleaned_modeling.csv`
- `outputs/day1/cleaned_synthcity_input.csv`
- `outputs/day1/synthcity_recommendations.json`
- `outputs/day1/baseline_metrics.json`
- `outputs/day1/class_distribution.csv`
- `outputs/day1/confusion_matrix_<model>.csv`
- `outputs/day1/group_metrics.csv` (if you set `--group-col`)
- `outputs/day1/fairness_gap_summary.json` (if you set `--group-col`)
- `outputs/day1/fig_feature_importance.png` (RandomForest)

## 0) Install deps
```bash
pip install -r requirements.txt
```

## 1) Put your internal BOM/DPP CSV somewhere accessible
Example: `data/my_bom_dpp.csv`

## 2) Run profiling + cleaning (replace column names)
```bash
python src/day1_profile_clean.py \
  --input-csv data/my_bom_dpp.csv \
  --output-dir outputs/day1 \
  --drop-cols supplier_name,partner_internal_id \
  --hash-cols product_id,component_id \
  --target-col repairability_bin \
  --group-col supplier_region
```

If you don’t have a target label yet, skip `--target-col` for now and just produce the cleaned table.

## 3) (Optional) Create a proxy target label if no label exists
Edit `src/make_proxy_label.py` rules, then run:
```bash
python src/make_proxy_label.py \
  --input-csv outputs/day1/cleaned_modeling.csv \
  --output-csv outputs/day1/cleaned_with_target.csv \
  --target-col recyclability_bin
```

## 4) Train baseline model on real data
```bash
python src/day1_baseline_classification.py \
  --input-csv outputs/day1/cleaned_modeling.csv \
  --target-col repairability_bin \
  --group-col supplier_region \
  --output-dir outputs/day1
```

If you used a proxy label file:
```bash
python src/day1_baseline_classification.py \
  --input-csv outputs/day1/cleaned_with_target.csv \
  --target-col recyclability_bin \
  --group-col supplier_region \
  --output-dir outputs/day1
```

## 5) Fill the docs templates for your report
- `docs/schema_template.md`
- `docs/release_card_template.md`

## 6) (Day 2) Generate synthetic rows with SynthCity
Install SynthCity dependencies:
```bash
pip install -r requirements_synthcity.txt
```

Generate synthetic rows:
```bash
python src/day2_synthcity_generate.py \
  --input-csv outputs/day1/cleaned_synthcity_input.csv \
  --output-csv outputs/day1/synthcity_generated_ctgan.csv \
  --plugin ctgan \
  --n-iter 200 \
  --target-col repairability_bin
```

## Minimal Day 1 success checklist
- [ ] `cleaned_modeling.csv` created
- [ ] schema + missingness exported
- [ ] one target label selected (real or proxy)
- [ ] baseline metrics saved
- [ ] fairness grouping chosen (e.g., region / supplier tier)

## Notes
- This starter intentionally keeps modeling simple for Day 1.
- It is **not** SynthCity yet (that is Day 2). It produces the clean tabular input that SynthCity needs.
- If your BOM is hierarchical/relational, start with **one flat table** (component-level rows).
- Codes like `NA` are preserved as categorical values (not auto-converted to missing) by the scripts.
