# Synthetic DPP Project

This repository contains a reproducible workflow for preparing, modeling, and evaluating synthetic Digital Product Passport (DPP) tabular data.

## Final outputs

The main final release artifacts are in `outputs/final/`:

- `outputs/final/synthetic_dpp.csv`: canonical final synthetic DPP dataset.
- `outputs/final/synthetic_dpp_5x.csv`: larger optional synthetic variant.
- `outputs/final/synthetic_dpp_release_card.md`: release notes and usage limits.
- `outputs/final/final_project_report.md`: consolidated project report.

If you want the packaged shareable bundle, use:

- `outputs/public_release_bundle.zip`
- `outputs/public_release_bundle/final/`

## Inputs

Raw source inputs are stored in `data/`:

- `data/my_bom_dpp.csv`: main seed tabular dataset.
- `data/*.xlsx`: BOM spreadsheets.
- `data/lexmark_*.txt`: text-based DPP examples.
- `data/real_train.csv`, `data/real_test.csv`, `data/seed_dpp_*.csv`: supporting source tables.

Processed and normalized inputs are stored in:

- `data/synthcity_ready/`: normalized CSV exports created from mixed raw sources.
- `outputs/day1/cleaned_modeling.csv`: cleaned modeling table for analysis.
- `outputs/day1/cleaned_synthcity_input.csv`: cleaned synthesis input used by the Day 2 pipeline.

## Project structure

- `src/`: Python scripts for preprocessing, profiling, baseline modeling, and synthetic generation.
- `config/`: example configuration files.
- `data/`: raw and normalized input data.
- `docs/`: schema and release-card templates.
- `notebooks/`: exploratory notebook.
- `outputs/day1/`: profiling, cleaning, and baseline classifier outputs.
- `outputs/day2/`: synthetic generation, utility, privacy, fairness, and validation outputs.
- `outputs/final/`: final selected release artifacts.
- `outputs/public_release_bundle/`: public-facing export bundle.
- `workspace/`: local cache files from metrics/plugins.

## How to run

Python 3.11 is the safest choice for this codebase.

Install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the full Day 2 synthetic pipeline, install the extra dependency set:

```bash
pip install -r requirements_synthcity.txt
```

Run the full workflow from the repository root:

```bash
# 1. Normalize mixed raw inputs
python src/prepare_synthcity_inputs.py \
  --input-dir data \
  --output-dir data/synthcity_ready

# 2. Profile and clean the main modeling table
python src/day1_profile_clean.py \
  --input-csv data/my_bom_dpp.csv \
  --output-dir outputs/day1 \
  --drop-cols supplier_name \
  --hash-cols product_id,component_id \
  --target-col repairability_bin \
  --group-col supplier_region

# 3. Train the baseline classifier on real data
python src/day1_baseline_classification.py \
  --input-csv outputs/day1/cleaned_modeling.csv \
  --target-col repairability_bin \
  --group-col supplier_region \
  --output-dir outputs/day1

# 4. Run the Day 2 synthetic data pipeline
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

## Quick checks

- Day 1 cleaned input: `outputs/day1/cleaned_synthcity_input.csv`
- Day 2 model comparison: `outputs/day2/comparison/model_comparison_table.csv`
- Final selected synthetic release: `outputs/final/synthetic_dpp.csv`

Load CSVs with `keep_default_na=False` if you need to preserve `NA` as a valid region code.
