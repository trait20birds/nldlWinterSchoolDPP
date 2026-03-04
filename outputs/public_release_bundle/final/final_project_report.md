# Final Project Report: Synthetic DPP Generation with SynthCity

## 1. Project motivation
This project targets a common CE-RISE/DPP challenge: limited shareable data for experimentation, benchmarking, and pipeline development.  
Goal: build a reproducible workflow that produces synthetic DPP-like tabular data with explicit utility, privacy, and fairness evaluation.

## 2. Scope and data
- Unit of analysis: one row = one BOM component.
- Core dataset: `data/seed_dpp_modeling.csv` (cleaned tabular DPP/BOM subset).
- Final synthesis feature space: 8 columns.
- Target task: `repairability_bin` classification.
- Fairness grouping: `supplier_region`.

## 3. Day 1 summary (data preparation and baseline)
Day 1 delivered the modeling substrate required for synthesis:
- Column normalization and type harmonization.
- Sensitive and ID-like fields removed from synthesis-ready table.
- Schema and constraints documented in `docs/schema.md`.
- Baseline model trained on real data.

Key Day 1 artifacts:
- `outputs/day1/cleaned_modeling.csv`
- `outputs/day1/cleaned_synthcity_input.csv`
- `outputs/day1/profile_summary.json`
- `outputs/day1/baseline_metrics.json`
- `docs/schema.md`

## 4. Day 2 methods
### 4.1 Train/test protocol
- Frozen snapshot: `data/real_train_snapshot.csv`.
- Split: `data/real_train.csv` (160 rows) / `data/real_test.csv` (40 rows), seed `42`, stratified by target.

### 4.2 SynthCity model families evaluated
- `ctgan`
- `tvae`
- `privbayes`

For each generator:
- Fit on real train.
- Generate 1x and 5x synthetic sets.
- Run validation, utility, privacy, fairness analyses.

### 4.3 Evaluation protocol
- Validation: rule-violation rate, invalid categories, duplicates, exact real overlap.
- Utility:
  - distribution similarity (numeric KS, categorical TVD),
  - TRTR real baseline vs TSTR synthetic-to-real.
- Privacy:
  - exact overlap rate,
  - nearest-neighbor distance profile,
  - suspiciously close rate vs real-real baseline.
- Fairness:
  - group proportion distortion (TVD),
  - group-wise TSTR macro-F1 gap.

## 5. Results summary
### 5.1 Real baseline
- TRTR accuracy: `0.9750`
- TRTR macro-F1: `0.9623`

### 5.2 Model comparison (1x)
- TVAE: TSTR macro-F1 `0.5319`, violation `0.0000`, overlap `0.0000`
- CTGAN: TSTR macro-F1 `0.4757`, violation `0.0000`, overlap `0.0000`
- PrivBayes: TSTR macro-F1 `0.3736`, violation `0.0000`, overlap `0.0000`

### 5.3 Optional Day 3 upgrade check (TVAE 5x)
- Additional TSTR run using `outputs/day2/synthetic/tvae_5x.csv`:
  - Accuracy `0.6500`
  - Macro-F1 `0.5365`
- This is a slight utility gain vs TVAE 1x (`+0.0046` macro-F1), but still far below real TRTR baseline.

### 5.4 Winner selection
Primary release model: **TVAE**  
Backup: **CTGAN**

Rationale:
- All candidates passed hard validity checks.
- TVAE achieved the highest TSTR macro-F1.
- Privacy indicators remained acceptable (no exact row overlap).

## 6. Final release package
Canonical release dataset:
- `outputs/final/synthetic_dpp.csv` (TVAE 1x, 160 rows)

Additional bundle artifacts:
- `outputs/final/synthetic_dpp_5x.csv`
- `outputs/final/synthetic_dpp_release_card.md`
- `outputs/final/model_comparison_table.csv`
- `outputs/final/run_context.json`
- `outputs/final/synthcity_run_log.json`
- `outputs/final/tvae_numeric_overlays.png`
- `outputs/final/tvae_nn_distance.png`

## 7. Limitations and risk notes
- **Major limitation:** utility gap remains substantial.  
  TVAE TSTR macro-F1 `0.5319` vs real TRTR `0.9623` (delta `-0.4304`).
- Fairness gap for TVAE is non-trivial (`0.4286` macro-F1 spread by group).
- Pilot dataset is small and simplified; this is a workflow validation, not a production release benchmark.
- The release data should be used for benchmarking/prototyping, not operational decisions.

## 8. Intended use and prohibited use
Intended:
- synthetic DPP benchmarking,
- reproducible pipeline demonstrations,
- educational/workshop use.

Prohibited:
- compliance/legal decisions,
- supplier-level business decisions,
- any re-identification or record reconstruction attempts.

## 9. Conclusion
The end-to-end synthetic DPP pipeline is operational and reproducible.  
The project successfully delivers validated synthetic datasets plus evaluation artifacts and a release package.  
The primary improvement needed after this milestone is utility uplift (reduce TSTR vs TRTR gap) while preserving current validity/privacy behavior.
