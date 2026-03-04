# Day 2 Execution Report
## Overall Verdict
- Day 2 pipeline executed end-to-end successfully.
- Required outputs for synthetic generation, validation, utility, privacy, fairness, and model comparison were produced.
- Process correctness: **PASS** (workflow completed as intended).
- Data-quality verdict: **Mixed** (synthetic validity/privacy are good; downstream utility is noticeably below real baseline).
## Data Freeze and Split
- Snapshot: `data/real_train_snapshot.csv`
- Split files: `data/real_train.csv` (160 rows), `data/real_test.csv` (40 rows)
- Seed: 42
- Target: `repairability_bin`
- Group: `supplier_region`
## Models Trained
- `tvae`: fit=3.43s, gen1x=0.024s, gen5x=0.044s
- `ctgan`: fit=3.39s, gen1x=0.023s, gen5x=0.028s
- `privbayes`: fit=0.39s, gen1x=0.030s, gen5x=0.108s
## Validation Results
- `tvae`: rule_violation=0.0000, duplicate_rate=0.0000, exact_overlap=0.0000
- `ctgan`: rule_violation=0.0000, duplicate_rate=0.0000, exact_overlap=0.0000
- `privbayes`: rule_violation=0.0000, duplicate_rate=0.0000, exact_overlap=0.0000
## Utility Results
- TRTR real baseline: accuracy=0.9750, macro_f1=0.9623
- TSTR `tvae`: accuracy=0.5250, macro_f1=0.5319, delta_vs_TRTR=-0.4304
- TSTR `ctgan`: accuracy=0.6750, macro_f1=0.4757, delta_vs_TRTR=-0.4866
- TSTR `privbayes`: accuracy=0.6250, macro_f1=0.3736, delta_vs_TRTR=-0.5887
## Privacy Results
- `tvae`: exact_overlap=0.0000, suspiciously_close_rate=0.0750
- `ctgan`: exact_overlap=0.0000, suspiciously_close_rate=0.0312
- `privbayes`: exact_overlap=0.0000, suspiciously_close_rate=0.0312
## Fairness Results
- `tvae`: group_proportion_tvd=0.0812, group_macro_f1_gap=0.4286
- `ctgan`: group_proportion_tvd=0.1437, group_macro_f1_gap=0.2864
- `privbayes`: group_proportion_tvd=0.0625, group_macro_f1_gap=0.2634
## Model Selection
- Primary winner for Day 3: **`tvae`**
- Backup model: **`ctgan`** (better fairness gap than TVAE but lower TSTR macro-F1).
## Checklist Status
- [x] Train 2-3 SynthCity generators
- [x] Generate 1 synthetic dataset per generator
- [x] Run quality checks (ranges/categories/duplicates/invalid combos)
- [x] Run utility evaluation (distribution + downstream TSTR/TRTR)
- [x] Run privacy checks (exact overlap + NN distance leakage)
- [x] Run fairness checks across group column
- [x] Save consolidated comparison outputs and logs
- [x] Select Day 3 winner model
## Notes
- Non-fatal SynthCity optional-plugin warnings were observed (missing `GReaT`/`dgl`), but selected generators ran correctly.
- Utility gap indicates synthetic data is not yet a replacement for real training data; this should be discussed in the Day 3 release card.
