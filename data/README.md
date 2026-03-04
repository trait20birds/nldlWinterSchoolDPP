# Data Folder Notes

## Day 1 Seed Files
- `seed_dpp_real.csv`: internal/raw BOM-DPP seed dataset used as source.
- `seed_dpp_modeling.csv`: cleaned flat modeling table (SynthCity-ready for tabular generation).

## Additional Source Files
- `my_bom_dpp.csv`: original starter CSV used for Day 1 pipeline.
- `lexmark_*_dpp_full_*.txt`: narrative DPP examples (converted to tabular in `data/synthcity_ready/`).
- `Vitocal*BOM.xlsx`: BOM spreadsheets (converted to tabular in `data/synthcity_ready/`).

## Generated Conversion Outputs
- `synthcity_ready/`: normalized per-file and combined tabular outputs created by `src/prepare_synthcity_inputs.py`.
