# Public Release Bundle

This bundle is intended for public sharing.

Included:
- `final/` (final synthetic release artifacts)
- `repro/day2_run_pipeline.py` (optional reproducibility script)
- `repro/requirements_synthcity.txt` (optional reproducibility dependencies)

Excluded by design:
- all raw/real/internal data files (`data/real_*`, `data/seed_dpp_real.csv`, original BOM/TXT/XLSX sources)
- intermediate Day 1/Day 2 working outputs outside `final/`
- cache/temp folders (`__pycache__`, `workspace`, `.DS_Store`)

CSV parsing note:
Use `keep_default_na=False` so category value `NA` is not converted to missing.
