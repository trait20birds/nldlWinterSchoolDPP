from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "day1"
DOCS_DIR = PROJECT_ROOT / "docs"

SEED_REAL_CSV = DATA_DIR / "seed_dpp_real.csv"
SEED_MODELING_CSV = DATA_DIR / "seed_dpp_modeling.csv"

CLEANED_MODELING_CSV = OUTPUTS_DIR / "cleaned_modeling.csv"
CLEANED_SYNTHCITY_INPUT_CSV = OUTPUTS_DIR / "cleaned_synthcity_input.csv"
BASELINE_METRICS_JSON = OUTPUTS_DIR / "baseline_metrics.json"
