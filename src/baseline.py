import subprocess
from pathlib import Path


def run_day1_baseline(
    project_root: Path,
    input_csv: str = "outputs/day1/cleaned_modeling.csv",
    target_col: str = "repairability_bin",
    group_col: str = "supplier_region",
    output_dir: str = "outputs/day1",
) -> None:
    cmd = [
        "python",
        "src/day1_baseline_classification.py",
        "--input-csv",
        input_csv,
        "--target-col",
        target_col,
        "--group-col",
        group_col,
        "--output-dir",
        output_dir,
    ]
    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    run_day1_baseline(Path(__file__).resolve().parents[1])
