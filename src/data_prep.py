import subprocess
from pathlib import Path


def run_day1_profile_clean(
    project_root: Path,
    input_csv: str = "data/my_bom_dpp.csv",
    output_dir: str = "outputs/day1",
    drop_cols: str = "supplier_name,partner_internal_id",
    hash_cols: str = "product_id,component_id",
    target_col: str = "repairability_bin",
    group_col: str = "supplier_region",
) -> None:
    cmd = [
        "python",
        "src/day1_profile_clean.py",
        "--input-csv",
        input_csv,
        "--output-dir",
        output_dir,
        "--drop-cols",
        drop_cols,
        "--hash-cols",
        hash_cols,
        "--target-col",
        target_col,
        "--group-col",
        group_col,
    ]
    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    run_day1_profile_clean(Path(__file__).resolve().parents[1])
