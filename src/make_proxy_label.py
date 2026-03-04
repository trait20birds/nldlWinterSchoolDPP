"""
Create a quick proxy target label for Day 1 if your real DPP/BOM table has no target column.

Edit the column names + rules below to fit your schema.
This script is intentionally simple and transparent.
"""

import argparse
import pandas as pd
import numpy as np


NA_VALUES = ['', ' ', 'nan', 'NaN', 'null', 'NULL', 'none', 'None']


def build_recyclability_proxy(df: pd.DataFrame) -> pd.Series:
    # --- EDIT THESE COLUMN NAMES TO MATCH YOUR CSV ---
    recycled_col = 'recycled_content_pct'        # numeric 0-100
    material_col = 'material_main'               # categorical
    rohs_col = 'compliance_rohs'                 # 0/1 or yes/no-like already cleaned
    hazardous_col = 'contains_hazardous_substance'  # 0/1 optional

    # If a column is missing, treat as unknown and degrade confidence conservatively
    recycled = pd.to_numeric(df.get(recycled_col, pd.Series(np.nan, index=df.index)), errors='coerce')
    material = df.get(material_col, pd.Series('<UNK>', index=df.index)).astype(str).str.lower()
    rohs = pd.to_numeric(df.get(rohs_col, pd.Series(np.nan, index=df.index)), errors='coerce')
    hazardous = pd.to_numeric(df.get(hazardous_col, pd.Series(0, index=df.index)), errors='coerce').fillna(0)

    highly_recyclable_materials = {
        'aluminum', 'aluminium', 'steel', 'copper', 'glass', 'paper', 'cardboard'
    }
    moderately_recyclable_materials = {
        'plastic', 'abs', 'pp', 'pet', 'pcb', 'mixed_metal', 'stainless_steel'
    }

    label = pd.Series('Medium', index=df.index)

    high_mask = (
        (recycled >= 50)
        & material.isin(highly_recyclable_materials)
        & ((rohs == 1) | rohs.isna())
        & (hazardous == 0)
    )

    low_mask = (
        (recycled < 10)
        | (hazardous == 1)
        | material.isin({'composite', 'mixed_composite', 'unknown'})
    )

    # Materials not in high set but plausible -> medium by default
    label[high_mask] = 'High'
    label[low_mask] = 'Low'

    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--target-col', default='recyclability_bin')
    args = ap.parse_args()

    # Keep codes like 'NA' as literal categories.
    df = pd.read_csv(args.input_csv, keep_default_na=False, na_values=NA_VALUES)
    df[args.target_col] = build_recyclability_proxy(df)
    df.to_csv(args.output_csv, index=False)
    print(f'Saved proxy-labeled file to {args.output_csv}')
    print(df[args.target_col].value_counts(dropna=False))


if __name__ == '__main__':
    main()
