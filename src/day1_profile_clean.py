import argparse
import hashlib
import json
import os
from typing import List

import numpy as np
import pandas as pd


NA_VALUES = ['', ' ', 'nan', 'NaN', 'null', 'NULL', 'none', 'None']


def parse_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


def normalize_string_series(s: pd.Series) -> pd.Series:
    # Preserve NaN; normalize whitespace and casing lightly
    s2 = s.astype('string')
    s2 = s2.str.strip()
    s2 = s2.str.replace(r'\s+', ' ', regex=True)
    s2 = s2.replace('', pd.NA)
    return s2


def hash_value(x: object) -> str | None:
    if pd.isna(x):
        return None
    h = hashlib.sha256(str(x).encode('utf-8')).hexdigest()
    return h[:16]


def infer_col_role(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return 'unknown'

    if pd.api.types.is_numeric_dtype(series):
        uniques = non_null.nunique(dropna=True)
        if uniques <= 2:
            return 'binary'
        return 'numeric'

    # String-like path
    s = non_null.astype(str).str.lower().str.strip()
    bool_vals = {'yes', 'no', 'true', 'false', '0', '1', 'y', 'n'}
    if set(s.unique()).issubset(bool_vals):
        return 'binary'

    nunique = s.nunique()
    # heuristic: low-cardinality as categorical
    if nunique <= max(20, int(0.2 * len(s))):
        return 'categorical'
    return 'text_or_id'


def is_id_like_column(col_name: str, series: pd.Series) -> bool:
    name = col_name.lower()
    id_keywords = ('id', 'uuid', 'serial', 'gtin', 'lot', 'batch', 'sku')
    if any(k in name for k in id_keywords):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    uniq_ratio = non_null.nunique(dropna=True) / len(non_null)
    return uniq_ratio >= 0.9


def coerce_bool_strings(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'yes': 1, 'y': 1, 'true': 1, 't': 1,
        'no': 0, 'n': 0, 'false': 0, 'f': 0
    }
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == 'object' or str(out[c].dtype).startswith('string'):
            s = out[c].astype('string').str.strip().str.lower()
            unique_non_null = set(s.dropna().unique().tolist())
            if unique_non_null and unique_non_null.issubset(set(mapping.keys()) | {'0', '1'}):
                out[c] = s.map(mapping).fillna(s)
                # convert '0','1' strings too
                out[c] = out[c].replace({'0': 0, '1': 1})
                try:
                    out[c] = pd.to_numeric(out[c])
                except Exception:
                    pass
    return out


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    seen = set()
    for c in df.columns:
        nc = str(c).strip().lower().replace(' ', '_')
        nc = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in nc)
        while '__' in nc:
            nc = nc.replace('__', '_')
        nc = nc.strip('_') or 'col'
        base = nc
        i = 1
        while nc in seen:
            i += 1
            nc = f"{base}_{i}"
        seen.add(nc)
        renamed[c] = nc
    return df.rename(columns=renamed)


def basic_range_flags(df: pd.DataFrame) -> dict:
    flags = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            col = c.lower()
            s = df[c].dropna()
            if s.empty:
                continue
            if ('pct' in col or 'percent' in col or 'percentage' in col) and ((s < 0).any() or (s > 100).any()):
                flags[c] = 'values_outside_0_100_for_percentage_like_column'
            if any(k in col for k in ['mass', 'weight', 'co2', 'amount', 'qty', 'quantity']) and (s < 0).any():
                flags[c] = 'negative_values_in_quantity_like_column'
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description='Profile and clean a flat BOM/DPP CSV for Day 1.')
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--drop-cols', default='')
    ap.add_argument('--hash-cols', default='')
    ap.add_argument('--target-col', default='')
    ap.add_argument('--group-col', default='')
    ap.add_argument('--dedupe-subset-cols', default='')
    ap.add_argument('--max-categorical-uniques', type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Keep codes like 'NA' as valid categories instead of auto-converting to missing.
    df = pd.read_csv(args.input_csv, keep_default_na=False, na_values=NA_VALUES)
    original_shape = df.shape
    original_columns = df.columns.tolist()

    df = standardize_column_names(df)

    # normalize strings
    for c in df.columns:
        if df[c].dtype == 'object' or str(df[c].dtype).startswith('string'):
            df[c] = normalize_string_series(df[c])

    df = coerce_bool_strings(df)

    drop_cols = parse_csv_list(args.drop_cols)
    hash_cols = parse_csv_list(args.hash_cols)
    dedupe_subset_cols = parse_csv_list(args.dedupe_subset_cols)

    # align provided columns to standardized names if user passed exact raw names not matching
    standardized_colset = set(df.columns)
    def resolve_cols(cols: List[str]) -> List[str]:
        resolved = []
        for c in cols:
            sc = str(c).strip().lower().replace(' ', '_')
            sc = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in sc)
            sc = sc.strip('_')
            if sc in standardized_colset:
                resolved.append(sc)
        return resolved

    drop_cols = resolve_cols(drop_cols)
    hash_cols = resolve_cols(hash_cols)
    dedupe_subset_cols = resolve_cols(dedupe_subset_cols)

    # Hash before drop (if both specified, hash is harmless but column might be dropped later)
    for c in hash_cols:
        df[c] = df[c].map(hash_value)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    before_dedup = len(df)
    if dedupe_subset_cols:
        df = df.drop_duplicates(subset=dedupe_subset_cols)
    else:
        df = df.drop_duplicates()
    duplicates_removed = before_dedup - len(df)

    # Convert obvious numerics stored as strings (best effort)
    for c in df.columns:
        if df[c].dtype == 'object' or str(df[c].dtype).startswith('string'):
            sample = df[c].dropna().astype(str)
            if sample.empty:
                continue
            # strip commas in numeric-like columns
            candidate = sample.str.replace(',', '', regex=False)
            num = pd.to_numeric(candidate, errors='coerce')
            conversion_rate = num.notna().mean() if len(candidate) > 0 else 0
            if conversion_rate > 0.95:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')

    # Missingness and schema inference
    rows, cols = df.shape
    missingness = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[c].dtype) for c in df.columns],
        'non_null_count': [int(df[c].notna().sum()) for c in df.columns],
        'missing_count': [int(df[c].isna().sum()) for c in df.columns],
        'missing_pct': [float(df[c].isna().mean() * 100) for c in df.columns],
        'nunique_non_null': [int(df[c].dropna().nunique()) for c in df.columns],
        'inferred_role': [infer_col_role(df[c]) for c in df.columns],
    })
    missingness['nunique_ratio'] = [
        float(df[c].dropna().nunique() / max(1, df[c].dropna().shape[0]))
        for c in df.columns
    ]
    missingness['is_id_like'] = [
        bool(is_id_like_column(c, df[c]))
        for c in df.columns
    ]

    # Mark target/group if provided
    target_col = str(args.target_col).strip().lower().replace(' ', '_') if args.target_col else ''
    group_col = str(args.group_col).strip().lower().replace(' ', '_') if args.group_col else ''
    missingness['is_target'] = missingness['column'].eq(target_col)
    missingness['is_group'] = missingness['column'].eq(group_col)

    # class distribution if target exists
    if target_col and target_col in df.columns:
        class_dist = (
            df[target_col]
            .astype('string')
            .fillna('<MISSING>')
            .value_counts(dropna=False)
            .rename_axis('label')
            .reset_index(name='count')
        )
        class_dist['pct'] = class_dist['count'] / class_dist['count'].sum() * 100
        class_dist.to_csv(os.path.join(args.output_dir, 'class_distribution.csv'), index=False)

    # Unique value samples for categoricals/text-like
    samples = []
    for c in df.columns:
        role = infer_col_role(df[c])
        if role in {'categorical', 'binary', 'text_or_id'}:
            vals = df[c].dropna().astype(str).unique().tolist()[:15]
            samples.append({'column': c, 'sample_values': vals})
    with open(os.path.join(args.output_dir, 'categorical_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Numeric summary
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        num_summary = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.reset_index()
        num_summary = num_summary.rename(columns={'index': 'column'})
        num_summary.to_csv(os.path.join(args.output_dir, 'numeric_summary.csv'), index=False)

    # IQR outlier counts for numerics
    outlier_rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[c] < low) | (df[c] > high)).sum())
        outlier_rows.append({'column': c, 'iqr_low': float(low), 'iqr_high': float(high), 'outlier_count': count})
    pd.DataFrame(outlier_rows).to_csv(os.path.join(args.output_dir, 'outlier_report_iqr.csv'), index=False)

    range_flags = basic_range_flags(df)
    recommended_drop_for_synthcity = [
        c for c in df.columns
        if is_id_like_column(c, df[c]) and c not in {target_col, group_col}
    ]

    profile_summary = {
        'input_csv': args.input_csv,
        'original_shape': {'rows': int(original_shape[0]), 'cols': int(original_shape[1])},
        'cleaned_shape': {'rows': int(rows), 'cols': int(cols)},
        'duplicates_removed': int(duplicates_removed),
        'columns_renamed_from': original_columns,
        'columns_final': df.columns.tolist(),
        'target_col': target_col if target_col in df.columns else None,
        'group_col': group_col if group_col in df.columns else None,
        'drop_cols_applied': drop_cols,
        'hash_cols_applied': hash_cols,
        'range_flags': range_flags,
        'recommended_drop_for_synthcity': recommended_drop_for_synthcity,
    }

    with open(os.path.join(args.output_dir, 'profile_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(profile_summary, f, indent=2, ensure_ascii=False)

    missingness.to_csv(os.path.join(args.output_dir, 'missingness.csv'), index=False)
    missingness.to_csv(os.path.join(args.output_dir, 'schema_inferred.csv'), index=False)
    df.to_csv(os.path.join(args.output_dir, 'cleaned_modeling.csv'), index=False)
    if recommended_drop_for_synthcity:
        synthcity_df = df.drop(columns=recommended_drop_for_synthcity, errors='ignore')
    else:
        synthcity_df = df.copy()
    synthcity_df.to_csv(os.path.join(args.output_dir, 'cleaned_synthcity_input.csv'), index=False)
    with open(os.path.join(args.output_dir, 'synthcity_recommendations.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'recommended_drop_for_synthcity': recommended_drop_for_synthcity,
                'output_synthcity_input_csv': os.path.join(args.output_dir, 'cleaned_synthcity_input.csv'),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print('Done.')
    print(f"Original shape: {original_shape} -> Cleaned shape: {df.shape}")
    print(f"Saved cleaned data to: {os.path.join(args.output_dir, 'cleaned_modeling.csv')}")
    print(f"Saved SynthCity input to: {os.path.join(args.output_dir, 'cleaned_synthcity_input.csv')}")
    if recommended_drop_for_synthcity:
        print(f"Recommended columns dropped for SynthCity: {', '.join(recommended_drop_for_synthcity)}")
    if range_flags:
        print('Range flags detected:')
        for k, v in range_flags.items():
            print(f'  - {k}: {v}')


if __name__ == '__main__':
    main()
