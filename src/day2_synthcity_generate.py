import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


NA_VALUES = ['', ' ', 'nan', 'NaN', 'null', 'NULL', 'none', 'None']


def parse_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


def read_csv_preserve_codes(path: str) -> pd.DataFrame:
    # Keep categorical codes like 'NA' as normal values.
    return pd.read_csv(path, keep_default_na=False, na_values=NA_VALUES)


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate synthetic DPP/BOM rows with SynthCity.')
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--plugin', default='ctgan', help='SynthCity plugin name, e.g., ctgan, tvae, dpgan')
    ap.add_argument('--n-iter', type=int, default=100, help='Training iterations for plugins that support n_iter')
    ap.add_argument('--count', type=int, default=0, help='Number of synthetic rows; 0 means same as input')
    ap.add_argument('--target-col', default='', help='Optional supervised target column name')
    ap.add_argument('--drop-cols', default='', help='Comma-separated columns to drop before fitting')
    ap.add_argument('--plugin-kwargs-json', default='', help='Optional extra plugin kwargs as JSON object')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)

    df = read_csv_preserve_codes(args.input_csv)
    input_rows = len(df)

    drop_cols = parse_csv_list(args.drop_cols)
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    target_col = args.target_col.strip()
    if target_col and target_col not in df.columns:
        normalized = target_col.lower().replace(' ', '_')
        if normalized in df.columns:
            target_col = normalized
        else:
            raise ValueError(f'target column not found in input: {args.target_col}')

    plugin_kwargs: Dict[str, Any] = {}
    if args.plugin_kwargs_json:
        plugin_kwargs = json.loads(args.plugin_kwargs_json)
        if not isinstance(plugin_kwargs, dict):
            raise ValueError('--plugin-kwargs-json must be a JSON object')
    plugin_kwargs.setdefault('n_iter', args.n_iter)

    loader = GenericDataLoader(df, target_column=target_col if target_col else None)
    try:
        plugin = Plugins().get(args.plugin, **plugin_kwargs)
    except TypeError as e:
        # Some plugins do not accept n_iter; retry without it.
        if 'n_iter' in plugin_kwargs and 'n_iter' in str(e):
            plugin_kwargs = {k: v for k, v in plugin_kwargs.items() if k != 'n_iter'}
            plugin = Plugins().get(args.plugin, **plugin_kwargs)
        else:
            raise
    plugin.fit(loader)

    count = args.count if args.count > 0 else input_rows
    syn = plugin.generate(count=count).dataframe()
    syn.to_csv(args.output_csv, index=False)

    meta = {
        'input_csv': args.input_csv,
        'output_csv': args.output_csv,
        'plugin': args.plugin,
        'plugin_kwargs': plugin_kwargs,
        'target_col': target_col or None,
        'drop_cols_applied': drop_cols,
        'input_rows': int(input_rows),
        'input_columns_after_drop': int(df.shape[1]),
        'generated_rows': int(len(syn)),
        'generated_columns': int(syn.shape[1]),
    }
    meta_path = os.path.splitext(args.output_csv)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('Done.')
    print(f'Input:  {args.input_csv} ({input_rows} rows)')
    print(f'Output: {args.output_csv} ({len(syn)} rows, {syn.shape[1]} cols)')
    print(f'Meta:   {meta_path}')


if __name__ == '__main__':
    main()
