import argparse
import json
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Generate a draft schema markdown from Day 1 outputs.')
    ap.add_argument('--schema-csv', required=True, help='Path to schema_inferred.csv')
    ap.add_argument('--profile-json', required=True, help='Path to profile_summary.json')
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    schema = pd.read_csv(args.schema_csv)
    with open(args.profile_json, 'r', encoding='utf-8') as f:
        profile = json.load(f)

    lines = []
    lines.append('# Auto-generated Day 1 Schema Draft\n')
    lines.append('## Dataset overview\n')
    lines.append(f"- Original shape: {profile.get('original_shape')}\n")
    lines.append(f"- Cleaned shape: {profile.get('cleaned_shape')}\n")
    lines.append(f"- Duplicates removed: {profile.get('duplicates_removed')}\n")
    lines.append(f"- Target column: {profile.get('target_col')}\n")
    lines.append(f"- Group column: {profile.get('group_col')}\n")
    lines.append('\n## Columns\n')
    lines.append('| Column | Dtype | Inferred role | Missing % | Unique (non-null) | Target? | Group? |\n')
    lines.append('|---|---|---|---:|---:|---:|---:|\n')

    for _, r in schema.iterrows():
        lines.append(
            f"| {r['column']} | {r['dtype']} | {r['inferred_role']} | {float(r['missing_pct']):.2f} | {int(r['nunique_non_null'])} | {bool(r.get('is_target', False))} | {bool(r.get('is_group', False))} |\n"
        )

    if profile.get('range_flags'):
        lines.append('\n## Automatic range flags (review manually)\n')
        for col, msg in profile['range_flags'].items():
            lines.append(f"- `{col}`: {msg}\n")

    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f'Saved schema markdown to {args.out_md}')


if __name__ == '__main__':
    main()
