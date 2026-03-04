import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


NA_VALUES = ['', ' ', 'nan', 'NaN', 'null', 'NULL', 'none', 'None']


@dataclass
class ModelRun:
    model_name: str
    plugin_kwargs: Dict[str, Any]
    fit_seconds: float
    generate_1x_seconds: float
    generate_5x_seconds: float
    output_1x_csv: str
    output_5x_csv: str


def ensure_dirs(base_out: str) -> Dict[str, str]:
    paths = {
        'base': base_out,
        'synthetic': os.path.join(base_out, 'synthetic'),
        'validation': os.path.join(base_out, 'validation'),
        'utility': os.path.join(base_out, 'utility'),
        'utility_plots': os.path.join(base_out, 'utility', 'plots'),
        'privacy': os.path.join(base_out, 'privacy'),
        'privacy_plots': os.path.join(base_out, 'privacy', 'nn_distance_plots'),
        'fairness': os.path.join(base_out, 'fairness'),
        'comparison': os.path.join(base_out, 'comparison'),
        'logs': os.path.join(base_out, 'logs'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=NA_VALUES)


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return feature_cols, num_cols, cat_cols


def build_classifier(train_df: pd.DataFrame, target_col: str) -> Pipeline:
    feature_cols, num_cols, cat_cols = infer_feature_types(train_df, target_col)
    pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ]), cat_cols),
    ])
    model = LogisticRegression(max_iter=2000)
    pipe = Pipeline([
        ('prep', pre),
        ('model', model),
    ])
    return pipe


def evaluate_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    clf = build_classifier(train_df, target_col)
    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(str)
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(str)

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'macro_f1': float(f1_score(y_test, preds, average='macro')),
        'classification_report': classification_report(y_test, preds, output_dict=True, zero_division=0),
        'labels': sorted(y_test.unique().tolist()),
    }
    return metrics, preds


def group_metrics(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series, min_n: int = 1) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = sorted(y_true.astype(str).unique().tolist())
    groups_filled = groups.fillna('<MISSING>').astype(str)
    for g in sorted(groups_filled.unique()):
        idx = groups_filled == g
        if int(idx.sum()) < min_n:
            continue
        yt = y_true[idx].astype(str)
        yp = pd.Series(y_pred, index=y_true.index)[idx].astype(str)
        rows.append({
            'group': g,
            'n': int(idx.sum()),
            'accuracy': float(accuracy_score(yt, yp)),
            'macro_f1': float(f1_score(yt, yp, average='macro', labels=labels)),
        })
    return pd.DataFrame(rows)


def get_plugin_with_fallback(name: str, kwargs: Dict[str, Any]):
    # Some SynthCity plugins do not accept n_iter.
    try:
        plugin = Plugins().get(name, **kwargs)
        return plugin, kwargs
    except TypeError as e:
        if 'n_iter' in kwargs and 'n_iter' in str(e):
            kwargs2 = dict(kwargs)
            kwargs2.pop('n_iter', None)
            plugin = Plugins().get(name, **kwargs2)
            return plugin, kwargs2
        raise


def fit_and_generate(
    model_name: str,
    train_df: pd.DataFrame,
    target_col: str,
    out_dir: str,
    plugin_kwargs: Dict[str, Any],
) -> ModelRun:
    loader = GenericDataLoader(train_df, target_column=target_col)
    plugin, used_kwargs = get_plugin_with_fallback(model_name, plugin_kwargs)

    t0 = time.perf_counter()
    plugin.fit(loader)
    fit_seconds = time.perf_counter() - t0

    out_1x = os.path.join(out_dir, f'{model_name}_1x.csv')
    out_5x = os.path.join(out_dir, f'{model_name}_5x.csv')

    t1 = time.perf_counter()
    syn_1x = plugin.generate(count=len(train_df)).dataframe()
    generate_1x_seconds = time.perf_counter() - t1
    syn_1x.to_csv(out_1x, index=False)

    t2 = time.perf_counter()
    syn_5x = plugin.generate(count=5 * len(train_df)).dataframe()
    generate_5x_seconds = time.perf_counter() - t2
    syn_5x.to_csv(out_5x, index=False)

    return ModelRun(
        model_name=model_name,
        plugin_kwargs=used_kwargs,
        fit_seconds=fit_seconds,
        generate_1x_seconds=generate_1x_seconds,
        generate_5x_seconds=generate_5x_seconds,
        output_1x_csv=out_1x,
        output_5x_csv=out_5x,
    )


def compute_overlap_rate(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    real_set = set(map(tuple, real_df.astype(str).to_numpy()))
    syn_rows = list(map(tuple, syn_df.astype(str).to_numpy()))
    if not syn_rows:
        return 0.0
    n_overlap = sum(r in real_set for r in syn_rows)
    return n_overlap / len(syn_rows)


def compute_validation(
    model_name: str,
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
    out_validation_dir: str,
) -> Dict[str, Any]:
    num_cols = [c for c in syn_df.columns if pd.api.types.is_numeric_dtype(syn_df[c])]
    cat_cols = [c for c in syn_df.columns if c not in num_cols]

    rules = pd.DataFrame(index=syn_df.index)
    for c in num_cols:
        real_min = pd.to_numeric(real_train[c], errors='coerce').min()
        real_max = pd.to_numeric(real_train[c], errors='coerce').max()
        vals = pd.to_numeric(syn_df[c], errors='coerce')
        rules[f'{c}__in_real_range'] = vals.between(real_min, real_max, inclusive='both')

    # Domain-aware hard rules
    if 'mass_g' in syn_df.columns:
        rules['mass_g_positive'] = pd.to_numeric(syn_df['mass_g'], errors='coerce') > 0
    if 'recycled_content_pct' in syn_df.columns:
        p = pd.to_numeric(syn_df['recycled_content_pct'], errors='coerce')
        rules['recycled_content_pct_0_100'] = p.between(0, 100, inclusive='both')
    for b in ['compliance_rohs', 'contains_hazardous_substance']:
        if b in syn_df.columns:
            rules[f'{b}_binary'] = pd.to_numeric(syn_df[b], errors='coerce').isin([0, 1])

    invalid_category_rates = {}
    for c in cat_cols:
        real_vals = set(real_train[c].dropna().astype(str).unique().tolist())
        syn_vals = syn_df[c].dropna().astype(str)
        if len(syn_vals) == 0:
            invalid = pd.Series([False] * len(syn_df), index=syn_df.index)
            invalid_rate = 0.0
        else:
            invalid = ~syn_df[c].astype(str).isin(real_vals)
            invalid_rate = float(invalid.mean())
        rules[f'{c}__valid_category'] = ~invalid
        invalid_category_rates[c] = invalid_rate

    rules['row_valid'] = rules.all(axis=1)
    rules['row_has_violation'] = ~rules['row_valid']

    violations_path = os.path.join(out_validation_dir, f'rule_violations_{model_name}.csv')
    rules.to_csv(violations_path, index=False)

    duplicate_rate = float(syn_df.duplicated().mean())
    exact_overlap_rate = float(compute_overlap_rate(real_train, syn_df))
    missing_rate = float(syn_df.isna().any(axis=1).mean())
    any_rule_violation_rate = float((~rules['row_valid']).mean())

    summary = {
        'model': model_name,
        'rows': int(len(syn_df)),
        'missing_row_rate': missing_rate,
        'duplicate_row_rate': duplicate_rate,
        'exact_overlap_with_real_train_rate': exact_overlap_rate,
        'any_rule_violation_rate': any_rule_violation_rate,
        'invalid_category_rate_by_column': invalid_category_rates,
        'violations_file': violations_path,
    }
    write_json(os.path.join(out_validation_dir, f'validation_summary_{model_name}.json'), summary)
    return summary


def plot_numeric_overlays(real_df: pd.DataFrame, syn_df: pd.DataFrame, numeric_cols: List[str], out_png: str) -> None:
    if not numeric_cols:
        return
    n = len(numeric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        r = pd.to_numeric(real_df[col], errors='coerce').dropna()
        s = pd.to_numeric(syn_df[col], errors='coerce').dropna()
        ax.hist(r, bins=20, alpha=0.6, label='real_train', density=True)
        ax.hist(s, bins=20, alpha=0.6, label='synthetic', density=True)
        ax.set_title(col)
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_nn_histogram(d_syn_real: np.ndarray, d_real_real: np.ndarray, out_png: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(d_syn_real, bins=30, alpha=0.6, label='synthetic->real NN', density=True)
    plt.hist(d_real_real, bins=30, alpha=0.6, label='real->real NN (2nd nearest)', density=True)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Nearest Neighbor Distance Distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def compute_utility(
    model_name: str,
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
    out_utility_dir: str,
) -> Dict[str, Any]:
    feature_cols, num_cols, cat_cols = infer_feature_types(real_train, target_col)
    numeric_stats: Dict[str, Any] = {}
    categorical_stats: Dict[str, Any] = {}

    for c in num_cols:
        r = pd.to_numeric(real_train[c], errors='coerce').dropna()
        s = pd.to_numeric(syn_df[c], errors='coerce').dropna()
        ks = ks_2samp(r, s)
        numeric_stats[c] = {
            'real_mean': safe_float(r.mean()),
            'syn_mean': safe_float(s.mean()),
            'real_std': safe_float(r.std()),
            'syn_std': safe_float(s.std()),
            'ks_stat': safe_float(ks.statistic),
            'ks_pvalue': safe_float(ks.pvalue),
        }

    for c in cat_cols + [target_col]:
        r = real_train[c].astype(str).value_counts(normalize=True)
        s = syn_df[c].astype(str).value_counts(normalize=True)
        idx = sorted(set(r.index.tolist()) | set(s.index.tolist()))
        r = r.reindex(idx, fill_value=0.0)
        s = s.reindex(idx, fill_value=0.0)
        tvd = 0.5 * float(np.abs(r - s).sum())
        categorical_stats[c] = {
            'tvd': tvd,
            'real_top': r.sort_values(ascending=False).head(5).to_dict(),
            'syn_top': s.sort_values(ascending=False).head(5).to_dict(),
        }

    corr_mae = None
    if len(num_cols) >= 2:
        cr = real_train[num_cols].corr().fillna(0.0)
        cs = syn_df[num_cols].corr().fillna(0.0)
        corr_mae = float((cr - cs).abs().values.mean())

    summary = {
        'model': model_name,
        'numeric_distribution_stats': numeric_stats,
        'categorical_distribution_stats': categorical_stats,
        'numeric_correlation_mae': corr_mae,
    }

    write_json(os.path.join(out_utility_dir, f'utility_stats_{model_name}.json'), summary)
    plot_numeric_overlays(
        real_train,
        syn_df,
        num_cols,
        os.path.join(out_utility_dir, 'plots', f'{model_name}_numeric_overlays.png'),
    )
    return summary


def compute_privacy(
    model_name: str,
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
    out_privacy_dir: str,
) -> Dict[str, Any]:
    feature_cols, num_cols, cat_cols = infer_feature_types(real_train, target_col)

    # Build dense feature matrix for distance-based privacy checks
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_cols),
    ])
    x_real = pre.fit_transform(real_train[feature_cols])
    x_syn = pre.transform(syn_df[feature_cols])

    # synthetic->real nearest neighbor distance
    nn_real = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(x_real)
    d_syn_real = nn_real.kneighbors(x_syn, return_distance=True)[0][:, 0]

    # real->real second-nearest baseline
    nn_rr = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(x_real)
    d_real_real = nn_rr.kneighbors(x_real, return_distance=True)[0][:, 1]

    # Rare-combination leakage
    key_cols = [c for c in ['supplier_region', 'component_type', 'material_main', target_col] if c in syn_df.columns]
    rare_combo_leakage_rate = None
    if key_cols:
        real_combo_counts = real_train[key_cols].astype(str).value_counts()
        rare_index = set(real_combo_counts[real_combo_counts <= 2].index.tolist())
        syn_combo = list(syn_df[key_cols].astype(str).itertuples(index=False, name=None))
        rare_hits = sum(c in rare_index for c in syn_combo)
        rare_combo_leakage_rate = float(rare_hits / max(1, len(syn_combo)))

    threshold = float(np.percentile(d_real_real, 1))
    suspiciously_close_rate = float((d_syn_real <= threshold).mean())
    exact_overlap_rate = float(compute_overlap_rate(real_train, syn_df))

    plot_nn_histogram(
        d_syn_real,
        d_real_real,
        os.path.join(out_privacy_dir, 'nn_distance_plots', f'{model_name}_nn_distance.png'),
    )

    summary = {
        'model': model_name,
        'exact_overlap_with_real_train_rate': exact_overlap_rate,
        'syn_to_real_nn_distance': {
            'min': float(np.min(d_syn_real)),
            'p01': float(np.percentile(d_syn_real, 1)),
            'median': float(np.median(d_syn_real)),
            'mean': float(np.mean(d_syn_real)),
        },
        'real_to_real_nn_distance': {
            'min': float(np.min(d_real_real)),
            'p01': float(np.percentile(d_real_real, 1)),
            'median': float(np.median(d_real_real)),
            'mean': float(np.mean(d_real_real)),
        },
        'suspiciously_close_rate_vs_real_real_p01': suspiciously_close_rate,
        'rare_combo_leakage_rate': rare_combo_leakage_rate,
    }
    write_json(os.path.join(out_privacy_dir, f'privacy_summary_{model_name}.json'), summary)
    return summary


def compute_fairness(
    model_name: str,
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    real_test: pd.DataFrame,
    target_col: str,
    group_col: str,
    tstr_predictions_on_real_test: np.ndarray,
    out_fairness_dir: str,
) -> Dict[str, Any]:
    if group_col not in real_train.columns or group_col not in syn_df.columns:
        summary = {
            'model': model_name,
            'skipped': True,
            'reason': f'group column not found: {group_col}',
        }
        write_json(os.path.join(out_fairness_dir, f'fairness_summary_{model_name}.json'), summary)
        return summary

    # Group proportion distortion
    gr = real_train[group_col].astype(str).value_counts(normalize=True)
    gs = syn_df[group_col].astype(str).value_counts(normalize=True)
    idx = sorted(set(gr.index.tolist()) | set(gs.index.tolist()))
    gr = gr.reindex(idx, fill_value=0.0)
    gs = gs.reindex(idx, fill_value=0.0)
    group_tvd = 0.5 * float(np.abs(gr - gs).sum())

    # Label distribution distortion inside each group
    labels = sorted(real_train[target_col].astype(str).unique().tolist())
    per_group_label_l1: Dict[str, float] = {}
    for g in idx:
        rr = real_train[real_train[group_col].astype(str) == g]
        ss = syn_df[syn_df[group_col].astype(str) == g]
        pr = rr[target_col].astype(str).value_counts(normalize=True).reindex(labels, fill_value=0.0)
        ps = ss[target_col].astype(str).value_counts(normalize=True).reindex(labels, fill_value=0.0)
        per_group_label_l1[g] = float(np.abs(pr - ps).sum())
    mean_group_label_l1 = float(np.mean(list(per_group_label_l1.values()))) if per_group_label_l1 else None

    # TSTR group-wise performance on real_test
    gm = group_metrics(
        y_true=real_test[target_col].astype(str),
        y_pred=tstr_predictions_on_real_test,
        groups=real_test[group_col].astype(str),
        min_n=1,
    )
    gm_path = os.path.join(out_fairness_dir, f'group_metrics_{model_name}.csv')
    gm.to_csv(gm_path, index=False)

    fairness_gap = None
    if not gm.empty:
        fairness_gap = float(gm['macro_f1'].max() - gm['macro_f1'].min())

    summary = {
        'model': model_name,
        'group_col': group_col,
        'group_proportion_tvd': group_tvd,
        'mean_group_label_distribution_l1': mean_group_label_l1,
        'group_macro_f1_gap_max_minus_min': fairness_gap,
        'group_metrics_csv': gm_path,
    }
    write_json(os.path.join(out_fairness_dir, f'fairness_summary_{model_name}.json'), summary)
    return summary


def decide_winner(comparison_df: pd.DataFrame) -> str:
    ranked = comparison_df.sort_values(
        by=[
            'any_rule_violation_rate',
            'tstr_macro_f1',
            'exact_overlap_with_real_train_rate',
            'group_macro_f1_gap_max_minus_min',
            'fit_seconds',
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    return str(ranked.loc[0, 'model'])


def main() -> None:
    ap = argparse.ArgumentParser(description='Day 2 SynthCity training/evaluation pipeline.')
    ap.add_argument('--input-csv', default='outputs/day1/cleaned_synthcity_input.csv')
    ap.add_argument('--target-col', default='repairability_bin')
    ap.add_argument('--group-col', default='supplier_region')
    ap.add_argument('--output-dir', default='outputs/day2')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--ctgan-iters', type=int, default=120)
    ap.add_argument('--tvae-iters', type=int, default=120)
    args = ap.parse_args()

    dirs = ensure_dirs(args.output_dir)
    errors: List[str] = []

    real_all = load_data(args.input_csv)
    if real_all.isna().any().any():
        raise ValueError('Input has missing values. Please impute/drop before running Day 2.')

    # Step 0: freeze snapshot and split
    snapshot_path = 'data/real_train_snapshot.csv'
    real_all.to_csv(snapshot_path, index=False)
    stratify = real_all[args.target_col] if args.target_col in real_all.columns else None
    real_train, real_test = train_test_split(
        real_all, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )
    real_train = real_train.reset_index(drop=True)
    real_test = real_test.reset_index(drop=True)
    real_train.to_csv('data/real_train.csv', index=False)
    real_test.to_csv('data/real_test.csv', index=False)

    run_context = {
        'seed': args.seed,
        'input_csv': args.input_csv,
        'snapshot_csv': snapshot_path,
        'real_train_csv': 'data/real_train.csv',
        'real_test_csv': 'data/real_test.csv',
        'rows_all': int(len(real_all)),
        'rows_train': int(len(real_train)),
        'rows_test': int(len(real_test)),
        'columns': real_all.columns.tolist(),
        'target_col': args.target_col,
        'group_col': args.group_col,
    }
    write_json(os.path.join(dirs['logs'], 'run_context.json'), run_context)

    # Step 1: plugin sanity
    available_plugins = Plugins().list()
    write_json(os.path.join(dirs['logs'], 'available_plugins.json'), {'plugins': available_plugins})

    selected_models = [m for m in ['ctgan', 'tvae', 'privbayes'] if m in available_plugins]
    if len(selected_models) < 2:
        raise RuntimeError(f'Not enough SynthCity generators available: found {selected_models}')

    # TRTR baseline on split (real_train -> real_test)
    trtr_metrics, trtr_preds = evaluate_classifier(real_train, real_test, args.target_col)
    write_json(os.path.join(dirs['utility'], 'trtr_metrics_real_baseline.json'), trtr_metrics)

    # Per-model loop
    comparison_rows: List[Dict[str, Any]] = []
    run_log: Dict[str, Any] = {
        'selected_models': selected_models,
        'model_runs': [],
        'errors': [],
    }

    for model_name in selected_models:
        try:
            kwargs = {}
            if model_name == 'ctgan':
                kwargs = {'n_iter': args.ctgan_iters}
            elif model_name == 'tvae':
                kwargs = {'n_iter': args.tvae_iters}
            elif model_name == 'privbayes':
                kwargs = {'epsilon': 1.0}

            run = fit_and_generate(
                model_name=model_name,
                train_df=real_train,
                target_col=args.target_col,
                out_dir=dirs['synthetic'],
                plugin_kwargs=kwargs,
            )

            syn_1x = load_data(run.output_1x_csv)
            # Enforce exact column order as real_train for all downstream comparisons.
            syn_1x = syn_1x[real_train.columns.tolist()]
            syn_1x.to_csv(run.output_1x_csv, index=False)

            # Validation
            validation_summary = compute_validation(
                model_name=model_name,
                real_train=real_train,
                syn_df=syn_1x,
                target_col=args.target_col,
                out_validation_dir=dirs['validation'],
            )

            # Utility stats + TSTR
            utility_stats = compute_utility(
                model_name=model_name,
                real_train=real_train,
                syn_df=syn_1x,
                target_col=args.target_col,
                out_utility_dir=dirs['utility'],
            )
            tstr_metrics, tstr_preds = evaluate_classifier(syn_1x, real_test, args.target_col)
            write_json(os.path.join(dirs['utility'], f'tstr_metrics_{model_name}.json'), tstr_metrics)

            # Privacy
            privacy_summary = compute_privacy(
                model_name=model_name,
                real_train=real_train,
                syn_df=syn_1x,
                target_col=args.target_col,
                out_privacy_dir=dirs['privacy'],
            )

            # Fairness
            fairness_summary = compute_fairness(
                model_name=model_name,
                real_train=real_train,
                syn_df=syn_1x,
                real_test=real_test,
                target_col=args.target_col,
                group_col=args.group_col,
                tstr_predictions_on_real_test=tstr_preds,
                out_fairness_dir=dirs['fairness'],
            )

            comparison_rows.append({
                'model': model_name,
                'fit_seconds': run.fit_seconds,
                'generate_1x_seconds': run.generate_1x_seconds,
                'generate_5x_seconds': run.generate_5x_seconds,
                'any_rule_violation_rate': validation_summary['any_rule_violation_rate'],
                'duplicate_row_rate': validation_summary['duplicate_row_rate'],
                'exact_overlap_with_real_train_rate': privacy_summary['exact_overlap_with_real_train_rate'],
                'tstr_accuracy': tstr_metrics['accuracy'],
                'tstr_macro_f1': tstr_metrics['macro_f1'],
                'trtr_macro_f1': trtr_metrics['macro_f1'],
                'tstr_minus_trtr_macro_f1': float(tstr_metrics['macro_f1'] - trtr_metrics['macro_f1']),
                'numeric_correlation_mae': utility_stats['numeric_correlation_mae'],
                'group_macro_f1_gap_max_minus_min': fairness_summary.get('group_macro_f1_gap_max_minus_min'),
                'group_proportion_tvd': fairness_summary.get('group_proportion_tvd'),
                'suspiciously_close_rate': privacy_summary['suspiciously_close_rate_vs_real_real_p01'],
                'output_1x_csv': run.output_1x_csv,
                'output_5x_csv': run.output_5x_csv,
            })

            run_log['model_runs'].append({
                'model': model_name,
                'plugin_kwargs': run.plugin_kwargs,
                'fit_seconds': run.fit_seconds,
                'generate_1x_seconds': run.generate_1x_seconds,
                'generate_5x_seconds': run.generate_5x_seconds,
                'output_1x_csv': run.output_1x_csv,
                'output_5x_csv': run.output_5x_csv,
            })
        except Exception as e:
            msg = f'{model_name} failed: {e}\n{traceback.format_exc()}'
            errors.append(msg)
            run_log['errors'].append(msg)
            continue

    if errors:
        with open(os.path.join(dirs['logs'], 'errors.txt'), 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(errors))

    if not comparison_rows:
        raise RuntimeError('All model runs failed. See outputs/day2/logs/errors.txt')

    comparison_df = pd.DataFrame(comparison_rows)
    winner = decide_winner(comparison_df)
    comparison_df = comparison_df.sort_values(
        by=['any_rule_violation_rate', 'tstr_macro_f1', 'exact_overlap_with_real_train_rate'],
        ascending=[True, False, True],
    )
    comparison_df.to_csv(os.path.join(dirs['comparison'], 'model_comparison_table.csv'), index=False)

    notes_lines = [
        '# Day 2 Decision Notes',
        '',
        f'- Selected primary model: **{winner}**',
        f"- Real baseline (TRTR) macro-F1: {trtr_metrics['macro_f1']:.4f}",
        '',
        '## Decision Criteria',
        '- 1) Lowest validation rule violation rate',
        '- 2) Highest TSTR macro-F1 (utility on held-out real test)',
        '- 3) Lower exact-overlap and suspiciously-close privacy risk',
        '- 4) Smaller fairness gap by group',
        '',
        '## Ranked Models',
    ]
    for _, r in comparison_df.reset_index(drop=True).iterrows():
        notes_lines.append(
            f"- {r['model']}: violation={r['any_rule_violation_rate']:.4f}, "
            f"TSTR_F1={r['tstr_macro_f1']:.4f}, overlap={r['exact_overlap_with_real_train_rate']:.4f}, "
            f"fairness_gap={r['group_macro_f1_gap_max_minus_min']:.4f}"
        )

    with open(os.path.join(dirs['comparison'], 'day2_decision_notes.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(notes_lines) + '\n')

    run_log['winner'] = winner
    run_log['trtr_metrics'] = trtr_metrics
    write_json(os.path.join(dirs['logs'], 'synthcity_run_log.json'), run_log)

    print('Done Day 2 pipeline.')
    print(f"Models run: {', '.join(comparison_df['model'].tolist())}")
    print(f'Winner: {winner}')
    print(f"Comparison table: {os.path.join(dirs['comparison'], 'model_comparison_table.csv')}")


if __name__ == '__main__':
    main()
