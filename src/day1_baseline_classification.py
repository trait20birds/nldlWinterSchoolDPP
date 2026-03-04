import argparse
import json
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NA_VALUES = ['', ' ', 'nan', 'NaN', 'null', 'NULL', 'none', 'None']


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


def fairness_group_metrics(y_true, y_pred, group_values: pd.Series):
    rows = []
    groups = group_values.fillna('<MISSING>').astype(str)
    for g in sorted(groups.unique()):
        idx = groups == g
        if idx.sum() < 5:
            continue
        yt = pd.Series(y_true)[idx]
        yp = pd.Series(y_pred)[idx]
        rows.append({
            'group': g,
            'n': int(idx.sum()),
            'accuracy': float(accuracy_score(yt, yp)),
            'macro_f1': float(f1_score(yt, yp, average='macro')),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Train a simple Day 1 baseline classifier for DPP/BOM data.')
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--target-col', required=True)
    ap.add_argument('--group-col', default='')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--min-class-count', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Keep categorical codes like 'NA' as values rather than converting to missing.
    df = pd.read_csv(args.input_csv, keep_default_na=False, na_values=NA_VALUES)

    target_col = args.target_col.strip().lower().replace(' ', '_')
    group_col = args.group_col.strip().lower().replace(' ', '_') if args.group_col else ''

    # Align if user passed exact but not normalized names
    if target_col not in df.columns and args.target_col in df.columns:
        target_col = args.target_col
    if group_col and group_col not in df.columns and args.group_col in df.columns:
        group_col = args.group_col

    if target_col not in df.columns:
        raise ValueError(f'Target column not found: {args.target_col}')

    # drop missing target
    work = df.copy()
    work = work[work[target_col].notna()].copy()

    # Convert target to string labels for classification robustness
    y = work[target_col].astype(str)

    # Filter ultra-rare classes (can break stratified split / F1)
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < args.min_class_count].index.tolist()
    if rare_classes:
        work = work[~work[target_col].astype(str).isin(rare_classes)].copy()
        y = work[target_col].astype(str)

    if y.nunique() < 2:
        raise ValueError('Need at least 2 target classes after filtering rare classes.')

    feature_cols = [c for c in work.columns if c != target_col]
    X = work[feature_cols].copy()

    # Exclude target leakage-ish columns by simple heuristic (safe, conservative)
    leakage_keywords = ['label', 'target', 'class']
    filtered_feature_cols = []
    dropped_id_like_cols = []
    for c in feature_cols:
        lc = c.lower()
        if any(k in lc for k in leakage_keywords) and c != group_col:
            # skip suspicious labels other than group col
            continue
        if c != group_col and is_id_like_column(c, work[c]):
            dropped_id_like_cols.append(c)
            continue
        filtered_feature_cols.append(c)
    X = X[filtered_feature_cols]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

    models = {
        'logreg': LogisticRegression(max_iter=2000, n_jobs=None),
        'rf': RandomForestClassifier(
            n_estimators=300, random_state=args.random_state, n_jobs=-1, class_weight='balanced'
        ),
    }

    all_metrics = {
        'input_csv': args.input_csv,
        'target_col': target_col,
        'group_col': group_col or None,
        'n_rows_used': int(len(work)),
        'n_features': int(X.shape[1]),
        'n_classes': int(y.nunique()),
        'class_distribution': y.value_counts().to_dict(),
        'dropped_id_like_feature_cols': dropped_id_like_cols,
        'models': {},
        'selected_best_model': None,
    }

    best_name: Optional[str] = None
    best_score = -1.0
    best_pipeline = None
    best_preds = None
    best_y_test = None

    for name, model in models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average='macro')
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))

        all_metrics['models'][name] = {
            'accuracy': float(acc),
            'macro_f1': float(mf1),
            'classification_report': report,
            'labels_order': sorted(y.unique().tolist()),
        }

        pd.DataFrame(
            cm,
            index=[f'true_{lbl}' for lbl in sorted(y.unique())],
            columns=[f'pred_{lbl}' for lbl in sorted(y.unique())],
        ).to_csv(os.path.join(args.output_dir, f'confusion_matrix_{name}.csv'))

        if mf1 > best_score:
            best_score = mf1
            best_name = name
            best_pipeline = pipe
            best_preds = preds
            best_y_test = y_test

    all_metrics['selected_best_model'] = best_name

    # Fairness-ish group metrics (performance by group) on test split only, if group col exists in original X
    if group_col and group_col in X.columns and best_preds is not None and best_y_test is not None:
        group_test_vals = X_test[group_col] if group_col in X_test.columns else pd.Series(['<NA>'] * len(X_test))
        gm = fairness_group_metrics(best_y_test.reset_index(drop=True), pd.Series(best_preds), group_test_vals.reset_index(drop=True))
        if not gm.empty:
            gm.to_csv(os.path.join(args.output_dir, 'group_metrics.csv'), index=False)
            fairness_summary = {
                'metric': 'macro_f1_and_accuracy_by_group',
                'groups_evaluated': gm['group'].tolist(),
                'macro_f1_gap_max_minus_min': float(gm['macro_f1'].max() - gm['macro_f1'].min()),
                'accuracy_gap_max_minus_min': float(gm['accuracy'].max() - gm['accuracy'].min()),
            }
            with open(os.path.join(args.output_dir, 'fairness_gap_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(fairness_summary, f, indent=2)

    # Save best model and metrics
    if best_pipeline is not None:
        joblib.dump(best_pipeline, os.path.join(args.output_dir, f'best_model_{best_name}.joblib'))

    with open(os.path.join(args.output_dir, 'baseline_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    # Feature importance for RF (best if RF selected, else still attempt if rf exists)
    try:
        rf_pipe = None
        if best_name == 'rf':
            rf_pipe = best_pipeline
        else:
            # retrain rf for plot (cheap enough)
            rf_pipe = Pipeline([('prep', preprocessor), ('model', models['rf'])])
            rf_pipe.fit(X_train, y_train)

        prep = rf_pipe.named_steps['prep']
        model = rf_pipe.named_steps['model']
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            ohe = prep.named_transformers_['cat'].named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

        importances = model.feature_importances_
        fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False).head(20)
        fi.to_csv(os.path.join(args.output_dir, 'feature_importance_top20.csv'), index=False)

        plt.figure(figsize=(9, 6))
        plt.barh(fi['feature'][::-1], fi['importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importances (RandomForest)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'fig_feature_importance.png'), dpi=150)
        plt.close()
    except Exception as e:
        with open(os.path.join(args.output_dir, 'feature_importance_warning.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Could not compute feature importance plot: {e}\n')

    print('Done.')
    print(f"Best model: {best_name} (macro_f1={best_score:.4f})")
    print(f"Saved metrics to {os.path.join(args.output_dir, 'baseline_metrics.json')}")


if __name__ == '__main__':
    main()
