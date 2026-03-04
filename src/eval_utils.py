import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_baseline_metrics(path: str | Path) -> Dict[str, Any]:
    metrics = load_json(path)
    best = metrics.get("selected_best_model")
    best_metrics = metrics.get("models", {}).get(best, {}) if best else {}
    return {
        "selected_best_model": best,
        "accuracy": best_metrics.get("accuracy"),
        "macro_f1": best_metrics.get("macro_f1"),
        "n_rows_used": metrics.get("n_rows_used"),
        "n_features": metrics.get("n_features"),
    }
