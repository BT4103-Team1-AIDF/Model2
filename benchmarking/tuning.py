"""Time-aware, train-only hyperparameter tuning utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from benchmarking.config import BenchmarkConfig
from benchmarking.features import ensure_parent_dir, split_time_aware
from benchmarking.metrics import default_auc_ovr
from benchmarking.models import align_proba_to_classes, build_model_pipeline


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _build_param_candidates(model_name: str, grid: dict[str, list]) -> list[dict[str, Any]]:
    """Build valid parameter combinations for a model."""
    candidates = [dict(row) for row in ParameterGrid(grid)]
    if model_name != "logistic_regression":
        return candidates

    valid: list[dict[str, Any]] = []
    for params in candidates:
        solver = str(params.get("solver", "lbfgs"))
        if solver not in {"lbfgs", "liblinear"}:
            continue
        valid.append(params)
    return valid


def tune_one_model(
    *,
    model_name: str,
    config: BenchmarkConfig,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    train_df: pd.DataFrame,
    tuning_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune one model on a chronological train/validation split from train only."""
    if not 0.0 < config.tuning_validation_fraction < 1.0:
        raise ValueError("tuning_validation_fraction must be in (0, 1).")

    year_col = config.time_columns[0] if len(config.time_columns) > 0 else "yyyy"
    month_col = config.time_columns[1] if len(config.time_columns) > 1 else "mm"
    fit_idx, val_idx = split_time_aware(
        train_df,
        validation_fraction=config.tuning_validation_fraction,
        year_col=year_col,
        month_col=month_col,
    )
    X_fit = X_train_full.iloc[fit_idx].copy()
    y_fit = y_train_full.iloc[fit_idx].copy()
    X_val = X_train_full.iloc[val_idx].copy()
    y_val = y_train_full.iloc[val_idx].copy()

    grid = config.tuning_param_grid.get(model_name, {})
    if not grid:
        raise ValueError(f"No tuning grid configured for model '{model_name}'.")

    all_candidates = _build_param_candidates(model_name, grid)
    if not all_candidates:
        raise ValueError(f"No valid tuning candidates for model '{model_name}'.")

    model_seed_offset = sum(ord(ch) for ch in model_name) % 10000
    rng = np.random.default_rng(config.random_state + model_seed_offset)
    order = rng.permutation(len(all_candidates)).tolist()
    selected_idx = order[: min(config.max_tuning_trials_per_model, len(order))]
    selected = [all_candidates[i] for i in selected_idx]

    base_params = dict(config.model_params.get(model_name, {}))
    default_class_idx = config.classes.index(config.default_class)
    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None

    for trial_id, delta in enumerate(selected, start=1):
        merged = {**base_params, **delta}
        status = "ok"
        error = ""
        score = float("nan")

        try:
            pipeline = build_model_pipeline(
                model_name,
                params=merged,
                random_state=config.random_state,
                X_fit=X_fit,
            )
            pipeline.fit(X_fit, y_fit)
            val_raw = pipeline.predict_proba(X_val)
            val_aligned = align_proba_to_classes(
                val_raw,
                model_classes=pipeline.named_steps["model"].classes_,
                expected_classes=config.classes,
            )
            score = default_auc_ovr(
                y_val,
                val_aligned[:, default_class_idx],
                default_class=config.default_class,
            )
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_params = delta
        except Exception as exc:
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "model_name": model_name,
                "trial_id": trial_id,
                "params_json": json.dumps(delta, sort_keys=True),
                "validation_default_auc": score,
                "status": status,
                "error": error,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["validation_default_auc", "trial_id"], ascending=[False, True], na_position="last"
    )
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    results_df["is_best"] = False
    if best_params is not None:
        best_match = results_df["params_json"] == json.dumps(best_params, sort_keys=True)
        if best_match.any():
            results_df.loc[best_match.idxmax(), "is_best"] = True
    else:
        best_params = {}

    tuning_dir.mkdir(parents=True, exist_ok=True)
    results_path = tuning_dir / f"tuning_results_{model_name}.csv"
    results_df.to_csv(results_path, index=False)

    best_params_payload = {
        "model_name": model_name,
        "best_params": best_params,
        "best_validation_default_auc": None if not np.isfinite(best_score) else float(best_score),
        "n_trials": int(len(results_df)),
        "validation_fraction": config.tuning_validation_fraction,
        "fit_rows": int(len(X_fit)),
        "validation_rows": int(len(X_val)),
        "time_columns": list(config.time_columns),
    }
    _safe_json_dump(tuning_dir / f"best_params_{model_name}.json", best_params_payload)
    return best_params, results_df
