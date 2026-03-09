"""End-to-end benchmarking runner for the static train/test split."""

from __future__ import annotations

from dataclasses import replace
import json
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from benchmarking.config import BenchmarkConfig
from benchmarking.data import load_table
from benchmarking.evaluation import (
    save_benchmark_summary,
    save_roc_outputs,
    save_yearly_default_auc_outputs,
)
from benchmarking.features import ensure_parent_dir, select_feature_columns, split_time_aware
from benchmarking.metrics import default_auc_ovr, multiclass_auc_macro_ovr, summarize_classification
from benchmarking.models import align_proba_to_classes, build_model_pipeline
from benchmarking.tuning import tune_one_model


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _save_predictions(
    out_path: Path,
    *,
    base_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    id_columns: tuple[str, ...],
    time_columns: tuple[str, ...],
) -> None:
    keep_cols = [c for c in [*id_columns, *time_columns] if c in base_df.columns]
    pred_df = base_df.loc[:, keep_cols].copy()
    pred_df["y_true"] = y_true.to_numpy()
    pred_df["y_pred"] = y_pred.astype(int)
    pred_df["prob_0"] = y_proba[:, 0]
    pred_df["prob_1"] = y_proba[:, 1]
    pred_df["prob_2"] = y_proba[:, 2]
    ensure_parent_dir(out_path)
    pred_df.to_csv(out_path, index=False)


def run_benchmark(config: BenchmarkConfig, model_names: list[str] | None = None) -> pd.DataFrame:
    """Run all benchmarks and persist outputs."""
    model_list = model_names if model_names else list(config.model_names)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_table(config.train_path)
    test_df = load_table(config.test_path)
    if config.target_col not in train_df.columns or config.target_col not in test_df.columns:
        raise KeyError(
            f"Target column '{config.target_col}' must exist in both train and test tables."
        )

    feature_cols = select_feature_columns(
        train_df,
        target_col=config.target_col,
        id_columns=config.id_columns,
        drop_other_horizon_targets=config.drop_other_horizon_targets,
    )
    if not feature_cols:
        raise ValueError("No feature columns selected after exclusions.")

    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df[config.target_col].astype(int).copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[config.target_col].astype(int).copy()

    if config.validation_fraction > 0.0:
        train_idx, val_idx = split_time_aware(
            train_df,
            validation_fraction=config.validation_fraction,
            year_col=config.time_columns[0] if len(config.time_columns) > 0 else "yyyy",
            month_col=config.time_columns[1] if len(config.time_columns) > 1 else "mm",
        )
        X_fit = X_train_full.iloc[train_idx].copy()
        y_fit = y_train_full.iloc[train_idx].copy()
        X_val = X_train_full.iloc[val_idx].copy()
        y_val = y_train_full.iloc[val_idx].copy()
    else:
        X_fit = X_train_full
        y_fit = y_train_full
        X_val = None
        y_val = None

    _safe_json_dump(
        config.output_dir / "run_config.json",
        {
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "target_col": config.target_col,
            "classes": list(config.classes),
            "default_class": config.default_class,
            "validation_fraction": config.validation_fraction,
            "drop_other_horizon_targets": config.drop_other_horizon_targets,
            "tune_hyperparameters": config.tune_hyperparameters,
            "tuning_validation_fraction": config.tuning_validation_fraction,
            "max_tuning_trials_per_model": config.max_tuning_trials_per_model,
            "features_used": feature_cols,
            "models": model_list,
        },
    )

    default_class_idx = config.classes.index(config.default_class)
    roc_inputs: dict[str, dict[str, Any]] = {}
    prediction_files: dict[str, str] = {}
    tuning_summary_rows: list[dict[str, Any]] = []
    tuning_dir = config.output_dir / "tuning"
    rows: list[dict[str, Any]] = []
    for model_name in model_list:
        result: dict[str, Any] = {"model": model_name, "status": "ok"}
        try:
            params = dict(config.model_params.get(model_name, {}))
            if config.tune_hyperparameters:
                best_delta, tuning_results_df = tune_one_model(
                    model_name=model_name,
                    config=config,
                    X_train_full=X_train_full,
                    y_train_full=y_train_full,
                    train_df=train_df,
                    tuning_dir=tuning_dir,
                )
                params = {**params, **best_delta}
                top_row = tuning_results_df.iloc[0].to_dict() if not tuning_results_df.empty else {}
                result["tuned"] = True
                result["tuning_best_val_default_auc"] = top_row.get("validation_default_auc", np.nan)
                result["best_params_json"] = json.dumps(best_delta, sort_keys=True)
                tuning_summary_rows.append(
                    {
                        "model_name": model_name,
                        "best_validation_default_auc": top_row.get("validation_default_auc", np.nan),
                        "n_trials": int(len(tuning_results_df)),
                        "best_params_json": json.dumps(best_delta, sort_keys=True),
                    }
                )

            X_final_fit = X_train_full if config.tune_hyperparameters else X_fit
            y_final_fit = y_train_full if config.tune_hyperparameters else y_fit
            pipeline = build_model_pipeline(
                model_name,
                params=params,
                random_state=config.random_state,
                X_fit=X_final_fit,
            )
            pipeline.fit(X_final_fit, y_final_fit)

            model_classes = pipeline.named_steps["model"].classes_

            if (not config.tune_hyperparameters) and X_val is not None and y_val is not None:
                val_proba = align_proba_to_classes(
                    pipeline.predict_proba(X_val),
                    model_classes=model_classes,
                    expected_classes=config.classes,
                )
                val_pred = np.asarray(config.classes)[np.argmax(val_proba, axis=1)]
                result["val_default_auc"] = default_auc_ovr(
                    y_val,
                    val_proba[:, config.classes.index(config.default_class)],
                    default_class=config.default_class,
                )
                result["val_multiclass_auc_macro"] = multiclass_auc_macro_ovr(
                    y_val,
                    val_proba,
                    labels=config.classes,
                )
                result.update({f"val_{k}": v for k, v in summarize_classification(y_val, val_pred).items()})

            test_proba = align_proba_to_classes(
                pipeline.predict_proba(X_test),
                model_classes=model_classes,
                expected_classes=config.classes,
            )
            test_pred = np.asarray(config.classes)[np.argmax(test_proba, axis=1)]

            result["test_default_auc"] = default_auc_ovr(
                y_test,
                test_proba[:, default_class_idx],
                default_class=config.default_class,
            )
            result["test_multiclass_auc_macro"] = multiclass_auc_macro_ovr(
                y_test,
                test_proba,
                labels=config.classes,
            )
            result.update({f"test_{k}": v for k, v in summarize_classification(y_test, test_pred).items()})

            pred_path = config.output_dir / "predictions" / f"{model_name}_test_predictions.csv"
            _save_predictions(
                pred_path,
                base_df=test_df,
                y_true=y_test,
                y_pred=test_pred,
                y_proba=test_proba,
                id_columns=config.id_columns,
                time_columns=config.time_columns,
            )
            result["prediction_file"] = str(pred_path)
            prediction_files[model_name] = str(pred_path)
            roc_inputs[model_name] = {
                "y_true": y_test.to_numpy(),
                "y_score": test_proba[:, default_class_idx],
            }

            if config.save_models:
                model_path = config.output_dir / "models" / f"{model_name}.joblib"
                try:
                    ensure_parent_dir(model_path)
                    joblib.dump(
                        {
                            "model_name": model_name,
                            "target_col": config.target_col,
                            "feature_columns": feature_cols,
                            "classes": config.classes,
                            "pipeline": pipeline,
                        },
                        model_path,
                    )
                    result["model_artifact"] = str(model_path)
                except Exception as exc:
                    result["model_artifact"] = ""
                    result["model_artifact_error"] = str(exc)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()

        rows.append(result)

    metrics_df = pd.DataFrame(rows)
    if "test_default_auc" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(by="test_default_auc", ascending=False, na_position="last")
    metrics_path = config.output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    tuning_summary_path = None
    if tuning_summary_rows:
        tuning_summary_path = tuning_dir / "tuning_summary.csv"
        pd.DataFrame(tuning_summary_rows).sort_values(
            by="best_validation_default_auc", ascending=False, na_position="last"
        ).to_csv(tuning_summary_path, index=False)

    results_dir = config.output_dir / "results"
    summary_path = results_dir / f"benchmark_summary_{config.target_col}.csv"
    save_benchmark_summary(metrics_df, output_path=summary_path, target_col=config.target_col)

    evaluation_outputs: dict[str, Any] = {
        "metrics_summary_csv": str(metrics_path),
        "benchmark_summary_csv": str(summary_path),
    }
    if tuning_summary_path is not None:
        evaluation_outputs["tuning_summary_csv"] = str(tuning_summary_path)
        evaluation_outputs["tuning_dir"] = str(tuning_dir)
    if config.generate_roc_plots and roc_inputs:
        plots_dir = results_dir / "plots"
        try:
            evaluation_outputs.update(
                save_roc_outputs(
                    roc_inputs,
                    output_dir=plots_dir,
                    target_col=config.target_col,
                    default_class=config.default_class,
                )
            )
        except ImportError as exc:
            evaluation_outputs["roc_plots_status"] = "skipped"
            evaluation_outputs["roc_plots_error"] = str(exc)

    if prediction_files:
        yearly_outputs = save_yearly_default_auc_outputs(
            prediction_files,
            output_dir=results_dir,
            plots_dir=results_dir / "plots",
            target_col=config.target_col,
            year_col=config.time_columns[0] if len(config.time_columns) > 0 else "yyyy",
            default_class=config.default_class,
        )
        evaluation_outputs.update(yearly_outputs)

    _safe_json_dump(results_dir / "evaluation_outputs.json", evaluation_outputs)
    return metrics_df


def run_benchmarks_for_horizons(
    config: BenchmarkConfig,
    *,
    horizons: list[int],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Run benchmark across multiple horizon targets and aggregate outputs."""
    if not horizons:
        raise ValueError("horizons cannot be empty.")

    deduped_horizons = sorted({int(h) for h in horizons})
    base_output_dir = config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for horizon in deduped_horizons:
        target_col = f"y_{horizon}m"
        horizon_output_dir = base_output_dir / f"h{horizon}"
        horizon_cfg = replace(
            config,
            target_col=target_col,
            output_dir=horizon_output_dir,
        )
        try:
            metrics_df = run_benchmark(horizon_cfg, model_names=model_names)
            tagged_df = metrics_df.copy()
            tagged_df.insert(0, "target_col", target_col)
            tagged_df.insert(0, "horizon_months", horizon)
            all_frames.append(tagged_df)
            run_rows.append(
                {
                    "horizon_months": horizon,
                    "target_col": target_col,
                    "status": "ok",
                    "output_dir": str(horizon_output_dir),
                }
            )
        except Exception as exc:
            run_rows.append(
                {
                    "horizon_months": horizon,
                    "target_col": target_col,
                    "status": "failed",
                    "error": str(exc),
                    "output_dir": str(horizon_output_dir),
                }
            )

    run_index_df = pd.DataFrame(run_rows).sort_values("horizon_months")
    run_index_path = base_output_dir / "horizon_run_index.csv"
    run_index_df.to_csv(run_index_path, index=False)
    _safe_json_dump(
        base_output_dir / "multi_horizon_runs.json",
        {
            "horizons": deduped_horizons,
            "target_columns": [f"y_{h}m" for h in deduped_horizons],
            "run_index_csv": str(run_index_path),
            "run_count": len(deduped_horizons),
        },
    )

    if not all_frames:
        raise RuntimeError("All horizon runs failed. Check horizon_run_index.csv for details.")

    combined_df = pd.concat(all_frames, ignore_index=True)
    if "test_default_auc" in combined_df.columns:
        combined_df = combined_df.sort_values(
            by=["horizon_months", "test_default_auc"],
            ascending=[True, False],
            na_position="last",
        )
    else:
        combined_df = combined_df.sort_values(by=["horizon_months", "model"], na_position="last")
    combined_df.to_csv(base_output_dir / "metrics_summary_all_horizons.csv", index=False)

    if "test_default_auc" in combined_df.columns:
        model_horizon_summary = (
            combined_df.groupby(["model", "horizon_months"], as_index=False)["test_default_auc"]
            .mean()
            .rename(columns={"test_default_auc": "mean_test_default_auc"})
            .sort_values(["horizon_months", "mean_test_default_auc"], ascending=[True, False])
        )
        model_horizon_summary.to_csv(
            base_output_dir / "benchmark_summary_all_horizons.csv",
            index=False,
        )

    return combined_df
