"""Evaluation outputs for multi-horizon benchmark comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve

from benchmarking.features import ensure_parent_dir
from benchmarking.metrics import default_auc_ovr


MODEL_FILE_STEM = {
    "logistic_regression": "logistic",
    "random_forest": "random_forest",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "lstm": "lstm",
}

MODEL_DISPLAY_NAME = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}


def save_benchmark_summary(
    metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    target_col: str,
) -> pd.DataFrame:
    """Save a clean benchmark summary table for one target horizon."""
    preferred_cols = [
        "model",
        "status",
        "test_default_auc",
        "test_multiclass_auc_macro",
        "test_accuracy",
        "test_macro_f1",
        "val_default_auc",
        "val_multiclass_auc_macro",
        "val_accuracy",
        "val_macro_f1",
    ]
    keep = [c for c in preferred_cols if c in metrics_df.columns]
    summary = metrics_df.loc[:, keep].copy()
    summary.insert(0, "target", target_col)
    summary = summary.sort_values(by="test_default_auc", ascending=False, na_position="last")
    ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    return summary


def _compute_roc_frame(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    default_class: int,
) -> tuple[pd.DataFrame, float]:
    y_binary = (y_true.astype(int) == int(default_class)).astype(int)
    fpr, tpr, thresholds = roc_curve(y_binary, y_score)
    auc = default_auc_ovr(y_true, y_score, default_class=default_class)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}), auc


def save_roc_outputs(
    roc_inputs: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    target_col: str,
    default_class: int,
) -> dict[str, str]:
    """Save per-model and combined ROC plots for class-1 default vs rest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_points_dir = output_dir / "roc_points"
    roc_points_dir.mkdir(parents=True, exist_ok=True)

    combined: list[tuple[str, str, pd.DataFrame, float]] = []
    palette = {
        "logistic_regression": (37, 99, 235),
        "random_forest": (16, 158, 103),
        "lightgbm": (245, 158, 11),
        "xgboost": (220, 38, 38),
        "lstm": (124, 58, 237),
    }

    for model_name, payload in roc_inputs.items():
        y_true = np.asarray(payload["y_true"])
        y_score = np.asarray(payload["y_score"])
        roc_df, auc = _compute_roc_frame(y_true, y_score, default_class=default_class)
        stem = MODEL_FILE_STEM.get(model_name, model_name)
        display = MODEL_DISPLAY_NAME.get(model_name, model_name)

        roc_points_path = roc_points_dir / f"roc_points_{stem}_{target_col}.csv"
        roc_df.to_csv(roc_points_path, index=False)

        out_path = output_dir / f"roc_{stem}_{target_col}.png"
        _render_roc_plot_pillow(
            curves=[(f"{display} (AUC={auc:.4f})", roc_df, palette.get(model_name, (30, 30, 30)))],
            title=f"ROC Curve - {display} ({target_col}, class {default_class} vs rest)",
            out_path=out_path,
        )

        combined.append((model_name, display, roc_df, auc))

    combined_path = output_dir / f"roc_combined_{target_col}.png"
    curves: list[tuple[str, pd.DataFrame, tuple[int, int, int]]] = []
    for model_name, display, roc_df, auc in combined:
        curves.append((f"{display} (AUC={auc:.4f})", roc_df, palette.get(model_name, (30, 30, 30))))
    _render_roc_plot_pillow(
        curves=curves,
        title=f"ROC Comparison ({target_col}, class {default_class} vs rest)",
        out_path=combined_path,
    )

    return {
        "plots_dir": str(output_dir),
        "roc_combined_plot": str(combined_path),
        "roc_points_dir": str(roc_points_dir),
    }


def save_yearly_default_auc_outputs(
    prediction_files: dict[str, str | Path],
    *,
    output_dir: Path,
    plots_dir: Path | None,
    target_col: str,
    year_col: str,
    default_class: int,
) -> dict[str, str]:
    """Save year-by-year default-class AUC table and line plots."""
    yearly_rows: list[dict[str, Any]] = []
    series_by_model: dict[str, pd.DataFrame] = {}
    palette = {
        "logistic_regression": (37, 99, 235),
        "random_forest": (16, 158, 103),
        "lightgbm": (245, 158, 11),
        "xgboost": (220, 38, 38),
        "lstm": (124, 58, 237),
    }

    for model_name, pred_path_like in prediction_files.items():
        pred_path = Path(pred_path_like)
        pred_df = pd.read_csv(pred_path)
        required = {year_col, "y_true", "prob_1"}
        missing = required.difference(pred_df.columns)
        if missing:
            raise KeyError(f"Missing required columns {sorted(missing)} in {pred_path}.")

        model_rows: list[dict[str, Any]] = []
        grouped = pred_df.groupby(year_col, dropna=False)
        for year, grp in grouped:
            y_true = grp["y_true"].to_numpy().astype(int)
            y_score = grp["prob_1"].to_numpy()
            y_binary = (y_true == int(default_class)).astype(int)
            n_samples = int(len(grp))
            n_default = int(y_binary.sum())
            n_non_default = int(n_samples - n_default)
            valid = bool(np.unique(y_binary).size >= 2)
            auc = default_auc_ovr(y_true, y_score, default_class=default_class) if valid else float("nan")

            row = {
                "year": year,
                "model_name": model_name,
                "default_auc": auc,
                "n_samples": n_samples,
                "n_default": n_default,
                "n_non_default": n_non_default,
                "valid_auc_flag": valid,
            }
            model_rows.append(row)
            yearly_rows.append(row)

        model_yearly_df = pd.DataFrame(model_rows).sort_values("year").reset_index(drop=True)
        series_by_model[model_name] = model_yearly_df

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_plots_dir = plots_dir if plots_dir is not None else output_dir
    effective_plots_dir.mkdir(parents=True, exist_ok=True)
    yearly_csv = output_dir / f"yearly_default_auc_{target_col}.csv"
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["model_name", "year"]).reset_index(drop=True)
    yearly_df.to_csv(yearly_csv, index=False)

    for model_name, model_df in series_by_model.items():
        stem = MODEL_FILE_STEM.get(model_name, model_name)
        display = MODEL_DISPLAY_NAME.get(model_name, model_name)
        plot_path = effective_plots_dir / f"yearly_auc_{stem}_{target_col}.png"
        _render_yearly_auc_plot_pillow(
            series=[(f"{display}", model_df, palette.get(model_name, (30, 30, 30)))],
            title=f"Yearly Default-Class AUC - {display} ({target_col})",
            out_path=plot_path,
        )

    combined_path = effective_plots_dir / f"yearly_auc_combined_{target_col}.png"
    combined_series: list[tuple[str, pd.DataFrame, tuple[int, int, int]]] = []
    for model_name, model_df in series_by_model.items():
        display = MODEL_DISPLAY_NAME.get(model_name, model_name)
        combined_series.append((display, model_df, palette.get(model_name, (30, 30, 30))))
    _render_yearly_auc_plot_pillow(
        series=combined_series,
        title=f"Yearly Default-Class AUC Comparison ({target_col})",
        out_path=combined_path,
    )

    return {
        "yearly_default_auc_csv": str(yearly_csv),
        "yearly_auc_combined_plot": str(combined_path),
        "yearly_auc_plots_dir": str(effective_plots_dir),
    }


def _render_roc_plot_pillow(
    curves: list[tuple[str, pd.DataFrame, tuple[int, int, int]]],
    *,
    title: str,
    out_path: Path,
) -> None:
    width, height = 1200, 900
    margin_left, margin_right = 120, 360
    margin_top, margin_bottom = 90, 110
    plot_left, plot_top = margin_left, margin_top
    plot_right, plot_bottom = width - margin_right, height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for tick in np.linspace(0.0, 1.0, 6):
        x = plot_left + int(tick * plot_w)
        y = plot_bottom - int(tick * plot_h)
        draw.line([(x, plot_top), (x, plot_bottom)], fill=(230, 230, 230), width=1)
        draw.line([(plot_left, y), (plot_right, y)], fill=(230, 230, 230), width=1)
        draw.text((x - 10, plot_bottom + 12), f"{tick:.1f}", fill=(60, 60, 60), font=font)
        draw.text((plot_left - 38, y - 6), f"{tick:.1f}", fill=(60, 60, 60), font=font)

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(0, 0, 0), width=2)
    draw.line([(plot_left, plot_bottom), (plot_left, plot_top)], fill=(0, 0, 0), width=2)

    chance_points = []
    for v in np.linspace(0.0, 1.0, 100):
        x = plot_left + int(v * plot_w)
        y = plot_bottom - int(v * plot_h)
        chance_points.append((x, y))
    for i in range(0, len(chance_points) - 1, 2):
        draw.line([chance_points[i], chance_points[i + 1]], fill=(150, 150, 150), width=2)

    for _, roc_df, color in curves:
        points: list[tuple[int, int]] = []
        for fpr, tpr in zip(roc_df["fpr"], roc_df["tpr"], strict=False):
            x = plot_left + int(float(fpr) * plot_w)
            y = plot_bottom - int(float(tpr) * plot_h)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)

    draw.text((margin_left, 30), title, fill=(20, 20, 20), font=font)
    draw.text((plot_left + plot_w // 2 - 75, height - 45), "False Positive Rate", fill=(20, 20, 20), font=font)
    draw.text((20, plot_top + plot_h // 2), "True Positive Rate", fill=(20, 20, 20), font=font)

    legend_x = plot_right + 25
    legend_y = plot_top
    draw.text((legend_x, legend_y), "Legend", fill=(20, 20, 20), font=font)
    legend_y += 25
    for label, _, color in curves:
        draw.line([(legend_x, legend_y + 6), (legend_x + 25, legend_y + 6)], fill=color, width=3)
        draw.text((legend_x + 35, legend_y), label, fill=(20, 20, 20), font=font)
        legend_y += 22
    draw.line([(legend_x, legend_y + 6), (legend_x + 25, legend_y + 6)], fill=(150, 150, 150), width=2)
    draw.text((legend_x + 35, legend_y), "Chance", fill=(20, 20, 20), font=font)

    ensure_parent_dir(out_path)
    img.save(out_path)


def _render_yearly_auc_plot_pillow(
    series: list[tuple[str, pd.DataFrame, tuple[int, int, int]]],
    *,
    title: str,
    out_path: Path,
) -> None:
    width, height = 1300, 900
    margin_left, margin_right = 120, 340
    margin_top, margin_bottom = 90, 120
    plot_left, plot_top = margin_left, margin_top
    plot_right, plot_bottom = width - margin_right, height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    years = sorted(
        {
            int(y)
            for _, df, _ in series
            for y in df["year"].tolist()
            if pd.notna(y)
        }
    )
    if not years:
        years = [0, 1]
    year_min, year_max = min(years), max(years)
    year_span = max(1, year_max - year_min)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for tick in np.linspace(0.0, 1.0, 6):
        y = plot_bottom - int(tick * plot_h)
        draw.line([(plot_left, y), (plot_right, y)], fill=(230, 230, 230), width=1)
        draw.text((plot_left - 42, y - 6), f"{tick:.1f}", fill=(60, 60, 60), font=font)

    n_ticks = min(8, len(years))
    year_ticks = sorted(set(np.linspace(0, len(years) - 1, num=n_ticks, dtype=int).tolist()))
    for idx in year_ticks:
        year = years[idx]
        x = plot_left + int(((year - year_min) / year_span) * plot_w)
        draw.line([(x, plot_top), (x, plot_bottom)], fill=(235, 235, 235), width=1)
        draw.text((x - 16, plot_bottom + 12), str(year), fill=(60, 60, 60), font=font)

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(0, 0, 0), width=2)
    draw.line([(plot_left, plot_bottom), (plot_left, plot_top)], fill=(0, 0, 0), width=2)

    for _, df, color in series:
        valid_df = df[df["valid_auc_flag"]].copy()
        if valid_df.empty:
            continue
        points: list[tuple[int, int]] = []
        for _, row in valid_df.iterrows():
            year = int(row["year"])
            auc = float(row["default_auc"])
            x = plot_left + int(((year - year_min) / year_span) * plot_w)
            y = plot_bottom - int(max(0.0, min(1.0, auc)) * plot_h)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=color, outline=color)

    draw.text((margin_left, 30), title, fill=(20, 20, 20), font=font)
    draw.text((plot_left + plot_w // 2 - 40, height - 45), "Year", fill=(20, 20, 20), font=font)
    draw.text((15, plot_top + plot_h // 2), "Default-Class ROC AUC", fill=(20, 20, 20), font=font)

    legend_x = plot_right + 25
    legend_y = plot_top
    draw.text((legend_x, legend_y), "Legend", fill=(20, 20, 20), font=font)
    legend_y += 25
    for label, df, color in series:
        n_invalid = int((~df["valid_auc_flag"]).sum()) if "valid_auc_flag" in df.columns else 0
        legend_label = f"{label}" if n_invalid == 0 else f"{label} (invalid years={n_invalid})"
        draw.line([(legend_x, legend_y + 6), (legend_x + 25, legend_y + 6)], fill=color, width=3)
        draw.ellipse([(legend_x + 9, legend_y + 2), (legend_x + 17, legend_y + 10)], fill=color, outline=color)
        draw.text((legend_x + 35, legend_y), legend_label, fill=(20, 20, 20), font=font)
        legend_y += 22

    ensure_parent_dir(out_path)
    img.save(out_path)
