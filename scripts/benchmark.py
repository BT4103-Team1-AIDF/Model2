"""Run static-split benchmark models for one or more horizons."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.config import BenchmarkConfig
from benchmarking.runner import run_benchmark, run_benchmarks_for_horizons


DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 36, 48, 60)
TARGET_HORIZON_PATTERN = re.compile(r"^y_(\d+)m$")


def _parse_horizons(raw: str, default_horizons: tuple[int, ...]) -> list[int]:
    value = raw.strip().lower()
    if value in {"all", "default"}:
        return list(default_horizons)

    horizons: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            horizon = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid horizon '{token}'. Use comma-separated integers or 'all'.") from exc
        if horizon <= 0:
            raise ValueError("Horizon values must be positive integers.")
        horizons.append(horizon)

    if not horizons:
        raise ValueError("No horizon values parsed.")
    return sorted(set(horizons))


def _horizon_from_target_col(target_col: str) -> int:
    match = TARGET_HORIZON_PATTERN.match(target_col.strip())
    if not match:
        raise ValueError(
            f"Unable to infer horizon from target_col='{target_col}'. Expected pattern like y_12m."
        )
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "benchmark")
    parser.add_argument("--target-col", type=str, default="y_12m")
    parser.add_argument(
        "--horizons",
        type=str,
        default="",
        help="Comma-separated horizons (e.g., 1,3,6,12,24,36,48,60) or 'all'. If empty, uses --target-col only.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated model list or 'all'. Supported: "
            "logistic_regression,random_forest,lightgbm,xgboost,lstm"
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Optional time-aware validation split fraction from train only (e.g., 0.2).",
    )
    parser.add_argument(
        "--keep-other-horizon-targets",
        action="store_true",
        help="Keep y_1m..y_60m columns as features. Default is to drop them to avoid leakage.",
    )
    parser.add_argument("--no-save-models", action="store_true", help="Skip writing model artifacts.")
    parser.add_argument("--skip-roc-plots", action="store_true", help="Skip generating ROC curve plots.")
    parser.add_argument("--tune", action="store_true", help="Enable leakage-safe hyperparameter tuning.")
    parser.add_argument(
        "--tuning-validation-fraction",
        type=float,
        default=0.2,
        help="Chronological validation fraction from train used for tuning.",
    )
    parser.add_argument(
        "--max-tuning-trials",
        type=int,
        default=20,
        help="Maximum tuning trials per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.models.strip().lower() == "all":
        model_names = None
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    parsed_horizons = (
        _parse_horizons(args.horizons, DEFAULT_HORIZONS)
        if args.horizons.strip()
        else [_horizon_from_target_col(args.target_col)]
    )

    config = BenchmarkConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        target_col=f"y_{parsed_horizons[0]}m",
        validation_fraction=args.validation_fraction,
        drop_other_horizon_targets=not args.keep_other_horizon_targets,
        save_models=not args.no_save_models,
        generate_roc_plots=not args.skip_roc_plots,
        tune_hyperparameters=args.tune,
        tuning_validation_fraction=args.tuning_validation_fraction,
        max_tuning_trials_per_model=args.max_tuning_trials,
    )
    metrics_df = (
        run_benchmarks_for_horizons(config, horizons=parsed_horizons, model_names=model_names)
        if len(parsed_horizons) > 1
        else run_benchmark(config, model_names=model_names)
    )
    print(metrics_df.to_string(index=False))
    print(f"\nSaved benchmark outputs under: {config.output_dir}")


if __name__ == "__main__":
    main()
