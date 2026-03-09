#!/usr/bin/env python
import argparse

from benchmark import run_benchmarks, run_submission_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="Run rolling-window benchmark models for corporate default prediction.")
    p.add_argument(
        "--mode",
        choices=["rolling", "submission"],
        default="rolling",
        help="rolling: train/test by rolling year windows; submission: train<=2014 and evaluate on test years",
    )
    p.add_argument("--data-path", required=True, help="CSV path with features + horizon label columns")
    p.add_argument("--test-data-path", default=None, help="Optional test CSV path for submission mode")
    p.add_argument("--output-dir", default="outputs", help="Directory for output CSV artifacts")
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 6, 12, 24, 36, 48, 60],
        help="Prediction horizons in months",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "random_forest", "xgboost", "lightgbm", "lstm"],
        help="Model list: logistic random_forest xgboost lightgbm lstm",
    )
    p.add_argument("--time-col", default="yyyy", help="Time column used for rolling window splits")
    p.add_argument(
        "--drop-cols",
        nargs="+",
        default=["CompNo", "yyyy", "mm"],
        help="Columns excluded from features",
    )
    p.add_argument("--min-train-years", type=int, default=8, help="Minimum initial train years for rolling evaluation")
    p.add_argument("--train-end-year", type=int, default=2014, help="Train/test split year used in submission mode")
    p.add_argument("--max-tuning-trials", type=int, default=12, help="Maximum hyperparameter tuning trials per model")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "submission":
        summary_df, yearly_df = run_submission_evaluation(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.models[0],
            horizon=args.horizons[0],
            time_col=args.time_col,
            train_end_year=args.train_end_year,
            test_data_path=args.test_data_path,
            drop_cols=args.drop_cols,
            random_state=args.random_state,
            max_tuning_trials=args.max_tuning_trials,
        )
    else:
        summary_df, yearly_df = run_benchmarks(
            data_path=args.data_path,
            output_dir=args.output_dir,
            horizons=args.horizons,
            model_names=args.models,
            time_col=args.time_col,
            drop_cols=args.drop_cols,
            min_train_years=args.min_train_years,
            random_state=args.random_state,
            max_tuning_trials=args.max_tuning_trials,
        )
    print("Saved benchmark outputs")
    print("Summary rows:", len(summary_df))
    print("Yearly rows:", len(yearly_df))
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
