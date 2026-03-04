#!/usr/bin/env python
import argparse

from benchmark import run_benchmarks


def parse_args():
    p = argparse.ArgumentParser(description="Run rolling-window benchmark models for corporate default prediction.")
    p.add_argument("--data-path", required=True, help="CSV path with features + horizon label columns")
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
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    summary_df, yearly_df = run_benchmarks(
        data_path=args.data_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        model_names=args.models,
        time_col=args.time_col,
        drop_cols=args.drop_cols,
        min_train_years=args.min_train_years,
        random_state=args.random_state,
    )
    print("Saved benchmark outputs")
    print("Summary rows:", len(summary_df))
    print("Yearly rows:", len(yearly_df))
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
