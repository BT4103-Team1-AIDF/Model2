# Corporate Default Modelling Repo

This folder is the **modelling workstream** for the BT4103 corporate default project.
It is intentionally independent of Codabench platform code.

## Scope

- Train and compare benchmark models across fixed prediction horizons:
  - `1, 3, 6, 12, 24, 36, 48, 60` months
- Use rolling time-window evaluation by year.
- Report:
  - Overall AUC for default-vs-rest (`y == 1`)
  - Mean yearly default AUC (years with no defaults are skipped)

Implemented model set:

- `logistic` (multinomial logistic regression)
- `random_forest` (multiclass RF)
- `xgboost` (multiclass XGB)
- `lightgbm` (multiclass LGBM)
- `lstm` (sequence-style NN with focal loss and class reweighting)

## Folder Structure

```text
models/
  README.md
  requirements.txt
  model.py
  benchmark.py
  __init__.py
  run_benchmarks.py
  MODEL_CARD_TEMPLATE.md
  outputs/
```

## Data Requirements

Input should be a CSV with:

- Time column: default `yyyy`
- Feature columns
- Label column per horizon, using one of:
  - `y_{h}m` (recommended), e.g. `y_12m`, `y_24m`
  - `y{h}m`, `label_{h}m`, or `target_{h}m`

For example, horizon 12 uses `y_12m` if present.

### Current local dataset note

`BT4103/Data/train_labeled_upto2014.csv` contains `y_12m` only, so only 12-month experiments are currently runnable unless you create the other horizon labels.

## Setup

```bash
cd /Users/yewj/School/BT4103/Corporate\ Default\ Practice/model_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Benchmarks

### Run all models for 12-month horizon only (works with current dataset)

```bash
python run_benchmarks.py \
  --data-path ../../Data/train_labeled_upto2014.csv \
  --output-dir outputs/h12 \
  --horizons 12 \
  --models logistic random_forest xgboost lightgbm \
  --max-tuning-trials 12
```

### Run full target horizons (requires corresponding label columns)

```bash
python run_benchmarks.py \
  --data-path /path/to/train_with_all_horizon_labels.csv \
  --output-dir outputs/full_horizon \
  --horizons 1 3 6 12 24 36 48 60 \
  --models logistic random_forest xgboost lightgbm lstm \
  --max-tuning-trials 12
```

### Run submission-style evaluation (train<=2014, test>=2015)

This mode produces CodaBench-like artifacts:
`predictions.csv`, `yearly_metrics.csv`, `yearly_auc.png`, `scores.txt`, `detailed_results.html`.

```bash
python run_benchmarks.py \
  --mode submission \
  --data-path /path/to/train_labeled_upto2014.csv \
  --test-data-path /path/to/test_labeled_from2015.csv \
  --output-dir outputs/submission_h12_lightgbm \
  --horizons 12 \
  --models lightgbm \
  --train-end-year 2014 \
  --max-tuning-trials 12
```

## Rolling Evaluation Logic

For each horizon and each model:

1. Sort distinct years from `yyyy`.
2. Use first `min_train_years` years as initial training window.
3. For each subsequent year `t`:
   - Train on years `< t`
   - Test on year `t`
4. Compute yearly default AUC from `p(default)`.
5. Aggregate:
   - `overall_auc` on all out-of-time predictions concatenated
   - `mean_yearly_auc` across valid yearly AUCs

`min_train_years` is configurable (default `8`).
For every model (`logistic`, `random_forest`, `xgboost`, `lightgbm`, `lstm`), tuning is done with a time-series validation split using only the training period (last train year as validation).

You can control tuning runtime with:

```bash
--max-tuning-trials 12
```

## Output Artifacts

Saved to `--output-dir`:

- `benchmark_summary.csv`
  - one row per `(horizon, model)`
  - includes `overall_auc`, `mean_yearly_auc`, `valid_years`, status
- `benchmark_yearly_aucs.csv`
  - per-year AUC records
- `benchmark_pivot_mean_yearly_auc.csv`
  - pivot table (rows=horizon, cols=model)
- `benchmark_best_params.csv`
  - tuned parameters and validation AUC per `(horizon, model)`

## Notes on Optional Dependencies

- If `xgboost` or `lightgbm` are not installed, those rows will be marked failed.
- If `tensorflow` is not installed, `lstm` will fail; other models still run.

### XGBoost environment fix

If benchmark output shows `Model 'xgboost' is unavailable in this environment.`, install it in the same Python interpreter:

```bash
python3 -m pip install --user xgboost
python3 -c "import xgboost; print(xgboost.__version__)"
```

## Suggested Next Modelling Steps

1. Add/verify horizon label generation (`y_1m` ... `y_60m`).
2. Expand tuning grids and trial budget per horizon as needed.
3. Add confidence intervals via repeated time splits.
4. Freeze best model configs for submission package.
