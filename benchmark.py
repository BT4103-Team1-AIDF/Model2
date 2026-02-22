from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception:
    tf = None
    layers = None


DEFAULT_HORIZONS = [1, 3, 6, 12, 24, 36, 48, 60]
DEFAULT_MODELS = ["logistic", "random_forest", "xgboost", "lightgbm", "rnn"]


@dataclass
class EvalResult:
    horizon: int
    model_name: str
    overall_auc: float
    mean_yearly_auc: float
    valid_years: int
    n_test: int
    n_default_test: int


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def _candidate_label_cols(h: int) -> List[str]:
    cands = [f"y_{h}m", f"y{h}m", f"label_{h}m", f"target_{h}m"]
    if h == 12:
        cands += ["y_12m", "y"]
    return cands


def resolve_label_col(df: pd.DataFrame, horizon: int, explicit_label_col: Optional[str] = None) -> str:
    if explicit_label_col is not None:
        if explicit_label_col not in df.columns:
            raise ValueError(f"Label column '{explicit_label_col}' not found for horizon {horizon}.")
        return explicit_label_col

    for col in _candidate_label_cols(horizon):
        if col in df.columns:
            return col
    raise ValueError(
        f"No label column found for horizon {horizon}. Expected one of {_candidate_label_cols(horizon)}"
    )


def build_feature_columns(df: pd.DataFrame, label_col: str, drop_cols: Sequence[str]) -> List[str]:
    drops = set(drop_cols) | {label_col}
    return [c for c in df.columns if c not in drops]


def _make_multiclass_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    classes = np.array([0, 1, 2])
    counts = {c: max(int(np.sum(y == c)), 1) for c in classes}
    total = float(len(y))
    weights = {c: total / (3.0 * counts[c]) for c in classes}
    return np.array([weights[int(v)] for v in y], dtype=float)


def build_model(name: str, random_state: int = 42):
    if name == "logistic":
        return Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                slice(0, None),
                            )
                        ],
                        remainder="drop",
                        sparse_threshold=0.0,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "xgboost":
        if XGBClassifier is None:
            return None
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        n_estimators=600,
                        learning_rate=0.03,
                        max_depth=5,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "lightgbm":
        if LGBMClassifier is None:
            return None
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=3,
                        n_estimators=700,
                        learning_rate=0.03,
                        num_leaves=31,
                        max_depth=-1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "rnn":
        return "rnn"

    raise ValueError(f"Unknown model '{name}'")


def _fit_predict_rnn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, random_state: int = 42) -> np.ndarray:
    if tf is None or layers is None:
        raise RuntimeError("TensorFlow is not installed. Cannot run RNN model.")

    tf.keras.utils.set_random_seed(random_state)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1], 1)),
            layers.SimpleRNN(32, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=256,
        verbose=0,
    )
    return model.predict(X_test, verbose=0)


def rolling_window_eval(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    model_name: str,
    year_col: str = "yyyy",
    min_train_years: int = 8,
    random_state: int = 42,
) -> Tuple[EvalResult, pd.DataFrame]:
    model = build_model(model_name, random_state=random_state)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' is unavailable in this environment.")

    years = sorted(pd.unique(df[year_col]))
    if len(years) < min_train_years + 1:
        raise ValueError("Not enough distinct years for rolling-window evaluation.")

    rows = []
    all_y = []
    all_p = []

    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]

        tr = df[df[year_col].isin(train_years)]
        te = df[df[year_col] == test_year]
        if len(tr) < 100 or len(te) < 10:
            continue

        X_tr = tr[feature_cols].values
        y_tr = tr[label_col].astype(int).values
        X_te = te[feature_cols].values
        y_te_default = (te[label_col].astype(int).values == 1).astype(int)

        try:
            if model_name == "rnn":
                proba = _fit_predict_rnn(X_tr, y_tr, X_te, random_state=random_state)
            else:
                sample_weight = _make_multiclass_sample_weight(y_tr)
                model.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
                proba = model.predict_proba(X_te)

            classes = np.array([0, 1, 2])
            p_default = np.zeros(len(te), dtype=float)
            if model_name != "rnn":
                clf = model.named_steps["clf"]
                cls = np.asarray(clf.classes_)
                if 1 in cls:
                    p_default = proba[:, int(np.where(cls == 1)[0][0])]
            else:
                p_default = proba[:, 1]

            auc = _safe_auc(y_te_default, p_default)

            rows.append(
                {
                    "year": int(test_year),
                    "auc_default": auc,
                    "n_default": int(np.sum(y_te_default)),
                    "n_total": int(len(te)),
                    "model": model_name,
                }
            )
            all_y.append(y_te_default)
            all_p.append(p_default)
        except Exception as ex:
            rows.append(
                {
                    "year": int(test_year),
                    "auc_default": np.nan,
                    "n_default": int(np.sum(y_te_default)),
                    "n_total": int(len(te)),
                    "model": model_name,
                    "error": str(ex),
                }
            )

    yearly_df = pd.DataFrame(rows)
    if len(all_y) == 0:
        overall_auc = np.nan
        n_test = 0
        n_default_test = 0
    else:
        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_p)
        overall_auc = _safe_auc(y_all, p_all)
        n_test = int(len(y_all))
        n_default_test = int(np.sum(y_all))

    mean_yearly_auc = float(np.nanmean(yearly_df["auc_default"].values)) if len(yearly_df) > 0 else np.nan
    valid_years = int(np.sum(~np.isnan(yearly_df["auc_default"].values))) if len(yearly_df) > 0 else 0

    return (
        EvalResult(
            horizon=-1,
            model_name=model_name,
            overall_auc=overall_auc,
            mean_yearly_auc=mean_yearly_auc,
            valid_years=valid_years,
            n_test=n_test,
            n_default_test=n_default_test,
        ),
        yearly_df,
    )


def run_benchmarks(
    data_path: str,
    output_dir: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    model_names: Sequence[str] = DEFAULT_MODELS,
    time_col: str = "yyyy",
    drop_cols: Sequence[str] = ("CompNo", "yyyy", "mm"),
    min_train_years: int = 8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    summary_rows: List[Dict] = []
    yearly_all: List[pd.DataFrame] = []

    for h in horizons:
        label_col = resolve_label_col(df, h)
        feature_cols = build_feature_columns(df, label_col=label_col, drop_cols=drop_cols)
        sub = df[feature_cols + [label_col, time_col]].copy()

        for m in model_names:
            try:
                res, yearly = rolling_window_eval(
                    sub,
                    label_col=label_col,
                    feature_cols=feature_cols,
                    model_name=m,
                    year_col=time_col,
                    min_train_years=min_train_years,
                    random_state=random_state,
                )
                res.horizon = int(h)
                summary_rows.append(
                    {
                        "horizon_months": int(h),
                        "model": m,
                        "overall_auc": res.overall_auc,
                        "mean_yearly_auc": res.mean_yearly_auc,
                        "valid_years": res.valid_years,
                        "n_test": res.n_test,
                        "n_default_test": res.n_default_test,
                        "status": "ok",
                    }
                )
                yearly["horizon_months"] = int(h)
                yearly_all.append(yearly)
            except Exception as ex:
                summary_rows.append(
                    {
                        "horizon_months": int(h),
                        "model": m,
                        "overall_auc": np.nan,
                        "mean_yearly_auc": np.nan,
                        "valid_years": 0,
                        "n_test": 0,
                        "n_default_test": 0,
                        "status": f"failed: {ex}",
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon_months", "mean_yearly_auc"], ascending=[True, False])
    yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()

    summary_path = output / "benchmark_summary.csv"
    yearly_path = output / "benchmark_yearly_aucs.csv"
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)

    pivot = summary_df.pivot(index="horizon_months", columns="model", values="mean_yearly_auc")
    pivot.to_csv(output / "benchmark_pivot_mean_yearly_auc.csv")

    return summary_df, yearly_df
