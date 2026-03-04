from __future__ import annotations

import json
import random
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
DEFAULT_MODELS = ["logistic", "random_forest", "xgboost", "lightgbm", "lstm"]


@dataclass
class EvalResult:
    horizon: int
    model_name: str
    overall_auc: float
    mean_yearly_auc: float
    valid_years: int
    n_test: int
    n_default_test: int


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)


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
    raise ValueError(f"No label column found for horizon {horizon}. Expected one of {_candidate_label_cols(horizon)}")


def _winsorize_series(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    low = s.quantile(q_low)
    high = s.quantile(q_high)
    return s.clip(lower=low, upper=high)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_cols:
        if col.startswith("y_") or col in {"CompNo", "yyyy", "mm"}:
            continue
        out[col] = _winsorize_series(out[col])

    if "dtdlevel" in out.columns:
        out["dtdlevel_log"] = np.log1p(np.maximum(out["dtdlevel"].astype(float), 0.0))
        out["dtdlevel_abs"] = np.abs(out["dtdlevel"].astype(float))
    if "dtdtrend" in out.columns:
        out["dtdtrend_abs"] = np.abs(out["dtdtrend"].astype(float))
    if "dtdlevel" in out.columns and "dtdtrend" in out.columns:
        denom = np.abs(out["dtdlevel"].astype(float)) + 1e-6
        out["dtd_interaction"] = out["dtdlevel"].astype(float) * out["dtdtrend"].astype(float)
        out["dtd_trend_ratio"] = out["dtdtrend"].astype(float) / denom

    for base_col in ["m2b", "sigma", "sizelevel", "liqnonfinlevel", "ni2talevel"]:
        if base_col in out.columns:
            v = out[base_col].astype(float)
            out[f"{base_col}_logabs"] = np.sign(v) * np.log1p(np.abs(v))

    return out


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


def build_model(name: str, random_state: int = 42, params: Optional[Dict] = None):
    params = params or {}

    if name == "logistic":
        return Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
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
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "random_forest":
        rf_defaults = dict(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        rf_defaults.update(params)
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(**rf_defaults))])

    if name == "xgboost":
        if XGBClassifier is None:
            return None
        xgb_defaults = dict(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=500,
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
        )
        xgb_defaults.update(params)
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", XGBClassifier(**xgb_defaults))])

    if name == "lightgbm":
        if LGBMClassifier is None:
            return None
        lgb_defaults = dict(
            objective="multiclass",
            num_class=3,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=4,
        )
        lgb_defaults.update(params)
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", LGBMClassifier(**lgb_defaults))])

    if name == "lstm":
        return "lstm"

    raise ValueError(f"Unknown model '{name}'")


def _sparse_multiclass_focal_loss(alpha: Sequence[float], gamma: float = 2.0):
    alpha_t = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_one_hot = tf.one_hot(y_true, depth=3)
        p_t = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        alpha_factor = tf.gather(alpha_t, y_true)
        focal = -alpha_factor * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)

    return loss


def _fit_predict_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    params: Optional[Dict] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> np.ndarray:
    if tf is None or layers is None:
        raise RuntimeError("TensorFlow is not installed. Cannot run LSTM model.")

    params = params or {}
    _set_global_seed(random_state)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    if X_val is not None and y_val is not None:
        X_val = scaler.transform(imputer.transform(X_val))

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    if X_val is not None:
        X_val = X_val[..., np.newaxis]

    units = int(params.get("units", 24))
    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.2))
    lr = float(params.get("learning_rate", 1e-3))
    epochs = int(params.get("epochs", 8))
    batch_size = int(params.get("batch_size", 512))

    classes = np.array([0, 1, 2], dtype=int)
    counts = np.array([max(int(np.sum(y_train == c)), 1) for c in classes], dtype=float)
    inv = 1.0 / counts
    alpha = (inv / np.sum(inv)).tolist()

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1], 1)),
            layers.LSTM(units, activation="tanh"),
            layers.Dropout(dropout),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=_sparse_multiclass_focal_loss(alpha=alpha, gamma=2.0),
        metrics=["accuracy"],
    )

    sample_weight = _make_multiclass_sample_weight(y_train)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        sample_weight=sample_weight,
        callbacks=callbacks,
    )
    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.1

    model.fit(**fit_kwargs)
    return model.predict(X_test, verbose=0)


def _extract_default_proba(model_name: str, model, proba: np.ndarray) -> np.ndarray:
    if model_name == "lstm":
        return proba[:, 1]

    clf = model.named_steps["clf"]
    classes = np.asarray(clf.classes_)
    if 1 not in classes:
        return np.zeros((proba.shape[0],), dtype=float)

    idx = int(np.where(classes == 1)[0][0])
    return proba[:, idx]


def _fit_predict_once(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    params: Optional[Dict] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> np.ndarray:
    if model_name == "lstm":
        return _fit_predict_lstm(
            X_train,
            y_train,
            X_test,
            random_state=random_state,
            params=params,
            X_val=X_val,
            y_val=y_val,
        )

    model = build_model(model_name, random_state=random_state, params=params)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' is unavailable in this environment.")

    sample_weight = _make_multiclass_sample_weight(y_train)
    model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    return model.predict_proba(X_test), model


def _split_train_val_by_year(df: pd.DataFrame, year_col: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    years = sorted(pd.unique(df[year_col]))
    if len(years) < 3:
        return None
    val_year = years[-1]
    tr = df[df[year_col] < val_year]
    val = df[df[year_col] == val_year]
    if len(tr) < 200 or len(val) < 50:
        return None
    return tr, val


def _tune_time_series_params(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    model_name: str,
    year_col: str,
    random_state: int,
) -> Tuple[Dict, float]:
    split = _split_train_val_by_year(df, year_col)
    if split is None:
        return {}, np.nan

    if model_name == "lightgbm":
        candidates = [
            {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 400},
            {"num_leaves": 63, "learning_rate": 0.02, "n_estimators": 600},
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
        ]
    elif model_name == "lstm":
        candidates = [
            {"units": 24, "dense_units": 24, "dropout": 0.2, "learning_rate": 1e-3, "epochs": 6, "batch_size": 512},
            {"units": 32, "dense_units": 32, "dropout": 0.25, "learning_rate": 8e-4, "epochs": 8, "batch_size": 512},
        ]
    else:
        return {}, np.nan

    tr, val = split
    X_tr = tr[feature_cols].values
    y_tr = tr[label_col].astype(int).values
    X_val = val[feature_cols].values
    y_val = val[label_col].astype(int).values
    y_val_default = (y_val == 1).astype(int)

    best_params = {}
    best_auc = -np.inf

    for params in candidates:
        try:
            if model_name == "lstm":
                proba = _fit_predict_lstm(
                    X_tr,
                    y_tr,
                    X_val,
                    random_state=random_state,
                    params=params,
                )
                p_default = proba[:, 1]
            else:
                pred_out = _fit_predict_once(
                    model_name=model_name,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_val,
                    random_state=random_state,
                    params=params,
                )
                proba, model = pred_out
                p_default = _extract_default_proba(model_name, model, proba)

            auc = _safe_auc(y_val_default, p_default)
            score = -np.inf if np.isnan(auc) else auc
            if score > best_auc:
                best_auc = score
                best_params = params
        except Exception:
            continue

    if best_auc == -np.inf:
        return {}, np.nan
    return best_params, float(best_auc)


def rolling_window_eval(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    model_name: str,
    year_col: str = "yyyy",
    min_train_years: int = 8,
    random_state: int = 42,
    params: Optional[Dict] = None,
) -> Tuple[EvalResult, pd.DataFrame]:
    params = params or {}

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
            if model_name == "lstm":
                inner_split = _split_train_val_by_year(tr, year_col)
                X_val = None
                y_val = None
                if inner_split is not None:
                    tr_inner, val_inner = inner_split
                    X_tr = tr_inner[feature_cols].values
                    y_tr = tr_inner[label_col].astype(int).values
                    X_val = val_inner[feature_cols].values
                    y_val = val_inner[label_col].astype(int).values

                proba = _fit_predict_lstm(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_te,
                    random_state=random_state,
                    params=params,
                    X_val=X_val,
                    y_val=y_val,
                )
                p_default = proba[:, 1]
            else:
                proba, model = _fit_predict_once(
                    model_name=model_name,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_te,
                    random_state=random_state,
                    params=params,
                )
                p_default = _extract_default_proba(model_name, model, proba)

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
    _set_global_seed(random_state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(data_path)
    df = engineer_features(raw_df)

    summary_rows: List[Dict] = []
    yearly_all: List[pd.DataFrame] = []
    best_param_rows: List[Dict] = []

    for h in horizons:
        label_col = resolve_label_col(df, h)
        feature_cols = build_feature_columns(df, label_col=label_col, drop_cols=drop_cols)
        sub = df[feature_cols + [label_col, time_col]].copy()

        for m in model_names:
            try:
                tuned_params, tuned_val_auc = _tune_time_series_params(
                    sub,
                    label_col=label_col,
                    feature_cols=feature_cols,
                    model_name=m,
                    year_col=time_col,
                    random_state=random_state,
                )

                res, yearly = rolling_window_eval(
                    sub,
                    label_col=label_col,
                    feature_cols=feature_cols,
                    model_name=m,
                    year_col=time_col,
                    min_train_years=min_train_years,
                    random_state=random_state,
                    params=tuned_params,
                )

                summary_rows.append(
                    {
                        "horizon_months": int(h),
                        "model": m,
                        "overall_auc": res.overall_auc,
                        "mean_yearly_auc": res.mean_yearly_auc,
                        "valid_years": res.valid_years,
                        "n_test": res.n_test,
                        "n_default_test": res.n_default_test,
                        "tuned_val_auc": tuned_val_auc,
                        "status": "ok",
                    }
                )

                best_param_rows.append(
                    {
                        "horizon_months": int(h),
                        "model": m,
                        "tuned_val_auc": tuned_val_auc,
                        "best_params": json.dumps(tuned_params, sort_keys=True),
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
                        "tuned_val_auc": np.nan,
                        "status": f"failed: {ex}",
                    }
                )
                best_param_rows.append(
                    {
                        "horizon_months": int(h),
                        "model": m,
                        "tuned_val_auc": np.nan,
                        "best_params": "{}",
                        "status": f"failed: {ex}",
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon_months", "mean_yearly_auc"], ascending=[True, False])
    yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()
    best_df = pd.DataFrame(best_param_rows).sort_values(["horizon_months", "tuned_val_auc"], ascending=[True, False])

    summary_df.to_csv(output / "benchmark_summary.csv", index=False)
    yearly_df.to_csv(output / "benchmark_yearly_aucs.csv", index=False)
    best_df.to_csv(output / "benchmark_best_params.csv", index=False)

    pivot = summary_df.pivot(index="horizon_months", columns="model", values="mean_yearly_auc")
    pivot.to_csv(output / "benchmark_pivot_mean_yearly_auc.csv")

    return summary_df, yearly_df
