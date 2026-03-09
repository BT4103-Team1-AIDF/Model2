"""Model builders with a consistent sklearn pipeline interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("tensorflow is required for lstm model.") from exc
    return tf


class TabularLSTMClassifier:
    """Minimal LSTM classifier for tabular inputs via pseudo-sequences."""

    def __init__(
        self,
        *,
        hidden_units: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 256,
        class_weight_mode: str = "balanced",
        verbose: int = 0,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight_mode = class_weight_mode
        self.verbose = verbose
        self.random_state = random_state
        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "hidden_units": self.hidden_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "class_weight_mode": self.class_weight_mode,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "TabularLSTMClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _set_seed(self, tf: Any) -> None:
        tf.keras.utils.set_random_seed(int(self.random_state))

    def _to_sequence(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32).reshape((X.shape[0], X.shape[1], 1))

    def _build_network(self, tf: Any, *, n_features: int, n_classes: int) -> Any:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features, 1)),
                tf.keras.layers.LSTM(int(self.hidden_units)),
                tf.keras.layers.Dropout(float(self.dropout)),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
        return model

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "TabularLSTMClassifier":
        tf = _import_tensorflow()
        self._set_seed(tf)

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).astype(int)
        if X_arr.ndim != 2:
            raise ValueError("LSTM expects a 2D feature array.")

        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = int(X_arr.shape[1])
        class_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[int(v)] for v in y_arr], dtype=np.int32)
        X_seq = self._to_sequence(X_arr)

        class_weight: dict[int, float] | None = None
        if self.class_weight_mode == "balanced":
            weights = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(len(self.classes_)),
                y=y_idx,
            )
            class_weight = {int(i): float(w) for i, w in enumerate(weights)}

        self.model_ = self._build_network(
            tf,
            n_features=self.n_features_in_,
            n_classes=len(self.classes_),
        )

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight)
        elif class_weight is not None:
            fit_kwargs["class_weight"] = class_weight

        self.model_.fit(
            X_seq,
            y_idx,
            epochs=int(self.epochs),
            batch_size=int(max(8, min(int(self.batch_size), len(X_seq)))),
            verbose=int(self.verbose),
            **fit_kwargs,
        )
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.classes_ is None or self.n_features_in_ is None:
            raise ValueError("LSTM model is not fitted yet.")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("Input shape mismatch for LSTM predict_proba.")
        X_seq = self._to_sequence(X_arr)
        return np.asarray(self.model_.predict(X_seq, verbose=0))

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LSTM model is not fitted yet.")
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model_"] = None
        return state


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    """Build shared preprocessing for numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_estimator(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    """Build one required benchmark estimator by name."""
    if model_name == "logistic_regression":
        base = {
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        return LogisticRegression(**{**base, **params})

    if model_name == "random_forest":
        base = {
            "n_estimators": 400,
            "random_state": random_state,
            "n_jobs": -1,
        }
        return RandomForestClassifier(**{**base, **params})

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed.") from exc

        base = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": random_state,
            "n_jobs": -1,
        }
        return LGBMClassifier(**{**base, **params})

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed.") from exc

        base = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        return XGBClassifier(**{**base, **params})

    if model_name == "lstm":
        base = {
            "hidden_units": 32,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "epochs": 8,
            "batch_size": 256,
            "class_weight_mode": "balanced",
            "verbose": 0,
            "random_state": random_state,
        }
        return TabularLSTMClassifier(**{**base, **params})

    raise ValueError(f"Unknown model: {model_name}")


def build_model_pipeline(
    model_name: str,
    params: dict[str, Any],
    *,
    random_state: int,
    X_fit: pd.DataFrame,
) -> Pipeline:
    """Build full trainable pipeline for a model."""
    estimator = build_estimator(model_name, params=params, random_state=random_state)
    preprocess = build_preprocessor(
        X_fit,
        scale_numeric=(model_name in {"logistic_regression", "lstm"}),
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def align_proba_to_classes(
    y_proba: np.ndarray,
    model_classes: Sequence[int],
    expected_classes: Sequence[int],
) -> np.ndarray:
    """Reorder prediction probabilities into expected class order."""
    class_to_idx = {int(c): i for i, c in enumerate(model_classes)}
    aligned = np.zeros((y_proba.shape[0], len(expected_classes)))
    for out_idx, c in enumerate(expected_classes):
        src_idx = class_to_idx.get(int(c))
        if src_idx is not None:
            aligned[:, out_idx] = y_proba[:, src_idx]
    return aligned
