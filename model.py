import random
from typing import Optional

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


class Model:
    def __init__(self):
        self.drop_cols = {"CompNo"}
        self.feature_cols = None
        self.used_cols = None

        self.rest_split = (0.5, 0.5)
        self.base_p1 = 0.05

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        self.lgb_model = None
        self.use_lgb = False

        self.lstm_model = None
        self.use_lstm = False

        self.fallback_model = LogisticRegression(
            solver="lbfgs",
            max_iter=800,
            class_weight="balanced",
            random_state=42,
        )

    def _set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        if tf is not None:
            tf.keras.utils.set_random_seed(seed)

    def _safe_float_matrix(self, X):
        try:
            arr = np.asarray(X)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            arr = arr.astype(np.float32, copy=False)
            arr[~np.isfinite(arr)] = 0.0
            return arr
        except Exception:
            return np.zeros((1, 1), dtype=np.float32)

    def _winsorize_np(self, X: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        low = np.nanquantile(X, low_q, axis=0)
        high = np.nanquantile(X, high_q, axis=0)
        return np.clip(X, low, high)

    def _extract_matrix(self, X, fit: bool = False):
        if hasattr(X, "columns"):
            if fit or self.feature_cols is None:
                self.feature_cols = list(X.columns)

            cols = [c for c in self.feature_cols if c in X.columns]
            if not cols:
                return self._safe_float_matrix(X)

            used = [c for c in cols if c not in self.drop_cols]
            if not used:
                used = cols

            if fit:
                self.used_cols = used

            use_cols = self.used_cols if self.used_cols is not None else used
            use_cols = [c for c in use_cols if c in X.columns]
            base = self._safe_float_matrix(X[use_cols].values)

            if "mm" in X.columns:
                mm = self._safe_float_matrix(X[["mm"]].values)
                mm = np.clip(mm, 1.0, 12.0)
                base = np.hstack(
                    [
                        base,
                        np.sin(2.0 * np.pi * mm / 12.0).astype(np.float32),
                        np.cos(2.0 * np.pi * mm / 12.0).astype(np.float32),
                    ]
                )

            return base

        return self._safe_float_matrix(X)

    def _feature_engineering_np(self, X: np.ndarray) -> np.ndarray:
        X = self._winsorize_np(X)
        X_logabs = np.sign(X) * np.log1p(np.abs(X))
        X_sq = X * X
        return np.hstack([X, X_logabs, X_sq]).astype(np.float32, copy=False)

    def _focal_loss(self, alpha, gamma=2.0):
        alpha_t = tf.constant(alpha, dtype=tf.float32)

        def loss(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            y_one = tf.one_hot(y_true, depth=3)
            p_t = tf.reduce_sum(y_one * y_pred, axis=-1)
            alpha_factor = tf.gather(alpha_t, y_true)
            focal = -alpha_factor * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
            return tf.reduce_mean(focal)

        return loss

    def _fit_lstm(self, X_train, y_train):
        if tf is None or layers is None:
            return
        if X_train.shape[0] < 1200:
            return

        counts = np.array([max(int(np.sum(y_train == c)), 1) for c in [0, 1, 2]], dtype=float)
        inv = 1.0 / counts
        alpha = (inv / np.sum(inv)).tolist()

        X_lstm = X_train[..., np.newaxis]

        model = tf.keras.Sequential(
            [
                layers.Input(shape=(X_lstm.shape[1], 1)),
                layers.LSTM(24, activation="tanh"),
                layers.Dropout(0.2),
                layers.Dense(24, activation="relu"),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=self._focal_loss(alpha=alpha, gamma=2.0),
            metrics=["accuracy"],
        )

        sample_weight = self._make_multiclass_sample_weight(y_train)
        model.fit(
            X_lstm,
            y_train,
            sample_weight=sample_weight,
            validation_split=0.1,
            epochs=6,
            batch_size=512,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        )
        self.lstm_model = model
        self.use_lstm = True

    def _make_multiclass_sample_weight(self, y):
        y = np.asarray(y).astype(int)
        classes = np.array([0, 1, 2])
        counts = {c: max(int(np.sum(y == c)), 1) for c in classes}
        total = float(len(y))
        weights = {c: total / (3.0 * counts[c]) for c in classes}
        return np.array([weights[int(v)] for v in y], dtype=float)

    def fit(self, X, y):
        self._set_seed(42)

        y = np.asarray(y).ravel().astype(int)
        y_bin = (y == 1).astype(int)

        if y.size > 0:
            self.base_p1 = float(np.clip(np.mean(y == 1), 1e-6, 1.0 - 1e-6))

        rest = y[y != 1]
        if rest.size > 0:
            p0 = float(np.mean(rest == 0))
            p2 = float(np.mean(rest == 2))
            s = p0 + p2
            if s > 0:
                self.rest_split = (p0 / s, p2 / s)

        X_raw = self._extract_matrix(X, fit=True)
        X_fe = self._feature_engineering_np(X_raw)
        X_imp = self.imputer.fit_transform(X_fe)
        X_scaled = self.scaler.fit_transform(X_imp)

        self.use_lgb = False
        if LGBMClassifier is not None and X_scaled.shape[0] >= 400:
            try:
                self.lgb_model = LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=220,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=8,
                    max_bin=127,
                    min_child_samples=40,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                )
                self.lgb_model.fit(X_scaled, y, sample_weight=self._make_multiclass_sample_weight(y))
                self.use_lgb = True
            except Exception:
                self.lgb_model = None
                self.use_lgb = False

        self.use_lstm = False
        try:
            self._fit_lstm(X_scaled, y)
        except Exception:
            self.lstm_model = None
            self.use_lstm = False

        self.fallback_model.fit(X_scaled, y_bin)
        return self

    def _force_shape_n3(self, proba, n_in):
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or proba.shape[1] != 3:
            return np.full((n_in, 3), 1.0 / 3.0, dtype=float)

        if proba.shape[0] != n_in:
            if proba.shape[0] == 1 and n_in > 1:
                proba = np.tile(proba, (n_in, 1))
            elif proba.shape[0] > n_in:
                proba = proba[:n_in, :]
            else:
                pad = np.full((n_in - proba.shape[0], 3), 1.0 / 3.0, dtype=float)
                proba = np.vstack([proba, pad])

        proba[~np.isfinite(proba)] = 1.0 / 3.0
        sums = np.sum(proba, axis=1, keepdims=True)
        return proba / np.maximum(sums, 1e-12)

    def predict_proba(self, X):
        n_in = int(len(X)) if hasattr(X, "__len__") else 1
        if n_in == 0:
            return np.zeros((0, 3), dtype=float)

        X_raw = self._extract_matrix(X, fit=False)
        X_fe = self._feature_engineering_np(X_raw)
        X_scaled = self.scaler.transform(self.imputer.transform(X_fe))

        p_mix = None

        if self.use_lgb and self.lgb_model is not None:
            try:
                p_mix = self.lgb_model.predict_proba(X_scaled)
            except Exception:
                p_mix = None

        if self.use_lstm and self.lstm_model is not None:
            try:
                p_lstm = self.lstm_model.predict(X_scaled[..., np.newaxis], verbose=0)
                p_mix = p_lstm if p_mix is None else (0.75 * p_mix + 0.25 * p_lstm)
            except Exception:
                pass

        if p_mix is None:
            p1 = self.fallback_model.predict_proba(X_scaled)[:, 1]
            p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
            rest = 1.0 - p1
            p0 = rest * float(self.rest_split[0])
            p2 = rest * float(self.rest_split[1])
            p_mix = np.vstack([p0, p1, p2]).T

        return self._force_shape_n3(p_mix, n_in)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(int)
