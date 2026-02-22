import numpy as np

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


class Model:
    def __init__(self):
        self.rest_split = (0.5, 0.5)
        self.base_p1 = 0.05

        self.drop_cols = {"CompNo"}
        self.feature_cols = None
        self.used_cols = None
        self.has_mm = False

        self.xgb_model = None
        self.using_xgb = False

        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0
        self.l2 = 1e-3
        self.max_steps = 1200

    def _n_samples_from_input(self, X):
        try:
            return int(len(X))
        except Exception:
            return 1

    def _safe_nan_to_num(self, a, fill=0.0):
        arr = np.asarray(a, dtype=float)
        arr[~np.isfinite(arr)] = fill
        return arr

    def _safe_float_matrix(self, X):
        try:
            arr = np.asarray(X)

            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)

            try:
                arr = arr.astype(np.float32, copy=False)
            except Exception:
                flat = arr.reshape(arr.shape[0], -1)
                out = np.zeros(flat.shape, dtype=np.float32)
                for i in range(flat.shape[0]):
                    for j in range(flat.shape[1]):
                        try:
                            out[i, j] = float(flat[i, j])
                        except Exception:
                            out[i, j] = 0.0
                arr = out

            return self._safe_nan_to_num(arr, fill=0.0).astype(np.float32, copy=False)
        except Exception:
            return np.zeros((1, 1), dtype=np.float32)

    def _sigmoid(self, z):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _force_shape_n3(self, proba, n_in):
        proba = np.asarray(proba, dtype=float)

        if proba.ndim != 2 or proba.shape[1] != 3:
            return np.full((n_in, 3), 1.0 / 3.0, dtype=float)

        r = int(proba.shape[0])
        if r == n_in:
            pass
        elif r == 1 and n_in > 1:
            proba = np.tile(proba, (n_in, 1))
        elif r > n_in:
            proba = proba[:n_in, :]
        else:
            pad = np.full((n_in - r, 3), 1.0 / 3.0, dtype=float)
            proba = np.vstack([proba, pad])

        proba = self._safe_nan_to_num(proba, fill=1.0 / 3.0)
        sums = np.sum(proba, axis=1, keepdims=True)
        proba = proba / np.maximum(sums, 1e-12)
        return proba

    def _extract_matrix(self, X, fit=False):
        # Preferred path for pandas DataFrame input from ingestion.
        if hasattr(X, "columns"):
            if fit or self.feature_cols is None:
                self.feature_cols = list(X.columns)

            cols = [c for c in self.feature_cols if c in X.columns]
            if len(cols) == 0:
                arr = self._safe_float_matrix(X)
                return arr

            df = X[cols].copy()

            used = [c for c in cols if c not in self.drop_cols]
            if len(used) == 0:
                used = cols

            if fit:
                self.used_cols = used
                self.has_mm = "mm" in used

            use_cols = self.used_cols if self.used_cols is not None else used
            use_cols = [c for c in use_cols if c in df.columns]
            if len(use_cols) == 0:
                use_cols = [c for c in cols if c in df.columns]

            base = self._safe_float_matrix(df[use_cols].values)

            # Add cyclic month terms if available.
            if "mm" in df.columns:
                mm = self._safe_float_matrix(df[["mm"]].values)
                mm = np.clip(mm, 1.0, 12.0)
                mm_sin = np.sin(2.0 * np.pi * mm / 12.0)
                mm_cos = np.cos(2.0 * np.pi * mm / 12.0)
                base = np.hstack([base, mm_sin.astype(np.float32), mm_cos.astype(np.float32)])

            return base.astype(np.float32, copy=False)

        # Fallback: array-like input.
        return self._safe_float_matrix(X)

    def _fit_numpy_logistic(self, X, y_bin):
        X = self._safe_float_matrix(X)
        n, d = X.shape
        if n < 5:
            self.w = None
            self.b = 0.0
            return

        self.mu = np.mean(X, axis=0, keepdims=True).astype(np.float32)
        self.sd = np.std(X, axis=0, keepdims=True).astype(np.float32)
        Xn = (X - self.mu) / np.maximum(self.sd, 1e-6)

        # Helpful nonlinearity for linear fallback model.
        Xn = np.hstack([Xn, Xn * Xn]).astype(np.float32, copy=False)

        n, d = Xn.shape
        y = y_bin.astype(np.float32)
        pos = float(np.sum(y))
        neg = float(n - pos)
        pos_w = neg / max(pos, 1.0)
        wt = np.where(y > 0.5, pos_w, 1.0).astype(np.float32)

        w = np.zeros(d, dtype=np.float32)
        b = float(np.log((np.mean(y) + 1e-6) / (1.0 - np.mean(y) + 1e-6)))

        lr = 0.03
        for t in range(self.max_steps):
            z = np.clip(np.dot(Xn, w) + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y) * wt
            gw = np.dot(Xn.T, err) / float(n) + self.l2 * w
            gb = float(np.mean(err))
            w -= lr * gw.astype(np.float32)
            b -= lr * gb
            if t in (300, 700, 1100):
                lr *= 0.5

        self.w = w
        self.b = b

    def fit(self, X, y):
        try:
            y = np.asarray(y).ravel()
            y_int = y.astype(int)
            y_bin = (y_int == 1).astype(np.float32)

            if y_int.size > 0:
                p1 = float(np.mean(y_int == 1))
                self.base_p1 = float(np.clip(p1, 1e-6, 1.0 - 1e-6))

            rest = y_int[y_int != 1]
            if rest.size > 0:
                p0 = float(np.mean(rest == 0))
                p2 = float(np.mean(rest == 2))
                s = p0 + p2
                if s > 0:
                    self.rest_split = (p0 / s, p2 / s)

            Xf = self._extract_matrix(X, fit=True)
            n = Xf.shape[0]
            pos = float(np.sum(y_bin))
            neg = float(n - pos)
            self.using_xgb = False
            self.xgb_model = None

            # XGBoost Model
            if XGBClassifier is not None and n >= 20 and pos >= 2:
                scale_pos_weight = neg / max(pos, 1.0)
                model = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=4,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    gamma=0.0,
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                )
                try:
                    model.fit(Xf, y_bin)
                    self.xgb_model = model
                    self.using_xgb = True
                    self.w = None
                    self.b = 0.0
                    return self
                except Exception:
                    self.xgb_model = None
                    self.using_xgb = False

            # Fallback model if XGBoost is unavailable.
            self._fit_numpy_logistic(Xf, y_bin)

        except Exception:
            self.xgb_model = None
            self.using_xgb = False
            self.w = None
            self.b = 0.0

        return self

    def _predict_p1_fallback(self, Xf):
        if self.w is None:
            return np.full((Xf.shape[0],), float(self.base_p1), dtype=float)

        Xf = self._safe_float_matrix(Xf)
        if self.mu is None or self.sd is None:
            mu = np.mean(Xf, axis=0, keepdims=True)
            sd = np.std(Xf, axis=0, keepdims=True)
        else:
            mu = self.mu
            sd = self.sd

        Xn = (Xf - mu) / np.maximum(sd, 1e-6)
        Xn = np.hstack([Xn, Xn * Xn]).astype(np.float32, copy=False)
        score = np.dot(Xn, self.w) + float(self.b)
        p1 = self._sigmoid(score)
        return np.clip(p1, 1e-6, 1.0 - 1e-6)

    def predict_proba(self, X):
        n_in = self._n_samples_from_input(X)
        if n_in == 0:
            return np.zeros((0, 3), dtype=float)

        Xf = self._extract_matrix(X, fit=False)

        try:
            if self.using_xgb and self.xgb_model is not None:
                p1 = self.xgb_model.predict_proba(Xf)[:, 1]
                p1 = self._safe_nan_to_num(p1, fill=self.base_p1)
                p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
            else:
                p1 = self._predict_p1_fallback(Xf)

            rest = 1.0 - p1
            p0 = rest * float(self.rest_split[0])
            p2 = rest * float(self.rest_split[1])
            proba = np.vstack([p0, p1, p2]).T

        except Exception:
            proba = np.full((1, 3), 1.0 / 3.0, dtype=float)

        return self._force_shape_n3(proba, n_in)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(int)
