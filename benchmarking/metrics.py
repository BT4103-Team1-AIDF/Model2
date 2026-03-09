"""Metrics for multi-horizon multiclass benchmarking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def default_auc_ovr(
    y_true: Sequence[int] | np.ndarray,
    y_proba_default: Sequence[float] | np.ndarray,
    *,
    default_class: int = 1,
) -> float:
    """Primary metric: ROC AUC for default class (class 1) vs rest."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_proba_default)
    y_binary = (y_true_arr == int(default_class)).astype(int)
    if np.unique(y_binary).size < 2:
        return float("nan")
    return float(roc_auc_score(y_binary, y_score_arr))


def multiclass_auc_macro_ovr(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    *,
    labels: Sequence[int],
) -> float:
    """Secondary metric: multiclass macro OVR ROC AUC."""
    return float(
        roc_auc_score(
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_proba),
            labels=list(labels),
            multi_class="ovr",
            average="macro",
        )
    )


def summarize_classification(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Secondary classification summary metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
