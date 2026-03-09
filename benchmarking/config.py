"""Central configuration for multi-horizon benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BenchmarkConfig:
    """Configuration for static-split multi-horizon benchmarking."""

    train_path: Path = PROJECT_ROOT / "data" / "train.csv"
    test_path: Path = PROJECT_ROOT / "data" / "test.csv"
    output_dir: Path = PROJECT_ROOT / "artifacts" / "benchmark"
    target_col: str = "y_12m"
    id_columns: tuple[str, ...] = ("CompNo",)
    time_columns: tuple[str, ...] = ("yyyy", "mm")
    classes: tuple[int, int, int] = (0, 1, 2)
    default_class: int = 1
    random_state: int = 42
    validation_fraction: float = 0.0
    drop_other_horizon_targets: bool = True
    save_models: bool = True
    generate_roc_plots: bool = True
    tune_hyperparameters: bool = False
    tuning_validation_fraction: float = 0.2
    max_tuning_trials_per_model: int = 20
    default_horizons: tuple[int, ...] = (1, 3, 6, 12, 24, 36, 48, 60)
    model_names: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        "lstm",
    )
    model_params: dict[str, dict] = field(
        default_factory=lambda: {
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 2000,
                "solver": "lbfgs",
            },
            "random_forest": {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 1,
                "n_jobs": -1,
            },
            "lightgbm": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": -1,
            },
            "xgboost": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": -1,
            },
            "lstm": {
                "hidden_units": 32,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "epochs": 8,
                "batch_size": 256,
                "class_weight_mode": "balanced",
                "verbose": 0,
            },
        }
    )
    tuning_param_grid: dict[str, dict[str, list]] = field(
        default_factory=lambda: {
            "logistic_regression": {
                "C": [0.1, 0.3, 1.0, 3.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
            "random_forest": {
                "n_estimators": [300, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 5],
                "max_features": ["sqrt", 0.7],
            },
            "lightgbm": {
                "n_estimators": [300, 500],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [20, 60],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "xgboost": {
                "n_estimators": [300, 500],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "min_child_weight": [1, 5],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "lstm": {
                "hidden_units": [16, 32, 64],
                "dropout": [0.1, 0.2, 0.3],
                "learning_rate": [5e-4, 1e-3],
                "epochs": [6, 10],
                "batch_size": [128, 256],
            },
        }
    )
