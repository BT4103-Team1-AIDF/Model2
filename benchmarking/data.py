"""Data I/O helpers for CSV/parquet benchmarking inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a table from CSV or parquet."""
    table_path = Path(path)
    suffix = table_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(table_path)
        except ImportError as exc:
            raise ImportError(
                "Parquet input requires a parquet engine such as pyarrow or fastparquet."
            ) from exc

    raise ValueError(f"Unsupported file type: {table_path}. Use .csv or .parquet.")

