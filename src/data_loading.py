from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Default column names for C-MAPSS FD001 training file
RAW_COLS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"s_{i}" for i in range(1, 22)],  # 21 sensors
]


def load_cmapps_fd001(raw_path: str | Path) -> pd.DataFrame:
    """
    Load the C-MAPSS FD001 training dataset from NASA.

    Parameters
    ----------
    raw_path:
        Path to the raw training file (e.g. 'data/raw/train_FD001.txt').

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        ['unit', 'cycle', 'setting_1..3', 's_1..s_21']
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(
        raw_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    df = df.iloc[:, : len(RAW_COLS)]
    df.columns = RAW_COLS
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) per row.

    RUL = max_cycle_per_unit - current_cycle

    Parameters
    ----------
    df:
        DataFrame with at least 'unit' and 'cycle' columns.

    Returns
    -------
    df_with_rul : pd.DataFrame
    """
    df = df.copy()
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]
    return df


def drop_low_variance_sensors(
    df: pd.DataFrame,
    sensor_prefix: str = "s_",
    threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Optionally drop sensors that have almost no variance (uninformative).

    Parameters
    ----------
    df:
        Input dataframe.
    sensor_prefix:
        Prefix for sensor columns, default 's_'.
    threshold:
        Minimum variance to keep a sensor.

    Returns
    -------
    df_reduced : pd.DataFrame
    """
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith(sensor_prefix)]
    variances = df[sensor_cols].var()
    keep_cols = variances[variances > threshold].index.tolist()

    drop_cols = sorted(set(sensor_cols) - set(keep_cols))
    if drop_cols:
        print(f"Dropping low-variance sensors: {drop_cols}")
        df = df.drop(columns=drop_cols)

    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save DataFrame as CSV with reasonable defaults.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataframe(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV DataFrame from disk.

    Parameters
    ----------
    path:
        File path.
    usecols:
        Optional subset of columns.

    Returns
    -------
    df : pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, usecols=usecols)
