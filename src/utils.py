from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def time_based_unit_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    unit_col: str = "unit",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/val/test by unit ID
    (to avoid leakage between units).

    Parameters
    ----------
    df:
        Features dataframe containing a 'unit' column.
    train_frac:
        Fraction of units used for training.
    val_frac:
        Fraction of units used for validation.
        Test fraction is implicitly 1 - train_frac - val_frac.
    unit_col:
        Column containing unit IDs.

    Returns
    -------
    df_train, df_val, df_test
    """
    units = sorted(df[unit_col].unique())
    n_units = len(units)
    n_train = int(n_units * train_frac)
    n_val = int(n_units * val_frac)

    train_units = units[:n_train]
    val_units = units[n_train : n_train + n_val]
    test_units = units[n_train + n_val :]

    df_train = df[df[unit_col].isin(train_units)].reset_index(drop=True)
    df_val = df[df[unit_col].isin(val_units)].reset_index(drop=True)
    df_test = df[df[unit_col].isin(test_units)].reset_index(drop=True)

    print(f"Total units: {n_units}")
    print(f"Train units: {len(train_units)}, Val units: {len(val_units)}, Test units: {len(test_units)}")

    return df_train, df_val, df_test


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
