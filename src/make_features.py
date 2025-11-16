from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class FeatureConfig:
    """
    Configuration for window-based feature engineering.
    """
    window_size: int = 30
    top_k_fft: int = 3


def _rolling_slope(x: np.ndarray) -> float:
    """
    Fit a linear trend (time index vs sensor values) and return slope.
    """
    if x.size < 2:
        return 0.0
    t = np.arange(x.size).reshape(-1, 1)
    model = LinearRegression()
    model.fit(t, x)
    return float(model.coef_[0])


def _fft_features(x: np.ndarray, top_k: int = 3) -> Dict[str, float]:
    """
    Compute simple FFT features from a 1D window.

    Returns
    -------
    {
        "fft_power": ...,
        "fft_top1": ...,
        "fft_top2": ...,
        "fft_top3": ...,
    }
    """
    if x.size < 2:
        return {
            "fft_power": 0.0,
            "fft_top1": 0.0,
            "fft_top2": 0.0,
            "fft_top3": 0.0,
        }

    x_centered = x - x.mean()
    fft_vals = np.fft.rfft(x_centered)
    magnitudes = np.abs(fft_vals)

    power = float((magnitudes ** 2).sum())

    # exclude DC component (index 0) when picking top-k
    mag_no_dc = magnitudes[1:]
    if mag_no_dc.size == 0:
        top = [0.0] * top_k
    else:
        idx_sorted = np.argsort(mag_no_dc)[::-1][:top_k]
        top = mag_no_dc[idx_sorted].tolist()
        top += [0.0] * (top_k - len(top))

    return {
        "fft_power": power,
        "fft_top1": float(top[0]),
        "fft_top2": float(top[1]),
        "fft_top3": float(top[2]),
    }


def generate_window_features_for_unit(
    df_unit: pd.DataFrame,
    config: FeatureConfig,
    sensor_cols: List[str],
    setting_cols: List[str],
) -> pd.DataFrame:
    """
    Generate one row of features per sliding window for a single unit.
    """
    rows: List[Dict[str, float]] = []
    values = df_unit.sort_values("cycle").reset_index(drop=True)
    W = config.window_size

    for end_idx in range(W - 1, len(values)):
        window = values.iloc[end_idx - W + 1 : end_idx + 1]
        row: Dict[str, float] = {}

        # meta
        row["unit"] = int(window["unit"].iloc[-1])
        row["cycle"] = int(window["cycle"].iloc[-1])
        # target
        row["RUL"] = float(window["RUL"].iloc[-1])

        # operating conditions: take last value in window
        for c in setting_cols:
            row[f"{c}_last"] = float(window[c].iloc[-1])

        # sensor features
        for c in sensor_cols:
            series = window[c].values.astype(float)
            row[f"{c}_mean"] = float(series.mean())
            row[f"{c}_std"] = float(series.std(ddof=1)) if series.size > 1 else 0.0
            row[f"{c}_min"] = float(series.min())
            row[f"{c}_max"] = float(series.max())
            row[f"{c}_slope"] = _rolling_slope(series)

            fft_feats = _fft_features(series, top_k=config.top_k_fft)
            for k, v in fft_feats.items():
                row[f"{c}_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def generate_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Generate window-based features for all units.

    Parameters
    ----------
    df:
        Input dataframe with columns:
        - 'unit', 'cycle', 'RUL'
        - 'setting_*'
        - 's_*' sensor columns
    config:
        FeatureConfig with window size etc.

    Returns
    -------
    features_df : pd.DataFrame
        One row per (unit, window-end-cycle) with engineered features and RUL.
    """
    if config is None:
        config = FeatureConfig()

    sensor_cols = [c for c in df.columns if c.startswith("s_")]
    setting_cols = [c for c in df.columns if c.startswith("setting_")]

    all_units: List[pd.DataFrame] = []
    for unit_id, df_unit in df.groupby("unit"):
        feats_unit = generate_window_features_for_unit(
            df_unit=df_unit,
            config=config,
            sensor_cols=sensor_cols,
            setting_cols=setting_cols,
        )
        all_units.append(feats_unit)

    features_df = pd.concat(all_units, ignore_index=True)
    return features_df


def add_classification_label(
    df: pd.DataFrame,
    horizon: int = 30,
    label_col: str = "fail_within_horizon",
) -> pd.DataFrame:
    """
    Add a binary label: 1 if RUL <= horizon, else 0.
    """
    df = df.copy()
    if "RUL" not in df.columns:
        raise ValueError("RUL column not found in features dataframe.")
    df[label_col] = (df["RUL"] <= horizon).astype(int)
    return df
