from __future__ import annotations

from pathlib import Path

from .data_loading import load_cmapps_fd001, add_rul, drop_low_variance_sensors, save_dataframe
from .make_features import FeatureConfig, generate_features
from .utils import ensure_dir


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw" / "train_FD001.txt"
    interim_dir = ensure_dir(project_root / "data" / "interim")
    processed_dir = ensure_dir(project_root / "data" / "processed")

    print(f"Loading raw data from {raw_path} ...")
    df_raw = load_cmapps_fd001(raw_path)

    print("Adding RUL ...")
    df_rul = add_rul(df_raw)

    print("Dropping low-variance sensors (optional) ...")
    df_clean = drop_low_variance_sensors(df_rul)

    interim_path = interim_dir / "train_fd001_with_rul.csv"
    print(f"Saving interim data to {interim_path} ...")
    save_dataframe(df_clean, interim_path)

    print("Generating window-based features ...")
    config = FeatureConfig(window_size=30, top_k_fft=3)
    df_features = generate_features(df_clean, config=config)

    processed_path = processed_dir / "features_fd001.csv"
    print(f"Saving processed features to {processed_path} ...")
    save_dataframe(df_features, processed_path)

    print("Done.")


if __name__ == "__main__":
    main()
