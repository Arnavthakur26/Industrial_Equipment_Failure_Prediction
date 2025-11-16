from __future__ import annotations

from pathlib import Path

from .train_models import run_training
from .utils import ensure_dir


def main():
    project_root = Path(__file__).resolve().parents[1]

    features_path = project_root / "data" / "processed" / "features_fd001.csv"
    models_dir = ensure_dir(project_root / "models")

    print(f"Training models using features from {features_path} ...")
    run_training(features_path=features_path, models_dir=models_dir, horizon=30)

    print("Training complete.")


if __name__ == "__main__":
    main()
