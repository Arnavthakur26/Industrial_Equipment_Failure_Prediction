from __future__ import annotations

from pathlib import Path

from .evaluate import run_evaluation


def main():
    project_root = Path(__file__).resolve().parents[1]

    test_features_path = project_root  / "test_features.csv"
    models_dir = project_root / "models"

    print(f"Evaluating models on test data from {test_features_path} ...")
    metrics = run_evaluation(test_features_path=test_features_path, models_dir=models_dir)

    print("\nAll test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
