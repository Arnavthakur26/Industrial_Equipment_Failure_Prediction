from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .train_models import TARGET_RUL_COL, CLASS_LABEL_COL, _select_feature_columns


def _load_models(models_dir: str | Path):
    models_dir = Path(models_dir)
    lin_reg = joblib.load(models_dir / "linear_regression_rul.pkl")
    xgb_reg = joblib.load(models_dir / "xgb_rul.pkl")
    scaler = joblib.load(models_dir / "feature_scaler.pkl")

    log_reg = None
    xgb_clf = None

    log_reg_path = models_dir / "logistic_failure.pkl"
    xgb_clf_path = models_dir / "xgb_failure.pkl"
    if log_reg_path.exists():
        log_reg = joblib.load(log_reg_path)
    if xgb_clf_path.exists():
        xgb_clf = joblib.load(xgb_clf_path)

    return lin_reg, xgb_reg, scaler, log_reg, xgb_clf


def evaluate_regression_models(
    df_test: pd.DataFrame,
    feature_cols: List[str],
    lin_reg,
    xgb_reg,
    scaler,
) -> Dict[str, float]:
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_RUL_COL].values

    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = lin_reg.predict(X_test_scaled)
    y_pred_xgb = xgb_reg.predict(X_test)

    metrics = {}

    for name, y_pred in [("lr", y_pred_lr), ("xgb", y_pred_xgb)]:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse

    print("Regression test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    return metrics

def evaluate_classification_models(
    df_test: pd.DataFrame,
    feature_cols: List[str],
    log_reg,
    xgb_clf,
) -> Dict[str, float]:
    y_test = df_test[CLASS_LABEL_COL].values
    X_test = df_test[feature_cols].values

    metrics: Dict[str, float] = {}
    print("Classification test metrics:")

    if log_reg is not None:
        y_pred_lr = log_reg.predict(X_test)
        y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

        metrics.update(_classification_metrics("lr", y_test, y_pred_lr, y_prob_lr))

    if xgb_clf is not None:
        y_pred_xgb = xgb_clf.predict(X_test)
        y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

        metrics.update(_classification_metrics("xgb", y_test, y_pred_xgb, y_prob_xgb))

    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    return metrics


def _classification_metrics(
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    return {
        f"{prefix}_acc": acc,
        f"{prefix}_f1": f1,
        f"{prefix}_prec": prec,
        f"{prefix}_rec": rec,
        f"{prefix}_roc_auc": roc,
    }


def run_evaluation(
    test_features_path: str | Path,
    models_dir: str | Path,
) -> Dict[str, float]:
    """
    High-level evaluation pipeline:

    - Load test features
    - Load trained models
    - Evaluate regression + classification
    """
    test_features_path = Path(test_features_path)
    df_test = pd.read_csv(test_features_path)

    lin_reg, xgb_reg, scaler, log_reg, xgb_clf = _load_models(models_dir)
    feature_cols = _select_feature_columns(df_test)

    metrics_reg = evaluate_regression_models(df_test, feature_cols, lin_reg, xgb_reg, scaler)
    metrics_clf = evaluate_classification_models(df_test, feature_cols, log_reg, xgb_clf)

    all_metrics = {**metrics_reg, **metrics_clf}
    return all_metrics
