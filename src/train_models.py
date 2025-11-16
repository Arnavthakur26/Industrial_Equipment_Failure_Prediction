from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .make_features import add_classification_label
from .utils import time_based_unit_split, ensure_dir


TARGET_RUL_COL = "RUL"
CLASS_LABEL_COL = "fail_within_horizon"


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select all columns that should be used as features.
    """
    blacklist = {"unit", "cycle", TARGET_RUL_COL, CLASS_LABEL_COL}
    feature_cols = [c for c in df.columns if c not in blacklist]
    return feature_cols


def train_regression_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[LinearRegression, XGBRegressor, StandardScaler]:
    """
    Train Linear Regression and XGBoost Regressor on RUL.
    """
    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_RUL_COL].values
    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_RUL_COL].values

    # Scale features for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1) Linear Regression baseline
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)

    y_pred_lr = lin_reg.predict(X_val_scaled)
    mae_lr = mean_absolute_error(y_val, y_pred_lr)
    mse_lr = mean_squared_error(y_val, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    print(f"[LinearRegression] Val MAE={mae_lr:.3f}, RMSE={rmse_lr:.3f}")

    # 2) XGBoost Regressor
    xgb_reg = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_xgb = xgb_reg.predict(X_val)
    mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
    mse_xgb = mean_squared_error(y_val, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    print(f"[XGBRegressor]   Val MAE={mae_xgb:.3f}, RMSE={rmse_xgb:.3f}")

    return lin_reg, xgb_reg, scaler

def train_classification_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[LogisticRegression, XGBClassifier]:
    """
    Train Logistic Regression and XGBoost Classifier on binary label.
    """
    X_train = df_train[feature_cols].values
    y_train = df_train[CLASS_LABEL_COL].values
    X_val = df_val[feature_cols].values
    y_val = df_val[CLASS_LABEL_COL].values

    # handle imbalance via class_weight
    class_weight = "balanced"

    # 1) Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        n_jobs=-1,
    )
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_val)
    y_prob_lr = log_reg.predict_proba(X_val)[:, 1]

    acc_lr = accuracy_score(y_val, y_pred_lr)
    f1_lr = f1_score(y_val, y_pred_lr)
    prec_lr = precision_score(y_val, y_pred_lr)
    rec_lr = recall_score(y_val, y_pred_lr)
    roc_lr = roc_auc_score(y_val, y_prob_lr)
    print(
        f"[LogisticRegression] "
        f"Val Acc={acc_lr:.3f}, F1={f1_lr:.3f}, "
        f"Prec={prec_lr:.3f}, Rec={rec_lr:.3f}, ROC-AUC={roc_lr:.3f}"
    )

    # 2) XGBoost Classifier
    # scale_pos_weight = negative / positive
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    if pos == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = float(neg) / float(pos)

    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_xgb = xgb_clf.predict(X_val)
    y_prob_xgb = xgb_clf.predict_proba(X_val)[:, 1]

    acc_xgb = accuracy_score(y_val, y_pred_xgb)
    f1_xgb = f1_score(y_val, y_pred_xgb)
    prec_xgb = precision_score(y_val, y_pred_xgb)
    rec_xgb = recall_score(y_val, y_pred_xgb)
    roc_xgb = roc_auc_score(y_val, y_prob_xgb)
    print(
        f"[XGBClassifier]     "
        f"Val Acc={acc_xgb:.3f}, F1={f1_xgb:.3f}, "
        f"Prec={prec_xgb:.3f}, Rec={rec_xgb:.3f}, ROC-AUC={roc_xgb:.3f}"
    )

    return log_reg, xgb_clf


def save_models(
    models_dir: str | Path,
    lin_reg: LinearRegression,
    xgb_reg: XGBRegressor,
    scaler: StandardScaler,
    log_reg: LogisticRegression | None = None,
    xgb_clf: XGBClassifier | None = None,
) -> None:
    """
    Save trained models and scaler to disk.
    """
    models_dir = ensure_dir(models_dir)

    joblib.dump(lin_reg, models_dir / "linear_regression_rul.pkl")
    joblib.dump(xgb_reg, models_dir / "xgb_rul.pkl")
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    if log_reg is not None:
        joblib.dump(log_reg, models_dir / "logistic_failure.pkl")
    if xgb_clf is not None:
        joblib.dump(xgb_clf, models_dir / "xgb_failure.pkl")


def run_training(features_path: str | Path, models_dir: str | Path, horizon: int = 30) -> None:
    """
    High-level training pipeline:
    - Load features
    - Add classification label
    - Split by unit
    - Train regression + classification models
    - Save them
    """
    features_path = Path(features_path)
    df = pd.read_csv(features_path)

    # Add binary classification label
    df = add_classification_label(df, horizon=horizon, label_col=CLASS_LABEL_COL)

    # Train/val/test split
    df_train, df_val, df_test = time_based_unit_split(df)

    feature_cols = _select_feature_columns(df_train)

    print(f"Using {len(feature_cols)} feature columns.")

    # Train regression models
    lin_reg, xgb_reg, scaler = train_regression_models(df_train, df_val, feature_cols)

    # Train classification models
    log_reg, xgb_clf = train_classification_models(df_train, df_val, feature_cols)

    # Save everything
    save_models(models_dir, lin_reg, xgb_reg, scaler, log_reg, xgb_clf)

    # Save test set for later evaluation
    out_test = Path(models_dir).parent / "test_features.csv"
    df_test.to_csv(out_test, index=False)
    print(f"Saved test features to {out_test}")
