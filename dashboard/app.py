import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# ==========================================================
# PATHS
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TEST_FEATURES_PATH = PROJECT_ROOT / "test_features.csv"

TARGET_RUL_COL = "RUL"
CLASS_LABEL_COL = "fail_within_horizon"


# ==========================================================
# LOADING FUNCTIONS
# ==========================================================
@st.cache_data
def load_test_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_models(models_dir: Path):
    xgb_reg = joblib.load(models_dir / "xgb_rul.pkl")

    try:
        xgb_clf = joblib.load(models_dir / "xgb_failure.pkl")
    except:
        xgb_clf = None

    try:
        lin_reg = joblib.load(models_dir / "linear_regression_rul.pkl")
    except:
        lin_reg = None

    try:
        log_reg = joblib.load(models_dir / "logistic_failure.pkl")
    except:
        log_reg = None

    try:
        scaler = joblib.load(models_dir / "feature_scaler.pkl")
    except:
        scaler = None

    return xgb_reg, xgb_clf, log_reg, lin_reg, scaler


def get_feature_columns(df: pd.DataFrame):
    blacklist = {"unit", "cycle", TARGET_RUL_COL, CLASS_LABEL_COL}
    return [c for c in df.columns if c not in blacklist]


# ==========================================================
# PREDICTION FUNCTIONS
# ==========================================================
def add_predictions(df: pd.DataFrame, xgb_reg, xgb_clf, feature_cols):
    df = df.copy()
    X = df[feature_cols].values

    # RUL prediction
    df["pred_RUL"] = xgb_reg.predict(X)

    # Failure classifier
    if xgb_clf is not None:
        df["failure_prob"] = xgb_clf.predict_proba(X)[:, 1]
    else:
        df["failure_prob"] = np.nan

    return df


def derive_risk_levels(df: pd.DataFrame, low: int, med: int):
    df = df.copy()
    max_rul = df["pred_RUL"].max()

    df["risk_level"] = pd.cut(
        df["pred_RUL"],
        bins=[-1, low, med, max_rul + 1],
        labels=["High", "Medium", "Low"]
    )
    return df


def get_latest_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["unit", "cycle"])
          .groupby("unit")
          .tail(1)
          .reset_index(drop=True)
    )


# ==========================================================
# METRICS
# ==========================================================
def compute_metrics(df, xgb_reg, xgb_clf, lin_reg, log_reg, scaler, feature_cols):
    metrics = {}

    X = df[feature_cols].values
    y_rul = df[TARGET_RUL_COL].values
    y_cls = df[CLASS_LABEL_COL].values

    # --- Regression XGB ---
    y_pred = xgb_reg.predict(X)
    metrics["xgb_rmse"] = np.sqrt(mean_squared_error(y_rul, y_pred))
    metrics["xgb_mae"] = mean_absolute_error(y_rul, y_pred)

    # --- Regression Linear ---
    if lin_reg is not None and scaler is not None:
        Xs = scaler.transform(X)
        y_linear = lin_reg.predict(Xs)
        metrics["lr_rmse"] = np.sqrt(mean_squared_error(y_rul, y_linear))
        metrics["lr_mae"] = mean_absolute_error(y_rul, y_linear)

    # --- Classification ---
    if xgb_clf is not None:
        y_pred_cls = xgb_clf.predict(X)
        y_prob = xgb_clf.predict_proba(X)[:, 1]
        metrics["xgb_acc"] = accuracy_score(y_cls, y_pred_cls)
        metrics["xgb_prec"] = precision_score(y_cls, y_pred_cls)
        metrics["xgb_rec"] = recall_score(y_cls, y_pred_cls)
        metrics["xgb_f1"] = f1_score(y_cls, y_pred_cls)
        metrics["xgb_roc_auc"] = roc_auc_score(y_cls, y_prob)

    if log_reg is not None:
        y_pred_lr = log_reg.predict(X)
        y_prob_lr = log_reg.predict_proba(X)[:, 1]
        metrics["lr_acc"] = accuracy_score(y_cls, y_pred_lr)
        metrics["lr_prec"] = precision_score(y_cls, y_pred_lr)
        metrics["lr_rec"] = recall_score(y_cls, y_pred_lr)
        metrics["lr_f1"] = f1_score(y_cls, y_pred_lr)
        metrics["lr_roc_auc"] = roc_auc_score(y_cls, y_prob_lr)

    return metrics


# ==========================================================
# UI COMPONENTS
# ==========================================================
def header(metrics, df):
    st.title("🔧 Predictive Maintenance Dashboard – Turbofan Engines")

    st.markdown(
        """
        This dashboard visualizes **Remaining Useful Life (RUL)** predictions,
        failure risk scores, feature insights, and model performance for
        industrial turbofan engines based on NASA C-MAPSS FD001 data.
        """
    )

    st.sidebar.header("Dataset Overview")
    st.sidebar.write(f"Engines in test set: **{df['unit'].nunique()}**")
    st.sidebar.write(f"Total windows: **{len(df)}**")

    if "xgb_rmse" in metrics:
        st.sidebar.metric("XGB RMSE (RUL)", f"{metrics['xgb_rmse']:.2f}")

    if "xgb_f1" in metrics:
        st.sidebar.metric("XGB F1-score", f"{metrics['xgb_f1']:.3f}")


def tab_summary(metrics):
    st.subheader("Executive Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### RUL Prediction (Regression - XGBoost)")
        st.write(f"- RMSE: `{metrics['xgb_rmse']:.2f}` cycles")
        st.write(f"- MAE: `{metrics['xgb_mae']:.2f}` cycles")

    with col2:
        st.markdown("### Failure Detection (Classification - XGBoost)")
        if "xgb_f1" in metrics:
            st.write(f"- Accuracy: `{metrics['xgb_acc']:.3f}`")
            st.write(f"- Precision: `{metrics['xgb_prec']:.3f}`")
            st.write(f"- Recall: `{metrics['xgb_rec']:.3f}`")
            st.write(f"- F1-score: `{metrics['xgb_f1']:.3f}`")
            st.write(f"- ROC–AUC: `{metrics['xgb_roc_auc']:.3f}`")
        else:
            st.info("XGB classifier not available.")


def tab_fleet(df_pred, df_latest):
    st.subheader("Fleet Overview")

    st.write("### Latest engine RUL predictions")
    st.dataframe(
        df_latest[["unit", "cycle", "RUL", "pred_RUL", "risk_level"]]
        .sort_values("pred_RUL")
    )

    st.bar_chart(df_latest.set_index("unit")["pred_RUL"])


def tab_engine(df_pred):
    st.subheader("Engine Explorer")

    unit_ids = sorted(df_pred["unit"].unique())
    unit = st.selectbox("Select Engine ID:", unit_ids)

    df_unit = df_pred[df_pred["unit"] == unit].sort_values("cycle")

    st.line_chart(
        df_unit.set_index("cycle")[["RUL", "pred_RUL"]]
        .rename(columns={"RUL": "True RUL", "pred_RUL": "Predicted RUL"})
    )

    st.markdown("### Feature Trends (Mean/Std Only)")
    candidates = [c for c in df_pred.columns if "_mean" in c or "_std" in c]
    feats = st.multiselect("Select features:", candidates, default=candidates[:3])

    if feats:
        st.line_chart(df_unit.set_index("cycle")[feats])


def tab_model_perf(df, feature_cols, xgb_reg, xgb_clf, metrics):
    st.subheader("Model Performance")

    X = df[feature_cols].values
    y_true = df[TARGET_RUL_COL].values
    y_pred = xgb_reg.predict(X)

    st.write("### Regression – True vs Predicted RUL")

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    st.pyplot(fig)

    if xgb_clf is not None:
        st.write("### Classification – Confusion Matrix")
        y_cls = df[CLASS_LABEL_COL].values
        y_pred_cls = xgb_clf.predict(X)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_cls, y_pred_cls, ax=ax)
        st.pyplot(fig)


def tab_feature_insights(xgb_reg):
    st.subheader("Feature Importance (XGBoost – Gain)")
    booster = xgb_reg.get_booster()
    importance = booster.get_score(importance_type="gain")

    imp_df = pd.DataFrame(
        {"feature": list(importance.keys()), "gain": list(importance.values())}
    ).sort_values("gain", ascending=False)

    st.dataframe(imp_df.head(20))
    st.bar_chart(imp_df.head(20).set_index("feature"))


# ==========================================================
# MAIN
# ==========================================================
def main():
    st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

    df = load_test_data(TEST_FEATURES_PATH)
    xgb_reg, xgb_clf, log_reg, lin_reg, scaler = load_models(MODELS_DIR)
    feature_cols = get_feature_columns(df)

    # Risk thresholds
    with st.sidebar:
        st.markdown("### Risk thresholds")
        low = st.slider("High risk if RUL ≤", 5, 60, 20)
        med = st.slider("Medium risk if RUL ≤", low + 5, 120, 50)

    # Predictions
    df_pred = add_predictions(df, xgb_reg, xgb_clf, feature_cols)
    df_pred = derive_risk_levels(df_pred, low, med)
    df_latest = get_latest_per_unit(df_pred)

    # Metrics
    metrics = compute_metrics(df, xgb_reg, xgb_clf, lin_reg, log_reg, scaler, feature_cols)

    header(metrics, df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏁 Summary", "📊 Fleet Monitor", "🔍 Engine Explorer", "📈 Model Performance", "🧠 Feature Insights"]
    )

    with tab1:
        tab_summary(metrics)

    with tab2:
        tab_fleet(df_pred, df_latest)

    with tab3:
        tab_engine(df_pred)

    with tab4:
        tab_model_perf(df, feature_cols, xgb_reg, xgb_clf, metrics)

    with tab5:
        tab_feature_insights(xgb_reg)


if __name__ == "__main__":
    main()
