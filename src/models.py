"""
Module Models: Huấn luyện, đánh giá mô hình Hồi quy và Phân loại.
Tuân thủ nghiêm ngặt nguyên tắc KHÔNG DATA LEAKAGE:
- Chia tập Train/Test theo thứ tự thời gian (shuffle=False, 80% Train, 20% Test).
- StandardScaler được đóng gói trong sklearn.pipeline.Pipeline, chỉ fit trên tập Train.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier


def split_time_series_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Phân tách dữ liệu theo đúng trình tự thời gian (Chronological Split).
    TUYỆT ĐỐI không xáo trộn (shuffle=False) để tránh rò rỉ dữ liệu tương lai vào quá khứ.
    """
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Huấn luyện và so sánh các mô hình Hồi quy:
    - Ridge Regression (hoặc OLS Trend)
    - XGBoost Regressor
    Đánh giá đầy đủ: MAE, RMSE, MAPE (%), R2.
    """
    # 1. Pipeline Ridge Regression
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_pred = ridge_pipeline.predict(X_test)

    # 2. Pipeline XGBoost Regressor
    xgb_reg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    xgb_reg_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_reg_pipeline.predict(X_test)

    def calc_reg_metrics(y_true, y_hat):
        mae = mean_absolute_error(y_true, y_hat)
        try:
            # scikit-learn >= 1.4
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(y_true, y_hat)
        except ImportError:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
        mape = mean_absolute_percentage_error(y_true, y_hat) * 100.0
        r2 = r2_score(y_true, y_hat)
        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2": round(r2, 4)
        }

    metrics_ridge = calc_reg_metrics(y_test, ridge_pred)
    metrics_xgb = calc_reg_metrics(y_test, xgb_pred)

    comparison_df = pd.DataFrame([
        {"Mô hình": "Ridge Regression", **metrics_ridge},
        {"Mô hình": "XGBoost Regressor", **metrics_xgb}
    ])

    return {
        "ridge_pipeline": ridge_pipeline,
        "xgb_pipeline": xgb_reg_pipeline,
        "ridge_pred": ridge_pred,
        "xgb_pred": xgb_pred,
        "metrics_ridge": metrics_ridge,
        "metrics_xgb": metrics_xgb,
        "comparison_df": comparison_df
    }


def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Huấn luyện mô hình Phân loại XGBoost Classifier dự đoán khả năng tăng giá vượt ngưỡng.
    Đánh giá: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    """
    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    clf_pipeline.fit(X_train, y_train)

    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    }

    return {
        "pipeline": clf_pipeline,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics
    }


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str]
) -> pd.DataFrame:
    """Trích xuất độ quan trọng của đặc trưng từ mô hình XGBoost trong Pipeline."""
    model = pipeline.named_steps.get("model")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.ravel()
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Độ quan trọng": importances
    }).sort_values(by="Độ quan trọng", ascending=False).reset_index(drop=True)

    return df_imp

