"""
Module Models: Huấn luyện, đánh giá mô hình Hồi quy và Phân loại.
Bổ sung cơ chế:
- Tự động tinh chỉnh siêu tham số (Auto-Hyperparameter Tuning) với TimeSeriesSplit & RandomizedSearchCV.
- Trọng số thời gian giảm dần (Exponential Time Decay Weights).
- Tự động hiệu chỉnh ngưỡng tối ưu (Auto-Threshold Calibration).
- Đánh giá độ trôi mô hình (Model Drift Detection) trong 20 phiên gần nhất.
Tuân thủ 100% nguyên tắc KHÔNG DATA LEAKAGE.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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


def compute_time_decay_weights(n_samples: int, half_life_ratio: float = 0.5) -> np.ndarray:
    """
    Tính trọng số thời gian phân rã hàm mũ (Exponential Time Decay Weights).
    Các phiên gần đây có trọng số lớn hơn để mô hình học hành vi thị trường mới nhất.
    """
    if n_samples <= 0:
        return np.array([])
    half_life = max(10, int(n_samples * half_life_ratio))
    decay_rate = np.log(2.0) / half_life
    steps = np.arange(n_samples)
    weights = np.exp(decay_rate * steps)
    # Chuẩn hóa để trung bình bằng 1
    weights = weights / np.mean(weights)
    return weights


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    auto_tune: bool = False,
    use_time_decay: bool = False
) -> Dict[str, Any]:
    """
    Huấn luyện và so sánh các mô hình Hồi quy: Ridge vs XGBoost Regressor.
    Hỗ trợ auto_tune (TimeSeriesSplit + RandomizedSearchCV) và time decay weights.
    Chỉ chạy fit trên Train 80%, không chạm vào Test 20%.
    """
    sample_weights = compute_time_decay_weights(len(X_train)) if use_time_decay else None

    # 1. Pipeline Ridge Regression
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    best_params_ridge = {"alpha": 1.0}
    if auto_tune:
        tscv = TimeSeriesSplit(n_splits=3)
        ridge_param_dist = {
            "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
        }
        rs_ridge = RandomizedSearchCV(
            ridge_pipeline,
            param_distributions=ridge_param_dist,
            n_iter=4,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=1
        )
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        rs_ridge.fit(X_train, y_train, **fit_kwargs)
        ridge_pipeline = rs_ridge.best_estimator_
        best_params_ridge = {k.replace("model__", ""): v for k, v in rs_ridge.best_params_.items()}
    else:
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        ridge_pipeline.fit(X_train, y_train, **fit_kwargs)

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

    best_params_xgb = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    if auto_tune:
        tscv = TimeSeriesSplit(n_splits=3)
        xgb_param_dist = {
            "model__n_estimators": [50, 80, 100],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0]
        }
        rs_xgb = RandomizedSearchCV(
            xgb_reg_pipeline,
            param_distributions=xgb_param_dist,
            n_iter=5,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=1
        )
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        rs_xgb.fit(X_train, y_train, **fit_kwargs)
        xgb_reg_pipeline = rs_xgb.best_estimator_
        best_params_xgb = {k.replace("model__", ""): v for k, v in rs_xgb.best_params_.items()}
    else:
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        xgb_reg_pipeline.fit(X_train, y_train, **fit_kwargs)

    xgb_pred = xgb_reg_pipeline.predict(X_test)

    def calc_reg_metrics(y_true, y_hat):
        mae = mean_absolute_error(y_true, y_hat)
        try:
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
        "comparison_df": comparison_df,
        "best_params_ridge": best_params_ridge,
        "best_params_xgb": best_params_xgb,
        "auto_tuned": auto_tune,
        "used_time_decay": use_time_decay
    }


def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    auto_tune: bool = False,
    use_time_decay: bool = False
) -> Dict[str, Any]:
    """
    Huấn luyện mô hình Phân loại XGBoost Classifier dự đoán khả năng tăng giá vượt ngưỡng.
    Hỗ trợ auto_tune (TimeSeriesSplit + RandomizedSearchCV) và time decay weights.
    """
    sample_weights = compute_time_decay_weights(len(X_train)) if use_time_decay else None

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos = round(float(neg_count / max(1, pos_count)), 2)

    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    best_params_clf = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos
    }

    if auto_tune:
        tscv = TimeSeriesSplit(n_splits=3)
        clf_param_dist = {
            "model__n_estimators": [50, 80, 100],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__scale_pos_weight": [1.0, scale_pos],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0]
        }
        rs_clf = RandomizedSearchCV(
            clf_pipeline,
            param_distributions=clf_param_dist,
            n_iter=5,
            cv=tscv,
            scoring="f1",
            random_state=42,
            n_jobs=1
        )
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        rs_clf.fit(X_train, y_train, **fit_kwargs)
        clf_pipeline = rs_clf.best_estimator_
        best_params_clf = {k.replace("model__", ""): v for k, v in rs_clf.best_params_.items()}
    else:
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["model__sample_weight"] = sample_weights
        clf_pipeline.fit(X_train, y_train, **fit_kwargs)

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
        "metrics": metrics,
        "best_params": best_params_clf,
        "auto_tuned": auto_tune,
        "used_time_decay": use_time_decay
    }


def calibrate_optimal_threshold(
    X_train: pd.DataFrame,
    y_train_clf: pd.Series,
    y_train_reg: pd.Series,
    current_closes_train: pd.Series,
    clf_pipeline: Pipeline,
    holding_period: int = 5,
    fee_per_trade: float = 0.0015
) -> Dict[str, Any]:
    """
    Tự động hiệu chuẩn ngưỡng xác suất tối ưu (Auto-Threshold Calibration).
    Dựa trên 25% cuối của tập Train (Validation nội bộ, No Leakage tới Test set),
    tìm ngưỡng trong khoảng [0.52, 0.70] bước nhảy 0.02 tối đa hóa Precision & Net Return
    với điều kiện số lệnh >= 5.
    """
    val_size = max(30, int(len(X_train) * 0.25))
    X_val = X_train.iloc[-val_size:]
    y_val_clf = y_train_clf.iloc[-val_size:]
    y_val_reg = y_train_reg.iloc[-val_size:]
    closes_val = current_closes_train.iloc[-val_size:]

    val_probs = clf_pipeline.predict_proba(X_val)[:, 1]

    best_thresh = 0.55
    best_score = -999.0
    best_info = {
        "optimal_threshold": 0.55,
        "precision": 0.0,
        "trades_count": 0,
        "avg_net_return_pct": 0.0
    }

    threshold_candidates = np.arange(0.52, 0.72, 0.02)

    for thresh in threshold_candidates:
        t_val = round(float(thresh), 2)
        signals = val_probs >= t_val
        n_trades = int(signals.sum())

        if n_trades < 5:
            continue

        # Precision
        prec = precision_score(y_val_clf, signals.astype(int), zero_division=0)

        # Lợi nhuận ròng trung bình của các lệnh kích hoạt
        trade_entry_prices = closes_val[signals].values
        trade_exit_prices = y_val_reg[signals].values
        if len(trade_entry_prices) > 0:
            gross_returns = (trade_exit_prices / trade_entry_prices) - 1.0
            net_returns = gross_returns - (2 * fee_per_trade)
            avg_net_ret = float(np.mean(net_returns)) * 100.0
        else:
            avg_net_ret = 0.0

        # Hàm mục tiêu định lượng: Kết hợp Precision (60%) và Net Return (40%)
        objective_score = (prec * 60.0) + (avg_net_ret * 2.0)

        if objective_score > best_score:
            best_score = objective_score
            best_thresh = t_val
            best_info = {
                "optimal_threshold": t_val,
                "precision": round(float(prec) * 100, 1),
                "trades_count": n_trades,
                "avg_net_return_pct": round(avg_net_ret, 2)
            }

    return best_info


def detect_model_drift(
    y_test_clf: pd.Series,
    y_pred_clf: np.ndarray,
    y_test_reg: pd.Series,
    y_pred_reg: np.ndarray,
    window: int = 20
) -> Dict[str, Any]:
    """
    Đánh giá độ trôi mô hình (Model Drift) trong 20 phiên giao dịch gần nhất của Test set.
    So sánh Hit Rate và sai số của 20 phiên gần nhất với toàn bộ tập Test.
    """
    n_test = len(y_test_clf)
    actual_window = min(window, n_test)

    # Toàn bộ Test
    overall_acc = accuracy_score(y_test_clf, y_pred_clf) * 100.0
    overall_mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg) * 100.0

    # 20 phiên gần nhất
    recent_y_true = y_test_clf.iloc[-actual_window:]
    recent_y_pred = y_pred_clf[-actual_window:]
    recent_acc = accuracy_score(recent_y_true, recent_y_pred) * 100.0

    recent_y_reg_true = y_test_reg.iloc[-actual_window:]
    recent_y_reg_pred = y_pred_reg[-actual_window:]
    recent_mape = mean_absolute_percentage_error(recent_y_reg_true, recent_y_reg_pred) * 100.0

    is_regime_shift = bool(recent_acc < 50.0)

    return {
        "recent_window": actual_window,
        "recent_accuracy": round(recent_acc, 1),
        "overall_accuracy": round(overall_acc, 1),
        "recent_mape": round(recent_mape, 2),
        "overall_mape": round(overall_mape, 2),
        "is_regime_shift": is_regime_shift,
        "warning_message": (
            "⚠️ Thị trường đang chuyển trạng thái (Regime Shift) - Đã kích hoạt chế độ ưu tiên trọng số ngắn hạn & khuyến nghị bật Auto-Tune"
            if is_regime_shift else None
        )
    }


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """Trích xuất độ quan trọng của đặc trưng từ mô hình trong Pipeline."""
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
