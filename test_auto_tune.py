"""
Script kiểm thử tự động các tính năng:
- Auto-Hyperparameter Tuning (TimeSeriesSplit + RandomizedSearchCV)
- Exponential Time Decay Weights
- Auto-Threshold Calibration
- Model Drift Detection
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data
from src.features import prepare_features_and_targets
from src.models import (
    split_time_series_data,
    train_regression_models,
    train_classification_model,
    calibrate_optimal_threshold,
    detect_model_drift
)


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ TÍNH NĂNG AUTO-HYPERPARAMETER TUNING & DRIFT")
    print("=" * 65)

    start_time = time.time()
    df = load_stock_data("FPT", start_date="2020-01-01", use_cache=True)
    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(df, n_forward=5)

    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X, y_reg, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X, y_clf, train_ratio=0.8)

    print(f"\n[1] Train samples: {len(X_train_r)}, Test samples: {len(X_test_r)}")

    # 1. Huấn luyện Hồi quy với auto_tune=True & use_time_decay=True
    print("\n[2] Huấn luyện Regression với auto_tune=True & use_time_decay=True...")
    t0 = time.time()
    reg_res = train_regression_models(
        X_train_r, y_train_r, X_test_r, y_test_r,
        auto_tune=True,
        use_time_decay=True
    )
    t_reg = time.time() - t0
    print(f"-> Thời gian tinh chỉnh hồi quy: {t_reg:.2f}s")
    print(f"-> Best Ridge params: {reg_res['best_params_ridge']}")
    print(f"-> Best XGB params: {reg_res['best_params_xgb']}")
    print(reg_res["comparison_df"].to_string(index=False))

    # 2. Huấn luyện Phân loại với auto_tune=True & use_time_decay=True
    print("\n[3] Huấn luyện Classification với auto_tune=True & use_time_decay=True...")
    t0 = time.time()
    clf_res = train_classification_model(
        X_train_c, y_train_c, X_test_c, y_test_c,
        auto_tune=True,
        use_time_decay=True
    )
    t_clf = time.time() - t0
    print(f"-> Thời gian tinh chỉnh phân loại: {t_clf:.2f}s")
    print(f"-> Best XGB Clf params: {clf_res['best_params']}")
    print(f"-> Metrics: {clf_res['metrics']}")

    # 3. Tự động điều chỉnh ngưỡng tối ưu
    print("\n[4] Tự động hiệu chỉnh ngưỡng tối ưu (Auto-Threshold Calibration)...")
    train_closes = df.loc[X_train_r.index, "close"]
    opt_info = calibrate_optimal_threshold(
        X_train=X_train_c,
        y_train_clf=y_train_c,
        y_train_reg=y_train_r,
        current_closes_train=train_closes,
        clf_pipeline=clf_res["pipeline"],
        holding_period=5
    )
    print(f"-> Ngưỡng tối ưu gợi ý: {opt_info['optimal_threshold']} (Precision: {opt_info['precision']}%, Số lệnh: {opt_info['trades_count']}, LN kỳ vọng: {opt_info['avg_net_return_pct']:+.2f}%)")

    # 4. Đánh giá độ trôi mô hình (Model Drift)
    print("\n[5] Đánh giá độ trôi mô hình (Model Drift) trong 20 phiên gần nhất...")
    drift_info = detect_model_drift(
        y_test_clf=y_test_c,
        y_pred_clf=clf_res["y_pred"],
        y_test_reg=y_test_r,
        y_pred_reg=reg_res["xgb_pred"],
        window=20
    )
    print(f"-> Độ chính xác 20 phiên gần nhất: {drift_info['recent_accuracy']}% (Tổng thể Test: {drift_info['overall_accuracy']}%)")
    print(f"-> Sai số MAPE 20 phiên gần nhất: {drift_info['recent_mape']}% (Tổng thể Test: {drift_info['overall_mape']}%)")
    print(f"-> Cảnh báo Regime Shift: {drift_info['is_regime_shift']}")
    if drift_info["warning_message"]:
        print(f"  {drift_info['warning_message']}")

    total_time = time.time() - start_time
    print(f"\n⏱️ TỔNG THỜI GIAN THỰC THI: {total_time:.2f}s (Yêu cầu: < 10s)")
    assert total_time < 10.0, f"Thời gian chạy vượt quá 10s: {total_time:.2f}s"
    print("\n✅ KIỂM THỬ AUTO-TUNE & MODEL DRIFT THÀNH CÔNG RỰC RỠ!")


if __name__ == "__main__":
    main()
