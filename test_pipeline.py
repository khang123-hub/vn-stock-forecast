"""
Script kiểm thử tự động toàn bộ Pipeline định lượng chứng khoán Việt Nam:
Tải dữ liệu -> Tạo đặc trưng -> Huấn luyện mô hình -> Đánh giá KPIs -> Backtest -> Dự báo.
"""

import sys
import os
import pandas as pd
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data
from src.features import prepare_features_and_targets
from src.models import (
    split_time_series_data,
    train_regression_models,
    train_classification_model,
    get_feature_importance,
    determine_consensus_signal
)
from src.backtest import run_backtest


def main():
    symbol = "FPT"
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU KIỂM THỬ PIPELINE ĐỊNH LƯỢNG CHO MÃ: {symbol}")
    print("=" * 65)

    # 1. Tải dữ liệu
    print("\n[Bước 1/6] Đang tải dữ liệu lịch sử...")
    df = load_stock_data(symbol, start_date="2020-01-01")
    print(f"-> Đã tải {len(df)} phiên giao dịch (Từ {df['date'].min().strftime('%d/%m/%Y')} đến {df['date'].max().strftime('%d/%m/%Y')})")
    assert len(df) > 100, "Dữ liệu quá ngắn để huấn luyện."

    # 2. Tạo đặc trưng kỹ thuật & biến mục tiêu
    print("\n[Bước 2/6] Đang tính toán các chỉ báo kỹ thuật & Target...")
    n_forward = 5
    profit_threshold = 0.015
    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(
        df, n_forward=n_forward, profit_threshold=profit_threshold
    )
    print(f"-> Số lượng đặc trưng: {len(feature_cols)}")
    print(f"-> Kích thước ma trận X: {X.shape}, y_reg: {y_reg.shape}, y_clf: {y_clf.shape}")
    print(f"-> Tỷ lệ nhãn tích cực (Target 2 = 1): {y_clf.mean() * 100:.2f}%")

    # 3. Phân tách Train / Test theo thời gian (Chronological Split - No Leakage)
    print("\n[Bước 3/6] Phân tách Train/Test theo thứ tự thời gian (80% Train, 20% Test)...")
    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X, y_reg, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X, y_clf, train_ratio=0.8)
    print(f"-> Train samples: {len(X_train_r)}, Test samples: {len(X_test_r)}")

    # 4. Huấn luyện & So sánh Mô hình Hồi quy trên Forward Return
    print("\n[Bước 4/6] Huấn luyện Mô hình Hồi quy trên Forward Return...")
    test_closes = df.loc[X_test_r.index, "close"]
    reg_results = train_regression_models(
        X_train_r, y_train_r, X_test_r, y_test_r,
        current_closes_test=test_closes
    )
    print("BẢNG SO SÁNH CHỈ SỐ HỒI QUY TRÊN TEST SET (QUY ĐỔI GIÁ):")
    print(reg_results["comparison_df"].to_string(index=False))

    # 5. Huấn luyện Mô hình Phân loại
    print("\n[Bước 5/6] Huấn luyện Mô hình Phân loại (XGBoost Classifier)...")
    clf_results = train_classification_model(X_train_c, y_train_c, X_test_c, y_test_c)
    print("CHỈ SỐ PHÂN LOẠI TRÊN TEST SET:")
    for k, v in clf_results["metrics"].items():
        print(f"  - {k}: {v}")

    # Top Features
    feat_imp = get_feature_importance(clf_results["pipeline"], feature_cols)
    print("\nTOP 5 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    print(feat_imp.head(5).to_string(index=False))

    # 6. Backtest Chiến lược
    print("\n[Bước 6/6] Backtest Chiến lược Giao dịch (Chu kỳ T+2.5, Phí 0.15%/chiều)...")
    test_sub_df = df.loc[X_test_r.index].copy()
    y_test_probs = clf_results["y_prob"]
    bt_res = run_backtest(
        test_df=test_sub_df,
        y_prob=y_test_probs,
        threshold=0.55,
        holding_period=n_forward,
        fee_per_trade=0.0015
    )

    bt_m = bt_res["metrics"]
    print(f"-> Tổng Lợi Nhuận AI: {bt_m['AI_Total_Return_Pct']:+.2f}% vs Buy & Hold: {bt_m['BH_Total_Return_Pct']:+.2f}%")
    print(f"-> Tỷ lệ Thắng (Win Rate): {bt_m['Win_Rate_Pct']:.1f}% ({bt_m['Win_Trades']}/{bt_m['Total_Trades']} lệnh)")
    print(f"-> Max Drawdown AI: {bt_m['AI_Max_Drawdown_Pct']:.2f}% vs B&H: {bt_m['BH_Max_Drawdown_Pct']:.2f}%")

    # 7. Dự báo cho phiên hiện tại
    print("\n" + "=" * 65)
    print("🔮 KẾT QUẢ DỰ BÁO CHO PHIÊN TIẾP THEO:")
    curr_close = float(df["close"].iloc[-1])
    latest_feat = latest_row[feature_cols]

    pred_xgb_return = float(reg_results["xgb_pipeline"].predict(latest_feat)[0])
    pred_xgb_price = curr_close * (1.0 + pred_xgb_return)
    pred_up_prob = float(clf_results["pipeline"].predict_proba(latest_feat)[0, 1])
    exp_change = pred_xgb_return * 100.0

    consensus = determine_consensus_signal(
        prob_up=pred_up_prob,
        exp_return_pct=exp_change,
        prob_threshold=0.55,
        min_return_pct=profit_threshold * 100.0
    )

    print(f"  * Giá đóng cửa hiện tại: {curr_close:,.0f} đ")
    print(f"  * Tỷ suất sinh lời kỳ vọng: {exp_change:+.2f}%")
    print(f"  * Giá dự báo sau {n_forward} phiên: {pred_xgb_price:,.0f} đ")
    print(f"  * Xác suất tăng giá > {profit_threshold*100:.1f}%: {pred_up_prob*100:.1f}%")
    print(f"  * KHUYẾN NGHỊ (Strict Consensus): {consensus['signal_badge']}")
    print("=" * 65)
    print("\n✅ KIỂM THỬ PIPELINE HOÀN TẤT THÀNH CÔNG VÀ KHÔNG CÓ LỖI!")


if __name__ == "__main__":
    main()

