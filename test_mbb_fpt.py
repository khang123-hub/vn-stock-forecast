"""
Script kiểm thử kiểm chứng tính nhất quán giữa Regressor và Classifier
cho 2 mã: MBB và FPT theo chuẩn Forward Return và Strict Consensus Rule.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data
from src.features import prepare_features_and_targets
from src.models import (
    split_time_series_data,
    train_regression_models,
    train_classification_model,
    determine_consensus_signal,
    predict_latest
)


def test_symbol(symbol: str, n_forward: int = 5, prob_threshold: float = 0.55):
    print("\n" + "=" * 65)
    print(f"🔍 KIỂM THỬ MÃ: {symbol}")
    print("=" * 65)

    df = load_stock_data(symbol, start_date="2020-01-01", use_cache=True)
    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(df, n_forward=n_forward)

    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X, y_reg, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X, y_clf, train_ratio=0.8)

    test_closes = df.loc[X_test_r.index, "close"]

    # Huấn luyện
    reg_res = train_regression_models(
        X_train_r, y_train_r, X_test_r, y_test_r,
        current_closes_test=test_closes,
        auto_tune=False
    )
    clf_res = train_classification_model(
        X_train_c, y_train_c, X_test_c, y_test_c,
        auto_tune=False
    )

    current_close = float(df["close"].iloc[-1])
    latest_feat = latest_row[feature_cols]

    # Gọi predict_latest duy nhất 1 lần
    pred_info = predict_latest(
        reg_pipeline=reg_res["xgb_pipeline"],
        clf_pipeline=clf_res["pipeline"],
        latest_features=latest_feat,
        current_price=current_close,
        horizon=n_forward,
        prob_threshold=prob_threshold,
        min_return_pct=1.5
    )

    pred_target_price = pred_info["predicted_price"]
    exp_pct_change = pred_info["expected_return_pct"]
    prob_up = pred_info["prob_up"]
    consensus = pred_info["consensus"]

    print(f"  * Giá đóng cửa hiện tại: {current_close:,.0f} đ")
    print(f"  * Kỳ vọng thay đổi sau {n_forward} phiên: {exp_pct_change:+.2f}%")
    print(f"  * Giá mục tiêu sau {n_forward} phiên: {pred_target_price:,.0f} đ")
    print(f"  * Xác suất tăng giá > 1.5%: {prob_up * 100:.1f}%")
    print(f"  * Khuyến nghị AI: {consensus['signal_badge']}")
    print(f"  * Phân loại tín hiệu: {consensus['signal_type']}")

    # Kiểm tra tính nhất quán (Assertions)
    # 1. Giá mục tiêu phải khớp chính xác với công thức: current_close * (1 + pred_return)
    expected_calc_price = current_close * (1.0 + exp_pct_change / 100.0)
    assert abs(pred_target_price - expected_calc_price) <= 10.0, "Lỗi tính giá mục tiêu!"

    # 2. Tuyệt đối KHÔNG báo MUA khi kỳ vọng thay đổi bị âm (< 0%)
    if exp_pct_change < 0.0:
        assert consensus["is_buy"] is False, "LỖI LOGIC: Không được báo MUA khi kỳ vọng thay đổi âm!"
        if prob_up >= prob_threshold:
            assert consensus["signal_type"] == "DIVERGENCE", "LỖI LOGIC: Phải là TÍN HIỆU PHÂN KỲ!"
            print("  -> Đã bắt đúng TÍN HIỆU PHÂN KỲ (Xác suất cao nhưng kỳ vọng giá âm)")

    # 3. Với MBB: Kiểm tra xem giá mục tiêu có nằm trong vùng biến động hợp lý không (không bị tụt 20%)
    pct_diff = abs(pred_target_price - current_close) / current_close * 100.0
    print(f"  -> Độ lệch giá mục tiêu so với giá hiện tại: {pct_diff:.2f}% (Hợp lý cho 5 phiên)")
    assert pct_diff < 30.0, f"Giá dự báo bị lệch bất thường: {pct_diff:.2f}%!"

    print(f"✅ Mã {symbol} HOÀN TOÀN NHẤT QUÁN VÀ ĐẠT CHUẨN ĐỊNH LƯỢNG!")


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ KIỂM CHỨNG NHẤT QUÁN LOGIC VỚI MBB VÀ FPT")
    print("=" * 65)

    test_symbol("MBB")
    test_symbol("FPT")

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ CÁC KIỂM THỬ NHẤT QUÁN ĐỀU THÀNH CÔNG RỰC RỠ!")
    print("=" * 65)


if __name__ == "__main__":
    main()

