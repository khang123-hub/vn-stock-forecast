"""
Kiểm thử bắt buộc: Xác thực loại bỏ hoàn toàn lỗi nhân trùng công thức (Double Multiplication Bug)
cho cả 2 mã FPT và MBB.
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
    predict_latest
)


def verify_symbol(symbol: str, horizon: int = 5):
    print("=" * 65)
    print(f"🔍 KIỂM CHỨNG TÍNH TOÁN CHO MÃ: {symbol}")
    print("=" * 65)

    df = load_stock_data(symbol, start_date="2020-01-01", use_cache=True)
    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(df, n_forward=horizon)

    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X, y_reg, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X, y_clf, train_ratio=0.8)

    test_closes = df.loc[X_test_r.index, "close"]

    reg_res = train_regression_models(
        X_train_r, y_train_r, X_test_r, y_test_r,
        current_closes_test=test_closes
    )
    clf_res = train_classification_model(X_train_c, y_train_c, X_test_c, y_test_c)

    current_price = float(df["close"].iloc[-1])
    latest_feat = latest_row[feature_cols]

    # Gọi predict_latest DUY NHẤT 1 LẦN
    pred_info = predict_latest(
        reg_pipeline=reg_res["xgb_pipeline"],
        clf_pipeline=clf_res["pipeline"],
        latest_features=latest_feat,
        current_price=current_price,
        horizon=horizon
    )

    predicted_price = pred_info["predicted_price"]
    expected_return_pct = pred_info["expected_return_pct"]
    pred_return = pred_info["pred_return"]
    prob_up_pct = pred_info["prob_up_pct"]
    consensus = pred_info["consensus"]

    print(f"  * Giá hiện tại (current_price):       {current_price:,.0f} đ")
    print(f"  * Giá dự báo (predicted_price):       {predicted_price:,.0f} đ")
    print(f"  * % Kỳ vọng (expected_return_pct):    {expected_return_pct:+.2f}%")
    print(f"  * Xác suất tăng giá > 1.5%:           {prob_up_pct:.1f}%")
    print(f"  * Khuyến nghị AI:                     {consensus['signal_badge']}")

    # Kiểm tra 1: % kỳ vọng thay đổi sau 5 phiên nằm trong khoảng [-15%, +15%]
    print(f"  -> Kiểm tra biên độ % kỳ vọng: -15% <= {expected_return_pct:+.2f}% <= +15% ?")
    assert -15.0 <= expected_return_pct <= 15.0, (
        f"LỖI: % kỳ vọng {expected_return_pct}% nằm ngoài biên độ hợp lý [-15%, +15%]!"
    )
    print("     ✅ ĐẠT: Biên độ kỳ vọng hoàn toàn hợp lý với thị trường chứng khoán!")

    # Kiểm tra 2: predicted_price khớp chính xác với current_price * (1 + expected_return_pct/100)
    expected_exact = round(current_price * (1.0 + expected_return_pct / 100.0), -1)
    diff = abs(predicted_price - expected_exact)
    print(f"  -> Kiểm tra công thức: predicted_price ({predicted_price:,.0f}) == {expected_exact:,.0f} đ ?")
    assert diff <= 10.0, (
        f"LỖI: predicted_price ({predicted_price}) không khớp với công thức ({expected_exact})!"
    )
    print("     ✅ ĐẠT: Giá dự báo khớp chính xác 100% công thức định lượng!")

    # Kiểm tra 3: Tuyệt đối không có chuyện giá bị nhân lên tiền tỷ hay hàng chục nghìn %
    assert predicted_price < current_price * 2.0, "LỖI NHÂN TRÙNG VẪN CÒN TỒN TẠI!"
    print("     ✅ ĐẠT: Không có lỗi nhân trùng công thức (No Double Multiplication)!\n")


def main():
    print("\n" + "#" * 65)
    print(" BẮT ĐẦU KIỂM TRA ĐỘ CHUẨN XÁC ĐƠN VỊ & CÔNG THỨC TOÀN HỆ THỐNG")
    print("#" * 65 + "\n")

    verify_symbol("FPT", horizon=5)
    verify_symbol("MBB", horizon=5)

    print("#" * 65)
    print("🎉 TẤT CẢ CÁC KIỂM TRA ĐỀU THÀNH CÔNG VÀ CHÍNH XÁC 100%!")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()

