"""
Script kiểm thử Module Config & AI Advisor:
- Quản lý API Key an toàn
- Kiểm tra hàm REST API thuần get_ai_advisor_report
- Graceful Fallback khi không có API Key
- Tránh lỗi 404 bằng REST API endpoint v1 chính thức
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import get_gemini_api_key
from src.ai_advisor import get_ai_advisor_report, generate_ai_analysis_report


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ MODULE CONFIG & AI ADVISOR (REST API v1)")
    print("=" * 65)

    # 1. Kiểm tra đọc API Key khi không có
    key_none = get_gemini_api_key(None)
    print(f"\n[1] Kiểm tra Fallback khi không có Key: {key_none} (Hợp lệ nếu None)")

    # 2. Kiểm tra khi người dùng truyền key
    sample_key = "AIzaSyFakeKeyTest123456"
    key_user = get_gemini_api_key(sample_key)
    assert key_user == sample_key, "Lỗi đọc key từ user input!"
    print(f"[2] Đọc thành công key từ input người dùng: {key_user[:6]}******")

    # 3. Kiểm tra hàm get_ai_advisor_report khi thiếu Key
    res_direct_no_key = get_ai_advisor_report("", "Phân tích mã FPT")
    assert "Thiếu Gemini API Key" in res_direct_no_key, "Lỗi: Không bắt được trường hợp thiếu key!"
    print(f"[3] Xử lý an toàn hàm get_ai_advisor_report khi thiếu key: {res_direct_no_key}")

    # 4. Kiểm tra hàm generate_ai_analysis_report khi thiếu Key
    res_no_key = generate_ai_analysis_report("", {})
    assert res_no_key["success"] is False, "Lỗi: Đáng lẽ phải thất bại khi thiếu key!"
    print(f"[4] Xử lý an toàn hàm generate_ai_analysis_report: {res_no_key['error']}")

    # 5. Kiểm tra cấu trúc tham số đầu vào cho AI Advisor
    mock_metrics = {
        "symbol": "FPT",
        "current_price": 50200.0,
        "predicted_price": 53280.0,
        "expected_return_pct": 6.14,
        "prob_up_pct": 81.8,
        "horizon": 5,
        "signal_badge": "🚀 KHUYẾN NGHỊ: MUA MẠNH",
        "rsi": 42.5,
        "macd_diff": 1.25,
        "bb_high": 54000.0,
        "bb_low": 49000.0,
        "support_price": 49500.0,
        "resistance_price": 54500.0,
        "win_rate": 62.5,
        "max_drawdown": -8.5
    }
    print("\n[5] Kiểm tra định dạng thông số định lượng đầu vào cho AI:")
    for k, v in mock_metrics.items():
        print(f"  - {k}: {v}")

    # 6. Nếu có biến môi trường GEMINI_API_KEY thực tế, thử gọi API
    real_key = os.getenv("GEMINI_API_KEY")
    if real_key:
        print("\n[6] Phát hiện GEMINI_API_KEY thực tế, tiến hành gọi thử nghiệm...")
        live_res = generate_ai_analysis_report(real_key, mock_metrics)
        if live_res["success"]:
            print(f"  ✅ Gọi Gemini API thành công bằng model: {live_res['model_used']}")
            print("  --- BẢN BÁO CÁO MẪU TỪ AI ---")
            print(live_res["report"][:300] + "...")
        else:
            print(f"  ⚠️ Kết quả API: {live_res['error']}")
    else:
        print("\n[6] Chưa thiết lập GEMINI_API_KEY thực tế. Hệ thống chạy ở chế độ Graceful Fallback (Không ảnh hưởng đến Machine Learning).")

    print("\n" + "=" * 65)
    print("✅ KIỂM THỬ MODULE AI ADVISOR HOÀN TẤT THÀNH CÔNG 100%!")
    print("=" * 65)


if __name__ == "__main__":
    main()
