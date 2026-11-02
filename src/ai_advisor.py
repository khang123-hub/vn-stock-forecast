"""
Module AI Advisor: Chuyên gia Phân tích & Nhận định Chuyên sâu bằng Google Gemini REST API.
Sử dụng trực tiếp REST API thuần (requests.post) tới endpoint chính thức v1 của Google để tránh hoàn toàn lỗi 404:
url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
"""

import requests
import json
from typing import Dict, Any, Optional


def get_ai_advisor_report(api_key: str, prompt_text: str) -> str:
    """
    Gọi trực tiếp Google Gemini REST API bằng requests.post tới endpoint chính thức v1.
    Loại bỏ hoàn toàn thư viện SDK cũ, đóng gói JSON chuẩn: contents -> parts -> text.
    """
    if not api_key or not str(api_key).strip():
        return "Thiếu Gemini API Key. Vui lòng cung cấp khóa API hợp lệ."

    clean_key = str(api_key).strip()
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={clean_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "Phản hồi API thành công nhưng không có nội dung văn bản."

        # Fallback thử endpoint v1beta nếu v1 trả về 404
        if response.status_code == 404:
            fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={clean_key}"
            fb_resp = requests.post(fb_url, headers=headers, data=json.dumps(data), timeout=30)
            if fb_resp.status_code == 200:
                fb_result = fb_resp.json()
                candidates = fb_result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")

        # Trích xuất thông báo lỗi chi tiết
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", response.text)
        except Exception:
            err_msg = response.text
        return f"Lỗi gọi API ({response.status_code}): {err_msg}"

    except Exception as e:
        return f"Lỗi kết nối API: {str(e)}"


def generate_ai_analysis_report(
    api_key: str,
    stock_metrics: Dict[str, Any],
    model_name: Optional[str] = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Chuẩn bị dữ liệu định lượng và gọi get_ai_advisor_report bằng REST API.
    Đảm bảo tương thích hoàn toàn với giao diện app.py và không bao giờ crash ứng dụng.
    """
    if not api_key or not str(api_key).strip():
        return {
            "success": False,
            "error": "Thiếu Gemini API Key. Vui lòng cung cấp khóa API hợp lệ."
        }

    symbol = stock_metrics.get("symbol", "UNKNOWN")
    current_price = stock_metrics.get("current_price", 0.0)
    predicted_price = stock_metrics.get("predicted_price", 0.0)
    exp_return_pct = stock_metrics.get("expected_return_pct", 0.0)
    prob_up_pct = stock_metrics.get("prob_up_pct", 50.0)
    horizon = stock_metrics.get("horizon", 5)
    signal_badge = stock_metrics.get("signal_badge", "TÍN HIỆU: THEO DÕI")
    rsi = stock_metrics.get("rsi", 50.0)
    macd_diff = stock_metrics.get("macd_diff", 0.0)
    bb_high = stock_metrics.get("bb_high", current_price * 1.05)
    bb_low = stock_metrics.get("bb_low", current_price * 0.95)
    support_price = stock_metrics.get("support_price", current_price * 0.95)
    resistance_price = stock_metrics.get("resistance_price", current_price * 1.05)
    win_rate = stock_metrics.get("win_rate", 50.0)
    max_drawdown = stock_metrics.get("max_drawdown", -15.0)

    prompt_text = f"""
Bạn là Giám đốc Định lượng & Chiến lược Đầu tư (Chief Quantitative Strategist) am hiểu sâu sắc thị trường chứng khoán Việt Nam (VN-Index, T+2.5, biên độ HOSE 7%).

Hãy phân tích mã cổ phiếu {symbol} dựa trên các thông số dữ liệu định lượng thực tế sau:
- Giá hiện tại: {current_price:,.0f} VNĐ
- Dự báo Machine Learning sau {horizon} phiên: {predicted_price:,.0f} VNĐ ({exp_return_pct:+.2f}%)
- Xác suất tăng giá (XGBoost Classifier): {prob_up_pct:.1f}%
- Tín hiệu Đồng thuận Hệ thống: {signal_badge}
- RSI (14 phiên): {rsi:.2f}
- MACD Histogram: {macd_diff:+.2f}
- Dải Bollinger Bands: [{bb_low:,.0f} - {bb_high:,.0f}] VNĐ
- Hỗ trợ gần nhất (20 phiên): {support_price:,.0f} VNĐ | Kháng cự gần nhất: {resistance_price:,.0f} VNĐ
- Hiệu suất Backtest mô hình: Win Rate = {win_rate:.1f}%, Max Drawdown = {max_drawdown:.1f}%

Hãy viết một bản nhận định đầu tư ngắn gọn, trực diện, giàu chuyên môn và có tính hành động cao theo cấu trúc sau:

### 1. 📊 Tóm tắt Kỹ thuật & Dòng tiền
(Nhận định nhanh sức mạnh dòng tiền, trạng thái RSI quá mua/quá bán, phân kỳ MACD và vị thế giá so với dải Bollinger Bands).

### 2. 🧠 Nhận định Tín hiệu Machine Learning & Độ tin cậy
(Đánh giá sự đồng thuận giữa mô hình Hồi quy và Phân loại. Lưu ý cảnh báo nếu có hiện tượng phân kỳ giữa xác suất và lợi nhuận kỳ vọng).

### 3. 🎯 Kịch bản Hành động & Quản trị Rủi ro (T+2.5)
(Đưa ra các mức giá cụ thể bằng VNĐ):
- **Vùng giá mua tham khảo:** ...
- **Mục tiêu chốt lời (Take Profit):** ...
- **Ngưỡng cắt lỗ (Stop Loss):** ...
- **Tỷ lệ R/R (Risk/Reward):** ...
- **Lời khuyên phân bổ tỷ trọng:** ...
"""

    report_output = get_ai_advisor_report(api_key, prompt_text)

    # Kiểm tra xem kết quả có phải là thông báo lỗi hay không
    if report_output.startswith("Lỗi gọi API") or report_output.startswith("Lỗi kết nối API") or report_output.startswith("Thiếu Gemini API Key"):
        return {
            "success": False,
            "error": report_output
        }

    return {
        "success": True,
        "report": report_output,
        "model_used": "gemini-1.5-flash"
    }
