"""
Module AI Advisor: Chuyên gia Phân tích & Nhận định Chuyên sâu bằng Google Gemini REST API.
Sử dụng trực tiếp endpoint chính thức v1 của Google để tránh hoàn toàn lỗi 404 do lệch version SDK:
url = https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}
"""

from typing import Dict, Any, List, Optional, Tuple
import requests

# Danh sách candidate models và endpoints ưu tiên
DEFAULT_MODEL = "gemini-1.5-flash"

# Thứ tự thử nghiệm: Ưu tiên chuẩn v1 chính thức, sau đó fallback dự phòng
FALLBACK_CONFIGS = [
    ("v1", "gemini-1.5-flash"),
    ("v1", "gemini-1.5-pro"),
    ("v1", "gemini-pro"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1beta", "gemini-2.0-flash"),
    ("v1beta", "gemini-2.5-flash"),
]


def _call_gemini_rest_api(
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    prompt: str = "",
    api_version: str = "v1"
) -> Tuple[bool, int, str]:
    """
    Gọi trực tiếp REST API (requests.post) tới endpoint chính thức của Google Generative Language.
    Trả về (success: bool, status_code: int, content_or_error: str).
    """
    clean_model = model_name.replace("models/", "").strip()
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{clean_model}:generateContent?key={clean_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        status_code = response.status_code

        if status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return True, 200, parts[0].get("text", "")
            return False, 200, "Phản hồi từ Google API không chứa nội dung phân tích."

        # Xử lý thông báo lỗi trả về từ Google
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", response.text)
        except Exception:
            err_msg = response.text
        return False, status_code, err_msg
    except Exception as e:
        return False, 500, f"Exception khi kết nối tới Google Gemini REST API: {str(e)}"


def generate_ai_analysis_report(
    api_key: str,
    stock_metrics: Dict[str, Any],
    model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Gửi các thông số định lượng vào Google Gemini API qua REST endpoint v1 để tạo báo cáo nhận định.
    Tự động fallback nếu model chính gặp lỗi 404.
    """
    if not api_key or not api_key.strip():
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

    prompt = f"""
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

    # Tạo danh sách các cấu hình cần thử: Ưu tiên model_name người dùng yêu cầu trên v1
    configs_to_try = [("v1", model_name)]
    for ver, m in FALLBACK_CONFIGS:
        if (ver, m) not in configs_to_try:
            configs_to_try.append((ver, m))

    last_error = ""

    for ver, mod in configs_to_try:
        success, code, res = _call_gemini_rest_api(
            api_key=api_key,
            model_name=mod,
            prompt=prompt,
            api_version=ver
        )

        if success:
            return {
                "success": True,
                "report": res,
                "model_used": f"{mod} ({ver})"
            }

        if code == 404:
            # 404: model không tồn tại trên endpoint này -> tiếp tục thử cấu hình kế tiếp
            last_error = f"Model {mod} trên {ver} không tìm thấy (404)"
            continue
        else:
            # Lỗi khác (ví dụ 400 API key invalid, 403 quota) -> trả về thông báo lỗi rõ ràng
            return {
                "success": False,
                "error": f"Lỗi Google Gemini API ({code}): {res}"
            }

    return {
        "success": False,
        "error": f"Không thể kết nối đến Gemini API qua REST endpoint. Chi tiết: {last_error}"
    }
