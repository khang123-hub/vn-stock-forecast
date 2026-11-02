"""
Module AI Advisor: Chuyên gia Phân tích & Nhận định Chuyên sâu bằng Google Gemini API.
Sử dụng REST API thuần requests để đảm bảo độ tin cậy và không xung đột thư viện.
"""

from typing import Dict, Any, Optional
import json
import requests


def generate_ai_analysis_report(
    api_key: str,
    stock_metrics: Dict[str, Any],
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Gửi các thông số định lượng vào Google Gemini API để tạo báo cáo nhận định chuyên sâu.
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
    prob_up_pct = stock_metrics.get("prob_up_pct", 0.0)
    horizon = stock_metrics.get("horizon", 5)
    signal = stock_metrics.get("signal_badge", "QUAN SÁT")

    rsi = stock_metrics.get("rsi", 50.0)
    macd_diff = stock_metrics.get("macd_diff", 0.0)
    bb_high = stock_metrics.get("bb_high", current_price * 1.05)
    bb_low = stock_metrics.get("bb_low", current_price * 0.95)
    support = stock_metrics.get("support_price", current_price * 0.96)
    resistance = stock_metrics.get("resistance_price", current_price * 1.05)
    win_rate = stock_metrics.get("win_rate", 50.0)
    mdd = stock_metrics.get("max_drawdown", -10.0)

    # Xây dựng prompt định lượng chuyên nghiệp
    prompt = f"""
Bạn là một Giám đốc Phân tích Đầu tư Định lượng (Chief Quantitative Strategist) chuyên nghiệp tại thị trường chứng khoán Việt Nam.
Dưới đây là toàn bộ thông số định lượng mới nhất của cổ phiếu **{symbol}** do hệ thống Machine Learning (XGBoost + Ridge Pipeline) và thuật toán phân tích kỹ thuật tính toán:

=== DỮ LIỆU ĐỊNH LƯỢNG CỔ PHIẾU {symbol} ===
- Giá đóng cửa phiên gần nhất: {current_price:,.0f} VNĐ
- Dự báo Machine Learning ({horizon} phiên tới):
  + Giá mục tiêu kỳ vọng: {predicted_price:,.0f} VNĐ
  + Tỷ suất sinh lời kỳ vọng: {exp_return_pct:+.2f}%
  + Xác suất tăng giá > 1.5%: {prob_up_pct:.1f}%
  + Tín hiệu thuật toán (Strict Consensus): {signal}
- Chỉ báo Kỹ thuật & Động lượng:
  + RSI(14): {rsi:.1f}
  + MACD Histogram (Diff): {macd_diff:+.2f}
  + Dải Bollinger Bands: Low = {bb_low:,.0f} VNĐ | High = {bb_high:,.0f} VNĐ
  + Vùng hỗ trợ 20 phiên: {support:,.0f} VNĐ
  + Vùng kháng cự 20 phiên: {resistance:,.0f} VNĐ
- Hiệu năng Backtest Chu kỳ T+2.5 trên dữ liệu lịch sử:
  + Tỷ lệ thắng (Win Rate): {win_rate:.1f}%
  + Mức sụt giảm vốn tối đa (Max Drawdown): {mdd:.2f}%

=== YÊU CẦU BẢN BÁO CÁO NHẬN ĐỊNH ===
Hãy viết một bản nhận định đầu tư ngắn gọn, trực diện, giàu tính hành động (khoảng 250 - 350 từ) bằng tiếng Việt theo đúng cấu trúc 3 phần sau:

### 1. 📊 Đánh giá Xu hướng & Động lượng Kỹ thuật
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

    # Danh sách các model fallback theo thứ tự ưu tiên
    candidate_models = [model_name, "gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = ""

    for target_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key.strip()}"
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
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    first_cand = candidates[0]
                    content_parts = first_cand.get("content", {}).get("parts", [])
                    if content_parts:
                        report_text = content_parts[0].get("text", "")
                        return {
                            "success": True,
                            "report": report_text,
                            "model_used": target_model
                        }
            else:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                last_error = f"API Error ({response.status_code}): {err_msg}"
        except Exception as e:
            last_error = f"Request Exception: {str(e)}"

    return {
        "success": False,
        "error": last_error or "Không thể kết nối đến Gemini API. Vui lòng kiểm tra lại API Key và kết nối mạng."
    }
