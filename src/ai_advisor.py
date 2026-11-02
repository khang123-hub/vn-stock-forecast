"""
Module AI Advisor: Chuyên gia Phân tích & Nhận định Chuyên sâu bằng Google Gemini API.
Tích hợp cơ chế Smart Fallback và Dynamic Model Discovery để loại bỏ triệt để lỗi 404 Not Found.
"""

from typing import Dict, Any, List, Optional, Tuple
import requests

# Danh sách các model theo thứ tự ưu tiên (chuẩn mới nhất đến các model dự phòng phổ biến)
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-pro"
]


def _call_gemini_endpoint(
    api_key: str,
    model_name: str,
    prompt: str,
    api_version: str = "v1beta"
) -> Tuple[bool, int, str]:
    """
    Gọi trực tiếp endpoint generateContent của Google Gemini.
    Trả về (is_success, status_code, result_text_or_error)
    """
    clean_model = model_name.replace("models/", "").strip()
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{clean_model}:generateContent?key={api_key.strip()}"
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
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        status_code = response.status_code

        if status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return True, 200, parts[0].get("text", "")
            return False, 200, "Phản hồi API không có nội dung văn bản."

        # Trích xuất thông báo lỗi nếu có
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", response.text)
        except Exception:
            err_msg = response.text
        return False, status_code, err_msg
    except Exception as e:
        return False, 500, f"Exception: {str(e)}"


def _discover_active_models(api_key: str) -> List[str]:
    """
    Tự động truy vấn danh sách model đang khả dụng cho API Key hiện tại (ListModels).
    Loại bỏ hoàn toàn trường hợp không biết tên model nào được hỗ trợ.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key.strip()}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            valid_models = []
            for item in data.get("models", []):
                methods = item.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    m_name = item.get("name", "").replace("models/", "").strip()
                    if m_name:
                        valid_models.append(m_name)
            return valid_models
    except Exception:
        pass
    return []


def generate_ai_analysis_report(
    api_key: str,
    stock_metrics: Dict[str, Any],
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Gửi các thông số định lượng vào Google Gemini API để tạo báo cáo nhận định chuyên sâu.
    Bảo đảm không bao giờ crash app khi gặp lỗi 404 với cơ chế đa tầng fallback.
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

    # Danh sách các model cần thử (ưu tiên model_name truyền vào trước)
    models_to_try = [model_name] + [m for m in CANDIDATE_MODELS if m != model_name]
    last_error_msg = ""

    # Tầng 1: Thử lần lượt các candidate model trên v1beta
    for candidate in models_to_try:
        try:
            success, code, res = _call_gemini_endpoint(api_key, candidate, prompt, api_version="v1beta")
            if success:
                return {
                    "success": True,
                    "report": res,
                    "model_used": candidate
                }
            if code == 404:
                # Nếu 404, tự động chuyển sang model dự phòng kế tiếp
                last_error_msg = f"Model '{candidate}' không tìm thấy (404), đang thử model dự phòng..."
                continue
            else:
                last_error_msg = f"Lỗi API ({code}): {res}"
        except Exception as e:
            last_error_msg = f"Lỗi kết nối model {candidate}: {str(e)}"
            continue

    # Tầng 2: Nếu các model trên v1beta đều bị 404, thử trên endpoint v1 với gemini-1.5-flash và gemini-pro
    for candidate in ["gemini-1.5-flash", "gemini-pro"]:
        try:
            success, code, res = _call_gemini_endpoint(api_key, candidate, prompt, api_version="v1")
            if success:
                return {
                    "success": True,
                    "report": res,
                    "model_used": f"{candidate} (v1)"
                }
        except Exception:
            pass

    # Tầng 3: Dynamic Discovery - Tự động hỏi Google API xem tài khoản có quyền truy cập model nào
    try:
        active_models = _discover_active_models(api_key)
        for disc_model in active_models:
            if disc_model not in models_to_try:
                success, code, res = _call_gemini_endpoint(api_key, disc_model, prompt, api_version="v1beta")
                if success:
                    return {
                        "success": True,
                        "report": res,
                        "model_used": disc_model
                    }
    except Exception:
        pass

    return {
        "success": False,
        "error": f"Không tìm thấy model Gemini khả dụng cho API Key này ({last_error_msg}). Vui lòng kiểm tra lại quyền hạn của khóa API tại Google AI Studio."
    }
