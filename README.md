# 📈 VN Stock Forecasting & Quant Backtesting System

Hệ thống Dự báo Chứng khoán Việt Nam và Kiểm thử Chiến lược Định lượng (Quantitative Trading System) được xây dựng theo chuẩn mực tài chính định lượng, tuân thủ 100% nguyên tắc chống rò rỉ dữ liệu (**No Data Leakage**).

---

## 🏗️ Cấu trúc Dự án

```
vn-stock-forecast/
├── .agents/rules/stock-forecasting.md # Quy tắc định lượng cốt lõi
├── data/                             # Thư mục lưu cache lịch sử OHLCV (.csv)
├── models/                           # Thư mục lưu trữ mô hình
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Tải dữ liệu OHLCV (vnstock / yfinance fallback / cache)
│   ├── features.py                   # Tính toán 23 đặc trưng kỹ thuật & Target (thư viện ta)
│   ├── models.py                     # Pipeline Ridge, XGBoost Regressor, XGBoost Classifier
│   ├── backtest.py                   # Backtest chiến lược T+2.5, phí 0.15%/chiều, Equity Curve
│   └── screener.py                   # Quét, phân tích & xếp hạng rổ VN30 chuẩn
├── app.py                            # Giao diện Dashboard tương tác Streamlit (4 Tabs)
├── test_pipeline.py                  # Script kiểm thử tự động toàn bộ quy trình đơn mã
├── test_screener.py                  # Script kiểm thử tự động bộ lọc đa mã VN30
├── requirements.txt                  # Danh sách thư viện (Tương thích Python 3.14 Windows)
└── README.md                         # Hướng dẫn chi tiết
```

---

## ⚡ Các Quy tắc Chống Rò rỉ Dữ liệu (No Data Leakage)
1. **Chronological Split:** Phân tách dữ liệu theo thứ tự thời gian (`shuffle=False`), 80% thời gian đầu dùng cho Train và 20% thời gian sau dùng cho Test.
2. **Feature Engineering an toàn:** Mọi chỉ báo kỹ thuật (SMA, EMA, MACD, RSI, Bollinger, ATR, Stoch, ROC, Volume Ratio, Return Lags) tại ngày $t$ chỉ được tính toán từ các phiên $\le t$.
3. **Target Shift chuẩn xác:** Biến mục tiêu dự báo $N$ phiên sau được tạo bằng `.shift(-N)` và tách riêng phiên mới nhất $t$ để dự báo thực tế phiên kế tiếp mà không làm rò rỉ nhãn vào quá khứ.
4. **StandardScaler Pipeline:** Bộ chuẩn hóa dữ liệu được đóng gói trong `sklearn.pipeline.Pipeline`, **chỉ fit trên tập Train** và transform Test.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt môi trường
Hệ thống tương thích hoàn toàn với Python 3.14 trên Windows:
```bash
pip install -r requirements.txt
```

### 2. Chạy thử nghiệm Pipeline qua Terminal
- Kiểm tra toàn bộ luồng xử lý đơn mã (`FPT`):
  ```bash
  python test_pipeline.py
  ```
- Kiểm tra bộ lọc đa mã VN30 (`FPT`, `VCB`, `HPG`):
  ```bash
  python test_screener.py
  ```

### 3. Khởi chạy Dashboard trực quan Streamlit
```bash
streamlit run app.py
```

---

## 📊 Các Tính năng Chính trên Dashboard (4 Tabs)
- **Hộp Khuyến nghị Định lượng:** Giá hiện tại, Giá dự báo sau $N$ phiên, Kỳ vọng tăng trưởng (%), Xác suất tăng giá, và Badge tín hiệu **🚀 MUA** hoặc **⏳ QUAN SÁT**.
- **Tab 1: Biểu đồ Kỹ thuật Tương tác:** Biểu đồ Nến (Candlestick) tích hợp SMA 20, SMA 50, dải Bollinger Bands, Khối lượng kèm MA20 Volume, và Động lượng RSI 14 (vùng 30-70).
- **Tab 2: Dự báo & Đánh giá Mô hình (KPIs):**
  - So sánh Ridge Regression vs XGBoost Regressor: MAE, RMSE, MAPE (%), $R^2$.
  - Đánh giá XGBoost Classifier: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
  - Biểu đồ đường Giá Thực tế vs Giá Dự báo trên tập Test.
  - Xếp hạng Top 12 Đặc trưng Kỹ thuật có trọng số quan trọng nhất (Feature Importance).
- **Tab 3: Backtest Chiến lược Định lượng (T+2.5):**
  - Mô phỏng chu kỳ nắm giữ $N$ phiên, trừ phí $0.15\%$ mỗi chiều (tương đương $0.3\%$ vòng quay).
  - So sánh trực tiếp Đường cong vốn (Equity Curve) giữa **AI Strategy** và **Buy & Hold**.
  - Các chỉ số: Tổng lợi nhuận ròng (%), Win Rate (%), Max Drawdown (%), và Nhật ký giao dịch chi tiết (Trade Log).
- **Tab 4: 🎯 Bộ lọc VN30 (Market Screener) [MỚI]:**
  - Quét danh mục rổ VN30 (Top 10 thanh khoản cao hoặc toàn bộ 30 mã VN30).
  - Tự động chạy Pipeline AI cho từng mã, bọc khối xử lý an toàn tránh gián đoạn tiến trình.
  - Bảng Xếp hạng định lượng sắp xếp theo **Xác suất tăng giá (%)** giảm dần, kèm RSI(14), $R^2$, Giá mục tiêu và Khuyến nghị (`🚀 MUA MẠNH`, `✅ MUA`, `⏳ QUAN SÁT`).
  - 3 Thẻ KPIs: Số mã cho tín hiệu Mua, Mã có xác suất tăng cao nhất, Mã có % lợi nhuận kỳ vọng cao nhất.
  - Biểu đồ cột ngang Top 10 cổ phiếu có xác suất tăng cao nhất kèm vạch ngưỡng (Threshold line).
  - Xuất báo cáo trực tiếp dưới định dạng file CSV (`utf-8-sig`).
