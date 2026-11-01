# Stock Forecasting System - Project Rules & Constraints

## 1. Tech Stack & Environment
- **Language:** Python (sử dụng môi trường ảo `venv`).
- **Data Source:** Thư viện `vnstock` để lấy dữ liệu lịch sử OHLCV thị trường chứng khoán Việt Nam (HOSE, HNX, UPCOM).
- **Data Processing & TA:** `pandas`, `numpy`, `ta` hoặc `pandas-ta` (tính chỉ báo kỹ thuật).
- **Machine Learning:** `scikit-learn`, `xgboost`, `statsmodels`.
- **Dashboard & UI:** `streamlit`, `plotly`.

## 2. Quy tắc CHỐNG DATA LEAKAGE (Bắt buộc tuân thủ tuyệt đối)
- **Chronological Split:** TUYỆT ĐỐI KHÔNG sử dụng `train_test_split(..., shuffle=True)`. Phải chia tập Train / Test theo đúng thứ tự thời gian (80% đầu là Train, 20% cuối là Test).
- **Feature Engineering an toàn:** Mọi đặc trưng trễ (lag features), trung bình trượt (rolling mean/std) và chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, EMA) tại ngày $t$ chỉ được phép tính toán từ dữ liệu của ngày $t$ trở về trước ($\le t$).
- **Target Shift chuẩn xác:** Biến mục tiêu (Target $Y$) dự báo cho tương lai $N$ phiên phải được dịch chuyển bằng `.shift(-N)` và `dropna()` các dòng cuối chưa có nhãn thực tế trước khi đánh giá.
- **Data Scaling:** `StandardScaler` CHỈ ĐƯỢC `.fit_transform()` trên tập Train, và CHỈ ĐƯỢC `.transform()` trên tập Test (đóng gói vào `sklearn.pipeline.Pipeline`).

## 3. Cấu trúc thư mục chuẩn
- `src/data_loader.py`: Tải và lưu cache dữ liệu từ `vnstock`.
- `src/features.py`: Tạo chỉ báo kỹ thuật, lag features, và biến mục tiêu (Target).
- `src/models.py`: Huấn luyện OLS Trend/Ridge, XGBoost Regressor (đánh giá MAE, RMSE, MAPE, R²) và XGBoost Classifier (đánh giá Precision, Recall, F1, ROC-AUC).
- `src/backtest.py`: Mô phỏng giao dịch theo tín hiệu dự báo (trừ phí 0.15%/chiều, chu kỳ T+2.5), tính Equity Curve, Win Rate và Max Drawdown so với Buy & Hold.
- `app.py`: Giao diện Streamlit 3 tab trực quan.