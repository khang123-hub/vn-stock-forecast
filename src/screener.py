"""
Module Screener: Quét và phân tích xếp hạng toàn bộ danh mục cổ phiếu VN30.
Tuân thủ chuẩn Quantitative Finance và nguyên tắc No Data Leakage.
"""

from typing import List, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import load_stock_data
from src.features import compute_technical_indicators, prepare_features_and_targets
from src.models import split_time_series_data

# Danh sách 30 mã rổ VN30 chuẩn
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

# Top 10 mã thanh khoản & vốn hóa cao nhất
VN30_TOP10 = ["FPT", "HPG", "SSI", "MWG", "TCB", "MBB", "VHM", "VIC", "STB", "VCB"]


def analyze_single_ticker(
    symbol: str,
    n_forward: int = 5,
    prob_threshold: float = 0.55,
    profit_threshold_pct: float = 1.5,
    start_date: str = "2020-01-01",
    force_reload: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Phân tích định lượng an toàn cho một mã cổ phiếu:
    - Lấy dữ liệu & kiểm tra độ dài tối thiểu
    - Tạo đặc trưng & Target (No Leakage)
    - Fit scaler & models trên Train
    - Dự báo phiên t+N và xác suất tăng giá
    """
    symbol = symbol.upper().strip()
    df = load_stock_data(
        symbol=symbol,
        start_date=start_date,
        use_cache=not force_reload,
        force_reload=force_reload
    )

    if df is None or len(df) < 60:
        return None

    # Tính toán đặc trưng kỹ thuật & biến mục tiêu
    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(
        df,
        n_forward=n_forward,
        profit_threshold=profit_threshold_pct / 100.0
    )

    if len(X) < 50:
        return None

    # Phân tách Chronological Train/Test (80/20)
    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X, y_reg, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X, y_clf, train_ratio=0.8)

    # 1. Pipeline Hồi quy (Ridge & XGBoost Regressor)
    # Dùng Ridge để tính R2 chuẩn xác và XGBoost dự báo giá
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    ridge_pipe.fit(X_train_r, y_train_r)
    y_test_pred_ridge = ridge_pipe.predict(X_test_r)
    r2_val = r2_score(y_test_r, y_test_pred_ridge)

    xgb_reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=70,
            max_depth=3,
            learning_rate=0.08,
            random_state=42
        ))
    ])
    xgb_reg_pipe.fit(X_train_r, y_train_r)

    # 2. Pipeline Phân loại (XGBoost Classifier)
    xgb_clf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    xgb_clf_pipe.fit(X_train_c, y_train_c)

    # Dự báo cho phiên hiện tại (t) -> t+N
    latest_features = latest_row[feature_cols]
    current_close = float(df["close"].iloc[-1])

    pred_target_price = float(xgb_reg_pipe.predict(latest_features)[0])
    exp_return_pct = ((pred_target_price / current_close) - 1.0) * 100.0
    prob_up_pct = float(xgb_clf_pipe.predict_proba(latest_features)[0, 1]) * 100.0

    # Lấy chỉ số RSI tại phiên t
    rsi_val = float(latest_row["rsi_14"].iloc[-1]) if "rsi_14" in latest_row else 50.0

    # Xác định khuyến nghị định lượng
    threshold_pct = prob_threshold * 100.0
    if prob_up_pct >= 65.0 and exp_return_pct > 2.0:
        recommendation = "🚀 MUA MẠNH"
    elif prob_up_pct >= threshold_pct:
        recommendation = "✅ MUA"
    else:
        recommendation = "⏳ QUAN SÁT"

    return {
        "Mã CP": symbol,
        "Giá hiện tại (VNĐ)": round(current_close, -1),
        "Giá mục tiêu sau N phiên": round(pred_target_price, -1),
        "% Kỳ vọng (+ / -)": round(exp_return_pct, 2),
        "Xác suất tăng > 1.5% (%)": round(prob_up_pct, 1),
        "RSI (14)": round(rsi_val, 1),
        "R² / Độ tin cậy": round(max(-1.0, r2_val), 3),
        "Khuyến nghị": recommendation,
        "is_buy": bool(recommendation in ["🚀 MUA MẠNH", "✅ MUA"])
    }


def scan_vn30_market(
    ticker_list: List[str],
    n_forward: int = 5,
    prob_threshold: float = 0.55,
    profit_threshold_pct: float = 1.5,
    start_date: str = "2020-01-01",
    force_reload: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Quét danh mục nhiều cổ phiếu VN30 và tổng hợp bảng xếp hạng.
    Bọc từng mã trong try-except để đảm bảo không làm sập tiến trình chung.
    """
    results: List[Dict[str, Any]] = []
    total = len(ticker_list)

    for i, symbol in enumerate(ticker_list):
        if progress_callback:
            progress_callback(i + 1, total, symbol)

        try:
            res = analyze_single_ticker(
                symbol=symbol,
                n_forward=n_forward,
                prob_threshold=prob_threshold,
                profit_threshold_pct=profit_threshold_pct,
                start_date=start_date,
                force_reload=force_reload
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            # Ghi nhận lỗi nhưng không dừng tiến trình
            print(f"[Screener] Bỏ qua mã {symbol} do lỗi: {e}")
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Sắp xếp mặc định theo Xác suất tăng > 1.5% (%) giảm dần
    df_results = df_results.sort_values(
        by="Xác suất tăng > 1.5% (%)",
        ascending=False
    ).reset_index(drop=True)

    # Thêm cột Hạng ở đầu
    df_results.insert(0, "Hạng", df_results.index + 1)

    return df_results

