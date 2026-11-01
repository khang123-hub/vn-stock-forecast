"""
Module Features: Tính toán các đặc trưng kỹ thuật và nhãn mục tiêu (Target).
Tuân thủ tuyệt đối quy tắc KHÔNG DATA LEAKAGE:
- Mọi đặc trưng tại ngày t chỉ dùng dữ liệu <= t.
- Tách dòng dữ liệu mới nhất (t) để dự báo phiên t+N.
- dropna() các dòng quá khứ để làm sạch tập train/test.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán các chỉ báo kỹ thuật tại thời điểm t dựa trên dữ liệu quá khứ (<= t).
    Sử dụng thư viện `ta` thuần Python, có fallback tính toán bằng pandas/numpy.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1. Nhóm Xu hướng (Trend)
    try:
        from ta.trend import SMAIndicator, EMAIndicator, MACD
        df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
        df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()

        df["ema_12"] = EMAIndicator(close=close, window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(close=close, window=26).ema_indicator()

        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        # Fallback pure pandas
        df["sma_10"] = close.rolling(10).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

    # 2. Nhóm Động lượng (Momentum)
    try:
        from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
        df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["roc_10"] = ROCIndicator(close=close, window=10).roc()
    except Exception:
        # Fallback RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Fallback Stoch
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        df["stoch_k"] = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        df["roc_10"] = close.pct_change(10) * 100

    # 3. Nhóm Biến động & Khối lượng (Volatility & Volume)
    try:
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.volume import OnBalanceVolumeIndicator

        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()

        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr_14"] = atr.average_true_range()

        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        df["obv"] = obv.on_balance_volume()
    except Exception:
        # Fallback BB
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["bb_high"] = mid + 2 * std
        df["bb_low"] = mid - 2 * std
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (mid + 1e-9)

        # Fallback ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Fallback OBV
        direction = np.sign(close.diff()).fillna(0)
        df["obv"] = (direction * volume).cumsum()

    # Tỷ lệ đột biến khối lượng: Volume / SMA20(Volume)
    sma_vol_20 = volume.rolling(20).mean()
    df["vol_sma20"] = sma_vol_20
    df["vol_spike_ratio"] = volume / (sma_vol_20 + 1e-9)

    # 4. Đặc trưng trễ (Lag features): Lợi suất các phiên trước
    df["return_1d"] = close.pct_change(1)
    df["return_2d"] = close.pct_change(2)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_lag1"] = df["return_1d"].shift(1)
    df["return_lag2"] = df["return_1d"].shift(2)

    return df


def prepare_features_and_targets(
    df: pd.DataFrame,
    n_forward: int = 5,
    profit_threshold: float = 0.015
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Xây dựng toàn bộ ma trận đặc trưng và các biến mục tiêu:
    - Target 1 (Regression): Giá đóng cửa N phiên sau (close.shift(-n_forward))
    - Target 2 (Classification): Nhãn 1 nếu lợi suất sau N phiên > profit_threshold (1.5%), ngược lại 0.

    Trả về:
    - X: Ma trận đặc trưng lịch sử
    - y_reg: Biến mục tiêu hồi quy
    - y_clf: Biến mục tiêu phân loại
    - latest_row: Dòng dữ liệu mới nhất (chưa có nhãn tương lai) dùng để dự báo thực tế
    - feature_cols: Danh sách tên cột đặc trưng
    """
    df_feat = compute_technical_indicators(df)

    # Xác định các cột đặc trưng mô hình sẽ dùng
    feature_cols = [
        "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_diff",
        "rsi_14", "stoch_k", "stoch_d", "roc_10",
        "bb_high", "bb_low", "bb_width", "atr_14",
        "vol_spike_ratio", "return_1d", "return_2d", "return_3d", "return_5d",
        "return_lag1", "return_lag2"
    ]

    # Tạo biến mục tiêu Hồi quy BẮT BUỘC là Forward Return (dạng số thập phân)
    horizon = n_forward
    raw_return = (df_feat["close"].shift(-horizon) - df_feat["close"]) / df_feat["close"]
    # Giới hạn (clip) trong khoảng [-0.35, 0.35] để loại bỏ các điểm dị biệt do chia tách cổ phiếu
    df_feat["target_reg"] = raw_return.clip(-0.35, 0.35)
    df_feat["target_clf"] = (df_feat["target_reg"] > profit_threshold).astype(int)

    # Dòng mới nhất để dự báo tương lai (phiên t hiện tại, dự báo t+N)
    # Cần đảm bảo các cột feature ở dòng này không bị NaN
    valid_features = df_feat.dropna(subset=feature_cols)
    if valid_features.empty:
        raise ValueError("Không đủ dữ liệu lịch sử để tính toán các chỉ báo kỹ thuật.")

    latest_row = valid_features.iloc[[-1]].copy()

    # Dữ liệu phục vụ huấn luyện: bỏ các dòng cuối cùng (thiếu target do shift(-N))
    # và các dòng đầu tiên (thiếu feature do window lag)
    trainable_df = df_feat.dropna(subset=feature_cols + ["target_reg", "target_clf"]).copy()

    X = trainable_df[feature_cols]
    y_reg = trainable_df["target_reg"]
    y_clf = trainable_df["target_clf"]

    return X, y_reg, y_clf, latest_row, feature_cols

