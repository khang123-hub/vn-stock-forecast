"""
Module Data Loader: Tải, làm sạch và lưu cache dữ liệu OHLCV chứng khoán Việt Nam.
Tích hợp cơ chế Smart Auto-Refresh Cache và xử lý triệt để lỗi exclusive end_date của yfinance.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột về: date, open, high, low, close, volume."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Xử lý trường hợp date nằm ở Index (yfinance)
    if not isinstance(df.index, pd.RangeIndex) and "date" not in [c.lower() for c in df.columns]:
        df = df.reset_index()

    col_map = {}
    for col in df.columns:
        c_lower = str(col).lower().strip()
        if c_lower in ["time", "date", "tradingdate", "datetime"]:
            col_map[col] = "date"
        elif c_lower in ["open", "gia_mo_cua", "giamocua", "openprice"]:
            col_map[col] = "open"
        elif c_lower in ["high", "gia_cao_nhat", "giacaonhat", "highprice"]:
            col_map[col] = "high"
        elif c_lower in ["low", "gia_thap_nhat", "giathapnhat", "lowprice"]:
            col_map[col] = "low"
        elif c_lower in ["close", "gia_dong_cua", "giadongcua", "closeprice", "adj close", "adjclose"]:
            if "close" not in col_map.values():
                col_map[col] = "close"
        elif c_lower in ["volume", "khoi_luong", "khoiluong", "totalvolume", "vol"]:
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    required = ["date", "open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        for c in missing:
            if c in ["open", "high", "low"] and "close" in df.columns:
                df[c] = df["close"]
            elif c == "volume":
                df["volume"] = 100000.0

    df = df[required].copy()

    # Chuẩn hóa kiểu dữ liệu
    df["date"] = pd.to_datetime(df["date"])
    # Loại bỏ timezone nếu có
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sắp xếp tăng dần theo thời gian và loại bỏ trùng lặp / rỗng
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def _fetch_from_vnstock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Thử lấy dữ liệu từ vnstock theo nhiều phiên bản API khác nhau."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol.upper(), source="VCI")
        df = stock.quote.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        from vnstock import stock_historical_data
        df = stock_historical_data(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            resolution="1D",
            type="stock"
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        import vnstock
        if hasattr(vnstock, "stock"):
            s = vnstock.stock(symbol=symbol.upper())
            df = s.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                return df
    except Exception:
        pass

    return None


def _fetch_from_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fallback 1: Tải dữ liệu qua Yahoo Finance với hậu tố .VN (ví dụ: FPT.VN, MBB.VN).
    Khắc phục tính chất exclusive của tham số end trong yfinance bằng cách cộng thêm ngày hoặc dùng period='5y'.
    """
    try:
        import yfinance as yf
        ticker_sym = f"{symbol.upper()}.VN"
        yf_ticker = yf.Ticker(ticker_sym)

        # yfinance end là exclusive, cần cộng thêm ít nhất 1-2 ngày để lấy trọn vẹn phiên hôm nay
        end_dt = pd.to_datetime(end_date)
        inclusive_end = (end_dt + timedelta(days=2)).strftime("%Y-%m-%d")

        # Thử lấy theo khoảng thời gian mở rộng
        df = yf_ticker.history(start=start_date, end=inclusive_end)
        if df is not None and not df.empty:
            return df

        # Nếu không có kết quả, thử lấy theo period="5y" (Yahoo Finance tự tính nến mới nhất)
        df = yf_ticker.history(period="5y")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[Data Loader] Lỗi yfinance cho {symbol}: {e}")
    return None


def _generate_synthetic_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback 2: Sinh dữ liệu mô phỏng chân thực khi mạng offline hoặc cả 2 API bị chặn."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        dates = pd.bdate_range(end=datetime.now(), periods=500)

    n = len(dates)
    np.random.seed(abs(hash(symbol)) % (2**32))

    symbol_upper = symbol.upper()
    base_price = 100000.0 if "FPT" in symbol_upper else (
        90000.0 if "VCB" in symbol_upper else (
            28000.0 if "HPG" in symbol_upper else 50000.0
        )
    )

    daily_returns = np.random.normal(0.0006, 0.018, n)
    price_series = base_price * np.exp(np.cumsum(daily_returns))

    high = price_series * (1 + np.abs(np.random.normal(0, 0.009, n)))
    low = price_series * (1 - np.abs(np.random.normal(0, 0.009, n)))
    open_p = low + np.random.uniform(0.1, 0.9, n) * (high - low)
    close = price_series
    volume = np.random.lognormal(mean=14.5, sigma=0.5, size=n)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_p, -1),
        "high": np.round(high, -1),
        "low": np.round(low, -1),
        "close": np.round(close, -1),
        "volume": np.round(volume, -2)
    })
    return df


def load_stock_data(
    symbol: str = "FPT",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Tải dữ liệu OHLCV của cổ phiếu:
    - Cơ chế Smart Auto-Refresh: Tự động phát hiện khi cache thiếu nến ngày hôm nay để tải mới.
    - Sửa lỗi exclusive end_date của Yahoo Finance để luôn cập nhật phiên mới nhất hôm nay.
    - Ghi đè cập nhật vào file cache CSV tại data/{symbol}_history.csv.
    """
    symbol = symbol.upper().strip()
    now = datetime.now()

    # Nếu không truyền end_date, mặc định lấy đến ngày mai để chắc chắn lấy nến hôm nay (inclusive)
    if end_date is None:
        end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Nếu truyền end_date là ngày hôm nay, cộng thêm 1 ngày cho API yfinance
        req_end_dt = pd.to_datetime(end_date)
        if req_end_dt.date() >= now.date():
            end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_history.csv")

    # Kiểm tra tính hợp lệ và độ mới của Cache (Smart Auto-Refresh Cache)
    needs_fetch = False
    cached_df = None

    if use_cache and not force_reload and os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file)
            cached_df = _standardize_columns(cached_df)
            if not cached_df.empty and len(cached_df) > 50:
                last_trade_date = pd.to_datetime(cached_df["date"].iloc[-1]).date()
                today_date = now.date()

                # Nếu là ngày làm việc (Thứ 2 - Thứ 6)
                is_weekday = (now.weekday() < 5)
                # Đã qua giờ đóng cửa khớp lệnh định kỳ ATC (sau 14h45)
                is_after_market_close = (now.hour > 14 or (now.hour == 14 and now.minute >= 45))

                # Nếu cache chưa có nến ngày hôm nay mà thị trường đã mở cửa / đã đóng cửa:
                if is_weekday and last_trade_date < today_date and (now.hour >= 9):
                    needs_fetch = True
                else:
                    needs_fetch = False
            else:
                needs_fetch = True
        except Exception:
            needs_fetch = True
    else:
        needs_fetch = True

    # Nếu cache còn mới và không cần tải lại
    if not needs_fetch and cached_df is not None:
        start_dt = pd.to_datetime(start_date)
        return cached_df[cached_df["date"] >= start_dt].reset_index(drop=True)

    # Thực hiện tải dữ liệu mới từ API
    # 1. Thử lấy từ vnstock
    raw_df = _fetch_from_vnstock(symbol, start_date, end_date)

    # 2. Thử lấy từ yfinance (hỗ trợ đuôi .VN và exclusive end_date fix)
    if raw_df is None or raw_df.empty:
        raw_df = _fetch_from_yfinance(symbol, start_date, end_date)

    # 3. Fallback dữ liệu mẫu nếu mất mạng hoàn toàn
    is_mock = False
    if raw_df is None or raw_df.empty:
        raw_df = _generate_synthetic_stock_data(symbol, start_date, end_date)
        is_mock = True

    df = _standardize_columns(raw_df)

    # Lọc từ ngày bắt đầu
    start_dt = pd.to_datetime(start_date)
    df = df[df["date"] >= start_dt].reset_index(drop=True)

    # Lưu và cập nhật cache nếu có dữ liệu hợp lệ
    if not df.empty:
        df.to_csv(cache_file, index=False)
        last_date_str = df["date"].iloc[-1].strftime("%d/%m/%Y")
        if is_mock:
            print(f"[Data Loader] Cảnh báo: Sử dụng dữ liệu mô phỏng cho {symbol}.")
        else:
            print(f"[Data Loader] Đã cập nhật thành công {len(df)} phiên cho {symbol} (Phiên gần nhất: {last_date_str}) và lưu cache.")

    return df


if __name__ == "__main__":
    test_df = load_stock_data("FPT", start_date="2020-01-01", force_reload=True)
    print("\n2 Phiên giao dịch gần nhất:")
    print(test_df.tail(2))
    print(f"\nTổng số phiên: {len(test_df)}")
