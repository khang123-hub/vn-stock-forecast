"""
Kiểm thử tự động cập nhật dữ liệu ngày hôm nay (02/11/2026) cho cả 2 mã FPT và MBB.
"""

import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data


def test_daily_data(symbol: str):
    print("=" * 65)
    print(f"🔄 ĐANG TẢI VÀ KIỂM TRA DỮ LIỆU MỚI NHẤT CHO MÃ: {symbol}")
    print("=" * 65)

    df = load_stock_data(symbol, start_date="2020-01-01", force_reload=True)

    print(f"\n2 DÒNG DỮ LIỆU CUỐI CÙNG CỦA {symbol}:")
    print(df.tail(2).to_string(index=False))

    last_date = df["date"].iloc[-1].strftime("%d/%m/%Y")
    last_close = df["close"].iloc[-1]
    last_volume = df["volume"].iloc[-1]

    print(f"\n  * Phiên giao dịch gần nhất: {last_date}")
    print(f"  * Giá đóng cửa:             {last_close:,.0f} đ")
    print(f"  * Khối lượng khớp lệnh:     {last_volume:,.0f} cổ phiếu")

    today_str = datetime.now().strftime("%d/%m/%Y")
    print(f"  * Ngày hôm nay trên hệ thống: {today_str}")

    # Xác nhận phiên cuối cùng chính là ngày hôm nay 02/11/2026
    assert last_date == today_str, (
        f"LỖI: Phiên gần nhất ({last_date}) chưa cập nhật được ngày hôm nay ({today_str})!"
    )
    print(f"  ✅ ĐẠT: Đã cập nhật thành công nến phiên hôm nay ({last_date})!")


def main():
    print("\n" + "#" * 65)
    print(" KIỂM THỬ CẬP NHẬT DỮ LIỆU PHIÊN GIAO DỊCH HÔM NAY (02/11/2026)")
    print("#" * 65 + "\n")

    test_daily_data("FPT")
    test_daily_data("MBB")

    print("\n" + "#" * 65)
    print("🎉 TẤT CẢ CÁC MÃ ĐÃ CẬP NHẬT DỮ LIỆU HÔM NAY (02/11/2026) THÀNH CÔNG 100%!")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()

