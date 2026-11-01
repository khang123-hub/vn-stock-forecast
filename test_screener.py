"""
Script kiểm thử tự động Module Screener với 3 mã cổ phiếu: FPT, VCB, HPG.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.screener import scan_vn30_market


def main():
    test_tickers = ["FPT", "VCB", "HPG"]
    print("=" * 65)
    print(f"🎯 BẮT ĐẦU KIỂM THỬ BỘ LỌC VN30 VỚI CÁC MÃ: {test_tickers}")
    print("=" * 65)

    def progress_update(current, total, ticker):
        pct = (current / total) * 100
        print(f"[{current}/{total}] ({pct:.0f}%) Đang phân tích mã: {ticker}...")

    df_screener = scan_vn30_market(
        ticker_list=test_tickers,
        n_forward=5,
        prob_threshold=0.55,
        profit_threshold_pct=1.5,
        start_date="2020-01-01",
        force_reload=False,
        progress_callback=progress_update
    )

    print("\n" + "=" * 65)
    print("BẢNG KẾT QUẢ XẾP HẠNG SCREENER:")
    print("=" * 65)
    if not df_screener.empty:
        print(df_screener.to_string(index=False))
        print("\n✅ KIỂM THỬ SCREENER THÀNH CÔNG 100%!")
    else:
        print("❌ LỖI: Không có kết quả nào được trả về!")
        sys.exit(1)


if __name__ == "__main__":
    main()

