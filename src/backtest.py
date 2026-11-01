"""
Module Backtest: Mô phỏng chiến lược giao dịch định lượng trên tập Test.
Tuân thủ cơ chế chu kỳ T+2.5, khấu trừ phí giao dịch 0.15%/chiều,
tính toán Equity Curve, Win Rate, Max Drawdown so với Buy & Hold.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """Tính mức sụt giảm vốn lớn nhất (Max Drawdown - MDD) theo %."""
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max * 100.0
    return float(drawdown.min())


def run_backtest(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.55,
    holding_period: int = 5,
    fee_per_trade: float = 0.0015,
    initial_capital: float = 100_000_000.0
) -> Dict[str, Any]:
    """
    Thực thi mô phỏng chiến lược giao dịch:
    - Mua khi y_prob >= threshold.
    - Giữ holding_period phiên (chu kỳ T+2.5 / N phiên).
    - Phí giao dịch: fee_per_trade mỗi chiều (0.15% mua, 0.15% bán).
    - So sánh với Buy & Hold.
    """
    df = test_df.copy().reset_index(drop=True)
    n = len(df)
    if n == 0 or len(y_prob) != n:
        raise ValueError("Dữ liệu test và xác suất dự đoán không khớp kích thước.")

    df["prob"] = y_prob
    closes = df["close"].values
    dates = df["date"].values

    # Khởi tạo chuỗi vốn
    ai_equity = np.full(n, initial_capital, dtype=float)
    trades: List[Dict[str, Any]] = []

    current_capital = initial_capital
    in_position = False
    entry_idx = -1
    entry_price = 0.0
    shares = 0.0

    i = 0
    while i < n:
        # Kiểm tra nếu đang có vị thế và đến thời điểm chốt lệnh (hết holding_period hoặc hết dữ liệu)
        if in_position:
            # Cập nhật vốn theo giá hiện tại
            unrealized_equity = shares * closes[i]
            ai_equity[i] = unrealized_equity

            # Nếu đủ số phiên nắm giữ hoặc phiên cuối cùng
            if (i - entry_idx >= holding_period) or (i == n - 1):
                exit_price = closes[i]
                exit_val_gross = shares * exit_price
                exit_val_net = exit_val_gross * (1.0 - fee_per_trade)
                current_capital = exit_val_net
                ai_equity[i] = current_capital

                # Tính tỷ suất sinh lời của lệnh
                trade_return = (exit_price / entry_price) - 1.0
                net_trade_return = (exit_val_net / (shares * entry_price / (1.0 - fee_per_trade))) - 1.0

                trades.append({
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "exit_date": dates[i],
                    "exit_price": exit_price,
                    "holding_days": i - entry_idx,
                    "gross_return_pct": round(trade_return * 100, 2),
                    "net_return_pct": round(net_trade_return * 100, 2),
                    "is_win": bool(net_trade_return > 0)
                })

                in_position = False
                entry_idx = -1
                shares = 0.0
            i += 1
            continue

        # Nếu chưa có vị thế: Xét tín hiệu Mua
        if not in_position:
            if df["prob"].iloc[i] >= threshold and i + 1 < n:
                # Vào lệnh mua tại phiên i
                entry_idx = i
                entry_price = closes[i]
                # Trừ phí mua ngay khi vào lệnh
                capital_after_buy_fee = current_capital * (1.0 - fee_per_trade)
                shares = capital_after_buy_fee / entry_price
                in_position = True
                ai_equity[i] = capital_after_buy_fee
            else:
                ai_equity[i] = current_capital

        i += 1

    # Buy & Hold Equity Curve (mua từ phiên 0, bán ở phiên cuối, có tính phí 2 chiều)
    bh_shares = (initial_capital * (1.0 - fee_per_trade)) / closes[0]
    bh_equity = bh_shares * closes
    bh_equity[-1] = bh_equity[-1] * (1.0 - fee_per_trade)

    df_equity = pd.DataFrame({
        "date": df["date"],
        "close": closes,
        "ai_strategy": ai_equity,
        "buy_and_hold": bh_equity
    })

    # Tính toán các chỉ số KPI
    # 1. Tổng lợi nhuận %
    ai_total_return = (ai_equity[-1] / initial_capital - 1.0) * 100.0
    bh_total_return = (bh_equity[-1] / initial_capital - 1.0) * 100.0

    # 2. Max Drawdown %
    ai_mdd = calculate_max_drawdown(df_equity["ai_strategy"])
    bh_mdd = calculate_max_drawdown(df_equity["buy_and_hold"])

    # 3. Win Rate %
    total_trades = len(trades)
    win_trades = sum(1 for t in trades if t["is_win"])
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    trades_df = pd.DataFrame(trades)

    metrics = {
        "AI_Total_Return_Pct": round(ai_total_return, 2),
        "BH_Total_Return_Pct": round(bh_total_return, 2),
        "AI_Max_Drawdown_Pct": round(ai_mdd, 2),
        "BH_Max_Drawdown_Pct": round(bh_mdd, 2),
        "Total_Trades": total_trades,
        "Win_Trades": win_trades,
        "Win_Rate_Pct": round(win_rate, 2),
        "Initial_Capital": initial_capital,
        "Final_AI_Capital": round(ai_equity[-1], 2),
        "Final_BH_Capital": round(bh_equity[-1], 2)
    }

    return {
        "df_equity": df_equity,
        "trades_df": trades_df,
        "metrics": metrics
    }

