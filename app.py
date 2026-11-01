"""
Dashboard Dự báo Chứng khoán Việt Nam - Streamlit Application
Chuẩn Kỹ thuật Định lượng Tài chính (Quant Developer)
Tích hợp Bộ lọc Rổ Cổ phiếu VN30 (Market Screener)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data_loader import load_stock_data
from src.features import compute_technical_indicators, prepare_features_and_targets
from src.models import (
    split_time_series_data,
    train_regression_models,
    train_classification_model,
    get_feature_importance
)
from src.backtest import run_backtest
from src.screener import scan_vn30_market, VN30_TICKERS, VN30_TOP10

# Thiết lập giao diện trang
st.set_page_config(
    page_title="VN Stock AI Forecasting & Quant Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách tài chính hiện đại
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .badge-buy {
        background-color: #065f46;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #10b981;
    }
    .badge-hold {
        background-color: #78350f;
        color: #fbbf24;
        padding: 8px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu hình Định lượng")

symbol = st.sidebar.text_input("Mã Cổ Phiếu Cá Nhân (Phân tích sâu)", value="FPT").upper().strip()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Ngày bắt đầu", value=pd.to_datetime("2020-01-01"))
with col_date2:
    end_date = st.date_input("Ngày kết thúc", value=pd.to_datetime("today"))

n_forward = st.sidebar.slider(
    "Khung dự báo N phiên tới",
    min_value=1,
    max_value=10,
    value=5,
    help="Dự báo giá đóng cửa và khả năng tăng giá sau N phiên (Chu kỳ T+2.5 thông dụng là 3-5 phiên)"
)

prob_threshold = st.sidebar.slider(
    "Ngưỡng xác suất vào lệnh (Threshold)",
    min_value=0.50,
    max_value=0.85,
    value=0.55,
    step=0.01,
    help="Chỉ mở vị thế Mua khi mô hình dự báo xác suất tăng giá >= ngưỡng này"
)

profit_threshold_pct = st.sidebar.slider(
    "Mục tiêu lợi nhuận tối thiểu (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Ngưỡng lợi nhuận tối thiểu sau N phiên để gán nhãn 1 (Tích cực)"
)

force_reload = st.sidebar.button("🔄 Làm mới dữ liệu mã hiện tại")

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **Quy tắc Quant:** No Data Leakage | Chronological Split | Phí 0.15%/chiều | T+2.5 Cycle")


# ----------------- HÀM TẢI VÀ CHUẨN BỊ DỮ LIỆU CÓ CACHE -----------------
@st.cache_data(show_spinner=False)
def get_processed_data(sym: str, start: str, end: str, n_fwd: int, p_thresh: float, reload: bool):
    df_raw = load_stock_data(
        symbol=sym,
        start_date=start,
        end_date=end,
        use_cache=not reload,
        force_reload=reload
    )
    if df_raw.empty or len(df_raw) < 60:
        return None, None, None, None, None, None

    X, y_reg, y_clf, latest_row, feature_cols = prepare_features_and_targets(
        df_raw,
        n_forward=n_fwd,
        profit_threshold=p_thresh / 100.0
    )
    return df_raw, X, y_reg, y_clf, latest_row, feature_cols


with st.spinner(f"Đang tải và xử lý dữ liệu cho cổ phiếu {symbol}..."):
    df_raw, X, y_reg, y_clf, latest_row, feature_cols = get_processed_data(
        symbol,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        n_forward,
        profit_threshold_pct,
        force_reload
    )

if df_raw is None or len(df_raw) < 60:
    st.error(f"Không đủ dữ liệu giao dịch cho mã '{symbol}'. Vui lòng kiểm tra lại mã hoặc mở rộng khoảng thời gian.")
    st.stop()


# ----------------- HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH -----------------
@st.cache_resource(show_spinner=False)
def train_and_evaluate_cached(X_df, y_r, y_c, f_cols):
    X_train_r, X_test_r, y_train_r, y_test_r = split_time_series_data(X_df, y_r, train_ratio=0.8)
    X_train_c, X_test_c, y_train_c, y_test_c = split_time_series_data(X_df, y_c, train_ratio=0.8)

    reg_results = train_regression_models(X_train_r, y_train_r, X_test_r, y_test_r)
    clf_results = train_classification_model(X_train_c, y_train_c, X_test_c, y_test_c)

    feat_imp = get_feature_importance(clf_results["pipeline"], f_cols)

    return {
        "reg": reg_results,
        "clf": clf_results,
        "feat_imp": feat_imp,
        "X_test": X_test_r,
        "y_test_reg": y_test_r,
        "y_test_clf": y_test_c
    }

with st.spinner("Đang huấn luyện mô hình Machine Learning định lượng (Ridge, XGBoost)..."):
    model_output = train_and_evaluate_cached(X, y_reg, y_clf, feature_cols)

# Dự báo cho phiên hiện tại (t) hướng tới (t+N)
current_close = float(df_raw["close"].iloc[-1])
current_date = df_raw["date"].iloc[-1].strftime("%d/%m/%Y")

latest_features = latest_row[feature_cols]
pred_price_ridge = float(model_output["reg"]["ridge_pipeline"].predict(latest_features)[0])
pred_price_xgb = float(model_output["reg"]["xgb_pipeline"].predict(latest_features)[0])
pred_prob_up = float(model_output["clf"]["pipeline"].predict_proba(latest_features)[0, 1])

exp_pct_change = ((pred_price_xgb / current_close) - 1.0) * 100.0
is_buy_signal = pred_prob_up >= prob_threshold


# ----------------- HỘP KHUYẾN NGHỊ NỔI BẬT ĐẦU TRANG -----------------
st.title(f"📊 Hệ thống Dự báo & Định lượng Chứng khoán: {symbol}")
st.caption(f"Dữ liệu cập nhật đến phiên gần nhất: **{current_date}** | Mô hình: Ensemble Pipeline (XGBoost + Ridge)")

rec_col1, rec_col2, rec_col3, rec_col4, rec_col5 = st.columns(5)

with rec_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Giá Hiện Tại (Đóng cửa)</div>
        <div class="metric-value">{current_close:,.0f} đ</div>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Dự báo sau {n_forward} phiên</div>
        <div class="metric-value">{pred_price_xgb:,.0f} đ</div>
    </div>
    """, unsafe_allow_html=True)

with rec_col3:
    color_ret = "#34d399" if exp_pct_change >= 0 else "#f87171"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Kỳ vọng Thay đổi</div>
        <div class="metric-value" style="color: {color_ret};">{exp_pct_change:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with rec_col4:
    color_prob = "#34d399" if pred_prob_up >= prob_threshold else "#fbbf24"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Xác suất Tăng > {profit_threshold_pct}%</div>
        <div class="metric-value" style="color: {color_prob};">{pred_prob_up * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with rec_col5:
    badge_html = f'<div class="badge-buy">🚀 KHUYẾN NGHỊ: MUA</div>' if is_buy_signal else f'<div class="badge-hold">⏳ QUAN SÁT / CHỜ</div>'
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">Tín hiệu AI (Ngưỡng {prob_threshold*100:.0f}%)</div>
        <div style="margin-top: 6px;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------- TABS GIAO DIỆN CHÍNH (4 TABS) -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tab 1: Biểu đồ Kỹ thuật Tương tác",
    "🎯 Tab 2: Dự báo & Đánh giá KPIs Mô hình",
    "💰 Tab 3: Backtest Chiến lược Định lượng (T+2.5)",
    "🎯 Tab 4: Bộ lọc VN30 (Market Screener)"
])


# ================= TAB 1: BIỂU ĐỒ KỸ THUẬT =================
with tab1:
    st.subheader(f"Biểu đồ Kỹ thuật Tổng quan & Động lượng: {symbol}")

    df_tech = compute_technical_indicators(df_raw).tail(250)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Nến Giá, Bollinger Bands & Đường Trung bình Động", "Khối lượng Giao dịch (Volume)", "Chỉ số Động lượng RSI(14)")
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df_tech["date"],
        open=df_tech["open"],
        high=df_tech["high"],
        low=df_tech["low"],
        close=df_tech["close"],
        name="OHLC",
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444"
    ), row=1, col=1)

    # SMA & Bollinger
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["sma_20"], name="SMA 20", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["sma_50"], name="SMA 50", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["bb_high"], name="BB High", line=dict(color="#94a3b8", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["bb_low"], name="BB Low", line=dict(color="#94a3b8", dash="dot", width=1), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)"), row=1, col=1)

    # 2. Volume
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df_tech["close"], df_tech["open"])]
    fig.add_trace(go.Bar(
        x=df_tech["date"],
        y=df_tech["volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.8
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["vol_sma20"], name="Vol SMA20", line=dict(color="#facc15", width=1.2)), row=2, col=1)

    # 3. RSI
    fig.add_trace(go.Scatter(x=df_tech["date"], y=df_tech["rsi_14"], name="RSI(14)", line=dict(color="#a855f7", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=3, col=1)

    fig.update_layout(
        height=750,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# ================= TAB 2: DỰ BÁO & KPIs MÔ HÌNH =================
with tab2:
    st.subheader("Đánh giá Định lượng Hiệu năng Mô hình (Test Set - 20% Dữ liệu mới nhất)")

    col_kpi1, col_kpi2 = st.columns([1, 1])

    with col_kpi1:
        st.markdown("#### 1. So sánh Mô hình Hồi quy (Dự báo Mức giá)")
        st.dataframe(model_output["reg"]["comparison_df"], use_container_width=True)

        st.caption("""
        - **MAE:** Sai số tuyệt đối trung bình (VNĐ).
        - **RMSE:** Căn bậc hai sai số bình phương trung bình (phạt nặng các sai số lớn).
        - **MAPE (%):** Tỷ lệ phần trăm sai số tuyệt đối trung bình.
        - **R²:** Hệ số xác định (Càng gần 1 mô hình giải thích phương sai giá càng tốt).
        """)

    with col_kpi2:
        st.markdown("#### 2. Chỉ số Mô hình Phân loại (XGBoost Classifier - Nhãn Tăng giá)")
        clf_metrics = model_output["clf"]["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{clf_metrics['Accuracy']*100:.1f}%")
        c2.metric("Precision", f"{clf_metrics['Precision']*100:.1f}%")
        c3.metric("Recall", f"{clf_metrics['Recall']*100:.1f}%")
        c4.metric("F1-Score", f"{clf_metrics['F1-Score']:.3f}")
        c5.metric("ROC-AUC", f"{clf_metrics['ROC-AUC']:.3f}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1.2, 0.8])

    with col_chart1:
        st.markdown(f"#### Biểu đồ Giá Thực tế vs Giá Dự báo trên tập Test ({n_forward} phiên)")
        test_indices = model_output["X_test"].index
        test_dates = df_raw.loc[test_indices, "date"]
        actual_prices = model_output["y_test_reg"].values
        pred_ridge = model_output["reg"]["ridge_pred"]
        pred_xgb = model_output["reg"]["xgb_pred"]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=test_dates, y=actual_prices, name="Giá Thực tế (Actual)", line=dict(color="#f8fafc", width=2.2)))
        fig_compare.add_trace(go.Scatter(x=test_dates, y=pred_xgb, name="XGBoost Dự báo", line=dict(color="#10b981", width=1.8, dash="solid")))
        fig_compare.add_trace(go.Scatter(x=test_dates, y=pred_ridge, name="Ridge Regression Dự báo", line=dict(color="#f59e0b", width=1.5, dash="dash")))

        fig_compare.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            template="plotly_dark",
            hovermode="x unified",
            yaxis_title="Giá (VNĐ)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_chart2:
        st.markdown("#### Top Đặc trưng Kỹ thuật Quan trọng Nhất")
        top_imp = model_output["feat_imp"].head(12)
        fig_imp = go.Figure(go.Bar(
            x=top_imp["Độ quan trọng"],
            y=top_imp["Đặc trưng"],
            orientation="h",
            marker=dict(color="#3b82f6")
        ))
        fig_imp.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            template="plotly_dark",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Importance Score"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ================= TAB 3: BACKTEST CHIẾN LƯỢC =================
with tab3:
    st.subheader(f"Mô phỏng Kiểm thử Chiến lược AI (Chu kỳ T+2.5 / {n_forward} phiên)")
    st.caption("Chiến lược: Mua khi xác suất tăng giá >= Threshold, nắm giữ N phiên, trừ 0.15% phí giao dịch mỗi chiều.")

    test_indices = model_output["X_test"].index
    test_sub_df = df_raw.loc[test_indices].copy()
    y_test_probs = model_output["clf"]["y_prob"]

    backtest_res = run_backtest(
        test_df=test_sub_df,
        y_prob=y_test_probs,
        threshold=prob_threshold,
        holding_period=n_forward,
        fee_per_trade=0.0015
    )

    bt_metrics = backtest_res["metrics"]
    df_equity = backtest_res["df_equity"]
    trades_df = backtest_res["trades_df"]

    # Thẻ KPI Chiến lược
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        ret_diff = bt_metrics['AI_Total_Return_Pct'] - bt_metrics['BH_Total_Return_Pct']
        st.metric("Tổng Lợi nhuận AI", f"{bt_metrics['AI_Total_Return_Pct']:+.2f}%", delta=f"{ret_diff:+.2f}% vs B&H")
    with kpi2:
        st.metric("Lợi nhuận Buy & Hold", f"{bt_metrics['BH_Total_Return_Pct']:+.2f}%")
    with kpi3:
        st.metric("Tỷ lệ thắng (Win Rate)", f"{bt_metrics['Win_Rate_Pct']:.1f}%", f"{bt_metrics['Win_Trades']}/{bt_metrics['Total_Trades']} lệnh")
    with kpi4:
        st.metric("Max Drawdown (AI)", f"{bt_metrics['AI_Max_Drawdown_Pct']:.2f}%")
    with kpi5:
        st.metric("Max Drawdown (B&H)", f"{bt_metrics['BH_Max_Drawdown_Pct']:.2f}%")

    st.markdown("---")

    # Đường cong vốn Equity Curve
    st.markdown("#### Đường cong Vốn Tích lũy (Equity Curve: AI Strategy vs Buy & Hold)")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=df_equity["date"],
        y=df_equity["ai_strategy"] / bt_metrics["Initial_Capital"] * 100,
        name="Chiến lược AI Quant",
        line=dict(color="#10b981", width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=df_equity["date"],
        y=df_equity["buy_and_hold"] / bt_metrics["Initial_Capital"] * 100,
        name="Chiến lược Mua & Nắm giữ (Buy & Hold)",
        line=dict(color="#94a3b8", width=1.8, dash="dot")
    ))

    fig_eq.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Giá trị Chuẩn hóa (Gốc = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Bảng nhật ký giao dịch
    st.markdown("#### Nhật ký Giao dịch Chi tiết (Trade Log)")
    if not trades_df.empty:
        trades_display = trades_df.copy()
        trades_display["entry_date"] = pd.to_datetime(trades_display["entry_date"]).dt.strftime("%d/%m/%Y")
        trades_display["exit_date"] = pd.to_datetime(trades_display["exit_date"]).dt.strftime("%d/%m/%Y")
        trades_display = trades_display.rename(columns={
            "entry_date": "Ngày Mua",
            "entry_price": "Giá Mua (VNĐ)",
            "exit_date": "Ngày Bán",
            "exit_price": "Giá Bán (VNĐ)",
            "holding_days": "Số phiên",
            "gross_return_pct": "LN Gộp (%)",
            "net_return_pct": "LN Ròng sau phí (%)",
            "is_win": "Thắng"
        })
        st.dataframe(trades_display.tail(20), use_container_width=True)
    else:
        st.info(f"Không có giao dịch nào được kích hoạt với ngưỡng xác suất {prob_threshold * 100:.0f}%. Hãy thử hạ ngưỡng xác suất ở thanh trượt bên trái.")


# ================= TAB 4: BỘ LỌC VN30 (MARKET SCREENER) =================
with tab4:
    st.subheader("🎯 Bộ lọc Định lượng Rổ VN30 (VN30 Market Screener)")
    st.caption(f"Quét toàn bộ cổ phiếu rổ VN30, huấn luyện mô hình dự báo {n_forward} phiên và xếp hạng theo xác suất tăng giá.")

    sc_col1, sc_col2, sc_col3 = st.columns([2, 1, 1])
    with sc_col1:
        screener_mode = st.radio(
            "Phạm vi quét danh mục:",
            options=["Top 10 mã thanh khoản cao (Nhanh)", "Toàn bộ 30 mã VN30"],
            horizontal=True
        )
    with sc_col2:
        force_refresh_screener = st.checkbox(
            "Ép cập nhật dữ liệu mới nhất",
            value=False,
            help="Bỏ qua cache cũ và tải trực tiếp dữ liệu mới nhất từ máy chủ."
        )
    with sc_col3:
        st.write("") # Căn lề nút bấm
        btn_start_scan = st.button("🚀 Bắt đầu Quét & Xếp hạng", type="primary", use_container_width=True)

    # Khởi tạo session state lưu kết quả quét
    if "screener_results" not in st.session_state:
        st.session_state["screener_results"] = None

    if btn_start_scan:
        chosen_tickers = VN30_TOP10 if "Top 10" in screener_mode else VN30_TICKERS
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_scan_progress(curr, total, ticker):
            pct = curr / total
            progress_bar.progress(pct)
            status_text.markdown(f"⏳ **Đang phân tích định lượng mã `{ticker}` ({curr}/{total})...**")

        with st.spinner("Đang chạy mô hình AI quét hàng loạt cổ phiếu..."):
            df_scanned = scan_vn30_market(
                ticker_list=chosen_tickers,
                n_forward=n_forward,
                prob_threshold=prob_threshold,
                profit_threshold_pct=profit_threshold_pct,
                start_date=start_date.strftime("%Y-%m-%d"),
                force_reload=force_refresh_screener,
                progress_callback=update_scan_progress
            )
            st.session_state["screener_results"] = df_scanned

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Đã quét và phân tích hoàn tất {len(st.session_state['screener_results'])} mã cổ phiếu!")

    # Hiển thị kết quả từ Session State
    df_results = st.session_state["screener_results"]

    if df_results is not None and not df_results.empty:
        st.markdown("---")

        # 1. Ba thẻ KPI nổi bật
        buy_count = int(df_results["is_buy"].sum())
        top_prob_row = df_results.sort_values(by="Xác suất tăng > 1.5% (%)", ascending=False).iloc[0]
        top_ret_row = df_results.sort_values(by="% Kỳ vọng (+ / -)", ascending=False).iloc[0]

        kpi_s1, kpi_s2, kpi_s3 = st.columns(3)
        with kpi_s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Số mã cho Tín hiệu Mua (Ngưỡng {prob_threshold*100:.0f}%)</div>
                <div class="metric-value" style="color: #34d399;">{buy_count} <span style="font-size: 1.1rem; color: #94a3b8;">/ {len(df_results)} mã</span></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_s2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Xác suất Tăng Cao Nhất</div>
                <div class="metric-value" style="color: #38bdf8;">{top_prob_row['Mã CP']} <span style="font-size: 1.2rem; color: #f8fafc;">({top_prob_row['Xác suất tăng > 1.5% (%)']}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_s3:
            ret_val = top_ret_row['% Kỳ vọng (+ / -)']
            ret_c = "#34d399" if ret_val >= 0 else "#f87171"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">% Lợi nhuận Kỳ vọng Cao Nhất</div>
                <div class="metric-value" style="color: {ret_c};">{top_ret_row['Mã CP']} <span style="font-size: 1.2rem; color: #f8fafc;">({ret_val:+.2f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Biểu đồ cột ngang Top 10 cổ phiếu có xác suất tăng cao nhất
        st.markdown(f"#### Biểu đồ Top 10 Cổ phiếu có Xác suất Tăng giá Cao nhất ({n_forward} phiên)")
        top10_plot = df_results.sort_values(by="Xác suất tăng > 1.5% (%)", ascending=True).tail(10)

        bar_colors = ["#10b981" if p >= (prob_threshold * 100.0) else "#64748b" for p in top10_plot["Xác suất tăng > 1.5% (%)"]]

        fig_screener = go.Figure(go.Bar(
            x=top10_plot["Xác suất tăng > 1.5% (%)"],
            y=top10_plot["Mã CP"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.1f}% ({rec})" for p, rec in zip(top10_plot["Xác suất tăng > 1.5% (%)"], top10_plot["Khuyến nghị"])],
            textposition="auto"
        ))

        # Vạch ngưỡng threshold
        threshold_val = prob_threshold * 100.0
        fig_screener.add_vline(
            x=threshold_val,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=2,
            annotation_text=f"Ngưỡng vào lệnh ({threshold_val:.0f}%)",
            annotation_position="top right"
        )

        fig_screener.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark",
            xaxis_title="Xác suất Tăng giá > 1.5% (%)",
            yaxis_title="Mã Cổ Phiếu"
        )
        st.plotly_chart(fig_screener, use_container_width=True)

        # 3. Bảng xếp hạng chi tiết
        st.markdown("#### Bảng Xếp hạng Chi tiết Rổ VN30")
        display_cols = [
            "Hạng", "Mã CP", "Giá hiện tại (VNĐ)", "Giá mục tiêu sau N phiên",
            "% Kỳ vọng (+ / -)", "Xác suất tăng > 1.5% (%)", "RSI (14)",
            "R² / Độ tin cậy", "Khuyến nghị"
        ]
        table_df = df_results[display_cols].copy()

        # Format hiển thị số liệu
        st.dataframe(
            table_df.style.format({
                "Giá hiện tại (VNĐ)": "{:,.0f}",
                "Giá mục tiêu sau N phiên": "{:,.0f}",
                "% Kỳ vọng (+ / -)": "{:+.2f}%",
                "Xác suất tăng > 1.5% (%)": "{:.1f}%",
                "RSI (14)": "{:.1f}",
                "R² / Độ tin cậy": "{:.3f}"
            }),
            use_container_width=True
        )

        # 4. Nút tải file CSV
        csv_data = table_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải Bảng Kết quả Bộ lọc VN30 (.csv)",
            data=csv_data,
            file_name=f"VN30_Screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 Hãy chọn phạm vi quét và bấm nút **'🚀 Bắt đầu Quét & Xếp hạng'** ở trên để khởi chạy bộ lọc AI cho rổ VN30.")
