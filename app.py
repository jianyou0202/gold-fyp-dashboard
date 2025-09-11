import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# --- Page Config ---
st.set_page_config(page_title="Gold FYP Backtest Dashboard", layout="wide")
st.markdown(
    """
    <h1 style="text-align:center; color:#FFD700;">
        🏆 Gold Multi-Strategy Backtest Dashboard
    </h1>
    <p style="text-align:center; color:gray;">
        ✨ Analyze and compare trading strategies on Gold Futures (GC=F) ✨
    </p>
    <hr style="border:1px solid #FFD700;">
    """,
    unsafe_allow_html=True
)

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy Parameters")

initial_capital = st.sidebar.number_input("💰 Initial Capital ($)", min_value=1000, max_value=5_000_000, value=10000, step=1000)

strategy_choices = st.sidebar.multiselect(
    "Select Strategies",
    ["EMA Crossover", "MACD Crossover", "Stochastic Oscillator", "Combined"],
    default=["EMA Crossover"]
)

# EMA
fast = st.sidebar.number_input("Fast EMA", min_value=5, max_value=50, value=12)
slow = st.sidebar.number_input("Slow EMA", min_value=10, max_value=200, value=26)

# MACD
short_window = st.sidebar.number_input("MACD Short EMA", min_value=5, max_value=20, value=12)
long_window = st.sidebar.number_input("MACD Long EMA", min_value=20, max_value=50, value=26)
signal_window = st.sidebar.number_input("MACD Signal EMA", min_value=5, max_value=20, value=9)

# Stochastic
stoch_k = st.sidebar.number_input("Stochastic %K Window", min_value=5, max_value=30, value=14)
stoch_d = st.sidebar.number_input("Stochastic %D Window", min_value=3, max_value=10, value=3)

# Indicator Weights (for Combined)
st.sidebar.subheader("📊 Indicator Weights (Combined Strategy)")
ema_weight = st.sidebar.slider("EMA Weight", 0.0, 1.0, 0.5)
macd_weight = st.sidebar.slider("MACD Weight", 0.0, 1.0, 0.3)
stoch_weight = st.sidebar.slider("Stochastic Weight", 0.0, 1.0, 0.2)

# Risk & cost params
st.sidebar.subheader("⚖️ Risk Settings")
target_annual_vol = st.sidebar.number_input("Target annual volatility (e.g. 0.15 = 15%)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
max_leverage = st.sidebar.number_input("Max leverage (position cap, e.g. 1 = 100%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
trade_cost = st.sidebar.number_input("Trade cost per turnover (fraction, e.g. 0.0005 = 0.05%)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)

# --- Fetch Data ---
symbol = "GC=F"
df = yf.download(symbol, period="5y", interval="1d")
df.dropna(inplace=True)

# flatten multiindex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

# --- Indicator Calculations ---
df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()

df["EMA_short"] = df["Close"].ewm(span=short_window, adjust=False).mean()
df["EMA_long"] = df["Close"].ewm(span=long_window, adjust=False).mean()
df["MACD"] = df["EMA_short"] - df["EMA_long"]
df["Signal_Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

df["Lk"] = df["Low"].rolling(window=stoch_k, min_periods=1).min()
df["Hk"] = df["High"].rolling(window=stoch_k, min_periods=1).max()
denom = (df["Hk"] - df["Lk"]).replace(0, np.nan)
df["%K"] = ((df["Close"] - df["Lk"]) / denom) * 100
df["%D"] = df["%K"].rolling(window=stoch_d, min_periods=1).mean()

# --- Warmup: drop first rows where long window indicators are NaN ---
warmup = max(slow, long_window, stoch_k) + 5
df = df.iloc[warmup:].copy()

# --- Helper: annualization factor (daily)
ANN_FACTOR = 252

# --- Improved Backtest Function ---
def run_backtest(strategy_name, initial_capital, cost=trade_cost, target_ann_vol=target_annual_vol, max_lev=max_leverage):
    temp = df.copy()

    # raw signals
    ema_signal = np.where(temp["EMA_fast"] > temp["EMA_slow"], 1, -1)
    macd_signal = np.where(temp["MACD"] > temp["Signal_Line"], 1, -1)
    stoch_signal = np.where(temp["%K"] > temp["%D"], 1, -1)

    if strategy_name == "EMA Crossover":
        temp["raw_signal"] = ema_signal
    elif strategy_name == "MACD Crossover":
        temp["raw_signal"] = macd_signal
    elif strategy_name == "Stochastic Oscillator":
        temp["raw_signal"] = stoch_signal
    elif strategy_name == "Combined":
        wsum = max(ema_weight + macd_weight + stoch_weight, 1e-9)
        score = (ema_signal * ema_weight + macd_signal * macd_weight + stoch_signal * stoch_weight) / wsum
        temp["raw_signal"] = np.where(score > 0.2, 1, np.where(score < -0.2, -1, 0))
        temp["Signal_Confidence"] = (abs(score) * 100)

    temp["signal"] = temp["raw_signal"].shift(1).fillna(0)  # shift to avoid lookahead
    temp["return"] = temp["Close"].pct_change().fillna(0)

    temp["daily_vol"] = temp["return"].rolling(window=14, min_periods=1).std()
    temp["daily_vol"] = temp["daily_vol"].clip(lower=1e-4)

    target_daily_vol = target_ann_vol / np.sqrt(ANN_FACTOR)
    temp["pos_size_raw"] = target_daily_vol / temp["daily_vol"]
    temp["pos_size_capped"] = temp["pos_size_raw"].clip(-max_lev, max_lev)

    temp["position"] = temp["signal"] * temp["pos_size_capped"]
    temp["prev_position"] = temp["position"].shift(1).fillna(0)

    temp["turnover"] = (temp["position"] - temp["prev_position"]).abs()
    temp["trade_costs"] = temp["turnover"] * cost

    temp["strategy_return"] = temp["prev_position"] * temp["return"] - temp["trade_costs"]
    temp["portfolio"] = (1 + temp["strategy_return"]).cumprod() * initial_capital

    temp["cum_returns"] = (1 + temp["strategy_return"]).cumprod()
    temp["rolling_max"] = temp["cum_returns"].cummax()
    temp["drawdown"] = (temp["cum_returns"] - temp["rolling_max"]) / temp["rolling_max"]

    total_days = len(temp)
    cagr = (temp["portfolio"].iloc[-1] / initial_capital) ** (ANN_FACTOR / total_days) - 1
    ann_vol = temp["strategy_return"].std() * np.sqrt(ANN_FACTOR)
    sharpe = (temp["strategy_return"].mean() / temp["strategy_return"].std()) * np.sqrt(ANN_FACTOR) if temp["strategy_return"].std() != 0 else 0
    max_dd = temp["drawdown"].min()

    trades_idx = temp.index[temp["turnover"] > 0]
    trade_log = []
    for t in trades_idx:
        trade_log.append({
            "Date": t,
            "Trade_Type": "Buy" if temp.at[t, "position"] > temp.at[t, "prev_position"] else "Sell",
            "Size": round(temp.at[t, "position"], 4),
            "Price": float(temp.at[t, "Close"]),
            "Turnover": float(temp.at[t, "turnover"])
        })
    trade_log_df = pd.DataFrame(trade_log).set_index("Date") if trade_log else pd.DataFrame(columns=["Trade_Type","Size","Price","Turnover"])

    return {
        "Strategy": strategy_name,
        "Initial Capital": initial_capital,
        "Final Portfolio Value": float(temp["portfolio"].iloc[-1]),
        "Net Profit": float(temp["portfolio"].iloc[-1] - initial_capital),
        "Total Trades": int((temp["turnover"] > 0).sum()),
        "CAGR": cagr,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Portfolio Curve": temp["portfolio"],
        "Risk Data": temp,
        "Trade Log": trade_log_df
    }

# --- Run Backtests ---
results = [run_backtest(strategy, initial_capital) for strategy in strategy_choices]

# --- KPIs ---
if len(results) > 0:
    st.markdown("### 📊 Key Metrics (first selected strategy)")
    first = results[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Capital", f"${first['Initial Capital']:,}")
    col2.metric("Final Value", f"${int(first['Final Portfolio Value']):,}")   
    col3.metric("Net Profit", f"${int(first['Net Profit']):,}")               
    col4.metric("Sharpe Ratio", round(first["Sharpe Ratio"], 4)) 

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Portfolio Growth",
    "📑 Trade Logs",
    "📂 Full Data",
    "⚠️ Risk Analysis",
    "📊 TradingView-Style Chart"
])

with tab1:
    st.subheader("Portfolio Growth Comparison")
    fig = px.line()
    for r in results:
        fig.add_scatter(x=r["Portfolio Curve"].index, y=r["Portfolio Curve"], mode="lines", name=r["Strategy"])
    fig.update_layout(template="plotly_dark", title="Portfolio Growth Over Time", xaxis_title="Date", yaxis_title="Portfolio Value ($)", legend_title="Strategy")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Trade Simulation Log")
    for r in results:
        st.write(f"📌 Strategy: **{r['Strategy']}**")
        st.dataframe(r["Trade Log"], use_container_width=True)

with tab3:
    with st.expander("🔎 Full Dataset (with indicators)"):
        st.dataframe(df, use_container_width=True)

with tab4:
    for r in results:
        st.markdown(f"### ⚠️ Risk Analysis - {r['Strategy']}")
        fig_dd = px.area(r["Risk Data"], x=r["Risk Data"].index, y="drawdown", title=f"Drawdown - {r['Strategy']}")
        st.plotly_chart(fig_dd, use_container_width=True)

with tab5:
    st.subheader("📊 TradingView-Style Gold Chart")
    try:
        recent_df = yf.download(symbol, period="6mo", interval="1d")
        if isinstance(recent_df.columns, pd.MultiIndex):
            recent_df.columns = [col[0] for col in recent_df.columns]

        recent_df["EMA_fast"] = recent_df["Close"].ewm(span=fast, adjust=False).mean()
        recent_df["EMA_slow"] = recent_df["Close"].ewm(span=slow, adjust=False).mean()
        recent_df["EMA_short"] = recent_df["Close"].ewm(span=short_window, adjust=False).mean()
        recent_df["EMA_long"] = recent_df["Close"].ewm(span=long_window, adjust=False).mean()
        recent_df["MACD"] = recent_df["EMA_short"] - recent_df["EMA_long"]
        recent_df["Signal_Line"] = recent_df["MACD"].ewm(span=signal_window, adjust=False).mean()
        recent_df["H14"] = recent_df["High"].rolling(window=stoch_k).max()
        recent_df["L14"] = recent_df["Low"].rolling(window=stoch_k).min()
        denom = (recent_df["H14"] - recent_df["L14"]).replace(0, np.nan)
        recent_df["%K"] = ((recent_df["Close"] - recent_df["L14"]) / denom) * 100
        recent_df["%D"] = recent_df["%K"].rolling(window=stoch_d).mean()

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.55,0.225,0.225], subplot_titles=("Price + EMA", "MACD", "Stochastic"))

        fig.add_trace(go.Candlestick(x=recent_df.index, open=recent_df["Open"], high=recent_df["High"],
                                     low=recent_df["Low"], close=recent_df["Close"], name="Candlestick"), row=1, col=1)
        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["EMA_fast"], mode="lines", name="EMA Fast"), row=1, col=1)
        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["EMA_slow"], mode="lines", name="EMA Slow"), row=1, col=1)

        fig.add_trace(go.Bar(x=recent_df.index, y=recent_df["MACD"]-recent_df["Signal_Line"], name="MACD Histogram"), row=2, col=1)
        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["MACD"], mode="lines", name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["Signal_Line"], mode="lines", name="Signal Line"), row=2, col=1)

        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["%K"], mode="lines", name="%K"), row=3, col=1)
        fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["%D"], mode="lines", name="%D"), row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            dragmode="pan",
            height=900,
            title="Gold Futures with EMA, MACD, and Stochastic",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not fetch chart data: {e}")

