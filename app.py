import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold FYP Backtest Dashboard", layout="wide")
st.title("📊 Gold Multi-Strategy Backtest Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy Parameters")

# Strategy choice
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

# --- Weighted Indicators (New Feature) ---
st.sidebar.subheader("Indicator Weights (for Combined Strategy)")
ema_weight = st.sidebar.slider("EMA Weight", 0.0, 1.0, 0.5)
macd_weight = st.sidebar.slider("MACD Weight", 0.0, 1.0, 0.3)
stoch_weight = st.sidebar.slider("Stochastic Weight", 0.0, 1.0, 0.2)

# --- Fetch Data ---
symbol = "GC=F"  # Gold Futures
df = yf.download(symbol, period="2y", interval="1d")
df.dropna(inplace=True)

# Flatten multi-level columns if exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# --- Indicator Calculations ---
# EMA
df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()

# MACD
df["EMA_short"] = df["Close"].ewm(span=short_window, adjust=False).mean()
df["EMA_long"] = df["Close"].ewm(span=long_window, adjust=False).mean()
df["MACD"] = df["EMA_short"] - df["EMA_long"]
df["Signal_Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

# Stochastic
df["L14"] = df["Low"].rolling(window=stoch_k).min()
df["H14"] = df["High"].rolling(window=stoch_k).max()
denom = df["H14"] - df["L14"]
denom = denom.replace(0, np.nan)
df["%K"] = ((df["Close"] - df["L14"]) / denom) * 100
df["%K"].fillna(0, inplace=True)
df["%D"] = df["%K"].rolling(window=stoch_d).mean()

# --- Backtest Function (Enhanced) ---
def run_backtest(strategy_name):
    temp = df.copy()

    # --- Signals ---
    ema_signal = np.where(temp["EMA_fast"] > temp["EMA_slow"], 1, -1)
    macd_signal = np.where(temp["MACD"] > temp["Signal_Line"], 1, -1)
    stoch_signal = np.where(temp["%K"] > temp["%D"], 1, -1)

    if strategy_name == "EMA Crossover":
        temp["Signal"] = ema_signal
    elif strategy_name == "MACD Crossover":
        temp["Signal"] = macd_signal
    elif strategy_name == "Stochastic Oscillator":
        temp["Signal"] = stoch_signal
    elif strategy_name == "Combined":
        # Weighted signal
        score = ema_signal*ema_weight + macd_signal*macd_weight + stoch_signal*stoch_weight
        temp["Signal"] = np.where(score > 0.5, 1, np.where(score < -0.5, -1, 0))
        temp["Signal_Confidence"] = (abs(score) / (ema_weight+macd_weight+stoch_weight)) * 100

    # --- Position & Strategy Returns ---
    temp["Position"] = temp["Signal"].replace(to_replace=0, method="ffill").fillna(0)
    temp["Return"] = temp["Close"].pct_change()

    # Volatility-adjusted position (New Feature)
    temp["Volatility"] = temp["Close"].pct_change().rolling(window=14).std()
    temp["Position"] = temp["Position"] * (0.02 / temp["Volatility"].replace(0,np.nan)).fillna(0)

    temp["Strategy"] = temp["Position"].shift(1) * temp["Return"]

    # --- Portfolio ---
    initial_capital = 10000
    temp["Portfolio"] = (1 + temp["Strategy"]).cumprod() * initial_capital

    # --- Trade Log (New Feature) ---
    temp["Trade_Type"] = np.where(temp["Signal"]==1, "Buy", np.where(temp["Signal"]==-1, "Sell", "Hold"))
    temp["Trade_Price"] = np.where(temp["Signal"]!=0, temp["Close"], np.nan)

    # --- Metrics ---
    mean_return = temp["Strategy"].mean()
    std_return = temp["Strategy"].std()
    sharpe_ratio = (mean_return/std_return)*np.sqrt(252) if std_return!=0 else 0
    final_value = temp["Portfolio"].iloc[-1]
    profit = final_value - initial_capital
    trades = temp["Signal"].diff().abs().sum()/2

    return {
        "Strategy": strategy_name,
        "Initial Capital": initial_capital,
        "Final Portfolio Value": round(final_value, 2),
        "Net Profit": round(profit, 2),
        "Total Trades": int(trades),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Portfolio Curve": temp["Portfolio"],
        "Trade Log": temp[["Trade_Type","Trade_Price","Signal_Confidence"]] if "Signal_Confidence" in temp else temp[["Trade_Type","Trade_Price"]]
    }

# --- Run Backtests ---
results = [run_backtest(strategy) for strategy in strategy_choices]

# --- Display Strategy Comparison Table ---
st.subheader("📊 Strategy Comparison")
results_df = pd.DataFrame([r for r in results if isinstance(r, dict)])
st.table(results_df.drop(columns=["Portfolio Curve","Trade Log"]))

# --- Plot Portfolio Growth ---
st.subheader("📈 Portfolio Growth Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
for r in results:
    ax.plot(r["Portfolio Curve"], label=r["Strategy"])
ax.set_title("Portfolio Growth Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
st.pyplot(fig)

# --- Display Trade Logs ---
st.subheader("📑 Trade Simulation Log")
for r in results:
    st.write(f"Strategy: {r['Strategy']}")
    st.dataframe(r["Trade Log"])

# --- Display Full Data ---
st.subheader("📑 Full Data (with indicators)")
st.dataframe(df)

