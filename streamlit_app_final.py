import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
LOG_FILE = 'logs/trade_log.csv'
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# === LOAD DATA ===
@st.cache_data
def load_data():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, header=None, names=["datetime", "coin", "price", "rsi", "volume_spike", "breakout", "signal"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values("datetime", ascending=False, inplace=True)
        return df
    else:
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("No trade logs found.")
    st.stop()

# === SIDEBAR NAVIGATION ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Alerts", "Strategy", "Backtest", "Settings"])

# === DARK MODE TOGGLE ===
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #111111;
                color: #ffffff;
            }
            .stDataFrame, .stMarkdown, .stButton, .stDownloadButton {
                background-color: #222222 !important;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)


# === Add Signal Quality Score Column to Alerts ===
def compute_quality(row):
    rsi_score = max(0, 100 - abs(40 - row['rsi']) * 2.5)  # Best if RSI ~ 40
    volume_score = min(100, (row['volume_spike'] - 2) * 25) if row['volume_spike'] > 2 else 0
    quality = 0.5 * rsi_score + 0.5 * volume_score
    return int(min(100, quality))

df_filtered["Signal Quality"] = df_filtered.apply(compute_quality, axis=1)

# === PAGE: ALERTS ===
if page == "Alerts":
    st.title("Trade Alerts")
    with st.sidebar.expander("Filters"):
        coins = st.multiselect("Select coins", options=sorted(df['coin'].unique()), default=sorted(df['coin'].unique()))
        date_range = st.date_input("Date range", [df['datetime'].min().date(), df['datetime'].max().date()])

    df_filtered = df[
        (df['coin'].isin(coins)) &
        (df['datetime'].dt.date >= date_range[0]) &
        (df['datetime'].dt.date <= date_range[1])
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", len(df_filtered))
    col2.metric("GO Signals", len(df_filtered[df_filtered['signal'] == "GO"]))
    col3.metric("Coins", df_filtered['coin'].nunique())

    st.subheader("Trade Log")
    styled_df = df_filtered.style.applymap(
        lambda val: 'color: green; font-weight: bold;' if val == "GO" else '', subset=["signal"]
    )
    st.dataframe(df_filtered[['datetime', 'coin', 'price', 'rsi', 'volume_spike', 'signal', 'Signal Quality']], use_container_width=True)

# === TOP SIGNALS HIGHLIGHT ===
top_signals = df_filtered[df_filtered['Signal Quality'] >= 80].sort_values("Signal Quality", ascending=False).head(5)
if not top_signals.empty:
    st.subheader("Top High-Quality Signals")
    st.dataframe(top_signals[['datetime', 'coin', 'price', 'rsi', 'volume_spike', 'signal', 'Signal Quality']], use_container_width=True)


# === PAGE: STRATEGY ===
elif page == "Strategy":
    st.title("Capital Compounding Model")
    go_signals = df[df['signal'] == "GO"]
    initial = st.number_input("Initial Capital (RM)", min_value=10.0, value=100.0)
    capital = [initial]

    for _ in range(len(go_signals)):
        capital.append(capital[-1] * 1.03)

    st.line_chart(pd.Series(capital, name="Capital Growth"))
    st.info(f"Total trades: {len(go_signals)} | Final capital: RM {capital[-1]:.2f}")


# === PAGE: BACKTEST with Scoring ===
elif page == "Backtest":
    st.title("Backtest & Coin Performance Score")

    coin_stats = []
    for coin in df['coin'].unique():
        subset = df[df['coin'] == coin]
        total_signals = len(subset)
        go_signals = subset[subset['signal'] == "GO"]
        successful = 0
        for i in range(len(go_signals) - 1):
            entry = go_signals.iloc[i]['price']
            next_price = go_signals.iloc[i + 1]['price']
            gain = (next_price - entry) / entry
            if gain >= 0.03:
                successful += 1
        success_rate = (successful / len(go_signals)) * 100 if len(go_signals) > 0 else 0
        coin_stats.append({"Coin": coin, "Total": total_signals, "GO": len(go_signals), "Success Rate (%)": round(success_rate, 2)})

    stats_df = pd.DataFrame(coin_stats).sort_values("Success Rate (%)", ascending=False)
    st.dataframe(stats_df, use_container_width=True)
    st.bar_chart(stats_df.set_index("Coin")["Success Rate (%)"])

    # === COIN LEADERBOARD ===
    st.subheader("Leaderboard")
    stats_df['Rank'] = stats_df['Success Rate (%)'].rank(method='min', ascending=False).astype(int)
    stats_df = stats_df.sort_values("Rank")
    styled_leaderboard = stats_df.style.apply(
        lambda row: ['background-color: gold' if row.Rank == 1 else 
                     'background-color: silver' if row.Rank == 2 else 
                     'background-color: peru' if row.Rank == 3 else '' 
                     for _ in row], axis=1
    )
    st.dataframe(styled_leaderboard, use_container_width=True)


elif page == "Backtest":
    st.title("RSI & Volume Backtest")
    coin = st.selectbox("Select coin", sorted(df['coin'].unique()))
    trend_data = df[df['coin'] == coin].sort_values("datetime")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(trend_data['datetime'], trend_data['rsi'], 'g-', label="RSI")
    ax2.plot(trend_data['datetime'], trend_data['volume_spike'], 'b--', label="Volume Spike")
    ax1.set_ylabel('RSI', color='g')
    ax2.set_ylabel('Volume Spike', color='b')
    plt.title(f"RSI & Volume Trend for {coin}")
    st.pyplot(fig)

# === PAGE: SETTINGS ===
elif page == "Settings":
    st.title("Export & Settings")
    st.write("Use the buttons below to export filtered data.")
    st.download_button("Download All Data as CSV", df.to_csv(index=False), file_name="trade_log.csv")