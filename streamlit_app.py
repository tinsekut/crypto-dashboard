import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
LOG_FILE = 'logs/trade_log.csv'

st.set_page_config(page_title="Crypto Trade Dashboard", layout="wide")

# === LOAD DATA ===
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE, header=None, names=["datetime", "coin", "price", "rsi", "volume_spike", "breakout", "signal"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values("datetime", ascending=False, inplace=True)
else:
    st.error("No trade logs found.")
    st.stop()

# === SIDEBAR ===
st.sidebar.title("Filter")
coins = st.sidebar.multiselect("Select coins", options=sorted(df['coin'].unique()), default=sorted(df['coin'].unique()))
date_range = st.sidebar.date_input("Date range", [df['datetime'].min().date(), df['datetime'].max().date()])

# === FILTER DATA ===
df_filtered = df[
    (df['coin'].isin(coins)) &
    (df['datetime'].dt.date >= date_range[0]) &
    (df['datetime'].dt.date <= date_range[1])
]

# === HEADER ===
st.title("Crypto Trade Readiness Dashboard")

# === METRICS ===
col1, col2, col3 = st.columns(3)
col1.metric("Total Alerts", len(df_filtered))
col2.metric("GO Signals", len(df_filtered[df_filtered['signal'] == "GO"]))
col3.metric("Unique Coins", df_filtered['coin'].nunique())

# === DATA TABLE ===
st.subheader("Trade Alert Log")
st.dataframe(df_filtered.style.highlight_max(axis=0), use_container_width=True)

# === COMPOUNDING MODEL ===
st.subheader("Capital Compounding (3% Gain / GO Trade)")
initial_capital = st.number_input("Initial Capital (RM)", min_value=10.0, value=100.0, step=10.0)
go_signals = df_filtered[df_filtered['signal'] == "GO"]
capital = [initial_capital]

for _ in range(len(go_signals)):
    capital.append(capital[-1] * 1.03)

st.line_chart(pd.Series(capital, name="Capital Growth"))

# === RSI & Volume Trend ===
st.subheader("RSI and Volume Trend")
selected_coin = st.selectbox("Choose coin to view trend", sorted(df_filtered['coin'].unique()))
trend_data = df_filtered[df_filtered['coin'] == selected_coin].sort_values("datetime")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(trend_data['datetime'], trend_data['rsi'], 'g-', label="RSI")
ax2.plot(trend_data['datetime'], trend_data['volume_spike'], 'b--', label="Volume Spike")

ax1.set_ylabel('RSI', color='g')
ax2.set_ylabel('Volume Spike', color='b')
plt.title(f"RSI & Volume Trend for {selected_coin}")
st.pyplot(fig)

# === EXPORT ===
st.subheader("Export Data")
st.download_button("Download CSV", df_filtered.to_csv(index=False), file_name="filtered_trade_log.csv", mime="text/csv")