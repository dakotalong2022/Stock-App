import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# App title
st.title('Stock App')

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Get stock data from Yahoo Finance
data = yf.download(ticker, period="1y", interval="1d")

# Calculate technical indicators using pandas_ta
data['SMA'] = ta.sma(data['Close'], 20)

# Display the data
st.write(f"Displaying data for {ticker}")
st.dataframe(data.tail())

# Display stock price chart with technical indicators
st.line_chart(data[['Close', 'SMA']])
