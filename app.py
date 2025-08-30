import streamlit as st
import yfinance as yf
import pandas as pd

# Set up page configuration
st.set_page_config(page_title="Stock Data App", layout="wide")

# Simple title
st.title("Stock Data App")

# User input for stock ticker
ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOG):", "AAPL")

# Fetch data for the selected ticker
@st.cache_data
def fetch_data(ticker):
    data = yf.download(ticker, period="7d", interval="1h")  # 7 days of data, 1 hour intervals
    return data

# Display fetched data
if ticker:
    data = fetch_data(ticker)
    st.write(f"Showing data for {ticker}")
    st.line_chart(data['Close'])
else:
    st.write("Please enter a stock ticker.")

