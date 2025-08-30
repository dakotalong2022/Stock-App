import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Streamlit app title
st.title("Stock Price Analysis")

# Sidebar for stock symbol and date range
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch the stock data
@st.cache_data  # Cache the data to avoid unnecessary reloading
def get_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

df = get_stock_data(symbol, start_date, end_date)

# Check if data is empty
if df.empty:
    st.error(f"No data found for {symbol} in the given date range.")
else:
    # Display the data
    st.write(f"Data for {symbol} from {start_date} to {end_date}")
    st.write(df.tail())

    # Calculate moving averages as an example of technical analysis
    df['SMA_50'] = ta.sma(df['Close'], 50)
    df['SMA_200'] = ta.sma(df['Close'], 200)

    # Plot the stock closing price and moving averages
    st.subheader("Stock Price with Moving Averages")
    st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])

    # Add more indicators as needed, for example RSI
    df['RSI'] = ta.rsi(df['Close'], 14)
    st.subheader("RSI (Relative Strength Index)")
    st.line_chart(df['RSI'])
