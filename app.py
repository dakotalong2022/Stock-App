
import yfinance as yf
import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import streamlit as st
from datetime import datetime

def get_stock_data(stock_symbol, real_time=False):
    if real_time:
        data = yf.download(stock_symbol, period='1d', interval='1m')
    else:
        data = yf.download(stock_symbol, start='2010-01-01', end=str(datetime.today().date()))
    
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
    data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
    
    data.dropna(inplace=True)
    return data

def create_lstm_model(X_train, y_train, X_test):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model.predict(X_test)

def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    return X, y, scaler

def random_forest_model(X_train, y_train, X_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model.predict(X_test)

def xgboost_model(X_train, y_train, X_test):
    xg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
    xg_model.fit(X_train, y_train)
    return xg_model.predict(X_test)

def main():
    st.title("Stock Price Prediction")
    st.sidebar.header("User Inputs")
    
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOG):", "AAPL")
    data_type = st.sidebar.radio("Select Data Type:", ["Historical Data", "Real-Time Data"])

    if data_type == "Real-Time Data":
        data = get_stock_data(stock_symbol, real_time=True)
    else:
        data = get_stock_data(stock_symbol, real_time=False)
    
    X, y, scaler = prepare_data(data[['Close','SMA_50','SMA_200','RSI','EMA_12','EMA_26']])

    lstm_predictions = create_lstm_model(X, y, X)
    rf_predictions = random_forest_model(X.reshape(X.shape[0], -1), y, X.reshape(X.shape[0], -1))
    xg_predictions = xgboost_model(X.reshape(X.shape[0], -1), y, X.reshape(X.shape[0], -1))

    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))
    xg_predictions = scaler.inverse_transform(xg_predictions.reshape(-1, 1))

    ensemble_predictions = (lstm_predictions + rf_predictions + xg_predictions) / 3

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], color='blue', label='Actual Price')
    ax.plot(ensemble_predictions, color='red', label='Predicted Price (Ensemble)')
    ax.set_title(f"Stock Price Prediction for {stock_symbol}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

    mse = np.mean((ensemble_predictions - data['Close'].values[-len(ensemble_predictions):])**2)
    r2 = 1 - mse / np.var(data['Close'].values[-len(ensemble_predictions):])
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()
