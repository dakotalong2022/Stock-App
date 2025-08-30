import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Utility functions
from utils import fetch_data, compute_indicators, prepare_features
from model import train_models, predict_ensemble

st.title("Stock Price Prediction App")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Period", ["1y", "2y", "5y"])

if st.button("Fetch Data"):
    df = fetch_data(symbol, period)
    df = compute_indicators(df)
    X, y = prepare_features(df)
    st.write(df.tail())
    
    st.subheader("Training Models...")
    models = train_models(X, y)
    st.success("Models trained!")
    st.subheader("Making Predictions...")
    preds = predict_ensemble(models, X)
    df['Predicted'] = preds
    st.line_chart(df[['Close', 'Predicted']])
    
    st.write("Performance Metrics:")
    st.write(f"RMSE: {np.sqrt(np.mean((df['Close'] - df['Predicted'])**2)):.2f}")
