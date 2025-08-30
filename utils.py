import yfinance as yf
import pandas as pd
import talib

def fetch_data(symbol, period):
    data = yf.download(symbol, period=period)
    return data

def compute_indicators(df):
    df['SMA'] = talib.SMA(df['Close'])
    df['EMA'] = talib.EMA(df['Close'])
    df['RSI'] = talib.RSI(df['Close'])
    return df

def prepare_features(df):
    features = df[['SMA', 'EMA', 'RSI']].fillna(0)
    target = df['Close']
    return features, target
