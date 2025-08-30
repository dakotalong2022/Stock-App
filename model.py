import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_models(X, y):
    rf = RandomForestRegressor()
    rf.fit(X, y)
    xgb = XGBRegressor()
    xgb.fit(X, y)
    lstm_model = Sequential()
    lstm_model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    X_lstm = np.expand_dims(X.values, axis=2)
    lstm_model.fit(X_lstm, y.values, epochs=10, batch_size=16, verbose=0)
    return {'rf': rf, 'xgb': xgb, 'lstm': lstm_model}

def predict_ensemble(models, X):
    rf_pred = models['rf'].predict(X)
    xgb_pred = models['xgb'].predict(X)
    X_lstm = np.expand_dims(X.values, axis=2)
    lstm_pred = models['lstm'].predict(X_lstm).flatten()
    return (rf_pred + xgb_pred + lstm_pred) / 3
