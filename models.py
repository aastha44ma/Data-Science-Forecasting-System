import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import holidays
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_sarima(train, validation_len):
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 52), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=validation_len)
        return fitted, forecast
    except Exception as e:
        print("SARIMA Failed", e)
        model = SARIMAX(train, order=(1, 1, 1))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=validation_len)
        return fitted, forecast

def train_prophet(train_df, validation_len):
    pdf = train_df.reset_index()[['Date', 'Total']]
    pdf.columns = ['ds', 'y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.add_country_holidays(country_name='US')
    m.fit(pdf)
    future = m.make_future_dataframe(periods=validation_len, freq='W')
    forecast = m.predict(future)
    return m, forecast['yhat'].tail(validation_len).values

def train_xgboost(train_feat, val_feat, target_col='Total'):
    X_train = train_feat.drop(columns=[target_col])
    y_train = train_feat[target_col]
    X_val = val_feat.drop(columns=[target_col])

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return model, preds

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(train_feat, val_feat, target_col='Total'):
    scaler = MinMaxScaler()
    cols = [target_col] + [c for c in train_feat.columns if c != target_col]
    train_data = train_feat[cols].values
    val_data = val_feat[cols].values

    train_scaled = scaler.fit_transform(train_data)

    seq_length = 8
    val_concat = np.vstack([train_scaled[-seq_length:], scaler.transform(val_data)])

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_concat, seq_length)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel(input_size=X_train.shape[2], hidden_size=32, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    preds = []
    current_seq = torch.tensor(train_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)

    for i in range(len(val_data)):
        with torch.no_grad():
            pred = model(current_seq)
            preds.append(pred.item())

            next_step = val_concat[seq_length + i].copy()
            next_step[0] = pred.item()

            next_step_t = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            current_seq = torch.cat([current_seq[:, 1:, :], next_step_t], dim=1)

    dummy = np.zeros((len(preds), train_scaled.shape[1]))
    dummy[:, 0] = preds
    preds_inv = scaler.inverse_transform(dummy)[:, 0]

    return model, preds_inv
