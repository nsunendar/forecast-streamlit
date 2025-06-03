
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# Transformer Model Components
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len]

class TransformerModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        encoder = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.decoder = nn.Linear(64, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.decoder(x[-1])

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}

# SARIMA Model
def forecast_sarima(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    results = model.fit(disp=False)
    pred = results.forecast(steps=6)
    return pred, evaluate(series[-6:], pred)

# Prophet Model
def forecast_prophet(series):
    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=6, freq='MS')
    forecast = model.predict(future)
    pred = forecast[['ds', 'yhat']].set_index('ds')[-6:]['yhat']
    return pred, evaluate(series[-6:], pred)

# LSTM / GRU Model
def forecast_dl(series, model_type='lstm'):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    def create_sequences(data, seq_length=6):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(50, activation='relu', input_shape=(6,1)))
    else:
        model.add(GRU(50, activation='relu', input_shape=(6,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    forecast_input = scaled[-6:].reshape((1,6,1))
    preds = []
    for _ in range(6):
        pred = model.predict(forecast_input)[0][0]
        preds.append(pred)
        forecast_input = np.append(forecast_input[:,1:,:], [[[pred]]], axis=1)

    pred_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return pd.Series(pred_inv, index=pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')), evaluate(series[-6:], pred_inv)

# Transformer Forecast
def forecast_transformer(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    seq_len = 6
    dataset = TimeSeriesDataset(scaled, seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TransformerModel(seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(100):
        for x_batch, y_batch in loader:
            x_batch = x_batch.unsqueeze(-1)
            output = model(x_batch.transpose(0,1))
            loss = criterion(output.squeeze(), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        seq = torch.FloatTensor(scaled[-seq_len:]).unsqueeze(-1).unsqueeze(1)
        preds = []
        for _ in range(6):
            out = model(seq.transpose(0,1))
            preds.append(out.item())
            seq = torch.cat((seq[1:], out.unsqueeze(0).unsqueeze(1)), dim=0)

    pred_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return pd.Series(pred_inv, index=pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')), evaluate(series[-6:], pred_inv)
