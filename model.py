import json
import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKENS, MODEL, CG_API_KEY


binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files, data_provider, token):
    price_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, header=None)  # 读取文件时不使用标题行
        price_df = pd.concat([price_df, df])

    # 设置列名
    print('before',price_df.columns)
    column_names = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    price_df.columns = column_names

    # 添加 date 列
    price_df['date'] = pd.to_datetime(price_df['open_time'], unit='ms')
    print('after',price_df.columns)
    # 保存处理后的数据
    output_file = f"{training_price_data_path}_{token}.csv"
    price_df.to_csv(output_file, index=False)
    print(f"Formatted data saved for token: {token}")
    print("Formatted data head:")
    print(price_df.head())
    print("\nFormatted data description:")
    print(price_df.describe())

def load_frame(frame, timeframe):
    #print("Loading data...")
    #print(frame.columns)
    
    required_columns = ['open', 'high', 'low', 'close', 'date']
    for col in required_columns:
        if col not in frame.columns:
            raise KeyError(f"Required column '{col}' not found in the data.")
    
    df = frame[required_columns].copy()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # 重采样数据
    df = df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()
    
    print(df.head())
    return df

def get_model(model_name, input_shape=None):
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "KernelRidge": KernelRidge(),
        "BayesianRidge": BayesianRidge(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "ElasticNet": ElasticNet(max_iter=10000, tol=1e-4),  # Increased iterations and lowered tolerance
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }
    if model_name == "LSTM":
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    return models.get(model_name)

def create_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(30, 4), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(timeframe, model_name, token):
    file_path = f"{training_price_data_path}_{token}.csv"
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"File not found or empty: {file_path}")
    
    price_data = pd.read_csv(file_path)
    df = load_frame(price_data, timeframe)

    scaler = StandardScaler()
    
    X = df[['open', 'high', 'low', 'close']]
    y = df['close']
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_prepared, y_prepared = prepare_data(X_scaled)

    if model_name == "LSTM":
        model = get_model(model_name, input_shape=(X_prepared.shape[1], X_prepared.shape[2]))
        model.fit(X_prepared, y_prepared, epochs=50, batch_size=32, verbose=0)
    else:
        model = get_model(model_name)
        model.fit(X_prepared.reshape(X_prepared.shape[0], -1), y_prepared)

    return model, scaler

def prepare_data(df, n_steps=1):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:(i + n_steps)].values)
        y.append(df['close'].iloc[i + n_steps])
    return np.array(X), np.array(y)

def get_inference(token, timeframe, region, data_provider, model, scaler):
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{token}USDT", region), timeframe)

    X_new = X_new[['open', 'high', 'low', 'close']]
    X_new_scaled = scaler.transform(X_new)
    X_new_prepared, _ = prepare_data(pd.DataFrame(X_new_scaled, columns=X_new.columns))

    if isinstance(model, tf.keras.Model):  # Check if the model is LSTM
        current_price_pred = model.predict(X_new_prepared)
    else:
        current_price_pred = model.predict(X_new_prepared.reshape(X_new_prepared.shape[0], -1))

    # Inverse transform the prediction
    current_price_pred = current_price_pred.reshape(-1, 1)
    current_price_pred = np.hstack([np.zeros((current_price_pred.shape[0], 3)), current_price_pred])
    current_price_pred = scaler.inverse_transform(current_price_pred)[:, -1]

    return current_price_pred[0]
