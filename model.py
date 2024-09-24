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
    print("Loading data...")
    print(frame.columns)
    
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

def get_model(model_name):
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
    return models.get(model_name)


def train_model(timeframe, model_name, token):
    file_path = f"{training_price_data_path}_{token}.csv"
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"File not found or empty: {file_path}")
    
    price_data = pd.read_csv(file_path)
    print(price_data.columns)
    print('xxxxx')
    df = load_frame(price_data, timeframe)

    print(df.tail())

    y_train = df['close'].shift(-1).dropna().values
    X_train = df[:-1]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Get the specified model
    model = get_model(model_name)
    if model is None:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Training model: {model_name} with data shape: {X_train_scaled.shape}")  # 添加调试输出

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def get_inference(token, timeframe, region, data_provider, model, scaler):
    """Predict current price using the provided model."""
    # Get current price
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{token}USDT", region), timeframe)

    print(X_new.tail())
    print(X_new.shape)

    # Scale the new data
    X_new_scaled = scaler.transform(X_new)

    current_price_pred = model.predict(X_new_scaled)

    return current_price_pred[0]
