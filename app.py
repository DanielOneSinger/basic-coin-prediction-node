import json
import os
import random
import pickle
import shutil
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKENS, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

def clean_data_directory():
    """Clean the data directory by removing all files and subdirectories."""
    data_dir = os.path.dirname(model_file_path)
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            shutil.rmtree(os.path.join(root, name))
    print("Data directory cleaned.")

# Clean data directory before starting the app
clean_data_directory()

app = Flask(__name__)

# Define the list of models we want to use
# skip:KernelRidge
MODELS = ["LinearRegression", "SVR", "BayesianRidge", "RandomForestRegressor",
          "GradientBoostingRegressor", "ElasticNet", "Lasso", "Ridge", "LSTM"]


def update_data():
    """Download price data, format data and train all models."""
    for token in TOKENS:
        files = download_data(token, TRAINING_DAYS, REGION, DATA_PROVIDER)
        format_data(files, DATA_PROVIDER, token)  # 确保传入 token

        for model_name in MODELS:
            model, scaler = train_model(TIMEFRAME, model_name, token)  # 添加 token 参数
            model_path = f"{model_file_path}_{model_name}_{token}"  # 确保路径正确
            with open(model_path, "wb") as f:
                pickle.dump((model, scaler), f)
            print(f"Trained model ({model_name}) for token ({token}) saved to {model_path}")

def get_random_model(token):
    """Randomly select and load a model for the given token, with 70% chance of selecting LSTM."""
    if random.random() < 0.4:
        model_name = "LSTM"
    else:
        model_name = random.choice([m for m in MODELS if m != "LSTM"])
    
    model_path = f"{model_file_path}_{model_name}_{token}"
    with open(model_path, "rb") as f:
        model, scaler = pickle.load(f)
    return model_name, model, scaler

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token using a randomly selected model."""
    if token.upper() not in TOKENS:
        error_msg = "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        model_name, model, scaler = get_random_model(token)
        print(f"Current Model: {model_name}")
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER, model, scaler)
        print(f"Current Model: {model_name}, inference: {str(inference)}")
        if model_name in ["LinearRegression", "SVR", "BayesianRidge","LSTM"]:
            random_factor = random.uniform(0.95, 1.05)
            adjusted_inference = inference * random_factor
            return Response(str(adjusted_inference), status=200)
        else:
            return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
