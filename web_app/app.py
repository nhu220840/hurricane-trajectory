import torch
import pickle
import numpy as np
import sys
from flask import Flask, render_template, jsonify, request
from pathlib import Path
import pandas as pd
import random

# --- PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"  # (NEW) Add RAW data directory path

# Import from src/
from src.models import LSTMForecaster, LSTMFromScratchForecaster
from src.config import (
    LSTM_TORCH, LSTM_SCRATCH,
    NUMERIC_X, SEED,
    PROCESSED_NPZ, SCALER_Y_PKL, PREPROCESSOR_X_PKL,  # Import file paths
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH,  # Import checkpoint paths
    RAW_CSV  # (NEW) Import RAW_CSV path
)

# --- INITIALIZE FLASK ---
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- GLOBAL VARIABLE TO STORE ARTIFACTS ---
artifacts = {}


# Updated load_artifacts function
def load_artifacts():
    """Load models, scalers, test data, AND raw data into RAM."""
    print("Starting to load artifacts...")

    # 1. Load TEST data (for demo prediction)
    # (MODIFIED: Load pre-split arrays from the new .npz file)
    try:
        data_path = DATA_PROCESSED_DIR / PROCESSED_NPZ.name
        if not data_path.exists():
            print(f"ERROR: File not found: {data_path}")
            print("Please run 'python main.py --process-data' first.")
            return False

        data = np.load(data_path, allow_pickle=True)

        X_test = data["X_test"]
        Y_test = data["Y_test"]  # Needed to determine out_dim
        artifacts["X_test"] = X_test
        artifacts["last_obs_test"] = data["last_obs_test"]
        print(f"Loaded {len(X_test)} test samples into RAM.")

    except Exception as e:
        print(f"ERROR while loading test data (.npz): {e}")
        print("The .npz file might be corrupted or in the wrong format. Please re-run --process-data.")
        return False

    # 2. Load Scalers
    try:
        scaler_y_path = DATA_PROCESSED_DIR / SCALER_Y_PKL.name
        with open(scaler_y_path, "rb") as f:
            artifacts["scaler_y"] = pickle.load(f)

        preprocessor_x_path = DATA_PROCESSED_DIR / PREPROCESSOR_X_PKL.name
        with open(preprocessor_x_path, "rb") as f:
            preprocessor_x = pickle.load(f)

        artifacts["num_scaler"] = preprocessor_x.named_transformers_['num']
        artifacts["num_features_count"] = len(NUMERIC_X)

        indices = {}
        for feat in ['lat', 'lon']:
            if feat in NUMERIC_X:
                indices[feat] = NUMERIC_X.index(feat)
        artifacts["feat_indices"] = indices

    except Exception as e:
        print(f"ERROR while loading scalers (.pkl): {e}")
        return False

    # 3. Load Models
    try:
        input_size = X_test.shape[-1]
        out_dim = Y_test.shape[-1]

        model_torch = LSTMForecaster(
            input_size,
            LSTM_TORCH["hidden_size"],
            LSTM_TORCH["num_layers"],
            LSTM_TORCH["dropout"]
        )
        ckpt_torch = MODELS_DIR / CHECKPOINT_LSTM_TORCH.name

        model_scratch = LSTMFromScratchForecaster(
            input_size,
            LSTM_SCRATCH["hidden_size"],
            LSTM_SCRATCH["num_layers"],
            out_dim,
            LSTM_SCRATCH["dropout"]
        )
        ckpt_scratch = MODELS_DIR / CHECKPOINT_LSTM_SCRATCH.name

        if not ckpt_torch.exists() or not ckpt_scratch.exists():
            print(f"ERROR: Model files not found in {MODELS_DIR}")
            print("Please run 'python main.py --train' first.")
            return False

        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))
        artifacts["model_torch"] = model_torch.to(device).eval()
        artifacts["model_scratch"] = model_scratch.to(device).eval()
    except Exception as e:
        print(f"ERROR while loading models (.pt): {e}")
        return False

    # 4. Load RAW data (for overview map)
    try:
        print("Loading raw data (ibtracs_track_ml.csv)...")
        raw_data_path = DATA_RAW_DIR / RAW_CSV.name
        if not raw_data_path.exists():
            print(f"WARNING: Raw data file not found at {raw_data_path}")
            print("The 'Overview Map' feature will be disabled.")
            artifacts["raw_storm_data"] = None  # Continue running anyway
        else:
            df_raw = pd.read_csv(raw_data_path)
            # Convert data types immediately
            df_raw['time'] = pd.to_datetime(df_raw['time'], errors='coerce')
            df_raw['wind'] = pd.to_numeric(df_raw['wind'], errors='coerce')
            df_raw['pres'] = pd.to_numeric(df_raw['pres'], errors='coerce')
            artifacts["raw_storm_data"] = df_raw
            print(f"Loaded {df_raw['sid'].nunique()} storms (raw) into RAM.")
    except Exception as e:
        print(f"ERROR while loading raw data: {e}")
        artifacts["raw_storm_data"] = None

    print("...Artifacts loaded successfully!")
    return True


# === WEB ROUTES ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/get_test_samples")
def get_test_samples():
    if "X_test" not in artifacts or artifacts["X_test"] is None:
        return jsonify({"error": "Test data has not been loaded"}), 500

    rng = np.random.default_rng()
    sample_indices = rng.choice(len(artifacts["X_test"]), 5, replace=False)
    samples = []
    for idx in sample_indices:
        lat, lon = artifacts["last_obs_test"][idx]
        samples.append({
            "id": int(idx),
            "name": f"Case Study #{idx} (Start: {lat:.1f}, {lon:.1f})"
        })
    return jsonify(samples)


@app.route("/api/predict")
def predict():
    sample_id = request.args.get("sample_id", default=0, type=int)
    print(f"Received prediction request for sample_id: {sample_id}")

    if "X_test" not in artifacts or sample_id >= len(artifacts["X_test"]):
        return jsonify({"error": "Invalid sample_id"}), 400

    input_window_scaled = artifacts["X_test"][sample_id]
    start_coord = artifacts["last_obs_test"][sample_id]
    true_coord_11th = artifacts["last_obs_test"][sample_id + 1] if sample_id + 1 < len(
        artifacts["last_obs_test"]) else start_coord

    num_scaler = artifacts["num_scaler"]
    num_count = artifacts["num_features_count"]
    idx_lat = artifacts["feat_indices"]['lat']
    idx_lon = artifacts["feat_indices"]['lon']

    history_scaled_nums = input_window_scaled[:, :num_count]
    history_unscaled_nums = num_scaler.inverse_transform(history_scaled_nums)
    history_coords = list(zip(history_unscaled_nums[:, idx_lat], history_unscaled_nums[:, idx_lon]))

    input_tensor = torch.tensor(input_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    scaler_y = artifacts["scaler_y"]

    with torch.no_grad():
        delta_torch = artifacts["model_torch"](input_tensor)
        delta_deg_torch = scaler_y.inverse_transform(delta_torch.cpu().numpy())[0]
        delta_scratch = artifacts["model_scratch"](input_tensor)
        delta_deg_scratch = scaler_y.inverse_transform(delta_scratch.cpu().numpy())[0]

    pred_coord_torch_np = (start_coord[0] + delta_deg_torch[0], start_coord[1] + delta_deg_torch[1])
    pred_coord_scratch_np = (start_coord[0] + delta_deg_scratch[0], start_coord[1] + delta_deg_scratch[1])

    history_coords_py = [[float(lat), float(lon)] for lat, lon in history_coords]
    start_point_py = [float(start_coord[0]), float(start_coord[1])]
    true_point_py = [float(true_coord_11th[0]), float(true_coord_11th[1])]
    pred_torch_py = [float(pred_coord_torch_np[0]), float(pred_coord_torch_np[1])]
    pred_scratch_py = [float(pred_coord_scratch_np[0]), float(pred_coord_scratch_np[1])]

    return jsonify({
        "history_coords": history_coords_py,
        "start_point": start_point_py,
        "true_point": true_point_py,
        "pred_torch": pred_torch_py,
        "pred_scratch": pred_scratch_py
    })


# --- API ENDPOINT FOR OVERVIEW MAP ---
@app.route("/api/get_all_tracks")
def get_all_tracks():
    """
    Process and return ALL storm data as JSON.
    """
    print("Received request for /api/get_all_tracks")
    if "raw_storm_data" not in artifacts or artifacts["raw_storm_data"] is None:
        print("ERROR: Raw data not loaded on the server.")
        return jsonify({"error": "Raw data not loaded on the server."}), 500

    df = artifacts["raw_storm_data"]
    grouped = df.groupby('sid')
    all_storms_json = []

    for sid, group in grouped:
        random_color = f"#{random.randint(0, 0xFFFFFF):06x}"

        points = []
        for _, row in group.iterrows():
            points.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "time": row['time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['time']) else 'N/A',
                "wind": row["wind"] if pd.notna(row["wind"]) else None,
                "pres": row["pres"] if pd.notna(row["pres"]) else None,
            })

        all_storms_json.append({
            "sid": sid,
            "color": random_color,
            "points": points,
            "count": len(points)
        })

    print(f"Sending data of {len(all_storms_json)} storms to client.")
    return jsonify(all_storms_json)  # Send JSON response


# --- RUN APPLICATION ---
if __name__ == "__main__":
    if load_artifacts():
        app.run(debug=True, port=5000)
    else:
        print("Server could not start due to missing artifacts.")
