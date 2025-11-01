import torch
import pickle
import numpy as np
import warnings
import sys
from flask import Flask, render_template, jsonify, request
from pathlib import Path

# --- (NEW) PATH CONFIGURATION ---
# 1. Add the project root directory to the system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 2. Define paths to directories containing "artifacts"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
# --- END OF NEW SECTION ---

# (CHANGED) Import from src/
from src.models import LSTMForecaster, LSTMFromScratchForecaster
from src.config import (
    LSTM_TORCH, LSTM_SCRATCH,
    NUMERIC_X, SEED,  # SEED is no longer needed for splitting here, but good to keep
    PROCESSED_NPZ, SCALER_Y_PKL, PREPROCESSOR_X_PKL,  # (NEW) Import paths
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH  # (NEW) Import paths
)

# (REMOVED) _split_by_sid - Logic moved to data_processing.py
# (REMOVED) _filter_by_sid_idx - Logic moved to data_processing.py


# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- GLOBAL VARIABLE TO HOLD ARTIFACTS ---
artifacts = {}


# (REVISED) Update load_artifacts function
def load_artifacts():
    """Load models, scalers, and test data into RAM."""
    print("Starting to load artifacts...")

    # 1. Load data
    data_path = DATA_PROCESSED_DIR / PROCESSED_NPZ.name  # Use config name
    if not data_path.exists():
        print(f"ERROR: File not found {data_path}")
        print("Please make sure you have run processing (e.g., python main.py --process-data)")
        return False

    try:
        data = np.load(data_path, allow_pickle=True)
        # (REVISED) Load only the pre-split test data
        X_test = data["X_test"]
        Y_test = data["Y_test"]  # Needed for out_dim
        last_obs_test = data["last_obs_test"]
    except KeyError as e:
        print(f"ERROR: Missing expected array {e} in {data_path}.")
        print("The .npz file might be old or corrupted. Please re-run data processing.")
        return False

    artifacts["X_test"] = X_test
    artifacts["last_obs_test"] = last_obs_test
    print(f"Loaded {len(X_test)} test samples into RAM.")

    # 2. Load scalers
    scaler_y_path = DATA_PROCESSED_DIR / SCALER_Y_PKL.name
    preprocessor_x_path = DATA_PROCESSED_DIR / PREPROCESSOR_X_PKL.name

    if not scaler_y_path.exists() or not preprocessor_x_path.exists():
        print(f"ERROR: Scaler file not found in {DATA_PROCESSED_DIR}")
        print("Please ensure scalers were saved correctly during data processing.")
        return False

    try:
        with open(scaler_y_path, "rb") as f:
            artifacts["scaler_y"] = pickle.load(f)
        with open(preprocessor_x_path, "rb") as f:
            preprocessor_x = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler/preprocessor: {e}")
        return False

    # Separate num_scaler (important)
    try:
        artifacts["num_scaler"] = preprocessor_x.named_transformers_['num']
    except KeyError:
        print("ERROR: Could not find 'num' transformer in preprocessor_x.")
        print("Check if NUMERIC_X in config.py matches data processing.")
        return False

    artifacts["num_features_count"] = len(NUMERIC_X)

    # Get indices for lat/lon
    indices = {}
    for feat in ['lat', 'lon']:
        if feat in NUMERIC_X:
            indices[feat] = NUMERIC_X.index(feat)
    if 'lat' not in indices or 'lon' not in indices:
        print("ERROR: 'lat' or 'lon' not found in NUMERIC_X config. Cannot get history.")
        return False
    artifacts["feat_indices"] = indices

    # 3. (REMOVED) Split test set (No longer needed)

    # 4. Load models
    input_size = X_test.shape[-1]
    out_dim = Y_test.shape[-1]  # Get out_dim from loaded Y_test

    # Model 1: PyTorch
    model_torch = LSTMForecaster(input_size, LSTM_TORCH["hidden_size"], LSTM_TORCH["num_layers"], LSTM_TORCH["dropout"])
    ckpt_torch = MODELS_DIR / CHECKPOINT_LSTM_TORCH.name

    # Model 2: Scratch
    model_scratch = LSTMFromScratchForecaster(
        in_dim=input_size,  # Use in_dim
        hidden=LSTM_SCRATCH["hidden_size"],
        num_layers=LSTM_SCRATCH["num_layers"],
        out_dim=out_dim,
        dropout=LSTM_SCRATCH["dropout"]
    )
    ckpt_scratch = MODELS_DIR / CHECKPOINT_LSTM_SCRATCH.name

    if not ckpt_torch.exists() or not ckpt_scratch.exists():
        print(f"ERROR: Model file not found in {MODELS_DIR} (checked for {ckpt_torch.name}, {ckpt_scratch.name})")
        print("Please make sure you have run training (e.g., python main.py --train)")
        return False

    try:
        # (NEW) Use weights_only=True for safer loading of .pt files
        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("This can happen if the model architecture in src/models.py does not match the saved checkpoint.")
        return False

    artifacts["model_torch"] = model_torch.to(device).eval()
    artifacts["model_scratch"] = model_scratch.to(device).eval()

    print("...Artifacts loaded successfully!")
    return True


# === WEB ROUTES ===
@app.route("/")
def index():
    """Serve the main web page (index.html)."""
    return render_template("index.html")


@app.route("/api/get_test_samples")
def get_test_samples():
    """Send a list of test sample IDs for the user to choose."""
    # Example: Send 5 random samples
    if "X_test" not in artifacts:
        return jsonify({"error": "Artifacts not loaded"}), 500

    rng = np.random.default_rng()
    num_samples = len(artifacts["X_test"])
    if num_samples == 0:
        return jsonify({"error": "No test samples loaded"}), 500

    # Ensure we don't request more samples than available
    k = min(5, num_samples)

    sample_indices = rng.choice(num_samples, k, replace=False)

    # Get starting coordinates for display
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
    """Run prediction for a sample and return coordinates (JSON)."""
    # Get sample_id from URL (e.g., /api/predict?sample_id=150)
    sample_id = request.args.get("sample_id", default=0, type=int)

    print(f"Received prediction request for sample_id: {sample_id}")

    # --- (NEW) Add boundary check ---
    if "X_test" not in artifacts or sample_id >= len(artifacts["X_test"]):
        print(f"ERROR: Invalid sample_id: {sample_id}")
        return jsonify({"error": "Invalid sample_id"}), 400

    # Get preloaded data
    try:
        input_window_scaled = artifacts["X_test"][sample_id]
        start_coord = artifacts["last_obs_test"][sample_id]

        # Check for the next point (true coord)
        # Handle edge case: if it's the very last sample, we can't get the next true coord
        if sample_id + 1 < len(artifacts["last_obs_test"]):
            true_coord_11th = artifacts["last_obs_test"][sample_id + 1]
        else:
            true_coord_11th = start_coord  # Fallback: use start_coord
    except Exception as e:
        print(f"Error accessing test data at index {sample_id}: {e}")
        return jsonify({"error": "Error accessing test data"}), 500

    # Get 10 history points (for plotting)
    num_scaler = artifacts["num_scaler"]
    num_count = artifacts["num_features_count"]
    idx_lat = artifacts["feat_indices"]['lat']
    idx_lon = artifacts["feat_indices"]['lon']

    # (NEW) Check if input_window_scaled has the expected shape
    if input_window_scaled.ndim != 2 or input_window_scaled.shape[1] < num_count:
        print(f"ERROR: input_window_scaled has unexpected shape: {input_window_scaled.shape}")
        return jsonify({"error": "Mismatch in data shape"}), 500

    history_scaled_nums = input_window_scaled[:, :num_count]
    history_unscaled_nums = num_scaler.inverse_transform(history_scaled_nums)
    history_coords = list(zip(history_unscaled_nums[:, idx_lat], history_unscaled_nums[:, idx_lon]))

    # Run prediction
    input_tensor = torch.tensor(input_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    scaler_y = artifacts["scaler_y"]

    with torch.no_grad():
        # PyTorch
        delta_torch = artifacts["model_torch"](input_tensor)
        delta_deg_torch = scaler_y.inverse_transform(delta_torch.cpu().numpy())[0]

        # Scratch
        delta_scratch = artifacts["model_scratch"](input_tensor)
        delta_deg_scratch = scaler_y.inverse_transform(delta_scratch.cpu().numpy())[0]

    # Calculate predicted coordinates
    pred_coord_torch_np = (start_coord[0] + delta_deg_torch[0], start_coord[1] + delta_deg_torch[1])
    pred_coord_scratch_np = (start_coord[0] + delta_deg_scratch[0], start_coord[1] + delta_deg_scratch[1])

    # Convert ALL to basic Python data types (float) so jsonify can handle it
    history_coords_py = [[float(lat), float(lon)] for lat, lon in history_coords]
    start_point_py = [float(start_coord[0]), float(start_coord[1])]
    true_point_py = [float(true_coord_11th[0]), float(true_coord_11th[1])]
    pred_torch_py = [float(pred_coord_torch_np[0]), float(pred_coord_torch_np[1])]
    pred_scratch_py = [float(pred_coord_scratch_np[0]), float(pred_coord_scratch_np[1])]

    # Return all coordinates as JSON
    return jsonify({
        "history_coords": history_coords_py,
        "start_point": start_point_py,
        "true_point": true_point_py,
        "pred_torch": pred_torch_py,
        "pred_scratch": pred_scratch_py
    })


# --- RUN APPLICATION ---
if __name__ == "__main__":
    # Add check if artifacts loaded successfully
    if load_artifacts():
        # Use port 5000 (standard for Flask development)
        app.run(debug=True, port=5001)
    else:
        print("Could not start server due to missing artifacts.")