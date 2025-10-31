import torch
import pickle
import numpy as np
import warnings
import sys  # (NEW) Added to modify the system path
from flask import Flask, render_template, jsonify, request, url_for  # (NEW) Added url_for
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
    NUMERIC_X, SEED
)


def _split_by_sid(window_sid_idx, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    uniq = np.unique(window_sid_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_sids = set(uniq[:n_train])
    val_sids = set(uniq[n_train:n_train + n_val])
    test_sids = set(uniq[n_train + n_val:])
    return train_sids, val_sids, test_sids


def _filter_by_sid_idx(arr, sid_idx, keep_sids):
    mask = np.isin(sid_idx, list(keep_sids))
    return arr[mask], mask


# --- FLASK APP INITIALIZATION ---
# Flask will automatically find 'templates' and 'static' in the same directory (web_app/)
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- GLOBAL VARIABLE TO HOLD ARTIFACTS ---
artifacts = {}


# (CHANGED) Update load_artifacts function
def load_artifacts():
    """Load models, scalers, and test data into RAM."""
    print("Starting to load artifacts...")

    # 1. Load data
    data_path = DATA_PROCESSED_DIR / "processed_data.npz"
    if not data_path.exists():
        print(f"ERROR: File not found {data_path}")
        print("Please make sure you have run processing (e.g., src/data_processing.py)")
        return False

    data = np.load(data_path, allow_pickle=True)
    X_all, Y_all, last_obs_all, sid_idx_all = data["X"], data["Y"], data["last_obs_latlon"], data["window_sid_idx"]

    # 2. Load scalers
    scaler_y_path = DATA_PROCESSED_DIR / "scaler_y.pkl"
    preprocessor_x_path = DATA_PROCESSED_DIR / "preprocessor_x.pkl"

    if not scaler_y_path.exists() or not preprocessor_x_path.exists():
        print(f"ERROR: Scaler file not found in {DATA_PROCESSED_DIR}")
        return False

    with open(scaler_y_path, "rb") as f:
        artifacts["scaler_y"] = pickle.load(f)
    with open(preprocessor_x_path, "rb") as f:
        preprocessor_x = pickle.load(f)

    # Separate num_scaler (important)
    artifacts["num_scaler"] = preprocessor_x.named_transformers_['num']
    artifacts["num_features_count"] = len(NUMERIC_X)

    # Get indices for lat/lon
    indices = {}
    for feat in ['lat', 'lon']:
        if feat in NUMERIC_X:
            indices[feat] = NUMERIC_X.index(feat)
    artifacts["feat_indices"] = indices

    # 3. Split test set (only get test data)
    _, _, test_sids = _split_by_sid(sid_idx_all, seed=SEED)
    X_test, m_te = _filter_by_sid_idx(X_all, sid_idx_all, test_sids)
    artifacts["X_test"] = X_test
    artifacts["last_obs_test"] = last_obs_all[m_te]
    print(f"Loaded {len(X_test)} test samples into RAM.")

    # 4. Load models
    input_size = X_test.shape[-1]
    out_dim = Y_all.shape[-1]

    # Model 1: PyTorch
    model_torch = LSTMForecaster(input_size, LSTM_TORCH["hidden_size"], LSTM_TORCH["num_layers"], LSTM_TORCH["dropout"])
    ckpt_torch = MODELS_DIR / "best_lstm_pytorch.pt"

    # Model 2: Scratch
    model_scratch = LSTMFromScratchForecaster(input_size, LSTM_SCRATCH["hidden_size"], LSTM_SCRATCH["num_layers"],
                                              out_dim, LSTM_SCRATCH["dropout"])
    ckpt_scratch = MODELS_DIR / "best_lstm_scratch.pt"

    if not ckpt_torch.exists() or not ckpt_scratch.exists():
        print(f"ERROR: Model file not found in {MODELS_DIR}")
        print("Please make sure you have run training (e.g., src/train.py)")
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))

    artifacts["model_torch"] = model_torch.to(device).eval()
    artifacts["model_scratch"] = model_scratch.to(device).eval()

    print("...Artifacts loaded successfully!")
    return True  # (NEW) Added return True


# === WEB ROUTES ===
# (This entire section is kept from your original code)
@app.route("/")
def index():
    """Serve the main web page (index.html)."""
    return render_template("index.html")


@app.route("/api/get_test_samples")
def get_test_samples():
    """Send a list of test sample IDs for the user to choose."""
    # Example: Send 5 random samples
    rng = np.random.default_rng()
    sample_indices = rng.choice(len(artifacts["X_test"]), 5, replace=False)

    # Get starting coordinates for display
    samples = []
    for idx in sample_indices:
        lat, lon = artifacts["last_obs_test"][idx]
        samples.append({
            "id": int(idx),
            "name": f"Case Study #{idx} (Starting at {lat:.1f}, {lon:.1f})"
        })
    return jsonify(samples)


@app.route("/api/predict")
def predict():
    """Run prediction for a sample and return coordinates (JSON)."""
    # Get sample_id from URL (e.g., /api/predict?sample_id=150)
    sample_id = request.args.get("sample_id", default=150, type=int)

    print(f"Received prediction request for sample_id: {sample_id}")

    # Get preloaded data
    input_window_scaled = artifacts["X_test"][sample_id]
    start_coord = artifacts["last_obs_test"][sample_id]
    # (NEW) Add boundary check
    true_coord_11th = artifacts["last_obs_test"][sample_id + 1] if sample_id + 1 < len(
        artifacts["last_obs_test"]) else start_coord

    # Get 10 history points (for plotting)
    num_scaler = artifacts["num_scaler"]
    num_count = artifacts["num_features_count"]
    idx_lat = artifacts["feat_indices"]['lat']
    idx_lon = artifacts["feat_indices"]['lon']

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

    # 8. Convert ALL to basic Python data types (float)
    #    so jsonify can handle it
    history_coords_py = [[float(lat), float(lon)] for lat, lon in history_coords]
    start_point_py = [float(start_coord[0]), float(start_coord[1])]
    true_point_py = [float(true_coord_11th[0]), float(true_coord_11th[1])]
    pred_torch_py = [float(pred_coord_torch_np[0]), float(pred_coord_torch_np[1])]
    pred_scratch_py = [float(pred_coord_scratch_np[0]), float(pred_coord_scratch_np[1])]

    # 9. Return all coordinates as JSON
    return jsonify({
        "history_coords": history_coords_py,
        "start_point": start_point_py,
        "true_point": true_point_py,
        "pred_torch": pred_torch_py,
        "pred_scratch": pred_scratch_py
    })


# --- RUN APPLICATION ---
if __name__ == "__main__":
    # (NEW) Add check if artifacts loaded successfully
    if load_artifacts():
        app.run(debug=True, port=5000)
    else:
        print("Could not start server due to missing artifacts.")