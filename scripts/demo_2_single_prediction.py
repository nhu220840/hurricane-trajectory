# scripts/demo_2_single_prediction.py
# (This script demonstrates a single 1-step prediction)
import torch
import pickle
import numpy as np
import folium
from pathlib import Path
import sys
import warnings

# Add repository root to path to import src
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Import everything from /src
from src.config import (
    PROCESSED_NPZ, SCALER_Y_PKL, PREPROCESSOR_X_PKL,
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH,
    N_IN, FEATURES_X, NUMERIC_X, CATEGORICAL_X, PLOTS_DIR,
    LSTM_TORCH, LSTM_SCRATCH
)
from src.models import LSTMForecaster, LSTMFromScratchForecaster

# (REMOVED) from src.train import _split_by_sid, _filter_by_sid_idx, SEED


# ===== DEMO CONFIGURATION =====
# Select a sample from the TEST SET to demonstrate
# (Change this number to try different storm tracks)
SAMPLE_INDEX_IN_TEST_SET = 100

OUT_HTML = PLOTS_DIR / f"demo_2_single_prediction_sample_{SAMPLE_INDEX_IN_TEST_SET}.html"


# ==============================


def get_model_and_checkpoint(choice, input_size, out_dim):
    """Loads the correct model class and corresponding checkpoint path."""
    if choice == "pytorch":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=LSTM_TORCH["hidden_size"],
            num_layers=LSTM_TORCH["num_layers"],
            dropout=LSTM_TORCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_TORCH
    elif choice == "scratch":
        model = LSTMFromScratchForecaster(
            in_dim=input_size,
            hidden=LSTM_SCRATCH["hidden_size"],
            num_layers=LSTM_SCRATCH["num_layers"],
            out_dim=out_dim,
            dropout=LSTM_SCRATCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_SCRATCH
    else:
        raise ValueError("Invalid model choice. Must be 'pytorch' or 'scratch'.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Have you run 'python main.py --train'?")

    return model, ckpt_path


def get_feature_indices(preprocessor, numeric_features):
    """Gets the (relative) index of lat/lon columns within the numeric block."""
    try:
        # Get the 'num' transformer from the ColumnTransformer
        num_transformer = preprocessor.named_transformers_['num']
        # Get the feature names *as seen by the num_transformer*
        num_features_list = num_transformer.feature_names_in_
    except Exception:
        # Fallback if feature_names_in_ is not available (older scikit-learn)
        print("Warning: Could not get feature_names_in_ from transformer. Falling back to NUMERIC_X.")
        num_features_list = numeric_features

    indices = {}
    for feat in ['lat', 'lon']:
        if feat in num_features_list:
            # Find the index of the feature
            indices[feat] = list(num_features_list).index(feat)

    if 'lat' not in indices or 'lon' not in indices:
        raise ValueError("Could not find 'lat'/'lon' in the preprocessor's numeric features. Check config.py.")

    print(f"Feature indices found (relative to NUMERIC block): {indices}")
    return indices


def main():
    print("Loading data, scalers, and models...")

    # 1. Load data and scalers
    try:
        data = np.load(PROCESSED_NPZ, allow_pickle=True)
        # (REVISED) Load pre-split test data directly
        X_test = data["X_test"]
        Y_test = data["Y_test"]
        last_obs_test = data["last_obs_test"]
        print(f"Loaded {len(X_test)} samples from the test set.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {PROCESSED_NPZ}. Have you run 'python main.py --process-data'?")
    except KeyError as e:
        raise KeyError(f"Missing array {e} in {PROCESSED_NPZ}. Please re-run data processing.")

    with open(SCALER_Y_PKL, "rb") as f:
        scaler_y = pickle.load(f)
    with open(PREPROCESSOR_X_PKL, "rb") as f:
        preprocessor_x = pickle.load(f)

    # 2. (REMOVED) Splitting logic (no longer needed)

    # 3. Load both models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_test.shape[-1]
    out_dim = Y_test.shape[-1]
    print(f"Using device: {device}. Input features: {input_size}, Output features: {out_dim}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Ignore warnings if any
        # Model 1: PyTorch
        model_torch, ckpt_torch = get_model_and_checkpoint("pytorch", input_size, out_dim)
        # (NEW) Use weights_only=True for safer loading
        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_torch.to(device).eval()
        print(f"Loaded 'pytorch' model from {ckpt_torch}.")

        # Model 2: Scratch
        model_scratch, ckpt_scratch = get_model_and_checkpoint("scratch", input_size, out_dim)
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))
        model_scratch.to(device).eval()
        print(f"Loaded 'scratch' model from {ckpt_scratch}.")

    # 4. Get Case Study data
    if SAMPLE_INDEX_IN_TEST_SET >= len(X_test):
        print(
            f"ERROR: SAMPLE_INDEX_IN_TEST_SET ({SAMPLE_INDEX_IN_TEST_SET}) is out of bounds for test set (size {len(X_test)}).")
        print("Using sample 0 instead.")
        SAMPLE_INDEX_IN_TEST_SET = 0

    # Get ONE 10-step window (scaled input)
    input_window_scaled = X_test[SAMPLE_INDEX_IN_TEST_SET]

    # Get ONE starting coordinate (point #10, unscaled)
    start_coord = last_obs_test[SAMPLE_INDEX_IN_TEST_SET]
    start_lat, start_lon = start_coord

    # Get ONE "ground truth" coordinate (point #11, unscaled)
    # (NEW) Add boundary check
    if SAMPLE_INDEX_IN_TEST_SET + 1 < len(last_obs_test):
        true_coord_11th = last_obs_test[SAMPLE_INDEX_IN_TEST_SET + 1]
    else:
        print(
            f"Warning: This is the last sample. Cannot get ground truth for point #11. Using start point as placeholder.")
        true_coord_11th = start_coord  # Fallback

    print(f"Loaded Case Study: Sample {SAMPLE_INDEX_IN_TEST_SET} (Using 10 points to predict point 11)")
    print(f"  Start point (10): ({start_lat:.2f}, {start_lon:.2f})")
    print(f"  Ground Truth (11): ({true_coord_11th[0]:.2f}, {true_coord_11th[1]:.2f})")

    # 5. Prepare parameters for history reconstruction
    feat_indices = get_feature_indices(preprocessor_x, NUMERIC_X)
    num_scaler = preprocessor_x.named_transformers_['num']
    # (NEW) Get the number of numeric features from the scaler itself
    num_features_count = num_scaler.n_features_in_

    # 6. RUN PREDICTION (1-step-only)
    input_tensor = torch.tensor(input_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Run PyTorch model
        pred_deltas_scaled_torch = model_torch(input_tensor)
        pred_deltas_deg_torch = scaler_y.inverse_transform(pred_deltas_scaled_torch.cpu().numpy())[0]

        # Run Scratch model
        pred_deltas_scaled_scratch = model_scratch(input_tensor)
        pred_deltas_deg_scratch = scaler_y.inverse_transform(pred_deltas_scaled_scratch.cpu().numpy())[0]

    # 7. Reconstruct predicted coordinates (point 11)
    pred_lat_torch = start_lat + pred_deltas_deg_torch[0]
    pred_lon_torch = start_lon + pred_deltas_deg_torch[1]
    pred_coord_torch = (pred_lat_torch, pred_lon_torch)

    pred_lat_scratch = start_lat + pred_deltas_deg_scratch[0]
    pred_lon_scratch = start_lon + pred_deltas_deg_scratch[1]
    pred_coord_scratch = (pred_lat_scratch, pred_lon_scratch)

    print(f"  PyTorch Prediction (11): ({pred_lat_torch:.2f}, {pred_lon_torch:.2f})")
    print(f"  Scratch Prediction (11): ({pred_lat_scratch:.2f}, {pred_lon_scratch:.2f})")

    # 8. Get 10 history points (for plotting)
    history_scaled_nums = input_window_scaled[:, :num_features_count]
    history_unscaled_nums = num_scaler.inverse_transform(history_scaled_nums)
    idx_lat = feat_indices['lat']
    idx_lon = feat_indices['lon']
    history_coords = list(zip(history_unscaled_nums[:, idx_lat], history_unscaled_nums[:, idx_lon]))

    # 9. PLOT MAP
    print("Generating Folium map...")
    start_point = tuple(start_coord)  # Point 10
    true_point_11th = tuple(true_coord_11th)  # Point 11

    m = folium.Map(location=start_point, zoom_start=7, tiles="CartoDB positron")

    # 1. History track (gray)
    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="History (10 points)").add_to(m)

    # 2. Start point (point 10)
    folium.Marker(start_point,
                  tooltip="Prediction Start (Point 10)",
                  icon=folium.Icon(color="green", icon="play", prefix='fa')
                  ).add_to(m)

    # 3. Ground Truth point (Navy)
    folium.CircleMarker(true_point_11th, radius=7, color='navy', fill=True,
                        tooltip=f"Ground Truth (Point 11)\n({true_point_11th[0]:.2f}, {true_point_11th[1]:.2f})").add_to(
        m)
    folium.PolyLine([start_point, true_point_11th], color='navy', weight=2,
                    tooltip="Actual Delta").add_to(m)

    # 4. PyTorch Prediction (Red)
    folium.CircleMarker(pred_coord_torch, radius=7, color='red', fill=True,
                        tooltip=f"PyTorch Prediction (Point 11)\n({pred_coord_torch[0]:.2f}, {pred_coord_torch[1]:.2f})").add_to(
        m)
    folium.PolyLine([start_point, pred_coord_torch], color='red', weight=2, dash_array="5, 5",
                    tooltip="PyTorch Predicted Delta").add_to(m)

    # 5. Scratch Prediction (Orange)
    folium.CircleMarker(pred_coord_scratch, radius=7, color='orange', fill=True,
                        tooltip=f"Scratch Prediction (Point 11)\n({pred_coord_scratch[0]:.2f}, {pred_coord_scratch[1]:.2f})").add_to(
        m)
    folium.PolyLine([start_point, pred_coord_scratch], color='orange', weight=2, dash_array="5, 5",
                    tooltip="Scratch Predicted Delta").add_to(m)

    # 10. Save file
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\n[SUCCESS] Saved 1-point prediction map to:")
    print(f"{OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()