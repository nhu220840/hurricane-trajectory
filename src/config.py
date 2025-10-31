# src/config.py

from pathlib import Path

# ========== Paths ==========
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
PLOTS_DIR = Path("results/plots")

RAW_CSV = RAW_DIR / "ibtracs_track_ml.csv"
PROCESSED_NPZ = PROCESSED_DIR / "processed_data.npz"
PREPROCESSOR_X_PKL = PROCESSED_DIR / "preprocessor_x.pkl"  # ColumnTransformer (num scaler + cat OHE)
SCALER_Y_PKL = PROCESSED_DIR / "scaler_y.pkl"              # separate scaler for y (delta)

CHECKPOINT_LSTM_TORCH = MODELS_DIR / "best_lstm_pytorch.pt"
CHECKPOINT_LSTM_SCRATCH = MODELS_DIR / "best_lstm_scratch.pt"

# ========== Data & Features ==========
# Keep the exact features you requested (renamed to lowercase in prepare_raw_data)
# - Numeric: lat, lon, wind, pres, dist2land
# - Categorical: basin
TIME_COLUMN = "time"
SID_COLUMN = "sid"

NUMERIC_X = ["lat", "lon", "wind", "pres", "dist2land"]
CATEGORICAL_X = ["basin"]

# Aggregated for easy reference
FEATURES_X = NUMERIC_X + CATEGORICAL_X

# FINAL: predict delta
TARGET_MODE = "delta"
TARGET_Y = ["delta_lat", "delta_lon"]

# ========== Windowing ==========
N_IN = 10
N_OUT = 1  # one-step delta

# ========== Training ==========
SEED = 1337
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
FACTOR = 0.5
CLIP_NORM = 1.0

# LSTM Torch
LSTM_TORCH = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
}

# LSTM From Scratch
LSTM_SCRATCH = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
}

# ========== Misc ==========
DEVICE = "cuda"  # falls back to cpu if cuda is not available