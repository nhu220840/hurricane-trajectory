# src/config.py
from pathlib import Path

# ----- PATHS -----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ibtracs_track_ml.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- SỬA LỖI Ở ĐÂY ---
# Xóa "/processed" bị trùng lặp
PROCESSED_DATA_NPZ = PROCESSED_DATA_DIR / "processed_data.npz"
# --- KẾT THÚC SỬA LỖI ---

# Sử dụng 2 scaler
SCALER_X_PKL = PROCESSED_DATA_DIR / "scaler_X.pkl"
SCALER_Y_PKL = PROCESSED_DATA_DIR / "scaler_y.pkl"

MODEL_DIR = ROOT_DIR / "models"
MODEL_CKPT_PATH = MODEL_DIR / "best_lstm_attention_model.pt"
RESULT_DIR = ROOT_DIR / "results" / "real_forecasts"

# ----- DATA PROCESSING -----
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
INPUT_TIMESTEPS = 10
OUTPUT_TIMESTEPS = 1

NUMERICAL_FEATURES = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND']
CATEGORICAL_FEATURES = ['BASIN', 'NATURE']

# Mục tiêu: Dự đoán delta của các đặc trưng vật lý
FINAL_INPUT_FEATURES = []
TARGET_FEATURES = [
    'LAT_delta',
    'LON_delta',
    'WMO_WIND_delta',
    'WMO_PRES_delta',
    'DIST2LAND_delta'
]

# ----- MODEL & TRAINING -----
BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MODEL_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3
}

# ----- FORECASTING (THEO YÊU CẦU CỦA BẠN) -----
N_STEPS_TO_FORECAST = 8     # 8 BƯỚC DỰ ĐOÁN
TIME_STEP_HOURS = 3         # MỖI BƯỚC CÁCH NHAU 3 GIỜ
# (Tổng cộng: 8 * 3 = 24 giờ)
TEST_INDICES_TO_VISUALIZE = [0, 50, 150]