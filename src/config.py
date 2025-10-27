# src/config.py (Cập nhật)
from pathlib import Path

# --- Đường dẫn ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Cập nhật đường dẫn thư mục RAW
RAW_DATA_DIR = DATA_DIR / "raw"
# File CSV lớn (input cho bước 1)
RAW_IBTRACS_FILE = RAW_DATA_DIR / "ibtracs.last3years.list.v04r01.csv"
# File CSV đã cắt (output bước 1, input bước 2)
RAW_DATA_PATH = RAW_DATA_DIR / "ibtracs_track_ml.csv"

# Cập nhật đường dẫn thư mục PROCESSED
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_CSV_PATH = PROCESSED_DIR / "ibtracs_processed_v2.csv" # File tạm do bước 2 tạo ra
PROCESSED_NPZ_PATH = PROCESSED_DIR / "processed_data.npz"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"

MODEL_DIR = ROOT_DIR / "models"
CKPT_PATH_PYTORCH = MODEL_DIR / "best_lstm_pytorch.pt"
CKPT_PATH_SCRATCH = MODEL_DIR / "best_lstm_scratch.pt"

RESULTS_DIR = ROOT_DIR / "results"
MAP_DIR = RESULTS_DIR / "maps"
PLOT_DIR = RESULTS_DIR / "plots"
COMPARISON_PLOT_PATH = PLOT_DIR / "model_comparison.png"

# --- Cấu hình Huấn luyện ---
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# --- Cấu hình Model ---
MODEL_PARAMS = {
    "pytorch": {
        "hidden": 20,
        "num_layers": 1,
        "dropout": 0.2
    },
    "scratch": {
        "hidden": 20,
        "num_layers": 2,
        "dropout": 0.2
    }
}

# --- Cấu hình Đánh giá & Trực quan hóa ---
# Số bước dự đoán (ví dụ: 3 bước = 3h, 6h, 9h)
FORECAST_STEPS = 3
# Tên file cho bản đồ so sánh
MAP_COMPARISON_PATH = MAP_DIR / "model_comparison_map.html"