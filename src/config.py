from pathlib import Path

# --- Đường dẫn ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Cập nhật đường dẫn thư mục RAW
RAW_DATA_DIR = DATA_DIR / "raw"
# File CSV lớn (input cho bước 1) [cite: ibtracs.last3years.list.v04r01.csv]
RAW_IBTRACS_FILE = RAW_DATA_DIR / "ibtracs.last3years.list.v04r01.csv"
# File CSV đã cắt (output bước 1, input bước 2) [cite: hurricane-trajectory-develop/data/ibtracs_track_ml.csv]
RAW_DATA_PATH = RAW_DATA_DIR / "ibtracs_track_ml.csv"

# Cập nhật đường dẫn thư mục PROCESSED
PROCESSED_DIR = DATA_DIR / "processed"
# File tạm do bước 2 tạo ra [cite: hurricane-trajectory-develop/ibtracs_processed_interpolated_pipeline_v2.csv]
PROCESSED_CSV_PATH = PROCESSED_DIR / "ibtracs_processed_v2.csv"
PROCESSED_NPZ_PATH = PROCESSED_DIR / "processed_data.npz" # [cite: hurricane-trajectory-develop/data/processed_data.npz]
SCALER_PATH = PROCESSED_DIR / "scaler.pkl" # [cite: hurricane-trajectory-develop/data/scaler.pkl]

MODEL_DIR = ROOT_DIR / "models"
CKPT_PATH_PYTORCH = MODEL_DIR / "best_lstm_pytorch.pt"
CKPT_PATH_SCRATCH = MODEL_DIR / "best_lstm_scratch.pt"

RESULTS_DIR = ROOT_DIR / "results"
MAP_DIR = RESULTS_DIR / "maps"
PLOT_DIR = RESULTS_DIR / "plots"
COMPARISON_PLOT_PATH = PLOT_DIR / "model_comparison.png"

# === KHỐI CẬP NHẬT (ĐÃ THÊM FEATURE MỚI) ===
# (Phần này đã đúng, giữ nguyên)

# Các đặc trưng thô sẽ được giữ lại từ file CSV
RAW_FEATURES_TO_KEEP = {
    'SID': 'sid',
    'ISO_TIME': 'time',
    'LAT': 'lat',
    'LON': 'lon',
    'WMO_WIND': 'wind',
    'WMO_PRES': 'pres',
    'DIST2LAND': 'dist2land', # <-- FEATURE MỚI
    'BASIN': 'basin'        # <-- FEATURE MỚI
}

# Các đặc trưng số SẼ DÙNG
NUMERIC_FEATURES = ['lat', 'lon', 'wind', 'pres', 'dist2land'] # <-- THÊM dist2land

# Các đặc trưng phân loại SẼ DÙNG
CATEGORICAL_FEATURES = ['basin'] # <-- THÊM basin

# Các đặc trưng sẽ được dùng làm INPUT cho mô hình
MODEL_INPUT_FEATURES = [] # Sẽ được tạo tự động trong data_processing.py

# Các đặc trưng sẽ được dùng làm TARGET (tọa độ)
TARGET_COORDS_FEATURES = ['lat', 'lon']

# Tên của các đặc trưng DELTA (mục tiêu cuối cùng)
TARGET_DELTAS_FEATURES = ['delta_lat', 'delta_lon']
# === KẾT THÚC KHỐI CẬP NHẬT ===


# --- Cấu hình Huấn luyện (ĐÃ SỬA) ---
BATCH_SIZE = 64
EPOCHS = 100            # Tăng từ 40 -> 100
PATIENCE = 20           # Tăng từ 10 -> 20
LEARNING_RATE = 1e-4    # Giảm từ 1e-3 -> 1e-4 (Quan trọng nhất)
WEIGHT_DECAY = 1e-4
N_IN_STEPS = 10  # Cửa sổ đầu vào (phải khớp với lúc train)
N_OUT_STEPS = 1 # Cửa sổ đầu ra (phải khớp với lúc train)

# --- Cấu hình Model (Giữ nguyên mô hình LỚN) ---
MODEL_PARAMS = {
    "pytorch": {
        "hidden": 128,
        "num_layers": 2,
        "dropout": 0.2
    },
    "scratch": {
        "hidden": 128,
        "num_layers": 2,
        "dropout": 0.2
    }
}

# --- Cấu hình Đánh giá & Trực quan hóa ---
# Số bước dự đoán (ví dụ: 3 bước = 3h, 6h, 9h)
FORECAST_STEPS = 3
MAP_COMPARISON_PATH = MAP_DIR / "model_comparison_map.html"
# ID Cơn bão để trực quan hóa (Giữ nguyên ID của bạn)
STORM_ID_TO_VISUALIZE = "2025129S08138"
# Index bắt đầu dự đoán TRONG cơn bão đó
STORM_SAMPLE_INDEX = 5

