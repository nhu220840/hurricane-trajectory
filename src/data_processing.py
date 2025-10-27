# src/data_processing.py
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from src import config  # Import từ config


# --- Phần 1: Lấy từ 'preprocessing.py' ---

class GroupedInterpolator(BaseEstimator, TransformerMixin):
    # ... (Sao chép y hệt class GroupedInterpolator từ file preprocessing.py) ...
    def __init__(self, group_col='sid', feature_cols=None, method='linear'):
        self.group_col = group_col
        self.feature_cols = feature_cols
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        interpolated_features = X_copy.groupby(self.group_col)[self.feature_cols].transform(
            lambda x: x.interpolate(method=self.method)
        )
        X_copy[self.feature_cols] = interpolated_features
        X_filled = X_copy.groupby(self.group_col, group_keys=False).apply(
            lambda group: group.fillna(method='bfill').fillna(method='ffill')
        )
        X_filled.dropna(inplace=True)
        return X_filled


def _run_sklearn_pipeline():
    """Đọc CSV thô, chạy pipeline và lưu CSV đã xử lý."""
    try:
        df = pd.read_csv(config.RAW_DATA_PATH, keep_default_na=False, na_values=[' '])
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {config.RAW_DATA_PATH}")
        return None

    columns_to_keep = {
        'SID': 'sid', 'ISO_TIME': 'time', 'LAT': 'lat', 'LON': 'lon',
        'WMO_WIND': 'wind', 'WMO_PRES': 'pres', 'STORM_SPEED': 'speed', 'STORM_DIR': 'direction'
    }
    df_clean = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

    df_clean['time'] = pd.to_datetime(df_clean['time'])
    feature_cols = ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.sort_values(by=['sid', 'time'])

    numeric_features = ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']
    full_pipeline = Pipeline(steps=[
        ('custom_interpolator', GroupedInterpolator(group_col='sid', feature_cols=numeric_features)),
        ('preprocessor', ColumnTransformer(transformers=[
            ('scaler', MinMaxScaler(), numeric_features)
        ], remainder='passthrough'))
    ])

    data_transformed = full_pipeline.fit_transform(df_clean)
    new_column_names = numeric_features + [col for col in df_clean.columns if col not in numeric_features]
    df_final = pd.DataFrame(data_transformed, columns=new_column_names)
    df_final = df_final[['sid', 'time'] + numeric_features]

    # Lưu scaler
    scaler = full_pipeline.named_steps['preprocessor'].named_transformers_['scaler']
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Đã lưu scaler vào {config.SCALER_PATH}")

    # Lưu file CSV đã xử lý
    df_final.to_csv(config.PROCESSED_CSV_PATH, index=False)
    print(f"Đã lưu dữ liệu đã xử lý vào {config.PROCESSED_CSV_PATH}")
    return df_final, feature_cols


# --- Phần 2: Lấy từ 'scripts/create_ts_data.py' ---
# (Tôi sẽ điều chỉnh lại một chút cho phù hợp)

def _create_sequences(df_group, input_features, target_features, n_in, n_out):
    # ... (Logic tạo chuỗi, bạn có thể copy từ file create_ts_data.py) ...
    # ... (Hãy đảm bảo nó trả về X, y cho nhóm đó) ...
    # Đây là một ví dụ đơn giản hóa:
    X, y = [], []
    data = df_group[input_features].values.astype(np.float32)
    target_data = df_group[target_features].values.astype(np.float32)

    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i: i + n_in])
        y.append(target_data[i + n_in: i + n_in + n_out])

    if not X:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)


def _convert_csv_to_npz(df, input_features):
    """Chuyển đổi DataFrame đã xử lý thành các chuỗi và lưu vào .npz"""
    N_IN, N_OUT = 10, 1  # Cấu hình cửa sổ thời gian
    TARGET_FEATURES = ['lat', 'lon']  # Mục tiêu dự đoán

    all_X, all_y = [], []
    grouped = df.groupby('sid')
    for sid, group in grouped:
        if len(group) < N_IN + N_OUT:
            continue
        X, y = _create_sequences(group, input_features, TARGET_FEATURES, N_IN, N_OUT)
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Tách dữ liệu (ví dụ: 70-15-15)
    n = len(X)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_valid, y_valid = X[n_train: n_train + n_valid], y[n_train: n_train + n_valid]
    X_test, y_test = X[n_train + n_valid:], y[n_train + n_valid:]

    # Lưu vào file NPZ
    np.savez_compressed(
        config.PROCESSED_NPZ_PATH,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
        INPUT_FEATURES=input_features,
        TARGET_FEATURES=TARGET_FEATURES
    )
    print(f"Đã lưu dữ liệu chuỗi vào {config.PROCESSED_NPZ_PATH}")


# --- Hàm chính để chạy từ main.py ---
def run_data_pipeline():
    """Hàm chính: Chạy toàn bộ pipeline tiền xử lý dữ liệu."""
    print("--- Bắt đầu Pipeline Tiền xử lý Dữ liệu ---")

    # 1. Chạy Sklearn Pipeline
    processed_df, feature_cols = _run_sklearn_pipeline()

    if processed_df is not None:
        # 2. Chuyển đổi sang NPZ
        _convert_csv_to_npz(processed_df, feature_cols)

    print("--- Pipeline Tiền xử lý Hoàn tất ---")


if __name__ == "__main__":
    # Cho phép chạy file này độc lập để test
    run_data_pipeline()