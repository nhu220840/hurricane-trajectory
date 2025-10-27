import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src import config  # <-- Đã import config


# Hàm này giữ nguyên
def _create_sequences(df_group, input_features, target_features, n_in, n_out):
    """Tạo chuỗi thời gian cho một nhóm (cơn bão)"""
    X, y = [], []
    data = df_group[input_features].values.astype(np.float32)
    target_data = df_group[target_features].values.astype(np.float32)

    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i: i + n_in])
        y.append(target_data[i + n_in: i + n_in + n_out])  # y là tọa độ tuyệt đối

    if not X:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)


# --- HÀM _run_sklearn_pipeline (ĐÃ SỬA LỖI) ---
def _run_sklearn_pipeline():
    """Đọc CSV thô, chạy pipeline và lưu CSV đã xử lý."""
    try:
        df = pd.read_csv(config.RAW_DATA_PATH, keep_default_na=False, na_values=[' '])
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {config.RAW_DATA_PATH}")
        return None, None

    # Sử dụng hằng số từ config
    df_clean = df[list(config.RAW_FEATURES_TO_KEEP.keys())].rename(columns=config.RAW_FEATURES_TO_KEEP)

    df_clean['time'] = pd.to_datetime(df_clean['time'])

    # Ép kiểu cho cột số
    for col in config.NUMERIC_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Ép kiểu cho cột phân loại (và điền giá trị 'UNKNOWN' nếu thiếu)
    for col in config.CATEGORICAL_FEATURES:
        df_clean[col] = df_clean[col].astype(str).fillna('UNKNOWN')

    df_clean = df_clean.sort_values(by=['sid', 'time'])

    # === SỬA LỖI LEAKAGE (TỪ PHIÊN BẢN TRƯỚC) ===
    print(f"Số dòng trước khi dropna (cho cột số): {len(df_clean)}")
    # Chỉ drop nếu các cột SỐ bị thiếu
    df_clean.dropna(subset=config.NUMERIC_FEATURES, inplace=True)
    print(f"Số dòng sau khi dropna: {len(df_clean)}")

    # === SỬA LỖI FutureWarning ===
    # Thay thế các giá trị rỗng còn lại trong cột phân loại (nếu có)
    for col in config.CATEGORICAL_FEATURES:
        # Gán lại trực tiếp thay vì dùng inplace=True trên một lát cắt
        df_clean[col] = df_clean[col].replace(r'^\s*$', 'UNKNOWN', regex=True)
    # === KẾT THÚC SỬA LỖI FutureWarning ===

    # === PIPELINE MỚI (Xử lý cả số và phân loại) ===
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Tạo preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERIC_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'  # Giữ lại các cột 'sid', 'time', v.v.
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # Fit và transform dữ liệu
    data_transformed = full_pipeline.fit_transform(df_clean)

    # === SỬA LỖI ValueError (Lấy tên cột mới) ===
    # Lấy tên cột one-hot
    ohe_feature_names = full_pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat'] \
        .named_steps['onehot'] \
        .get_feature_names_out(config.CATEGORICAL_FEATURES)

    # Lấy tên các cột còn lại (remainder) một cách chính xác
    # Đây là các cột không nằm trong NUMERIC hoặc CATEGORICAL
    remainder_cols = [col for col in df_clean.columns if
                      col not in config.NUMERIC_FEATURES and col not in config.CATEGORICAL_FEATURES]

    # Tên cột cuối cùng (Phải khớp với thứ tự của ColumnTransformer)
    # 1. Numeric, 2. Categorical, 3. Remainder
    new_column_names = config.NUMERIC_FEATURES + list(ohe_feature_names) + remainder_cols
    # === KẾT THÚC SỬA LỖI ValueError ===

    # Các đặc trưng input cho mô hình
    MODEL_INPUT_FEATURES = config.NUMERIC_FEATURES + list(ohe_feature_names)
    print(f"Các đặc trưng đầu vào (features) mới: {MODEL_INPUT_FEATURES}")

    # Tạo DataFrame
    df_final = pd.DataFrame(data_transformed, columns=new_column_names)

    # Sắp xếp lại (chỉ lấy các cột cần thiết cho bước sau)
    final_cols_order = ['sid', 'time'] + MODEL_INPUT_FEATURES
    df_final = df_final[final_cols_order]

    # --- Lưu scaler và file CSV ---
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Lưu toàn bộ pipeline (vì nó chứa cả scaler và one-hot)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(full_pipeline, f)
    print(f"Đã lưu toàn bộ pipeline (scaler + onehot) vào {config.SCALER_PATH}")

    df_final.to_csv(config.PROCESSED_CSV_PATH, index=False)
    print(f"Đã lưu dữ liệu đã xử lý vào {config.PROCESSED_CSV_PATH}")

    return df_final, MODEL_INPUT_FEATURES


# --- HÀM _convert_csv_to_npz (ĐÃ CẬP NHẬT ĐỂ CHIA THEO SID) ---
def _convert_csv_to_npz(df, input_features):
    """
    Chuyển đổi DataFrame đã xử lý thành các chuỗi (X, y_delta)
    VÀ chia train/valid/test theo SID.
    """

    # 1. Lấy danh sách SID duy nhất và xáo trộn
    all_sids = df['sid'].unique()
    np.random.seed(42)  # Thêm seed để đảm bảo chia ổn định
    np.random.shuffle(all_sids)

    # 2. Chia danh sách SID
    n = len(all_sids)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)

    train_sids = set(all_sids[:n_train])
    valid_sids = set(all_sids[n_train: n_train + n_valid])
    test_sids = set(all_sids[n_train + n_valid:])

    print(f"Tổng số cơn bão: {n}")
    print(f"Train (SIDs): {len(train_sids)}, Valid (SIDs): {len(valid_sids)}, Test (SIDs): {len(test_sids)}")

    # 3. Chuẩn bị các list để chứa dữ liệu
    all_X_train, all_y_train = [], []
    all_X_valid, all_y_valid = [], []
    all_X_test, all_y_test = [], []

    grouped = df.groupby('sid')

    # Sử dụng hằng số từ config
    try:
        lat_idx = input_features.index('lat')
        lon_idx = input_features.index('lon')
        coord_indices_in_X = [lat_idx, lon_idx]
    except ValueError as e:
        print(f"LỖI: 'lat' hoặc 'lon' không có trong input_features: {input_features}")
        return

    print("Đang tạo chuỗi (sequence) và tính toán deltas theo SID...")
    for sid, group in grouped:
        if len(group) < config.N_IN_STEPS + config.N_OUT_STEPS:
            continue

        # 1. Lấy X và y (tọa độ tuyệt đối) từ hàm
        X, y_abs = _create_sequences(
            group,
            input_features,
            config.TARGET_COORDS_FEATURES,  # <-- Sử dụng config
            config.N_IN_STEPS,
            config.N_OUT_STEPS
        )

        if X.shape[0] == 0:
            continue

        # 2. Tính toán Delta
        last_coords_in_X = X[:, -1, coord_indices_in_X]
        target_coords = np.squeeze(y_abs, axis=1)
        y_delta = target_coords - last_coords_in_X

        # 3. Reshape y_delta
        y_delta = np.expand_dims(y_delta, axis=1)

        # 4. Phân bổ X và y_delta vào đúng bộ (train/valid/test)
        if sid in train_sids:
            all_X_train.append(X)
            all_y_train.append(y_delta)
        elif sid in valid_sids:
            all_X_valid.append(X)
            all_y_valid.append(y_delta)
        elif sid in test_sids:
            all_X_test.append(X)
            all_y_test.append(y_delta)

    # 5. Gộp (concatenate) các bộ lại
    X_train = np.concatenate(all_X_train, axis=0) if all_X_train else np.array([])
    y_train = np.concatenate(all_y_train, axis=0) if all_y_train else np.array([])

    X_valid = np.concatenate(all_X_valid, axis=0) if all_X_valid else np.array([])
    y_valid = np.concatenate(all_y_valid, axis=0) if all_y_valid else np.array([])

    X_test = np.concatenate(all_X_test, axis=0) if all_X_test else np.array([])
    y_test = np.concatenate(all_y_test, axis=0) if all_y_test else np.array([])

    if len(X_train) == 0:
        print("LỖI: Không có dữ liệu huấn luyện. Kiểm tra lại logic chia SID.")
        return

    print(f"Tổng số chuỗi Train: {len(X_train)}")
    print(f"Tổng số chuỗi Valid: {len(X_valid)}")
    print(f"Tổng số chuỗi Test: {len(X_test)}")

    # 6. Lưu vào file NPZ
    np.savez_compressed(
        config.PROCESSED_NPZ_PATH,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
        INPUT_FEATURES=input_features,
        TARGET_FEATURES=config.TARGET_DELTAS_FEATURES
    )
    print(f"Đã lưu dữ liệu chuỗi (delta, chia theo SID) vào {config.PROCESSED_NPZ_PATH}")


# --- Hàm main để chạy từ main.py (Giữ nguyên) ---
def run_data_pipeline():
    """Hàm chính: Chạy toàn bộ pipeline tiền xử lý dữ liệu."""
    print("--- Bắt đầu Bước 2: Tiền xử lý Dữ liệu ---")

    # 1. Chạy Sklearn Pipeline
    processed_df, feature_cols = _run_sklearn_pipeline()

    if processed_df is not None:
        # 2. Chuyển đổi sang NPZ
        _convert_csv_to_npz(processed_df, feature_cols)

    print("--- Bước 2: Tiền xử lý Hoàn tất ---")


if __name__ == "__main__":
    run_data_pipeline()