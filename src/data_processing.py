# src/data_processing.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from src import config


def feature_engineering(df):
    """Tạo các đặc trưng mới, bao gồm 5 đặc trưng delta."""
    df = df.copy()
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df.dropna(subset=['ISO_TIME'], inplace=True)
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)

    # Điền giá trị thiếu cho các cột số trước khi tính delta
    for col in config.NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df.groupby('SID')[col].ffill().bfill()

    # Tạo đặc trưng tuần hoàn
    df['month'] = df['ISO_TIME'].dt.month
    df['day'] = df['ISO_TIME'].dt.day
    df['hour'] = df['ISO_TIME'].dt.hour
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # --- THAY ĐỔI: Tạo 5 đặc trưng delta ---
    dfg = df.groupby('SID')
    df['LAT_delta'] = dfg['LAT'].diff().fillna(0)
    df['LON_delta'] = dfg['LON'].diff().fillna(0)
    df['WMO_WIND_delta'] = dfg['WMO_WIND'].diff().fillna(0)
    df['WMO_PRES_delta'] = dfg['WMO_PRES'].diff().fillna(0)
    df['DIST2LAND_delta'] = dfg['DIST2LAND'].diff().fillna(0)
    # ------------------------------------

    df = pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES, prefix=config.CATEGORICAL_FEATURES)
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    # Xử lý các giá trị vô hạn (nếu có)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def create_windows(df, input_features, target_features, in_steps, out_steps):
    """
    Trả về: Xs (Inputs), ys (Targets), X_last_datetimes (để dự báo)
    """
    Xs, ys, X_last_datetimes = [], [], []

    df['SID'] = df['SID'].astype('category')

    cols_to_group = list(dict.fromkeys(input_features + target_features + ['ISO_TIME']))

    for sid, group in df.groupby('SID', observed=False):
        n = len(group)
        if n < in_steps + out_steps:
            continue

        group_data = group[cols_to_group]

        for i in range(n - in_steps - out_steps + 1):
            input_window_data = group_data.iloc[i: i + in_steps]
            target_window_data = group_data.iloc[i + in_steps: i + in_steps + out_steps]

            Xs.append(input_window_data[input_features].to_numpy(dtype=np.float32))
            ys.append(target_window_data[target_features].to_numpy(dtype=np.float32))
            X_last_datetimes.append(input_window_data.iloc[-1]['ISO_TIME'])

    # squeeze ys để loại bỏ chiều không cần thiết (n, 1, 5) -> (n, 5)
    return np.array(Xs), np.squeeze(np.array(ys), axis=1), np.array(X_last_datetimes)


def run_processing():
    """Hàm chính: Sử dụng 2 scaler (scaler_X và scaler_y)"""
    print("Bắt đầu pipeline xử lý dữ liệu nâng cao...")
    df = pd.read_csv(config.RAW_DATA_PATH)

    print("1. Đang tạo các đặc trưng (Feature Engineering)...")
    df_featured = feature_engineering(df)

    # Xác định Input Features (22 features)
    final_input_features = config.NUMERICAL_FEATURES + \
                           [col for col in df_featured.columns if 'sin_' in col or 'cos_' in col] + \
                           [col for col in df_featured.columns if
                            any(cat_feat in col for cat_feat in config.CATEGORICAL_FEATURES)]

    final_input_features = [f for f in final_input_features if
                            f in df_featured.columns and df_featured[f].dtype != 'object']

    # Xác định Target Features (5 features)
    target_features = config.TARGET_FEATURES

    # Đảm bảo tất cả các cột đều là số và điền NA
    for col in final_input_features + target_features:
        df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)

    # Chia dữ liệu
    sids = df_featured['SID'].unique()
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(sids)
    n_tr = int(len(sids) * config.TRAIN_RATIO)
    n_va = int(len(sids) * config.VALID_RATIO)
    tr_sids, va_sids, te_sids = sids[:n_tr], sids[n_tr:n_tr + n_va], sids[n_tr + n_va:]
    df_tr = df_featured[df_featured.SID.isin(tr_sids)].copy()
    df_va = df_featured[df_featured.SID.isin(va_sids)].copy()
    df_te = df_featured[df_featured.SID.isin(te_sids)].copy()
    print(f"2. Đã chia dữ liệu: {len(tr_sids)} train, {len(va_sids)} valid, {len(te_sids)} test.")

    # Ép kiểu float64 để dập tắt cảnh báo
    for col in final_input_features:
        df_tr[col] = df_tr[col].astype('float64')
        df_va[col] = df_va[col].astype('float64')
        df_te[col] = df_te[col].astype('float64')
    for col in target_features:
        df_tr[col] = df_tr[col].astype('float64')
        df_va[col] = df_va[col].astype('float64')
        df_te[col] = df_te[col].astype('float64')

    # --- THAY ĐỔI: SỬ DỤNG 2 SCALER ---
    print(f"3. Đang chuẩn hóa {len(final_input_features)} Input Features (scaler_X)...")
    scaler_X = StandardScaler()
    df_tr.loc[:, final_input_features] = scaler_X.fit_transform(df_tr[final_input_features])
    df_va.loc[:, final_input_features] = scaler_X.transform(df_va[final_input_features])
    df_te.loc[:, final_input_features] = scaler_X.transform(df_te[final_input_features])

    print(f"4. Đang chuẩn hóa {len(target_features)} Target Features (scaler_y)...")
    scaler_y = StandardScaler()
    df_tr.loc[:, target_features] = scaler_y.fit_transform(df_tr[target_features])
    df_va.loc[:, target_features] = scaler_y.transform(df_va[target_features])
    df_te.loc[:, target_features] = scaler_y.transform(df_te[target_features])
    # -----------------------------------

    print("5. Đang tạo cửa sổ dữ liệu...")
    X_train, y_train, X_train_last_dt = create_windows(df_tr, final_input_features, target_features,
                                                       config.INPUT_TIMESTEPS, config.OUTPUT_TIMESTEPS)
    X_valid, y_valid, X_valid_last_dt = create_windows(df_va, final_input_features, target_features,
                                                       config.INPUT_TIMESTEPS, config.OUTPUT_TIMESTEPS)
    X_test, y_test, X_test_last_dt = create_windows(df_te, final_input_features, target_features,
                                                    config.INPUT_TIMESTEPS, config.OUTPUT_TIMESTEPS)

    print(f"   - Train shapes: X={X_train.shape}, y={y_train.shape}, dt={X_train_last_dt.shape}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.PROCESSED_DATA_NPZ,
        X_train=X_train, y_train=y_train, X_train_last_dt=X_train_last_dt,
        X_valid=X_valid, y_valid=y_valid, X_valid_last_dt=X_valid_last_dt,
        X_test=X_test, y_test=y_test, X_test_last_dt=X_test_last_dt,
        INPUT_FEATURES=np.array(final_input_features, dtype=object),
        TARGET_FEATURES=np.array(target_features, dtype=object)
    )
    # --- THAY ĐỔI: Lưu 2 scaler ---
    with open(config.SCALER_X_PKL, "wb") as f:
        pickle.dump(scaler_X, f)
    with open(config.SCALER_Y_PKL, "wb") as f:
        pickle.dump(scaler_y, f)
    print(f"6. Đã lưu thành công dữ liệu và 2 scaler (X và y).")