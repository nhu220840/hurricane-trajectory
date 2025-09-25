# scripts/create_ts_data.py
import pandas as pd, numpy as np, pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ===== CẤU HÌNH MỚI =====
CSV_PATH = Path("data/ibtracs_track_ml.csv")
OUT_NPZ = Path("data/processed_splits_delta.npz")  # Đổi tên file output để tránh nhầm lẫn
OUT_PKL = Path("data/scaler_delta.pkl")
RANDOM_SEED = 42

# --- THAY ĐỔI QUAN TRỌNG ---
# Giữ nguyên input features
INPUT_FEATURES = ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']
# Target bây giờ là sự thay đổi của lat và lon
TARGET_FEATURES = ['lat_change', 'lon_change']
# Giữ nguyên độ dài input
INPUT_TIMESTEPS = 16
# Chỉ dự đoán 1 bước tiếp theo
OUTPUT_TIMESTEPS = 1
# ========================

TRAIN_RATIO, VALID_RATIO = 0.70, 0.15


def read_clean_and_create_deltas(csv_path: Path):
    df = pd.read_csv(csv_path)
    cols = {'SID': 'sid', 'ISO_TIME': 'time', 'LAT': 'lat', 'LON': 'lon',
            'WMO_WIND': 'wind', 'WMO_PRES': 'pres', 'STORM_SPEED': 'speed', 'STORM_DIR': 'direction'}
    df = df[list(cols)].rename(columns=cols)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    for c in ['lat', 'lon', 'wind', 'pres', 'speed', 'direction']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.sort_values(['sid', 'time']).reset_index(drop=True)

    # --- TÍNH TOÁN DELTA ---
    # groupby('sid') để đảm bảo .diff() tính toán riêng cho mỗi cơn bão
    df['lat_change'] = df.groupby('sid')['lat'].diff().fillna(0)
    df['lon_change'] = df.groupby('sid')['lon'].diff().fillna(0)

    return df


def split_by_sid(df, train_ratio=0.7, valid_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    sids = df['sid'].dropna().unique()
    rng.shuffle(sids)
    n_tr = int(len(sids) * train_ratio)
    n_va = int(len(sids) * valid_ratio)
    tr, va, te = sids[:n_tr], sids[n_tr:n_tr + n_va], sids[n_tr + n_va:]
    return (df[df.sid.isin(tr)].copy(),
            df[df.sid.isin(va)].copy(),
            df[df.sid.isin(te)].copy())


class GroupedInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='sid', feature_cols=None):
        self.group_col = group_col
        self.feature_cols = feature_cols or []

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        X = X.copy()
        # Bao gồm cả các cột delta mới trong việc nội suy
        all_features = list(set(self.feature_cols + ['lat_change', 'lon_change']))
        X[all_features] = X.groupby(self.group_col)[all_features] \
            .transform(lambda g: g.interpolate('linear'))
        X[all_features] = X.groupby(self.group_col)[all_features] \
            .transform(lambda g: g.bfill().ffill())
        X = X.dropna(subset=all_features + ['time', 'sid']).sort_values(['sid', 'time'])
        return X


def create_windows(df, in_steps, out_steps, in_feats, tgt_feats):
    Xs, ys = [], []
    for sid in df['sid'].unique():
        s = df[df.sid == sid].sort_values('time').reset_index(drop=True)
        n = len(s)
        if n < in_steps + out_steps: continue
        for i in range(n - in_steps - out_steps + 1):
            Xs.append(s.loc[i:i + in_steps - 1, in_feats].to_numpy(dtype=np.float32))
            ys.append(s.loc[i + in_steps:i + in_steps + out_steps - 1, tgt_feats].to_numpy(dtype=np.float32))
    if not Xs:
        return (np.empty((0, in_steps, len(in_feats)), dtype=np.float32),
                np.empty((0, out_steps, len(tgt_feats)), dtype=np.float32))
    return np.stack(Xs), np.stack(ys)


if __name__ == "__main__":
    df = read_clean_and_create_deltas(CSV_PATH)
    df_tr, df_va, df_te = split_by_sid(df, TRAIN_RATIO, VALID_RATIO, RANDOM_SEED)

    # Tất cả các cột cần được nội suy
    all_numeric_features = INPUT_FEATURES + TARGET_FEATURES
    interp = GroupedInterpolator(feature_cols=all_numeric_features)
    tr = interp.fit_transform(df_tr)
    va = interp.transform(df_va)
    te = interp.transform(df_te)

    # Chỉ fit scaler trên INPUT_FEATURES của tập train
    scaler = MinMaxScaler().fit(tr[INPUT_FEATURES])


    def apply_scale(d):
        # Scale các input features
        scaled_inputs = scaler.transform(d[INPUT_FEATURES])
        df_scaled_inputs = pd.DataFrame(scaled_inputs, columns=INPUT_FEATURES, index=d.index)

        # Giữ lại các cột khác (bao gồm target deltas không được scale)
        other_cols = d.drop(columns=INPUT_FEATURES)
        return pd.concat([other_cols, df_scaled_inputs], axis=1)


    tr_f, va_f, te_f = apply_scale(tr), apply_scale(va), apply_scale(te)

    X_train, y_train = create_windows(tr_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)
    X_valid, y_valid = create_windows(va_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)
    X_test, y_test = create_windows(te_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, X_train=X_train, y_train=y_train,
             X_valid=X_valid, y_valid=y_valid,
             X_test=X_test, y_test=y_test,
             INPUT_FEATURES=np.array(INPUT_FEATURES, dtype=object),
             TARGET_FEATURES=np.array(TARGET_FEATURES, dtype=object))
    with open(OUT_PKL, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved:", OUT_NPZ, "and", OUT_PKL)
    print("Shapes:", X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)