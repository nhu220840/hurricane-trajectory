import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ===== CẤU HÌNH =====
CSV_PATH = Path("data/ibtracs_track_ml.csv")
OUT_DIR = Path("data/")
OUT_NPZ = OUT_DIR / "processed_data.npz"
OUT_PKL = OUT_DIR / "scaler.pkl"
RANDOM_SEED = 42

INPUT_FEATURES  = [
    'lat', 'lon', 'wind', 'pres',
    'lat_change', 'lon_change', 'wind_change', 'pres_change'
]
TARGET_FEATURES = ['lat_change', 'lon_change']
INPUT_TIMESTEPS = 10
OUTPUT_TIMESTEPS = 1
TRAIN_RATIO, VALID_RATIO = 0.70, 0.15

def read_clean_and_create_deltas(csv_path: Path):
    print("Bắt đầu đọc và xử lý dữ liệu thô...")
    df = pd.read_csv(csv_path)
    cols = {'SID':'sid','ISO_TIME':'time','LAT':'lat','LON':'lon',
            'WMO_WIND':'wind','WMO_PRES':'pres'}
    df = df[list(cols)].rename(columns=cols)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    for c in ['lat', 'lon', 'wind', 'pres']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(['sid','time']).reset_index(drop=True)

    print("Tạo các feature 'delta'...")
    dfg = df.groupby('sid')
    df['lat_change'] = dfg['lat'].diff().fillna(0)
    df['lon_change'] = dfg['lon'].diff().fillna(0)
    df['wind_change'] = dfg['wind'].diff().fillna(0)
    df['pres_change'] = dfg['pres'].diff().fillna(0)
    return df

def split_by_sid(df, train_ratio, valid_ratio, seed):
    rng = np.random.RandomState(seed)
    sids = df['sid'].dropna().unique()
    rng.shuffle(sids)
    n_tr = int(len(sids) * train_ratio)
    n_va = int(len(sids) * valid_ratio)
    tr_sids, va_sids, te_sids = sids[:n_tr], sids[n_tr:n_tr+n_va], sids[n_tr+n_va:]
    print(f"Chia dữ liệu: {len(tr_sids)} sids train, {len(va_sids)} sids valid, {len(te_sids)} sids test.")
    return (df[df.sid.isin(tr_sids)].copy(),
            df[df.sid.isin(va_sids)].copy(),
            df[df.sid.isin(te_sids)].copy())

class GroupedInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='sid', feature_cols=None):
        self.group_col = group_col; self.feature_cols = feature_cols or []
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        X = X.copy()
        X[self.feature_cols] = X.groupby(self.group_col)[self.feature_cols].transform(lambda g: g.interpolate('linear'))
        X[self.feature_cols] = X.groupby(self.group_col)[self.feature_cols].transform(lambda g: g.bfill().ffill())
        X = X.dropna(subset=self.feature_cols + ['time','sid']).sort_values(['sid','time'])
        return X

def create_windows(df, in_steps, out_steps, in_feats, tgt_feats):
    Xs, ys = [], []
    for sid in df['sid'].unique():
        s = df[df.sid==sid].sort_values('time').reset_index(drop=True)
        n = len(s)
        if n < in_steps + out_steps: continue
        for i in range(n - in_steps - out_steps + 1):
            Xs.append(s.loc[i : i+in_steps-1, in_feats].to_numpy(dtype=np.float32))
            ys.append(s.loc[i+in_steps : i+in_steps+out_steps-1, tgt_feats].to_numpy(dtype=np.float32))
    return np.stack(Xs), np.stack(ys)

if __name__ == "__main__":
    df = read_clean_and_create_deltas(CSV_PATH)
    df_tr, df_va, df_te = split_by_sid(df, TRAIN_RATIO, VALID_RATIO, RANDOM_SEED)
    interp = GroupedInterpolator(feature_cols=INPUT_FEATURES)
    tr = interp.fit_transform(df_tr); va = interp.transform(df_va); te = interp.transform(df_te)
    scaler = MinMaxScaler().fit(tr[INPUT_FEATURES])
    def apply_scale(d):
        scaled = scaler.transform(d[INPUT_FEATURES])
        df_scaled = pd.DataFrame(scaled, columns=INPUT_FEATURES, index=d.index)
        return pd.concat([d[['sid','time']], df_scaled], axis=1)
    tr_f, va_f, te_f = apply_scale(tr), apply_scale(va), apply_scale(te)
    X_train, y_train = create_windows(tr_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)
    X_valid, y_valid = create_windows(va_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)
    X_test,  y_test  = create_windows(te_f, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, INPUT_FEATURES, TARGET_FEATURES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test,  y_test=y_test, INPUT_FEATURES=np.array(INPUT_FEATURES, dtype=object), TARGET_FEATURES=np.array(TARGET_FEATURES, dtype=object))
    with open(OUT_PKL, "wb") as f: pickle.dump(scaler, f)
    print(f"\nĐã lưu thành công dữ liệu và scaler.")
    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")