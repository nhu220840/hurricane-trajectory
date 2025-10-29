# src/data_processing.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

from .config import (
    RAW_CSV, PROCESSED_DIR, PROCESSED_NPZ,
    NUMERIC_X, CATEGORICAL_X, FEATURES_X,
    TARGET_Y, TIME_COLUMN, SID_COLUMN,
    N_IN, N_OUT, PREPROCESSOR_X_PKL, SCALER_Y_PKL
)

# ================= Helpers =================

def _sort_and_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    required = set([SID_COLUMN, TIME_COLUMN, "lat", "lon"] + FEATURES_X)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong RAW_CSV: {missing}")

    df = df[list(required)].copy()
    df = df.sort_values([SID_COLUMN, TIME_COLUMN]).reset_index(drop=True)

    if "basin" in df.columns:
        df["basin"] = df["basin"].astype(str).fillna("UNK")
    return df


def _coerce_numeric_columns(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _interpolate_numeric_by_sid(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    numeric_cols = list(dict.fromkeys(numeric_cols))  # khử trùng lặp

    def _interp_block(g):
        g = g.copy()
        for c in numeric_cols:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            g[c] = g[c].interpolate(method="linear", limit_direction="both")
        filled = g[numeric_cols].ffill().bfill()
        g.loc[:, numeric_cols] = filled.values
        return g

    return df.groupby(SID_COLUMN, group_keys=False, sort=False).apply(_interp_block)


def _final_impute_numeric(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    """
    Sau interpolate/ffill/bfill vẫn có thể còn NaN nếu cả group trống.
    Dùng median theo cột, nếu vẫn NaN thì điền 0.0.
    """
    cols = [c for c in numeric_cols if c in df.columns]
    med = df[cols].median(numeric_only=True)
    df.loc[:, cols] = df[cols].fillna(med)
    df.loc[:, cols] = df[cols].fillna(0.0)
    return df


def _add_deltas_per_sid(df: pd.DataFrame) -> pd.DataFrame:
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["delta_lat"] = df.groupby(SID_COLUMN)["lat"].shift(-1) - df["lat"]
    df["delta_lon"] = df.groupby(SID_COLUMN)["lon"].shift(-1) - df["lon"]
    df = df.dropna(subset=["delta_lat", "delta_lon"]).reset_index(drop=True)
    return df


def _encode_sid_to_index(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df[SID_COLUMN].astype(str).unique().tolist()
    sid2idx = {s: i for i, s in enumerate(uniq)}
    df["sid_idx"] = df[SID_COLUMN].astype(str).map(sid2idx).astype(int)
    return df


def _fit_x_preprocessor_and_transform(df: pd.DataFrame):
    num_cols = [c for c in NUMERIC_X if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_X if c in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    X_all = preprocessor.fit_transform(df[num_cols + cat_cols])
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()
    X_all = X_all.astype(np.float32)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = []
    return X_all, preprocessor, feature_names


def _fit_y_scaler_and_transform(df: pd.DataFrame):
    scaler_y = MinMaxScaler()
    Y_all = scaler_y.fit_transform(df[TARGET_Y].values.astype(np.float32)).astype(np.float32)
    return Y_all, scaler_y


def _make_windows(X_all, Y_all, sids_idx, lat_arr, lon_arr, N_in, N_out):
    assert N_out == 1, "Code hiện tại giả định one-step (N_OUT=1)."
    X_seq, Y_seq, last_obs_latlon, window_sid_idx = [], [], [], []

    uniq, idx = np.unique(sids_idx, return_index=True)
    order = np.argsort(idx)
    uniq, starts = uniq[order], idx[order]
    ends = list(starts[1:]) + [len(sids_idx)]

    for sid_i, start, end in zip(uniq, starts, ends):
        for t in range(start + N_in - 1, end - N_out):
            Xw = X_all[t - (N_in - 1): t + 1, :]
            Yw = Y_all[t: t + N_out, :]  # (1, 2)
            Yw = Yw[0, :]                # -> (2,)
            last_lat = lat_arr[t]
            last_lon = lon_arr[t]
            X_seq.append(Xw)
            Y_seq.append(Yw)
            last_obs_latlon.append([last_lat, last_lon])
            window_sid_idx.append(sid_i)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    Y_seq = np.asarray(Y_seq, dtype=np.float32)
    last_obs_latlon = np.asarray(last_obs_latlon, dtype=np.float32)
    window_sid_idx = np.asarray(window_sid_idx, dtype=np.int32)
    return X_seq, Y_seq, last_obs_latlon, window_sid_idx


def _filter_invalid_windows(X, Y, last_obs, sid_idx):
    """
    Loại bỏ mọi cửa sổ có NaN/Inf trong X, Y, hoặc last_obs.
    """
    mask = (
        np.isfinite(X).all(axis=(1, 2)) &
        np.isfinite(Y).all(axis=1) &
        np.isfinite(last_obs).all(axis=1)
    )
    dropped = int(X.shape[0] - mask.sum())
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} / {X.shape[0]} windows due to NaN/Inf.")
    return X[mask], Y[mask], last_obs[mask], sid_idx[mask]

# ================= Public API =================

def process_and_save_npz():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    # 1) Clean & sort
    df = _sort_and_basic_clean(df)

    # 2) Coerce numeric
    numeric_all = list(dict.fromkeys(list(NUMERIC_X) + ["lat", "lon"]))
    df = _coerce_numeric_columns(df, numeric_all)

    # lọc giá trị vô lý
    df.loc[(df["lat"] < -90) | (df["lat"] > 90), "lat"] = np.nan
    df.loc[(df["lon"] < -180) | (df["lon"] > 180), "lon"] = np.nan

    # 3) Interpolate by sid
    numeric_for_interp = list(dict.fromkeys([c for c in NUMERIC_X if c in df.columns] + ["lat", "lon"]))
    df = _interpolate_numeric_by_sid(df, numeric_cols=numeric_for_interp)

    # 4) Impute cuối để đảm bảo không còn NaN ở numeric
    df = _final_impute_numeric(df, numeric_for_interp)

    # 5) Deltas
    df = _add_deltas_per_sid(df)

    # 6) sid index
    df = _encode_sid_to_index(df)

    # 7) X preprocessor & transform
    X_all, preprocessor_x, x_feature_names = _fit_x_preprocessor_and_transform(df)

    # 8) Y scaler & transform (delta)
    Y_all, scaler_y = _fit_y_scaler_and_transform(df)

    # 9) Windows
    X_seq, Y_seq, last_obs, win_sid_idx = _make_windows(
        X_all, Y_all, df["sid_idx"].values,
        df["lat"].values, df["lon"].values,
        N_IN, N_OUT
    )

    # 10) Lọc cửa sổ lỗi
    X_seq, Y_seq, last_obs, win_sid_idx = _filter_invalid_windows(X_seq, Y_seq, last_obs, win_sid_idx)

    # 11) Save npz
    np.savez_compressed(
        PROCESSED_NPZ,
        X=X_seq, Y=Y_seq,
        last_obs_latlon=last_obs,
        window_sid_idx=win_sid_idx,
        x_feature_names=np.array(x_feature_names, dtype=object),
        target_y=np.array(TARGET_Y, dtype=object),
        N_IN=np.array(N_IN),
        N_OUT=np.array(N_OUT)
    )

    # 12) Save preprocessors
    with open(PREPROCESSOR_X_PKL, "wb") as f:
        pickle.dump(preprocessor_x, f)
    with open(SCALER_Y_PKL, "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"[OK] Saved processed arrays to {PROCESSED_NPZ}")
    print(f"[OK] Saved X preprocessor to {PREPROCESSOR_X_PKL}")
    print(f"[OK] Saved Y scaler to {SCALER_Y_PKL}")
