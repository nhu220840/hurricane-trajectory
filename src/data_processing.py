# src/data_processing.py

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
import random  # (NEW) Added for splitting

from .config import (
    RAW_CSV, PROCESSED_DIR, PROCESSED_NPZ,
    NUMERIC_X, CATEGORICAL_X, FEATURES_X,
    TARGET_Y, TIME_COLUMN, SID_COLUMN,
    N_IN, N_OUT, PREPROCESSOR_X_PKL, SCALER_Y_PKL,
    SEED  # (NEW) Import SEED
)


# ================= (NEW) Splitting Function =================

def _split_by_sid(window_sid_idx, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    """
    Splits a list of SIDs into train, val, and test sets based on unique SIDs.
    """
    uniq = np.unique(window_sid_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_sids = set(uniq[:n_train])
    val_sids = set(uniq[n_train:n_train + n_val])
    test_sids = set(uniq[n_train + n_val:])
    print(f"[Split] Total SIDs: {n}. Train: {len(train_sids)}, Val: {len(val_sids)}, Test: {len(test_sids)}")
    return train_sids, val_sids, test_sids


# ================= Helpers (Unchanged "safe" functions) =================

def _sort_and_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    required = set([SID_COLUMN, TIME_COLUMN, "lat", "lon"] + FEATURES_X)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in RAW_CSV: {missing}")

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
    # (This function is safe as it groups by SID)
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    numeric_cols = list(dict.fromkeys(numeric_cols))  # remove duplicates

    def _interp_block(g):
        g = g.copy()
        for c in numeric_cols:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            g[c] = g[c].interpolate(method="linear", limit_direction="both")
        filled = g[numeric_cols].ffill().bfill()
        g.loc[:, numeric_cols] = filled.values
        return g

    return df.groupby(SID_COLUMN, group_keys=False, sort=False).apply(_interp_block)


def _add_deltas_per_sid(df: pd.DataFrame) -> pd.DataFrame:
    # (This function is safe as it groups by SID)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["delta_lat"] = df.groupby(SID_COLUMN)["lat"].shift(-1) - df["lat"]
    df["delta_lon"] = df.groupby(SID_COLUMN)["lon"].shift(-1) - df["lon"]
    df = df.dropna(subset=["delta_lat", "delta_lon"]).reset_index(drop=True)
    return df


def _encode_sid_to_index(df: pd.DataFrame) -> pd.DataFrame:
    # (This function is safe, just mapping)
    uniq = df[SID_COLUMN].astype(str).unique().tolist()
    sid2idx = {s: i for i, s in enumerate(uniq)}
    df["sid_idx"] = df[SID_COLUMN].astype(str).map(sid2idx).astype(int)
    return df


# ================= (REVISED) Fit/Apply Helper Functions =================

def _fit_final_imputer(df_train: pd.DataFrame, numeric_cols):
    """
    (NEW) Fits the imputer (median) ONLY on training data.
    Returns the median map.
    """
    cols = [c for c in numeric_cols if c in df_train.columns]
    median_map = df_train[cols].median(numeric_only=True)
    # Handle case where median might be NaN (if train col is all NaN)
    median_map = median_map.fillna(0.0)
    return median_map


def _apply_final_imputer(df: pd.DataFrame, numeric_cols, median_map) -> pd.DataFrame:
    """
    (NEW) Applies the pre-fitted median map to any DataFrame (train, val, or test).
    """
    cols = [c for c in numeric_cols if c in df.columns]
    df.loc[:, cols] = df[cols].fillna(median_map)
    # Ensure no NaNs remain after fillna (if median_map had a NaN)
    df.loc[:, cols] = df[cols].fillna(0.0)
    return df


def _fit_x_preprocessor(df_train: pd.DataFrame):
    """
    (REVISED) Fits the ColumnTransformer ONLY on training data.
    """
    num_cols = [c for c in NUMERIC_X if c in df_train.columns]
    cat_cols = [c for c in CATEGORICAL_X if c in df_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    # FIT only on df_train
    preprocessor.fit(df_train[num_cols + cat_cols])
    return preprocessor


def _fit_y_scaler(df_train: pd.DataFrame):
    """
    (REVISED) Fits the Y scaler ONLY on training data.
    """
    scaler_y = MinMaxScaler()
    # FIT only on df_train
    scaler_y.fit(df_train[TARGET_Y].values.astype(np.float32))
    return scaler_y


def _make_windows(X_all, Y_all, sids_idx, lat_arr, lon_arr, N_in, N_out):
    # (This function is unchanged, it just processes arrays)
    assert N_out == 1, "Current code assumes one-step (N_OUT=1)."
    X_seq, Y_seq, last_obs_latlon, window_sid_idx = [], [], [], []

    uniq, idx = np.unique(sids_idx, return_index=True)
    order = np.argsort(idx)
    uniq, starts = uniq[order], idx[order]
    ends = list(starts[1:]) + [len(sids_idx)]

    for sid_i, start, end in zip(uniq, starts, ends):
        for t in range(start + N_in - 1, end - N_out):
            Xw = X_all[t - (N_in - 1): t + 1, :]
            Yw = Y_all[t: t + N_out, :]  # (1, 2)
            Yw = Yw[0, :]  # -> (2,)
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
    Remove any windows with NaN/Inf in X, Y, or last_obs.
    (Unchanged)
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


# ================= (REVISED) Public API =================

def process_and_save_npz():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")

    df = pd.read_csv(RAW_CSV)

    # === "SAFE" PREPROCESSING STEPS (run on full dataframe) ===
    # These steps are safe because they either map values (encode)
    # or group by SID (interpolate, deltas).

    # 1) Clean & sort
    df = _sort_and_basic_clean(df)

    # 2) Coerce numeric
    numeric_all = list(dict.fromkeys(list(NUMERIC_X) + ["lat", "lon"]))
    df = _coerce_numeric_columns(df, numeric_all)

    # filter absurd values
    df.loc[(df["lat"] < -90) | (df["lat"] > 90), "lat"] = np.nan
    df.loc[(df["lon"] < -180) | (df["lon"] > 180), "lon"] = np.nan

    # 3) Interpolate by sid
    numeric_for_interp = list(dict.fromkeys([c for c in NUMERIC_X if c in df.columns] + ["lat", "lon"]))
    df = _interpolate_numeric_by_sid(df, numeric_cols=numeric_for_interp)

    # 4) Deltas
    df = _add_deltas_per_sid(df)

    # 5) sid index
    df = _encode_sid_to_index(df)

    # === (NEW) DATA SPLIT ===
    # Split the DataFrame *before* fitting any scalers or median imputers.
    print("[Split] Performing split based on SID...")
    all_sids_idx = df["sid_idx"].values
    train_sids, val_sids, test_sids = _split_by_sid(all_sids_idx, seed=SEED)

    df_train = df[df['sid_idx'].isin(train_sids)].copy()
    df_val = df[df['sid_idx'].isin(val_sids)].copy()
    df_test = df[df['sid_idx'].isin(test_sids)].copy()
    print(f"[Split] df_train: {len(df_train)}, df_val: {len(df_val)}, df_test: {len(df_test)}")
    if len(df_train) == 0:
        raise ValueError("Training set is empty. Check splitting logic or data source.")

    # === (NEW) FIT PREPROCESSORS (ONLY ON TRAIN DATA) ===

    # 6) Final impute (Fit on train, apply to all)
    print("[Fit] Fitting imputer on train data...")
    median_map = _fit_final_imputer(df_train, numeric_for_interp)
    df_train = _apply_final_imputer(df_train, numeric_for_interp, median_map)
    df_val = _apply_final_imputer(df_val, numeric_for_interp, median_map)
    df_test = _apply_final_imputer(df_test, numeric_for_interp, median_map)

    # 7) X preprocessor (Fit on train, transform all)
    print("[Fit] Fitting X preprocessor on train data...")
    preprocessor_x = _fit_x_preprocessor(df_train)
    X_train_all = preprocessor_x.transform(df_train[NUMERIC_X + CATEGORICAL_X])
    X_val_all = preprocessor_x.transform(df_val[NUMERIC_X + CATEGORICAL_X])
    X_test_all = preprocessor_x.transform(df_test[NUMERIC_X + CATEGORICAL_X])

    if hasattr(X_train_all, "toarray"):
        X_train_all = X_train_all.toarray()
        X_val_all = X_val_all.toarray()
        X_test_all = X_test_all.toarray()

    X_train_all = X_train_all.astype(np.float32)
    X_val_all = X_val_all.astype(np.float32)
    X_test_all = X_test_all.astype(np.float32)

    # Get feature names (after fitting)
    try:
        x_feature_names = preprocessor_x.get_feature_names_out().tolist()
    except Exception:
        x_feature_names = []

    # 8) Y scaler (Fit on train, transform all)
    print("[Fit] Fitting Y scaler on train data...")
    scaler_y = _fit_y_scaler(df_train)
    Y_train_all = scaler_y.transform(df_train[TARGET_Y].values.astype(np.float32))
    Y_val_all = scaler_y.transform(df_val[TARGET_Y].values.astype(np.float32))
    Y_test_all = scaler_y.transform(df_test[TARGET_Y].values.astype(np.float32))

    # === (NEW) CREATE WINDOWS FOR EACH SPLIT ===

    # 9) Windows
    print("[Window] Creating windows for Train set...")
    X_tr, Y_tr, last_tr, sid_tr = _make_windows(
        X_train_all, Y_train_all, df_train["sid_idx"].values,
        df_train["lat"].values, df_train["lon"].values,
        N_IN, N_OUT
    )
    print(f"[Window] Train windows: {X_tr.shape}")

    print("[Window] Creating windows for Validation set...")
    X_v, Y_v, last_v, sid_v = _make_windows(
        X_val_all, Y_val_all, df_val["sid_idx"].values,
        df_val["lat"].values, df_val["lon"].values,
        N_IN, N_OUT
    )
    print(f"[Window] Validation windows: {X_v.shape}")

    print("[Window] Creating windows for Test set...")
    X_te, Y_te, last_te, sid_te = _make_windows(
        X_test_all, Y_test_all, df_test["sid_idx"].values,
        df_test["lat"].values, df_test["lon"].values,
        N_IN, N_OUT
    )
    print(f"[Window] Test windows: {X_te.shape}")

    # 10) Filter invalid windows (for each split)
    print("[Filter] Filtering invalid windows from splits...")
    X_tr, Y_tr, last_tr, sid_tr = _filter_invalid_windows(X_tr, Y_tr, last_tr, sid_tr)
    X_v, Y_v, last_v, sid_v = _filter_invalid_windows(X_v, Y_v, last_v, sid_v)
    X_te, Y_te, last_te, sid_te = _filter_invalid_windows(X_te, Y_te, last_te, sid_te)

    # 11) Save npz (Save all splits)
    print(f"[Save] Saving all splits to {PROCESSED_NPZ}...")
    np.savez_compressed(
        PROCESSED_NPZ,
        # Train
        X_train=X_tr, Y_train=Y_tr,
        last_obs_train=last_tr,
        window_sid_idx_train=sid_tr,
        # Validation
        X_val=X_v, Y_val=Y_v,
        last_obs_val=last_v,
        window_sid_idx_val=sid_v,
        # Test
        X_test=X_te, Y_test=Y_te,
        last_obs_test=last_te,
        window_sid_idx_test=sid_te,
        # Metadata
        x_feature_names=np.array(x_feature_names, dtype=object),
        target_y=np.array(TARGET_Y, dtype=object),
        N_IN=np.array(N_IN),
        N_OUT=np.array(N_OUT)
    )

    # 12) Save preprocessors (These are correctly fit on train data)
    with open(PREPROCESSOR_X_PKL, "wb") as f:
        pickle.dump(preprocessor_x, f)
    with open(SCALER_Y_PKL, "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"[OK] Saved processed arrays to {PROCESSED_NPZ}")
    print(f"[OK] Saved X preprocessor (fit on train) to {PREPROCESSOR_X_PKL}")
    print(f"[OK] Saved Y scaler (fit on train) to {SCALER_Y_PKL}")