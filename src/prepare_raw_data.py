# src/prepare_raw_data.py

from pathlib import Path
import pandas as pd

from .config import RAW_DIR, RAW_CSV

# KEEP THESE FEATURES (mapping: original column -> standard name)
RAW_FEATURES_TO_KEEP = {
    'SID': 'sid',
    'ISO_TIME': 'time',
    'LAT': 'lat',
    'LON': 'lon',
    'WMO_WIND': 'wind',
    'WMO_PRES': 'pres',
    'DIST2LAND': 'dist2land',  # new feature
    'BASIN': 'basin'           # new feature (categorical)
}

def run_raw_data_preparation():
    """
    Step 1: From the large original file in data/raw (e.g., ibtracs.last3years.list.v04r01.csv),
    filter and rename columns to a concise ML format: sid, time, lat, lon, wind, pres, dist2land, basin.
    Result is written to data/raw/ibtracs_track_ml.csv
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Put the original IBTrACS filename here
    source_csv = RAW_DIR / "ibtracs.last3years.list.v04r01.csv"
    if not source_csv.exists():
        print(f"[WARN] {source_csv} not found. Skipping raw preparation step.")
        return

    df = pd.read_csv(source_csv, skiprows=[1])

    # Standardize column names to UPPER to match key mapping, then map to standard name (lower)
    col_upper_map = {c: c.upper() for c in df.columns}
    df = df.rename(columns=col_upper_map)

    keep_cols_upper = [c for c in RAW_FEATURES_TO_KEEP.keys() if c in df.columns]
    if not keep_cols_upper:
        raise ValueError("Could not match any columns in RAW_FEATURES_TO_KEEP with the original file.")

    df = df[keep_cols_upper].copy()
    df = df.rename(columns=RAW_FEATURES_TO_KEEP)

    # Basic data types
    # time remains a sortable string (ISO), basin is a string
    if "basin" in df.columns:
        df["basin"] = df["basin"].astype(str).fillna("UNK")

    # Sort approximately by sid, time (if time is standard ISO format)
    df = df.sort_values(["sid", "time"]).reset_index(drop=True)

    df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Wrote prepared raw CSV to {RAW_CSV}")