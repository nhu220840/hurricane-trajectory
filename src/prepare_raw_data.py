# src/prepare_raw_data.py

from pathlib import Path
import pandas as pd

from .config import RAW_DIR, RAW_CSV

# GIỮ NGUYÊN CÁC FEATURE NÀY (mapping: cột gốc -> tên chuẩn)
RAW_FEATURES_TO_KEEP = {
    'SID': 'sid',
    'ISO_TIME': 'time',
    'LAT': 'lat',
    'LON': 'lon',
    'WMO_WIND': 'wind',
    'WMO_PRES': 'pres',
    'DIST2LAND': 'dist2land',  # feature mới
    'BASIN': 'basin'           # feature mới (categorical)
}

def run_raw_data_preparation():
    """
    Bước 1: Từ file lớn gốc trong data/raw (ví dụ: ibtracs.last3years.list.v04r01.csv),
    lọc và đổi tên cột về format ML gọn: sid, time, lat, lon, wind, pres, dist2land, basin.
    Kết quả ghi ra data/raw/ibtracs_track_ml.csv
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Bạn đặt tên file gốc IBTrACS ở đây
    source_csv = RAW_DIR / "ibtracs.last3years.list.v04r01.csv"
    if not source_csv.exists():
        print(f"[WARN] Không tìm thấy {source_csv}. Bỏ qua bước chuẩn bị raw.")
        return

    df = pd.read_csv(source_csv)

    # Chuẩn hoá tên cột về UPPER để khớp key mapping, rồi map về tên chuẩn (lower)
    col_upper_map = {c: c.upper() for c in df.columns}
    df = df.rename(columns=col_upper_map)

    keep_cols_upper = [c for c in RAW_FEATURES_TO_KEEP.keys() if c in df.columns]
    if not keep_cols_upper:
        raise ValueError("Không khớp được cột nào trong RAW_FEATURES_TO_KEEP với file gốc.")

    df = df[keep_cols_upper].copy()
    df = df.rename(columns=RAW_FEATURES_TO_KEEP)

    # Kiểu dữ liệu cơ bản
    # time giữ dạng string sortable (ISO), basin là string
    if "basin" in df.columns:
        df["basin"] = df["basin"].astype(str).fillna("UNK")

    # Sắp xếp tương đối theo sid, time (nếu time có format ISO chuẩn)
    df = df.sort_values(["sid", "time"]).reset_index(drop=True)

    df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Wrote prepared raw CSV to {RAW_CSV}")
