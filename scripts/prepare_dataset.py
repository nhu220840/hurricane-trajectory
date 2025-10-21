# scripts/prepare_dataset.py
import pandas as pd
from pathlib import Path

# ----- CẤU HÌNH -----
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "data" / "raw" / "ibtracs.last3years.list.v04r01.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "raw" / "ibtracs_track_ml.csv"

# Giữ lại nhiều cột hơn cho feature engineering
COLUMNS_TO_KEEP = [
    'SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME', 'ISO_TIME',
    'NATURE', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'LANDFALL'
]

def prepare_raw_data():
    """
    Đọc file dữ liệu thô, chọn các cột cần thiết cho feature engineering nâng cao,
    và lưu ra một file CSV mới.
    """
    print(f"Bắt đầu đọc file dữ liệu lớn tại: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, skiprows=[1], low_memory=False)
        print("Đọc file thành công.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{INPUT_CSV}'.")
        return

    # Chỉ giữ lại các cột đã định nghĩa
    df_filtered = df[COLUMNS_TO_KEEP].copy()
    print(f"Đã giữ lại {len(df_filtered.columns)} cột cần thiết.")

    # Xử lý giá trị thiếu (IBTrACS dùng khoảng trắng)
    df_filtered = df_filtered.replace(r'^\s*$', pd.NA, regex=True)
    df_filtered.dropna(subset=['LAT', 'LON', 'ISO_TIME'], inplace=True)
    print(f"Số dòng dữ liệu sau khi làm sạch cơ bản: {len(df_filtered)}")

    # Lưu ra file CSV mới
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Đã lưu thành công file dữ liệu đã được cắt gọn tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    prepare_raw_data()