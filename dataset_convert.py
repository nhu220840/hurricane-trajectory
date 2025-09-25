import pandas as pd
from pathlib import Path

# Input / output paths
src = Path("ibtracs.last3years.list.v04r01.csv")   # đổi đường dẫn nếu cần
dst = Path("data/ibtracs_track_ml_2.csv")

# Read CSV (IBTrACS đôi khi có dòng đầu chứa 'units', sẽ lọc sau)
df = pd.read_csv(src, low_memory=False)

# Danh sách cột cần giữ cho bài toán dự báo quỹ đạo
desired_cols = [
    # Định danh & thời gian
    "SID", "SEASON", "NUMBER", "ISO_TIME",
    # Ngữ cảnh (hữu ích cho phân tích/feature engineering)
    "BASIN", "SUBBASIN", "NAME",
    # Mục tiêu vị trí
    "LAT", "LON",
    # Cường độ/cấu trúc (giúp học động lực di chuyển)
    "WMO_WIND", "WMO_PRES",
    "USA_WIND", "USA_PRES", "USA_SSHS",
    # Động học di chuyển
    "STORM_SPEED", "STORM_DIR",
    # Cờ/trạng thái hữu ích
    "NATURE", "USA_STATUS", "DIST2LAND"
]

# Chỉ giữ các cột thực sự có trong file
keep = [c for c in desired_cols if c in df.columns]

# Nếu dòng đầu là dòng "units" (ví dụ ISO_TIME == 'Year' hoặc LAT = 'degrees_north'), bỏ nó đi
if len(df) and (str(df.iloc[0].get("ISO_TIME","")).lower() == "year" or str(df.iloc[0].get("LAT","")).startswith("degrees_")):
    df = df.iloc[1:].reset_index(drop=True)

# Chọn cột
out = df[keep].copy()

# Ép kiểu số cho các cột kỳ vọng là numeric
numeric_cols = [
    "LAT","LON","WMO_WIND","WMO_PRES","USA_WIND","USA_PRES","USA_SSHS",
    "STORM_SPEED","STORM_DIR","DIST2LAND","SEASON","NUMBER"
]
for c in numeric_cols:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

# Parse thời gian
if "ISO_TIME" in out.columns:
    out["ISO_TIME"] = pd.to_datetime(out["ISO_TIME"], errors="coerce")

# Ghi file rút gọn
out.to_csv(dst, index=False)

print("Saved:", dst, "with shape:", out.shape)
print("Columns:", list(out.columns))
