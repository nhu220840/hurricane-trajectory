# scripts/demo_1_all_tracks.py
import pandas as pd
import folium
from pathlib import Path
import sys

# Thêm thư mục gốc vào path để import src.config
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Import đường dẫn từ config
try:
    from src.config import RAW_CSV, PLOTS_DIR
except ImportError:
    print("Lỗi: Không thể import src.config. Đảm bảo bạn chạy script từ thư mục gốc hoặc src/ nằm trong PYTHONPATH.")
    # Fallback nếu không import được
    RAW_CSV = REPO_ROOT / "data" / "raw" / "ibtracs_track_ml.csv"
    PLOTS_DIR = REPO_ROOT / "results" / "plots" # Sửa 'results' nếu config của bạn khác

OUT_HTML = PLOTS_DIR / "demo_1_all_tracks_overview.html"

def create_overview_map():
    print(f"Đang đọc dữ liệu từ: {RAW_CSV}")
    if not RAW_CSV.exists():
        print(f"LỖI: Không tìm thấy file {RAW_CSV}. Bạn đã chạy 'python main.py --prepare-raw-data' chưa?")
        return

    df = pd.read_csv(RAW_CSV)
    print(f"Đã tải {len(df)} điểm dữ liệu của {df['sid'].nunique()} cơn bão.")

    # Lấy một vị trí trung tâm để bắt đầu bản đồ
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    # Gom nhóm theo SID (ID cơn bão)
    grouped = df.groupby('sid')

    # Lặp qua một số cơn bão để vẽ (ví dụ: 500 cơn bão đầu tiên cho nhanh)
    for sid, group in list(grouped)[:500]:
        coords = list(zip(group['lat'], group['lon']))
        folium.PolyLine(
            coords,
            color='navy',
            weight=1,
            opacity=0.3
        ).add_to(m)

    # Tạo thư mục output nếu chưa có
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\n[THÀNH CÔNG] Đã lưu bản đồ 'Big Picture' tại:")
    print(f"{OUT_HTML.resolve()}")

if __name__ == "__main__":
    create_overview_map()