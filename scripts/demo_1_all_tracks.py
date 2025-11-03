import pandas as pd
import folium
from pathlib import Path
import sys
import io
import random  # (MỚI) Import thư viện random
from folium.plugins import MarkerCluster

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
    PLOTS_DIR = REPO_ROOT / "results" / "plots"

# (MỚI) Đổi tên file output để phản ánh các thay đổi
OUT_HTML = PLOTS_DIR / "demo_1_all_tracks_colored_clustered.html"


def create_overview_map():
    print(f"Đang đọc dữ liệu từ: {RAW_CSV}")
    if not RAW_CSV.exists():
        print(f"LỖI: Không tìm thấy file {RAW_CSV}. Bạn đã chạy 'python main.py --prepare-raw-data' chưa?")
        return

    df = pd.read_csv(RAW_CSV)

    # Chuyển đổi các cột sang dạng số (giữ nguyên từ lần sửa trước)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['pres'] = pd.to_numeric(df['pres'], errors='coerce')

    print(f"Đã tải {len(df)} điểm dữ liệu của {df['sid'].nunique()} cơn bão.")

    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    # Tạo một lớp MarkerCluster và thêm vào bản đồ
    marker_cluster = MarkerCluster(name="Tất cả các điểm bão").add_to(m)

    # Gom nhóm theo SID (ID cơn bão)
    grouped = df.groupby('sid')

    print("Đang tạo các đường đi (theo màu) và các cụm điểm (có thể mất một lúc)...")
    for sid, group in grouped:

        # 1. (MỚI) TẠO MỘT MÀU NGẪU NHIÊN CHO CƠN BÃO NÀY
        # Tạo một chuỗi hex color ngẫu nhiên, ví dụ: '#A033FF'
        random_color = f"#{random.randint(0, 0xFFFFFF):06x}"

        # 2. Tạo danh sách tọa độ (Lat, Lon) cho đường PolyLine
        coords = list(zip(group['lat'], group['lon']))
        if not coords:
            continue

        # 3. (MỚI) Tạo bảng HTML chi tiết cho Popup (giống như trước)
        html_io = io.StringIO()
        html_io.write(f'<h4 style="margin:5px;">Storm SID: {sid}</h4>')
        html_io.write(f'<p style="margin:5px;">Total Points: {len(group)}</p>')
        # ... (Toàn bộ code tạo bảng HTML giữ nguyên) ...
        html_io.write('<div style="max-height: 200px; overflow-y: scroll; border: 1px solid #ccc; margin-top: 10px;">')
        html_io.write('<table style="width:100%; font-size: 11px; border-collapse: collapse;">')
        html_io.write('<thead style="position: sticky; top: 0; background-color: #f2f2f2;">')
        html_io.write('<tr>')
        html_io.write('<th style="padding: 4px; border: 1px solid #ddd;">Time</th>')
        html_io.write('<th style="padding: 4px; border: 1px solid #ddd;">Lat</th>')
        html_io.write('<th style="padding: 4px; border: 1px solid #ddd;">Lon</th>')
        html_io.write('<th style="padding: 4px; border: 1px solid #ddd;">Wind (kt)</th>')
        html_io.write('<th style="padding: 4px; border: 1px solid #ddd;">Pres (mb)</th>')
        html_io.write('</tr></thead>')
        html_io.write('<tbody>')
        for _, row in group.iterrows():
            time_str = row['time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['time']) else str(row['time'])
            wind_str = f'{row["wind"]:.0f}' if pd.notna(row["wind"]) else "N/A"
            pres_str = f'{row["pres"]:.0f}' if pd.notna(row["pres"]) else "N/A"
            html_io.write('<tr>')
            html_io.write(f'<td style="padding: 4px; border: 1px solid #ddd;">{time_str}</td>')
            html_io.write(f'<td style="padding: 4px; border: 1px solid #ddd;">{row["lat"]:.2f}</td>')
            html_io.write(f'<td style="padding: 4px; border: 1px solid #ddd;">{row["lon"]:.2f}</td>')
            html_io.write(f'<td style="padding: 4px; border: 1px solid #ddd;">{wind_str}</td>')
            html_io.write(f'<td style="padding: 4px; border: 1px solid #ddd;">{pres_str}</td>')
            html_io.write('</tr>')
        html_io.write('</tbody></table></div>')

        # Tạo IFrame và Popup cho Bảng
        iframe_table = folium.IFrame(html_io.getvalue(), width=400, height=280)
        popup_table = folium.Popup(iframe_table)

        # 4. (SỬA ĐỔI) Vẽ PolyLine
        folium.PolyLine(
            coords,
            color=random_color,  # <-- (MỚI) Sử dụng màu ngẫu nhiên
            weight=2.5,  # <-- (MỚI) Làm đường dày lên
            opacity=0.8,  # <-- (MỚI) Làm đường rõ nét hơn
            popup=popup_table,  # Gắn popup (bảng) vào đường line
            tooltip=f"Click for full table: {sid}"
        ).add_to(m)  # Thêm đường line vào bản đồ chính

        # 5. (SỬA ĐỔI) Lặp qua từng điểm và thêm vào MarkerCluster
        for _, row in group.iterrows():
            # Tạo popup cho *từng điểm*
            time_str = row['time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['time']) else str(row['time'])
            wind_str = f'{row["wind"]:.0f} kt' if pd.notna(row["wind"]) else "N/A"
            pres_str = f'{row["pres"]:.0f} mb' if pd.notna(row["pres"]) else "N/A"
            point_html = f"""
            <b>SID:</b> {sid}<br>
            <b>Thời gian:</b> {time_str}<br>
            <hr style="margin: 3px 0;">
            <b>Lat:</b> {row['lat']:.2f}, <b>Lon:</b> {row['lon']:.2f}<br>
            <b>Gió:</b> {wind_str}, <b>Áp suất:</b> {pres_str}
            """
            popup_point = folium.Popup(point_html, max_width=250)

            # Thêm dấu chấm (CircleMarker)
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=1,
                color=random_color,  # <-- (MỚI) Tô màu dấu chấm trùng với màu đường
                fill=True,
                fill_color=random_color,
                fill_opacity=0.7,
                popup=popup_point,  # Gắn popup chi tiết của điểm
                tooltip=f"{sid} ({time_str})"
            ).add_to(marker_cluster)  # Thêm vào CLUSTER

    # Thêm bộ điều khiển Lớp (Layer Control)
    folium.LayerControl().add_to(m)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\n[THÀNH CÔNG] Đã lưu bản đồ Cụm điểm (Clustered) tại:")
    print(f"{OUT_HTML.resolve()}")


if __name__ == "__main__":
    create_overview_map()