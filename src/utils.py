# src/utils.py
import folium
import branca
import pandas as pd
import numpy as np
from folium.plugins import AntPath


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách Haversine giữa 2 điểm (lat, lon) trên Trái Đất.
    Kết quả trả về bằng Kilômét (km).
    """
    R = 6371  # Bán kính Trái Đất (km)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = R * c
    return distance_km


def get_color(wind_speed):
    """Trả về màu sắc dựa trên tốc độ gió (thang Saffir-Simpson)."""
    if pd.isna(wind_speed): wind_speed = 0
    if wind_speed < 34: return '#008000'  # Green - Tropical Depression
    if wind_speed < 64: return '#FFFF00'  # Yellow - Tropical Storm
    if wind_speed < 83: return '#FFC0CB'  # Pink - Category 1
    if wind_speed < 96: return '#ADD8E6'  # Light Blue - Category 2
    if wind_speed < 113: return '#FFA500'  # Orange - Category 3
    if wind_speed < 137: return '#FF0000'  # Red - Category 4
    return '#800080'  # Purple - Category 5


def draw_advanced_forecast_map(history_df, truth_df, pred_df, out_html_path):
    """
    Vẽ bản đồ dự báo nâng cao, mô phỏng chính xác file Colab
    với các đường sai số và vòng tròn tại mỗi bước.
    """
    if history_df.empty:
        print("Cảnh báo: Không có dữ liệu lịch sử để vẽ.")
        return

    start_point = (history_df['LAT'].iloc[-1], history_df['LON'].iloc[-1])
    m = folium.Map(location=start_point, zoom_start=5, tiles="CartoDB positron")

    # --- 1. Vẽ Quỹ đạo Lịch sử (HISTORY) ---
    history_points = list(zip(history_df['LAT'], history_df['LON']))
    folium.PolyLine(history_points, color="gray", weight=3, tooltip="Lịch sử").add_to(m)

    # Vẽ các điểm tròn có màu theo sức gió cho History
    for _, row in history_df.iterrows():
        popup_html = f"""
        <b>Thời gian (Lịch sử):</b><br>
        <b>Vị trí:</b> ({row['LAT']:.2f}, {row['LON']:.2f})<br>
        <b>Sức gió:</b> {row.get('WMO_WIND', 'N/A')} knots
        """
        folium.CircleMarker(
            location=(row['LAT'], row['LON']),
            radius=3,
            color=get_color(row.get('WMO_WIND')),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    # --- 2. Chuẩn bị & Vẽ các Quỹ đạo chính (TRUTH & PREDICTION) ---
    # Nối điểm cuối của history vào đầu truth và pred
    truth_points = [start_point] + list(zip(truth_df['LAT'], truth_df['LON']))
    pred_points = [start_point] + list(zip(pred_df['LAT'], pred_df['LON']))

    # Vẽ đường Truth (Xanh Navy)
    folium.PolyLine(truth_points, color="navy", weight=4, tooltip="Quỹ đạo Thực tế").add_to(m)
    # Vẽ đường Prediction (Đỏ, AntPath)
    AntPath(pred_points, color="red", weight=4, dash_array="10, 5", tooltip="Quỹ đạo Dự báo").add_to(m)

    # --- 3. Vẽ Chi tiết Sai số (Step-by-Step) ---
    # Lặp qua từng bước dự báo
    for i in range(len(truth_df)):
        truth_loc = (truth_df.iloc[i]['LAT'], truth_df.iloc[i]['LON'])
        pred_loc = (pred_df.iloc[i]['LAT'], pred_df.iloc[i]['LON'])
        error_km = truth_df.iloc[i].get('error_km', 0)
        step = i + 1

        popup_html = f"<b>Bước dự báo: {step}</b><br>Sai số: {error_km:.2f} km"

        # Đánh dấu điểm Thực tế (Xanh)
        folium.CircleMarker(
            location=truth_loc,
            radius=3,
            color='navy',
            fill=True,
            tooltip=f"Thực tế (Bước {step})"
        ).add_to(m)

        # Đánh dấu điểm Dự báo (Đỏ)
        folium.CircleMarker(
            location=pred_loc,
            radius=3,
            color='red',
            fill=True,
            tooltip=f"Dự báo (Bước {step})"
        ).add_to(m)

        # Vẽ đường gạch nối biểu thị sai số (Đen)
        folium.PolyLine(
            locations=[truth_loc, pred_loc],
            color='black',
            weight=1,
            dash_array='5, 5',
            popup=popup_html,
            tooltip=f"Sai số (Bước {step}): {error_km:.2f} km"
        ).add_to(m)

        # Vẽ vòng tròn sai số (Tâm tại điểm DỰ BÁO)
        folium.Circle(
            location=pred_loc,
            radius=error_km * 1000,  # Chuyển km sang mét
            color='#00000000',  # Viền trong suốt
            fill_color='gray',
            fill_opacity=0.3,
            popup=popup_html
        ).add_to(m)

    # --- 4. Hoàn Thiện Bản Đồ ---

    # Đánh dấu điểm bắt đầu
    folium.Marker(
        start_point,
        tooltip="Điểm bắt đầu dự báo",
        icon=folium.Icon(color="green", icon="play", prefix='fa')
    ).add_to(m)

    # Tạo thang màu
    colormap = branca.colormap.StepColormap(
        colors=['#008000', '#FFFF00', '#FFC0CB', '#ADD8E6', '#FFA500', '#FF0000', '#800080'],
        index=[0, 34, 64, 83, 96, 113, 137],
        vmin=0, vmax=160,
        caption='Sức gió (knots) - Thang Saffir-Simpson'
    )
    m.add_child(colormap)

    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html_path))
    print(f"Đã lưu bản đồ (kiểu Colab) tại: '{out_html_path}'")