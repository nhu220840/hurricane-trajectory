# scripts/4_run_forecast.py
import numpy as np
import torch
import pickle
import folium
from pathlib import Path
from torch import nn

# ===== CẤU HÌNH =====
ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data/processed_data.npz"
SCALER_PKL = ROOT / "data/scaler.pkl"
CKPT_PATH = ROOT / "models/best_lstm_model.pt"
OUT_DIR = ROOT / "results/real_forecasts"
N_STEPS_TO_FORECAST = 24  # <-- Dự báo 24 giờ trong tương lai


# ========================

# Phải định nghĩa lại class model ở đây
class LSTMForecaster(nn.Module):
    def __init__(self, in_dim, hidden=20, num_layers=1, out_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        return self.head(last_h)


def forecast_iteratively(model, initial_window_scaled, n_steps, scaler, device, input_features):
    """
    Dự báo lặp lại n bước trong tương lai bằng cách sử dụng chính
    kết quả dự đoán của mô hình để làm đầu vào cho bước tiếp theo.
    """
    model.eval()
    predicted_points_abs = []
    current_window_tensor = torch.tensor(initial_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    idx = {name: i for i, name in enumerate(input_features)}

    with torch.no_grad():
        for _ in range(n_steps):
            predicted_delta = model(current_window_tensor).cpu().numpy().flatten()
            last_scaled_row = current_window_tensor.cpu().numpy().squeeze(0)[-1, :]
            last_unscaled_row = scaler.inverse_transform(last_scaled_row.reshape(1, -1)).flatten()
            new_unscaled_row = np.zeros_like(last_unscaled_row)
            new_unscaled_row[idx['lat']] = last_unscaled_row[idx['lat']] + predicted_delta[0]
            new_unscaled_row[idx['lon']] = last_unscaled_row[idx['lon']] + predicted_delta[1]
            new_unscaled_row[idx['wind']] = last_unscaled_row[idx['wind']]
            new_unscaled_row[idx['pres']] = last_unscaled_row[idx['pres']]
            new_unscaled_row[idx['lat_change']] = new_unscaled_row[idx['lat']] - last_unscaled_row[idx['lat']]
            new_unscaled_row[idx['lon_change']] = new_unscaled_row[idx['lon']] - last_unscaled_row[idx['lon']]
            new_unscaled_row[idx['wind_change']] = 0
            new_unscaled_row[idx['pres_change']] = 0
            predicted_points_abs.append((new_unscaled_row[idx['lat']], new_unscaled_row[idx['lon']]))
            new_scaled_row = scaler.transform(new_unscaled_row.reshape(1, -1))
            new_window_np = np.vstack([current_window_tensor.cpu().numpy().squeeze(0)[1:, :], new_scaled_row])
            current_window_tensor = torch.tensor(new_window_np, dtype=torch.float32).unsqueeze(0).to(device)

    return np.array(predicted_points_abs)


def reconstruct_path_from_deltas(start_lat, start_lon, deltas):
    path = []
    current_lat, current_lon = start_lat, start_lon
    for d_lat, d_lon in deltas:
        current_lat += d_lat
        current_lon += d_lon
        path.append((current_lat, current_lon))
    return path


def draw_forecast_map(history_coords, truth_coords, pred_coords, out_html):
    start_point = history_coords[-1]
    m = folium.Map(location=start_point, zoom_start=5, tiles="CartoDB positron")
    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="Lịch sử").add_to(m)
    folium.PolyLine([start_point] + truth_coords, color="navy", weight=4, tooltip="Thực tế (để so sánh)").add_to(m)
    folium.PolyLine([start_point] + pred_coords, color="red", weight=4, dash_array="10, 5", tooltip="Dự báo").add_to(m)
    folium.Marker(start_point, tooltip="Điểm bắt đầu dự báo",
                  icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)
    m.save(out_html)
    print(f"Đã lưu bản đồ dự báo tại: '{out_html}'")


def main():
    # --- SỬA LỖI: Thêm lại phần tải dữ liệu và khởi tạo model ---
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_test, y_test_delta = npz["X_test"], npz["y_test"]
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    INPUT_FEATURES = checkpoint["input_features"]

    model = LSTMForecaster(in_dim=len(INPUT_FEATURES), out_dim=y_test_delta.shape[2]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lat_idx = INPUT_FEATURES.index('lat')
    lon_idx = INPUT_FEATURES.index('lon')
    # --- KẾT THÚC SỬA LỖI ---

    test_indices_to_visualize = [0, 50, 150]

    print(
        f"\nBắt đầu chạy dự báo thực tế cho {len(test_indices_to_visualize)} mẫu test tại các chỉ số: {test_indices_to_visualize}...")

    for i, test_sample_index in enumerate(test_indices_to_visualize):
        print(f"\n--- Đang xử lý mẫu thử {i + 1} (chỉ số {test_sample_index}) ---")
        initial_window_scaled = X_test[test_sample_index]

        predicted_points_abs = forecast_iteratively(
            model, initial_window_scaled, N_STEPS_TO_FORECAST, scaler, device, INPUT_FEATURES
        )

        history_unscaled = scaler.inverse_transform(initial_window_scaled)
        history_coords = list(zip(history_unscaled[:, lat_idx], history_unscaled[:, lon_idx]))

        ground_truth_deltas_full = y_test_delta[test_sample_index: test_sample_index + N_STEPS_TO_FORECAST].squeeze(1)
        truth_coords = reconstruct_path_from_deltas(
            history_unscaled[-1, lat_idx],
            history_unscaled[-1, lon_idx],
            ground_truth_deltas_full
        )

        out_html = OUT_DIR / f"test_sample_index_{test_sample_index}_real_forecast.html"
        draw_forecast_map(history_coords, truth_coords,
                          list(zip(predicted_points_abs[:, 0], predicted_points_abs[:, 1])), out_html)


if __name__ == "__main__":
    main()