# scripts/3_evaluate_and_visualize.py
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
OUT_DIR = ROOT / "folium_maps"
N_STEPS_TO_TEST = 8  # Số bước muốn kiểm tra


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


def reconstruct_path_from_deltas(start_lat, start_lon, deltas):
    path = []
    current_lat, current_lon = start_lat, start_lon
    for d_lat, d_lon in deltas:
        current_lat += d_lat
        current_lon += d_lon
        path.append((current_lat, current_lon))
    return path


def draw_comparison_map(history_coords, truth_path, pred_path, out_html):
    start_point = history_coords[-1]
    m = folium.Map(location=start_point, zoom_start=7, tiles="CartoDB positron")

    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="Lịch sử").add_to(m)

    folium.PolyLine([start_point] + truth_path, color="navy", weight=5,
                    tooltip=f"Thực tế ({len(truth_path)} bước)").add_to(m)
    for i, p in enumerate(truth_path):
        folium.CircleMarker(location=p, radius=5, color='navy', fill=True, tooltip=f"Thực tế - Bước {i + 1}").add_to(m)

    folium.PolyLine([start_point] + pred_path, color="red", weight=3, dash_array="10, 5",
                    tooltip=f"Dự đoán ({len(pred_path)} bước)").add_to(m)
    for i, p in enumerate(pred_path):
        folium.CircleMarker(location=p, radius=5, color='red', fill=True, tooltip=f"Dự đoán - Bước {i + 1}").add_to(m)

    m.save(out_html)
    print(f"\nĐã lưu bản đồ so sánh tại: '{out_html}'")


def main():
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_test, y_test = npz["X_test"], npz["y_test"]
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    INPUT_FEATURES = checkpoint["input_features"]

    model = LSTMForecaster(in_dim=X_test.shape[2], out_dim=y_test.shape[2]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_sample_index = 0
    initial_window_scaled = X_test[test_sample_index]
    ground_truth_deltas = y_test[test_sample_index: test_sample_index + N_STEPS_TO_TEST].squeeze(1)

    predicted_deltas = []
    current_window_tensor = torch.tensor(initial_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"--- BẮT ĐẦU DỰ ĐOÁN TỪNG BƯỚC ({N_STEPS_TO_TEST} BƯỚC) ---")

    with torch.no_grad():
        for i in range(N_STEPS_TO_TEST):
            pred_delta = model(current_window_tensor).cpu().numpy().flatten()
            predicted_deltas.append(pred_delta)

            actual_delta = ground_truth_deltas[i]
            print(f"\n--- Bước {i + 1} ---")
            print(f"  Dự đoán:  ∆lat={pred_delta[0]:.4f}, ∆lon={pred_delta[1]:.4f}")
            print(f"  Thực tế:    ∆lat={actual_delta[0]:.4f}, ∆lon={actual_delta[1]:.4f}")

            if i < N_STEPS_TO_TEST - 1:
                next_actual_point_scaled = X_test[test_sample_index + i + 1][-1, :]
                new_window_np = np.vstack(
                    [current_window_tensor.cpu().numpy().squeeze(0)[1:, :], next_actual_point_scaled])
                current_window_tensor = torch.tensor(new_window_np, dtype=torch.float32).unsqueeze(0).to(device)

    history_unscaled = scaler.inverse_transform(initial_window_scaled)
    lat_idx = INPUT_FEATURES.index('lat')
    lon_idx = INPUT_FEATURES.index('lon')
    last_lat, last_lon = history_unscaled[-1, lat_idx], history_unscaled[-1, lon_idx]

    history_coords = list(zip(history_unscaled[:, lat_idx], history_unscaled[:, lon_idx]))
    truth_path = reconstruct_path_from_deltas(last_lat, last_lon, ground_truth_deltas)
    pred_path = reconstruct_path_from_deltas(last_lat, last_lon, np.array(predicted_deltas))

    out_html = OUT_DIR / f"test_sample_{test_sample_index}_step_by_step_{N_STEPS_TO_TEST}_steps.html"
    draw_comparison_map(history_coords, truth_path, pred_path, out_html)


if __name__ == "__main__":
    main()