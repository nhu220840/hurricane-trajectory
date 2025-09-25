# scripts/visualize_folium.py
import numpy as np
import torch
import pickle
import folium
from pathlib import Path
# --- SỬA LỖI 1: Import đúng class model ---
from train_model import RNNForecaster  # Import RNNForecaster thay vì LSTMForecaster

# ===== CẤU HÌNH MỚI =====
ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data" / "processed_splits_delta.npz"
SCALER_PKL = ROOT / "data" / "scaler_delta.pkl"
# --- SỬA LỖI 2: Trỏ đến đúng file checkpoint của RNN ---
CKPT = ROOT / "models" / "best_rnn_delta.pt"
OUT_DIR = ROOT / "folium_maps"  # Đổi tên thư mục output cho rõ ràng


# ========================

def inverse_transform_features(scaled_features, scaler):
    """Chuyển đổi một mảng numpy đã scale về thang đo gốc."""
    return scaler.inverse_transform(scaled_features)


def predict_iteratively(model, initial_window_scaled, n_steps_to_predict, scaler, device):
    """Dự đoán lặp lại n bước trong tương lai."""
    model.eval()
    predicted_points_abs = []

    # --- SỬA LỖI 3: Dùng model.rnn thay vì model.lstm ---
    last_known_features_scaled = initial_window_scaled[-1, :]
    last_known_features_unscaled = inverse_transform_features(
        last_known_features_scaled.reshape(1, -1), scaler
    )
    last_lat = last_known_features_unscaled[0, 0]
    last_lon = last_known_features_unscaled[0, 1]

    current_window_scaled = torch.tensor(initial_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(n_steps_to_predict):
            predicted_delta = model(current_window_scaled).cpu().numpy().flatten()
            new_lat = last_lat + predicted_delta[0]
            new_lon = last_lon + predicted_delta[1]
            predicted_points_abs.append((new_lat, new_lon))

            new_features_unscaled = last_known_features_unscaled.copy()
            new_features_unscaled[0, 0] = new_lat
            new_features_unscaled[0, 1] = new_lon

            new_features_scaled = scaler.transform(new_features_unscaled)

            new_window_np = np.vstack([current_window_scaled.cpu().numpy().squeeze(0)[1:, :], new_features_scaled])
            current_window_scaled = torch.tensor(new_window_np, dtype=torch.float32).unsqueeze(0).to(device)

            last_lat, last_lon = new_lat, new_lon

    return np.array(predicted_points_abs)


def draw_map(history_unscaled, ground_truth_unscaled, prediction_abs, out_html):
    lat_h, lon_h = history_unscaled[:, 0], history_unscaled[:, 1]
    lat_t, lon_t = ground_truth_unscaled[:, 0], ground_truth_unscaled[:, 1]
    lat_p, lon_p = prediction_abs[:, 0], prediction_abs[:, 1]

    lat0, lon0 = lat_h[-1], lon_h[-1]
    m = folium.Map(location=[lat0, lon0], zoom_start=5, tiles="CartoDB positron")

    folium.PolyLine(list(zip(lat_h, lon_h)), color="gray", weight=3, opacity=0.8, tooltip="Lịch sử").add_to(m)
    full_actual_path = list(zip(lat_h[-1:], lon_h[-1:])) + list(zip(lat_t, lon_t))
    folium.PolyLine(full_actual_path, color="navy", weight=4, opacity=0.9, tooltip="Thực tế").add_to(m)
    full_pred_path = list(zip(lat_h[-1:], lon_h[-1:])) + list(zip(lat_p, lon_p))
    folium.PolyLine(full_pred_path, color="red", weight=4, tooltip="Dự đoán", dash_array="10, 5").add_to(m)
    folium.Marker([lat0, lon0], tooltip="Điểm bắt đầu dự báo",
                  icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)

    m.save(out_html)
    print("Đã lưu bản đồ tại:", out_html)


def main():
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_test, y_test_delta = npz["X_test"], npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sử dụng weights_only=True để an toàn hơn
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)

    # --- SỬA LỖI 4: Khởi tạo đúng class model ---
    model = RNNForecaster(in_dim=len(INPUT_FEATURES), out_dim=y_test_delta.shape[2]).to(device)
    model.load_state_dict(ckpt["model"])

    OUT_DIR.mkdir(exist_ok=True)

    for i in range(3):
        initial_window_scaled = X_test[i]
        predicted_points_abs = predict_iteratively(model, initial_window_scaled, n_steps_to_predict=8, scaler=scaler,
                                                   device=device)

        history_unscaled = inverse_transform_features(initial_window_scaled, scaler)

        ground_truth_abs = []
        last_lat, last_lon = history_unscaled[-1, 0], history_unscaled[-1, 1]
        for step in range(y_test_delta[i].shape[0]):
            delta = y_test_delta[i][step]
            new_lat = last_lat + delta[0]
            new_lon = last_lon + delta[1]
            ground_truth_abs.append((new_lat, new_lon))
            last_lat, last_lon = new_lat, new_lon
        ground_truth_abs = np.array(ground_truth_abs)

        out_html = OUT_DIR / f"test_sample_{i}.html"
        draw_map(history_unscaled, ground_truth_abs, predicted_points_abs, out_html)


if __name__ == "__main__":
    main()