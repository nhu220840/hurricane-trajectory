# scripts/demo_5_single_prediction.py
# (PHIÊN BẢN SỬA LỖI - V5 - Chỉ dự đoán 1 điểm 11)
import torch
import pickle
import numpy as np
import folium
from pathlib import Path
import sys
import warnings

# Thêm thư mục gốc vào path để import src
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Import mọi thứ từ /src
from src.config import (
    PROCESSED_NPZ, SCALER_Y_PKL, PREPROCESSOR_X_PKL,
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH,
    N_IN, FEATURES_X, NUMERIC_X, CATEGORICAL_X, PLOTS_DIR,
    LSTM_TORCH, LSTM_SCRATCH
)
from src.models import LSTMForecaster, LSTMFromScratchForecaster
from src.train import _split_by_sid, _filter_by_sid_idx, SEED


# ===== CẤU HÌNH DEMO =====
# Chọn một mẫu bão từ tập TEST để demo (thay đổi số này để thử các cơn bão khác)
SAMPLE_INDEX_IN_TEST_SET = 100

OUT_HTML = PLOTS_DIR / f"demo_2_single_prediction_sample_{SAMPLE_INDEX_IN_TEST_SET}.html"
# ==========================


def get_model_and_checkpoint(choice, input_size, out_dim):
    """Tải đúng class model và checkpoint tương ứng."""
    if choice == "pytorch":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=LSTM_TORCH["hidden_size"],
            num_layers=LSTM_TORCH["num_layers"],
            dropout=LSTM_TORCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_TORCH
    elif choice == "scratch":
        model = LSTMFromScratchForecaster(
            in_dim=input_size,
            hidden=LSTM_SCRATCH["hidden_size"],
            num_layers=LSTM_SCRATCH["num_layers"],
            out_dim=out_dim,
            dropout=LSTM_SCRATCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_SCRATCH
    else:
        raise ValueError("Model choice không hợp lệ.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}. Bạn đã chạy 'python main.py --train' chưa?")

    return model, ckpt_path

def get_feature_indices(preprocessor, numeric_features):
    """Lấy vị trí (index) của các cột số (lat/lon)."""
    indices = {}
    for feat in ['lat', 'lon']:
        if feat in numeric_features:
            indices[feat] = numeric_features.index(feat)

    if 'lat' not in indices or 'lon' not in indices:
        raise ValueError("Không tìm thấy 'lat'/'lon' trong NUMERIC_X. Kiểm tra config.py.")

    print(f"Feature indices found (relative to NUMERIC block): {indices}")
    return indices


def main():
    print("Đang tải dữ liệu và scaler...")
    # 1. Tải dữ liệu và scalers
    if not PROCESSED_NPZ.exists():
        raise FileNotFoundError(f"Không tìm thấy {PROCESSED_NPZ}. Bạn đã chạy 'python main.py --process-data' chưa?")
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    X_all, Y_all, last_obs_all, sid_idx_all = data["X"], data["Y"], data["last_obs_latlon"], data["window_sid_idx"]
    with open(SCALER_Y_PKL, "rb") as f: scaler_y = pickle.load(f)
    with open(PREPROCESSOR_X_PKL, "rb") as f: preprocessor_x = pickle.load(f)

    # 2. Chia tập test
    _, _, test_sids = _split_by_sid(sid_idx_all, seed=SEED)
    X_test, m_te = _filter_by_sid_idx(X_all, sid_idx_all, test_sids)
    Y_test = Y_all[m_te]
    last_obs_test = last_obs_all[m_te]
    print(f"Tổng cộng có {len(X_test)} mẫu trong tập test.")

    # 3. Tải cả 2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_test.shape[-1]
    out_dim = Y_test.shape[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Model 1: PyTorch
        model_torch, ckpt_torch = get_model_and_checkpoint("pytorch", input_size, out_dim)
        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_torch.to(device).eval()
        print(f"Đã tải model 'pytorch' từ {ckpt_torch} lên {device}.")

        # Model 2: Scratch
        model_scratch, ckpt_scratch = get_model_and_checkpoint("scratch", input_size, out_dim)
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))
        model_scratch.to(device).eval()
        print(f"Đã tải model 'scratch' từ {ckpt_scratch} lên {device}.")

    # 4. Lấy dữ liệu Case Study
    # Lấy DUY NHẤT 1 cửa sổ 10 điểm (10 điểm ban đầu)
    input_window_scaled = X_test[SAMPLE_INDEX_IN_TEST_SET]

    # Lấy DUY NHẤT 1 điểm bắt đầu (điểm thứ 10)
    start_coord = last_obs_test[SAMPLE_INDEX_IN_TEST_SET]
    start_lat, start_lon = start_coord

    # Lấy DUY NHẤT 1 điểm "sự thật" (điểm thứ 11)
    true_coord_11th = last_obs_test[SAMPLE_INDEX_IN_TEST_SET + 1]

    print(f"Đã lấy Case Study: Sample {SAMPLE_INDEX_IN_TEST_SET} (Dùng 10 điểm, dự đoán điểm 11)")
    print(f"  Điểm bắt đầu (10): ({start_lat:.2f}, {start_lon:.2f})")
    print(f"  Điểm sự thật (11): ({true_coord_11th[0]:.2f}, {true_coord_11th[1]:.2f})")

    # 5. Chuẩn bị các thông số
    feat_indices = get_feature_indices(preprocessor_x, NUMERIC_X)
    num_scaler = preprocessor_x.named_transformers_['num']
    num_features_count = len(NUMERIC_X)

    # 6. CHẠY DỰ BÁO (1-bước-duy-nhất)
    input_tensor = torch.tensor(input_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Chạy 1-lần cho PyTorch
        pred_deltas_scaled_torch = model_torch(input_tensor)
        pred_deltas_deg_torch = scaler_y.inverse_transform(pred_deltas_scaled_torch.cpu().numpy())[0]

        # Chạy 1-lần cho Scratch
        pred_deltas_scaled_scratch = model_scratch(input_tensor)
        pred_deltas_deg_scratch = scaler_y.inverse_transform(pred_deltas_scaled_scratch.cpu().numpy())[0]

    # 7. Tái tạo các điểm dự đoán (điểm 11)
    pred_lat_torch = start_lat + pred_deltas_deg_torch[0]
    pred_lon_torch = start_lon + pred_deltas_deg_torch[1]
    pred_coord_torch = (pred_lat_torch, pred_lon_torch)

    pred_lat_scratch = start_lat + pred_deltas_deg_scratch[0]
    pred_lon_scratch = start_lon + pred_deltas_deg_scratch[1]
    pred_coord_scratch = (pred_lat_scratch, pred_lon_scratch)

    print(f"  Dự đoán PyTorch (11): ({pred_lat_torch:.2f}, {pred_lon_torch:.2f})")
    print(f"  Dự đoán Scratch (11): ({pred_lat_scratch:.2f}, {pred_lon_scratch:.2f})")

    # 8. Lấy 10 điểm lịch sử (để vẽ)
    history_scaled_nums = input_window_scaled[:, :num_features_count]
    history_unscaled_nums = num_scaler.inverse_transform(history_scaled_nums)
    idx_lat = feat_indices['lat']
    idx_lon = feat_indices['lon']
    history_coords = list(zip(history_unscaled_nums[:, idx_lat], history_unscaled_nums[:, idx_lon]))

    # 9. VẼ BẢN ĐỒ
    print("Đang vẽ bản đồ Folium dự đoán 1-điểm...")
    start_point = tuple(start_coord) # Điểm 10
    true_point_11th = tuple(true_coord_11th) # Điểm 11

    m = folium.Map(location=start_point, zoom_start=7, tiles="CartoDB positron")

    # 1. Đường lịch sử (màu xám)
    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="Lịch sử (10 điểm)").add_to(m)

    # 2. Điểm bắt đầu (điểm 10)
    folium.Marker(start_point,
                  tooltip="Điểm bắt đầu dự báo (Điểm 10)",
                  icon=folium.Icon(color="green", icon="play", prefix='fa')
                 ).add_to(m)

    # 3. Điểm "Sự thật" (Navy)
    folium.CircleMarker(true_point_11th, radius=7, color='navy', fill=True,
                        tooltip=f"Sự thật (Điểm 11)\n({true_point_11th[0]:.2f}, {true_point_11th[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, true_point_11th], color='navy', weight=2,
                    tooltip="Delta Thực tế").add_to(m)

    # 4. Điểm "Dự báo PyTorch" (Red)
    folium.CircleMarker(pred_coord_torch, radius=7, color='red', fill=True,
                        tooltip=f"Dự báo PyTorch (Điểm 11)\n({pred_coord_torch[0]:.2f}, {pred_coord_torch[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, pred_coord_torch], color='red', weight=2, dash_array="5, 5",
                    tooltip="Delta Dự đoán PyTorch").add_to(m)

    # 5. Điểm "Dự báo Scratch" (Orange)
    folium.CircleMarker(pred_coord_scratch, radius=7, color='orange', fill=True,
                        tooltip=f"Dự báo Scratch (Điểm 11)\n({pred_coord_scratch[0]:.2f}, {pred_coord_scratch[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, pred_coord_scratch], color='orange', weight=2, dash_array="5, 5",
                    tooltip="Delta Dự đoán Scratch").add_to(m)

    # 10. Lưu file
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\n[THÀNH CÔNG] Đã lưu bản đồ dự đoán 1-điểm tại:")
    print(f"{OUT_HTML.resolve()}")

if __name__ == "__main__":
    main()