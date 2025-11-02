# scripts/demo_2_single_prediction.py
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

# (ĐÃ XÓA) from src.train import _split_by_sid, _filter_by_sid_idx, SEED

# ===== CẤU HÌNH DEMO =====
SAMPLE_INDEX_IN_TEST_SET = 100
OUT_HTML = PLOTS_DIR / f"demo_2_single_prediction_sample_{SAMPLE_INDEX_IN_TEST_SET}.html"


# ==========================

def get_model_and_checkpoint(choice, input_size, out_dim):
    # (Hàm này giữ nguyên)
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
    # (Hàm này giữ nguyên, nhưng cải tiến để an toàn hơn)
    try:
        num_transformer = preprocessor.named_transformers_['num']
        num_features_list = num_transformer.feature_names_in_
    except Exception:
        print("Cảnh báo: Không thể lấy feature_names_in_ từ transformer. Dùng NUMERIC_X.")
        num_features_list = numeric_features

    indices = {}
    for feat in ['lat', 'lon']:
        if feat in num_features_list:
            indices[feat] = list(num_features_list).index(feat)
    if 'lat' not in indices or 'lon' not in indices:
        raise ValueError("Không tìm thấy 'lat'/'lon' trong các đặc trưng số của preprocessor.")
    print(f"Indices đặc trưng (trong khối numeric): {indices}")
    return indices


def main():
    print("Đang tải dữ liệu, scalers, và models...")

    # 1. Tải dữ liệu và scalers
    try:
        data = np.load(PROCESSED_NPZ, allow_pickle=True)
        # (SỬA ĐỔI) Tải trực tiếp dữ liệu test đã chia sẵn
        X_test = data["X_test"]
        Y_test = data["Y_test"]
        last_obs_test = data["last_obs_test"]
        print(f"Đã tải {len(X_test)} mẫu từ tập test.")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Không tìm thấy file: {PROCESSED_NPZ}. Bạn đã chạy 'python main.py --process-data' chưa?")
    except KeyError as e:
        raise KeyError(f"Thiếu mảng {e} trong {PROCESSED_NPZ}. Vui lòng chạy lại --process-data.")

    with open(SCALER_Y_PKL, "rb") as f:
        scaler_y = pickle.load(f)
    with open(PREPROCESSOR_X_PKL, "rb") as f:
        preprocessor_x = pickle.load(f)

    # 2. (ĐÃ XÓA) Logic chia dữ liệu (không cần nữa)

    # 3. Tải cả 2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_test.shape[-1]
    out_dim = Y_test.shape[-1]
    print(f"Sử dụng thiết bị: {device}. Input features: {input_size}, Output features: {out_dim}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model_torch, ckpt_torch = get_model_and_checkpoint("pytorch", input_size, out_dim)
        model_torch.load_state_dict(torch.load(ckpt_torch, map_location=device, weights_only=True))
        model_torch.to(device).eval()
        print(f"Đã tải model 'pytorch' từ {ckpt_torch}.")

        model_scratch, ckpt_scratch = get_model_and_checkpoint("scratch", input_size, out_dim)
        model_scratch.load_state_dict(torch.load(ckpt_scratch, map_location=device, weights_only=True))
        model_scratch.to(device).eval()
        print(f"Đã tải model 'scratch' từ {ckpt_scratch}.")

    # 4. Lấy dữ liệu Case Study
    if SAMPLE_INDEX_IN_TEST_SET >= len(X_test):
        print(
            f"LỖI: SAMPLE_INDEX_IN_TEST_SET ({SAMPLE_INDEX_IN_TEST_SET}) vượt quá giới hạn (tối đa {len(X_test) - 1}).")
        print("Sử dụng mẫu 0.")
        SAMPLE_INDEX_IN_TEST_SET = 0

    input_window_scaled = X_test[SAMPLE_INDEX_IN_TEST_SET]
    start_coord = last_obs_test[SAMPLE_INDEX_IN_TEST_SET]
    start_lat, start_lon = start_coord

    if SAMPLE_INDEX_IN_TEST_SET + 1 < len(last_obs_test):
        true_coord_11th = last_obs_test[SAMPLE_INDEX_IN_TEST_SET + 1]
    else:
        print(f"Cảnh báo: Đây là mẫu cuối cùng. Không thể lấy 'sự thật' cho điểm 11.")
        true_coord_11th = start_coord

    print(f"Đã tải Case Study: Sample {SAMPLE_INDEX_IN_TEST_SET}")
    print(f"  Điểm bắt đầu (10): ({start_lat:.2f}, {start_lon:.2f})")
    print(f"  Điểm sự thật (11): ({true_coord_11th[0]:.2f}, {true_coord_11th[1]:.2f})")

    # 5. Chuẩn bị thông số
    feat_indices = get_feature_indices(preprocessor_x, NUMERIC_X)
    num_scaler = preprocessor_x.named_transformers_['num']
    num_features_count = num_scaler.n_features_in_

    # 6. CHẠY DỰ BÁO
    input_tensor = torch.tensor(input_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_deltas_scaled_torch = model_torch(input_tensor)
        pred_deltas_deg_torch = scaler_y.inverse_transform(pred_deltas_scaled_torch.cpu().numpy())[0]
        pred_deltas_scaled_scratch = model_scratch(input_tensor)
        pred_deltas_deg_scratch = scaler_y.inverse_transform(pred_deltas_scaled_scratch.cpu().numpy())[0]

    # 7. Tái tạo tọa độ
    pred_lat_torch = start_lat + pred_deltas_deg_torch[0]
    pred_lon_torch = start_lon + pred_deltas_deg_torch[1]
    pred_coord_torch = (pred_lat_torch, pred_lon_torch)
    pred_lat_scratch = start_lat + pred_deltas_deg_scratch[0]
    pred_lon_scratch = start_lon + pred_deltas_deg_scratch[1]
    pred_coord_scratch = (pred_lat_scratch, pred_lon_scratch)

    print(f"  Dự đoán PyTorch (11): ({pred_lat_torch:.2f}, {pred_lon_torch:.2f})")
    print(f"  Dự đoán Scratch (11): ({pred_lat_scratch:.2f}, {pred_lon_scratch:.2f})")

    # 8. Lấy 10 điểm lịch sử
    history_scaled_nums = input_window_scaled[:, :num_features_count]
    history_unscaled_nums = num_scaler.inverse_transform(history_scaled_nums)
    idx_lat = feat_indices['lat']
    idx_lon = feat_indices['lon']
    history_coords = list(zip(history_unscaled_nums[:, idx_lat], history_unscaled_nums[:, idx_lon]))

    # 9. VẼ BẢN ĐỒ
    print("Đang vẽ bản đồ Folium...")
    start_point = tuple(start_coord)
    true_point_11th = tuple(true_coord_11th)

    m = folium.Map(location=start_point, zoom_start=7, tiles="CartoDB positron")
    # ... (Toàn bộ code vẽ Folium giữ nguyên) ...
    folium.PolyLine(history_coords, color="gray", weight=3, tooltip="Lịch sử (10 điểm)").add_to(m)
    folium.Marker(start_point, tooltip="Điểm bắt đầu (10)",
                  icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)
    folium.CircleMarker(true_point_11th, radius=7, color='navy', fill=True,
                        tooltip=f"Sự thật (11)\n({true_point_11th[0]:.2f}, {true_point_11th[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, true_point_11th], color='navy', weight=2, tooltip="Delta Thực tế").add_to(m)
    folium.CircleMarker(pred_coord_torch, radius=7, color='red', fill=True,
                        tooltip=f"PyTorch (11)\n({pred_coord_torch[0]:.2f}, {pred_coord_torch[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, pred_coord_torch], color='red', weight=2, dash_array="5, 5",
                    tooltip="Delta PyTorch").add_to(m)
    folium.CircleMarker(pred_coord_scratch, radius=7, color='orange', fill=True,
                        tooltip=f"Scratch (11)\n({pred_coord_scratch[0]:.2f}, {pred_coord_scratch[1]:.2f})").add_to(m)
    folium.PolyLine([start_point, pred_coord_scratch], color='orange', weight=2, dash_array="5, 5",
                    tooltip="Delta Scratch").add_to(m)

    # 10. Lưu file
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\n[THÀNH CÔNG] Đã lưu bản đồ dự đoán 1-điểm tại:")
    print(f"{OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()