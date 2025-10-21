# src/forecast.py
import numpy as np
import torch
import pickle
import pandas as pd
from src import config
from src.model import LSTMAttention
from src.utils import draw_advanced_forecast_map, haversine_distance


def calculate_time_features(dt_object):
    """Tính toán các đặc trưng sin/cos từ một đối tượng datetime."""
    month = dt_object.month
    day = dt_object.day
    hour = dt_object.hour
    return {
        'sin_month': np.sin(2 * np.pi * month / 12),
        'cos_month': np.cos(2 * np.pi * month / 12),
        'sin_day': np.sin(2 * np.pi * day / 31),
        'cos_day': np.cos(2 * np.pi * day / 31),
        'sin_hour': np.sin(2 * np.pi * hour / 24),
        'cos_hour': np.cos(2 * np.pi * hour / 24),
    }


def forecast_iteratively(model, initial_window_scaled, last_known_datetime, n_steps, scaler_X, scaler_y, device,
                         input_features):
    """
    Dự báo lặp n bước, sử dụng 2 scaler (X và y) và cập nhật động các đặc trưng.
    Đây chính là logic dự báo tiên tiến.
    """
    model.eval()
    predicted_points_abs = []  # Lưu (lat, lon)

    current_window_tensor = torch.tensor(initial_window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    current_window_df_scaled = pd.DataFrame(initial_window_scaled, columns=input_features)

    with torch.no_grad():
        for i in range(n_steps):
            # 1. Dự báo 5 giá trị DELTA (đã scale)
            predicted_deltas_scaled = model(current_window_tensor).cpu().numpy()  # Shape (1, 5)

            # 2. GIẢI CHUẨN HÓA (unscale) các DELTA này bằng scaler_y
            predicted_deltas_unscaled = scaler_y.inverse_transform(predicted_deltas_scaled).flatten()  # Shape (5,)

            # 3. Lấy dòng cuối cùng (đã scale) từ cửa sổ
            last_row_scaled = current_window_df_scaled.iloc[-1].values.reshape(1, -1)  # Shape (1, 22)

            # 4. GIẢI CHUẨN HÓA (unscale) dòng cuối cùng bằng scaler_X
            last_row_unscaled_array = scaler_X.inverse_transform(last_row_scaled).flatten()  # Shape (22,)
            last_row_unscaled_df = pd.DataFrame([last_row_unscaled_array], columns=input_features)

            # 5. Tạo dòng feature mới (chưa scale) bằng cách CỘNG DELTA (unscaled + unscaled)
            new_row_unscaled_dict = {}
            d_lat, d_lon, d_wind, d_pres, d_dist = predicted_deltas_unscaled

            # 5a. Cập nhật các giá trị vật lý
            new_row_unscaled_dict['LAT'] = last_row_unscaled_df['LAT'].iloc[0] + d_lat
            new_row_unscaled_dict['LON'] = last_row_unscaled_df['LON'].iloc[0] + d_lon
            new_row_unscaled_dict['WMO_WIND'] = max(0, last_row_unscaled_df['WMO_WIND'].iloc[0] + d_wind)
            new_row_unscaled_dict['WMO_PRES'] = last_row_unscaled_df['WMO_PRES'].iloc[0] + d_pres
            new_row_unscaled_dict['DIST2LAND'] = max(0, last_row_unscaled_df['DIST2LAND'].iloc[0] + d_dist)

            predicted_points_abs.append((new_row_unscaled_dict['LAT'], new_row_unscaled_dict['LON']))

            # 5b. Tính toán lại các đặc trưng THỜI GIAN (Giải quyết "đóng băng")
            current_dt = last_known_datetime + pd.Timedelta(hours=config.TIME_STEP_HOURS * (i + 1))
            new_time_features = calculate_time_features(current_dt)
            new_row_unscaled_dict.update(new_time_features)

            # 5c. Giữ nguyên các đặc trưng CATEGORICAL (Tĩnh)
            for col in input_features:
                if any(cat_feat in col for cat_feat in config.CATEGORICAL_FEATURES):
                    new_row_unscaled_dict[col] = last_row_unscaled_df[col].iloc[0]

            # 6. Chuyển dict thành DataFrame đúng thứ tự 22 cột
            new_row_unscaled_df = pd.DataFrame([new_row_unscaled_dict])[input_features]

            # 7. CHUẨN HÓA (scale) lại TOÀN BỘ dòng feature mới bằng scaler_X
            new_row_scaled_array = scaler_X.transform(new_row_unscaled_df)[0]

            # 8. Cập nhật cửa sổ (dạng DataFrame đã scale)
            new_window_df_scaled = pd.DataFrame(np.vstack([
                current_window_df_scaled.iloc[1:].values,
                new_row_scaled_array
            ]), columns=input_features)

            # 9. Cập nhật tensor cho vòng lặp tiếp theo
            current_window_tensor = torch.tensor(new_window_df_scaled.values, dtype=torch.float32).unsqueeze(0).to(
                device)
            current_window_df_scaled = new_window_df_scaled

    return np.array(predicted_points_abs)


def reconstruct_path_from_deltas(start_lat, start_lon, deltas_array):
    """Tái tạo quỹ đạo từ mảng các giá trị delta (unscaled)."""
    path = []
    current_lat, current_lon = start_lat, start_lon

    for delta_step in deltas_array:
        d_lat = delta_step[0]
        d_lon = delta_step[1]

        current_lat += d_lat
        current_lon += d_lon
        path.append((current_lat, current_lon))
    return path


def run_forecasting():
    """Hàm chính: Tải 2 scaler và chạy dự báo động."""
    npz = np.load(config.PROCESSED_DATA_NPZ, allow_pickle=True)
    X_test, y_test_scaled = npz["X_test"], npz["y_test"]
    X_test_last_dt = npz["X_test_last_dt"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])

    # Tải 2 scaler
    with open(config.SCALER_X_PKL, "rb") as f:
        scaler_X = pickle.load(f)
    with open(config.SCALER_Y_PKL, "rb") as f:
        scaler_y = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.MODEL_CKPT_PATH, map_location=device)
    model_params = checkpoint["model_params"]

    model = LSTMAttention(
        in_dim=X_test.shape[2],
        out_dim=len(TARGET_FEATURES),
        **model_params
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"\nBắt đầu chạy dự báo động cho {config.N_STEPS_TO_FORECAST} bước (tương đương {config.N_STEPS_TO_FORECAST * config.TIME_STEP_HOURS} giờ)...")

    for i, test_idx in enumerate(config.TEST_INDICES_TO_VISUALIZE):
        print(f"\n--- Đang xử lý mẫu thử {i + 1} (chỉ số {test_idx}) ---")
        initial_window_scaled = X_test[test_idx]
        last_known_datetime = X_test_last_dt[test_idx]

        # 1. Lấy LỊCH SỬ (unscaled)
        history_unscaled_array = scaler_X.inverse_transform(initial_window_scaled)
        history_df_unscaled = pd.DataFrame(history_unscaled_array, columns=INPUT_FEATURES)

        # 2. Chạy DỰ BÁO
        predicted_points_abs = forecast_iteratively(
            model, initial_window_scaled, last_known_datetime,
            config.N_STEPS_TO_FORECAST, scaler_X, scaler_y, device, INPUT_FEATURES
        )
        pred_df = pd.DataFrame(predicted_points_abs, columns=['LAT', 'LON'])
        pred_df['WMO_WIND'] = 0

        # 3. Lấy THỰC TẾ (ground truth)
        ground_truth_scaled_deltas = y_test_scaled[test_idx: test_idx + config.N_STEPS_TO_FORECAST]
        ground_truth_unscaled_deltas = scaler_y.inverse_transform(ground_truth_scaled_deltas)

        truth_path = reconstruct_path_from_deltas(
            history_df_unscaled['LAT'].iloc[-1],
            history_df_unscaled['LON'].iloc[-1],
            ground_truth_unscaled_deltas
        )
        truth_df = pd.DataFrame(truth_path, columns=['LAT', 'LON'])
        truth_wind = np.cumsum(ground_truth_unscaled_deltas[:, 2]) + history_df_unscaled['WMO_WIND'].iloc[-1]
        truth_df['WMO_WIND'] = np.maximum(0, truth_wind)

        # 4. Tính toán sai số
        if len(truth_df) == len(pred_df):
            errors = []
            for j in range(len(truth_df)):
                err = haversine_distance(
                    truth_df.iloc[j]['LAT'], truth_df.iloc[j]['LON'],
                    pred_df.iloc[j]['LAT'], pred_df.iloc[j]['LON']
                )
                errors.append(err)
            truth_df['error_km'] = errors
            print(f"   -> Sai số trung bình {np.mean(errors):.2f} km, Sai số cuối cùng {errors[-1]:.2f} km")
        else:
            truth_df['error_km'] = 0

        out_html = config.RESULT_DIR / f"test_sample_index_{test_idx}_delta_forecast_8steps.html"
        draw_advanced_forecast_map(history_df_unscaled, truth_df, pred_df, out_html)